import { validateWithZod } from "@cline/shared";
import {
	type EditFileInput,
	INPUT_ARG_CHAR_LIMIT,
	type ReadFileRequest,
	type StructuredCommandInput,
	StructuredCommandsInputUnionSchema,
} from "./schemas";
import type { ToolOperationResult } from "./types";

/**
 * Format an error into a string message
 */
export function formatError(error: unknown): string {
	if (error instanceof Error) {
		return error.message;
	}
	return String(error);
}

export function getEditorSizeError(input: EditFileInput): string | null {
	if (
		typeof input.old_text === "string" &&
		input.old_text.length > INPUT_ARG_CHAR_LIMIT
	) {
		return `Editor input too large: old_text was ${input.old_text.length} characters, exceeding the recommended limit of ${INPUT_ARG_CHAR_LIMIT}. Split the edit into smaller tool calls so later tool calls are less likely to be truncated or time out.`;
	}

	if (input.new_text.length > INPUT_ARG_CHAR_LIMIT) {
		return `Editor input too large: new_text was ${input.new_text.length} characters, exceeding the recommended limit of ${INPUT_ARG_CHAR_LIMIT}. Split the edit into smaller tool calls so later tool calls are less likely to be truncated or time out.`;
	}

	return null;
}

/**
 * Create a timeout-wrapped promise
 */
export class TimeoutError extends Error {
	readonly timeoutMs: number;

	constructor(message: string, timeoutMs: number) {
		super(message);
		this.name = "TimeoutError";
		this.timeoutMs = timeoutMs;
	}
}

export function withTimeout<T>(
	promise: Promise<T>,
	ms: number,
	message: string,
): Promise<T> {
	return Promise.race([
		promise,
		new Promise<never>((_, reject) => {
			setTimeout(() => reject(new TimeoutError(message, ms)), ms);
		}),
	]);
}

export function formatReadFileQuery(request: ReadFileRequest): string {
	const { path, start_line, end_line } = request;
	if (start_line == null && end_line == null) {
		return path;
	}
	const start = start_line ?? 1;
	const end = end_line ?? "EOF";
	return `${path}:${start}-${end}`;
}

export function getReadFileRangeError(request: ReadFileRequest): string | null {
	const { start_line, end_line } = request;
	if (start_line == null || end_line == null || start_line <= end_line) {
		return null;
	}

	return `start_line must be less than or equal to end_line (received start_line: ${start_line}, end_line: ${end_line})`;
}

export function normalizeRunCommandsInput(
	input: unknown,
): Array<string | StructuredCommandInput> {
	const validate = validateWithZod(StructuredCommandsInputUnionSchema, input);

	if (typeof validate === "string") {
		return [validate];
	}

	if (Array.isArray(validate)) {
		return validate;
	}

	if ("commands" in validate) {
		return Array.isArray(validate.commands)
			? validate.commands
			: [validate.commands];
	}

	if ("command" in validate) {
		return "args" in validate ? [validate] : [validate.command];
	}

	if ("cmd" in validate) {
		return [validate.cmd];
	}

	return [validate];
}

export function formatRunCommandQuery(
	command: string | StructuredCommandInput,
): string {
	if (typeof command === "string") {
		return command;
	}

	const args = command.args ?? [];
	if (args.length === 0) {
		return command.command;
	}

	const renderedArgs = args.map((arg) =>
		/[\s"]/u.test(arg) ? JSON.stringify(arg) : arg,
	);
	return `${command.command} ${renderedArgs.join(" ")}`;
}

/**
 * Max characters of the executed command echoed back in the tool result's
 * `query` field. The full command already exists in the assistant tool-call
 * input, so repeating it in the result only duplicates tokens in the
 * provider request (expensive for large heredoc/file-generation commands).
 */
export const RUN_COMMAND_QUERY_PREVIEW_LIMIT = 200;

/**
 * Bound the command echo placed in a provider-facing tool result.
 * Short commands pass through unchanged; long commands keep a short
 * prefix plus a truncation note so the result is still identifiable.
 */
export function formatRunCommandQueryPreview(
	command: string | StructuredCommandInput,
): string {
	const rendered = formatRunCommandQuery(command);
	if (rendered.length <= RUN_COMMAND_QUERY_PREVIEW_LIMIT) {
		return rendered;
	}
	const truncatedChars = rendered.length - RUN_COMMAND_QUERY_PREVIEW_LIMIT;
	return `${rendered.slice(0, RUN_COMMAND_QUERY_PREVIEW_LIMIT)} ... [command truncated: ${truncatedChars} more chars; full command is in the tool call input]`;
}

/**
 * Measure how many characters an entry contributes to the conversation.
 * Structured results (e.g. read_files image reads return arrays of
 * text/image blocks) are walked recursively so base64 image payloads and
 * nested text count too — a non-string result is content, not free.
 */
function measureToolResultChars(value: unknown): number {
	if (typeof value === "string") {
		return value.length;
	}
	if (Array.isArray(value)) {
		let total = 0;
		for (const item of value) {
			total += measureToolResultChars(item);
		}
		return total;
	}
	if (value !== null && typeof value === "object") {
		let total = 0;
		for (const item of Object.values(value)) {
			total += measureToolResultChars(item);
		}
		return total;
	}
	return 0;
}

/**
 * Enforce a combined output budget across all entries of one batched tool
 * call. Per-entry caps bound each command/file/query individually, but a
 * single call can batch many entries that are each under their own cap and
 * still sum to hundreds of kilobytes. Entries are charged against the
 * budget in order; once an entry does not fit, it and every later entry
 * are replaced with a placeholder that preserves the entry's identity
 * (query, success/error status) and says how to re-fetch the content.
 *
 * The switch is one-way on purpose: back-filling smaller later entries
 * would make what the model sees depend on accidental ordering, and the
 * predictable rule is the one the model can learn from.
 *
 * Placeholders are always emitted in full, even when the budget is already
 * exactly consumed — a clipped or empty placeholder would be
 * indistinguishable from a legitimately empty result and would swallow
 * error messages. The overshoot is bounded by entries × placeholder length,
 * which is noise next to the content the budget exists to bound.
 */
export function applyAggregateToolOutputBudget(
	entries: ToolOperationResult[],
	maxChars: number,
	makePlaceholder: (entry: ToolOperationResult, omittedChars: number) => string,
): ToolOperationResult[] {
	let remaining = Math.max(0, maxChars);
	let exhausted = false;

	return entries.map((entry) => {
		const size =
			measureToolResultChars(entry.result) +
			(typeof entry.error === "string" ? entry.error.length : 0);
		if (!exhausted && size <= remaining) {
			remaining -= size;
			return entry;
		}

		exhausted = true;
		const placeholder = makePlaceholder(entry, size);
		if (entry.error !== undefined) {
			return { ...entry, result: "", error: placeholder };
		}
		return { ...entry, result: placeholder };
	});
}
