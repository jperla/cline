import type { AgentToolContext } from "@cline/shared";
import { describe, expect, it } from "vitest";
import { createBashExecutor } from "./bash";

const ctx: AgentToolContext = {
	agentId: "agent-1",
	conversationId: "conv-1",
	iteration: 1,
};

describe("createBashExecutor", () => {
	it("runs a simple command and returns stdout", async () => {
		const bash = createBashExecutor();
		const output = await bash("echo hello", process.cwd(), ctx);
		expect(output.trim()).toBe("hello");
	});

	it("rejects on non-zero exit code", async () => {
		const bash = createBashExecutor();
		await expect(bash("exit 1", process.cwd(), ctx)).rejects.toThrow();
	});

	it("includes stderr in combined output on success", async () => {
		const bash = createBashExecutor({ combineOutput: true });
		const output = await bash(
			{
				command: process.execPath,
				args: [
					"-e",
					"process.stdout.write('ok'); process.stderr.write('warn')",
				],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toContain("ok");
		expect(output).toContain("[stderr]");
		expect(output).toContain("warn");
	});

	it("excludes stderr when combineOutput is false", async () => {
		const bash = createBashExecutor({ combineOutput: false });
		const output = await bash(
			{
				command: process.execPath,
				args: [
					"-e",
					"process.stdout.write('ok'); process.stderr.write('warn')",
				],
			},
			process.cwd(),
			ctx,
		);
		expect(output.trim()).toBe("ok");
	});

	it("rejects on timeout", async () => {
		const bash = createBashExecutor({ timeoutMs: 50 });
		await expect(bash("sleep 10", process.cwd(), ctx)).rejects.toThrow(
			"timed out",
		);
	});

	it("middle-truncates output exceeding maxOutputBytes, keeping head and tail", async () => {
		const bash = createBashExecutor({ maxOutputBytes: 20 });
		const output = await bash(
			{
				command: process.execPath,
				args: ["-e", "process.stdout.write('HEAD' + 'x'.repeat(100) + 'TAIL')"],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toContain("HEAD");
		expect(output).toContain("TAIL");
		expect(output).toContain("[... output truncated: 108 chars total");
		expect(output.length).toBeLessThan(300);
	});

	it("keeps default-capped output bounded with the notice in the preserved head/tail", async () => {
		// Provider-request building (session/services/message-builder.ts)
		// may middle-cut long tool-result strings again with its own
		// backstop. The executor keeps its truncation notice in the head and
		// tail halves, so the recovery guidance survives any such cut.
		const bash = createBashExecutor();
		const output = await bash(
			{
				command: process.execPath,
				args: ["-e", "process.stdout.write('x'.repeat(60_000))"],
			},
			process.cwd(),
			ctx,
		);
		expect(output.length).toBeLessThanOrEqual(50_000);
		expect(output).toContain("output truncated: 60000 chars total");
	});

	it("does not truncate output within maxOutputBytes", async () => {
		const bash = createBashExecutor({ maxOutputBytes: 1000 });
		const payload = "b".repeat(500);
		const output = await bash(
			{
				command: process.execPath,
				args: ["-e", `process.stdout.write('${payload}')`],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toBe(payload);
	});

	it("marks truncation in the error when a failing command floods stderr", async () => {
		const bash = createBashExecutor({ maxOutputBytes: 20 });
		await expect(
			bash(
				{
					command: process.execPath,
					args: [
						"-e",
						"process.stderr.write('ERR' + 'x'.repeat(100) + 'TAIL'); process.exit(1)",
					],
				},
				process.cwd(),
				ctx,
			),
		).rejects.toThrow("output truncated");
	});

	it("keeps the tail of streamed output written in many chunks", async () => {
		const bash = createBashExecutor({ maxOutputBytes: 40 });
		const output = await bash(
			{
				command: process.execPath,
				args: [
					"-e",
					"for (let i = 0; i < 50; i++) process.stdout.write('line' + i + '\\n'); process.stdout.write('FINAL')",
				],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toContain("line0");
		expect(output).toContain("FINAL");
		expect(output).toContain("output truncated");
	});

	it("rejects when abort signal fires", async () => {
		const ac = new AbortController();
		const abortCtx: AgentToolContext = { ...ctx, signal: ac.signal };
		const bash = createBashExecutor();

		setTimeout(() => ac.abort(), 50);
		await expect(bash("sleep 10", process.cwd(), abortCtx)).rejects.toThrow(
			"aborted",
		);
	});

	it("flushes a trailing incomplete multibyte sequence instead of dropping it", async () => {
		const bash = createBashExecutor();
		// Output ends with the first byte of a two-byte UTF-8 sequence; the
		// decoder must flush it at end-of-stream (as U+FFFD) rather than
		// silently dropping buffered bytes.
		const output = await bash(
			{
				command: process.execPath,
				args: ["-e", "process.stdout.write(Buffer.from([0x61, 0x62, 0xc3]))"],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toHaveLength(3);
		expect(output.startsWith("ab")).toBe(true);
	});

	it("collapses carriage-return progress rewrites to the final frame", async () => {
		const bash = createBashExecutor();
		const output = await bash(
			{
				command: process.execPath,
				args: [
					"-e",
					String.raw`process.stdout.write('0%\r50%\r100% done\nfinished')`,
				],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toBe("100% done\nfinished");
	});

	it("appends the notice without duplicating output when CR collapse shrinks dropped output under the cap", async () => {
		// 'a%\r' x 200 overflows the 100-char rolling buffer (so the stream
		// genuinely dropped its middle), but collapses to just 'done'.
		// Slicing that short remainder would emit it twice.
		const bash = createBashExecutor({ maxOutputChars: 100 });
		const output = await bash(
			{
				command: process.execPath,
				args: [
					"-e",
					String.raw`process.stdout.write('a%\r'.repeat(200) + 'done')`,
				],
			},
			process.cwd(),
			ctx,
		);
		expect(output.startsWith("done")).toBe(true);
		expect(output).toContain("output truncated: 604 chars total");
		expect(output.match(/done/g)).toHaveLength(1);
	});

	it("honors maxOutputChars and the deprecated maxOutputBytes alias", async () => {
		const emit = {
			command: process.execPath,
			args: ["-e", "process.stdout.write('x'.repeat(500))"],
		};
		const renamed = await createBashExecutor({ maxOutputChars: 100 })(
			emit,
			process.cwd(),
			ctx,
		);
		const alias = await createBashExecutor({ maxOutputBytes: 100 })(
			emit,
			process.cwd(),
			ctx,
		);
		expect(renamed).toContain("output truncated: 500 chars total");
		expect(alias).toContain("output truncated: 500 chars total");
	});
});

describe.runIf(process.platform === "win32")("createWindowsExecutor", () => {
	it("runs structured commands without shell parsing", async () => {
		const executor = createBashExecutor();
		const output = await executor(
			{
				command: process.execPath,
				args: ["-e", "process.stdout.write(process.argv[1])", "argv-ok"],
			},
			process.cwd(),
			ctx,
		);
		expect(output).toBe("argv-ok");
	});

	it("runs string commands through the shell", async () => {
		const executor = createBashExecutor();
		const output = await executor("echo shell-ok", process.cwd(), ctx);
		expect(output.trim()).toBe("shell-ok");
	});
});
