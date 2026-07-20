import { describe, expect, it } from "vitest";

import type { JsonValue } from "../inspector/jsonTree";
import type { CanonicalCandidate } from "./canonical";
import { messageMatches, messagesArePrefix, normalizeMessage, selectCanonicalFlow } from "./canonical";

const user = (text: string, extra: Record<string, JsonValue> = {}) =>
  ({ role: "user", content: [{ type: "text", text, ...extra }] }) as JsonValue;
const assistant = (text: string) => ({ role: "assistant", content: [{ type: "text", text }] }) as JsonValue;
const toolResult = (id: string, text: string) =>
  ({ role: "user", content: [{ type: "tool_result", tool_use_id: id, content: text }] }) as JsonValue;

function candidate(flowId: string, messages: readonly JsonValue[], order: number, suggestion = false): CanonicalCandidate {
  return { flowId, messages, suggestion, order };
}

describe("message normalization and prefix matching", () => {
  it("ignores cache_control markers moving between requests", () => {
    const earlier = normalizeMessage(user("hello", { cache_control: { type: "ephemeral" } }));
    const later = normalizeMessage(user("hello"));
    expect(messageMatches(earlier, later)).toBe(true);
    expect(messagesArePrefix([earlier], [later, normalizeMessage(assistant("hi"))])).toBe(true);
  });

  it("expands string content shorthand before comparing", () => {
    const shorthand = normalizeMessage({ role: "user", content: "hello" } as JsonValue);
    const expanded = normalizeMessage(user("hello"));
    expect(messageMatches(shorthand, expanded)).toBe(true);
  });

  it("tolerates system-reminder text blocks appended to an earlier message in a later request", () => {
    const earlier = normalizeMessage(toolResult("t1", "ok"));
    const later = normalizeMessage({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: "ok" },
        { type: "text", text: "<system-reminder>injected</system-reminder>" },
      ],
    } as JsonValue);
    expect(messageMatches(earlier, later)).toBe(true);
  });

  it("rejects diverging turns: appended non-text blocks or different content", () => {
    const earlier = normalizeMessage(user("hello"));
    expect(messageMatches(earlier, normalizeMessage(user("goodbye")))).toBe(false);
    const gainedToolUse = normalizeMessage({
      role: "user",
      content: [{ type: "text", text: "hello" }, { type: "tool_use", name: "Bash", input: {} }],
    } as JsonValue);
    expect(messageMatches(earlier, gainedToolUse)).toBe(false);
  });
});

describe("selectCanonicalFlow", () => {
  const u1 = user("fix the bug");
  const a1 = assistant("looking at it");
  const chainEnd = toolResult("t1", "tests pass");

  it("prefix-dedupe regression: a newer equal-count off-chain branch must not beat the dominant chain", () => {
    // Old heuristic (message_count desc, then recency) picks `branch`: it is
    // newer than `main` with the same count and carries no suggestion marker.
    // The prefix chains disagree: {old-main -> main} has two members while
    // the diverging branch is a singleton, so `main` is canonical.
    const branch = candidate("branch", [u1, assistant("different reply"), user("what quota remains?")], 0);
    const main = candidate("main", [u1, a1, chainEnd], 1);
    const oldMain = candidate("old-main", [u1], 2);
    const selection = selectCanonicalFlow([branch, main, oldMain]);
    expect(selection.canonicalId).toBe("main");
    expect(selection.chainIds).toEqual(["old-main", "main"]);
  });

  it("never selects a suggestion-mode request even when it is the newest and longest", () => {
    const suggestion = candidate("suggestion", [u1, a1, chainEnd, user("[SUGGESTION MODE: propose]")], 0, true);
    const main = candidate("main", [u1, a1, chainEnd], 1);
    const oldMain = candidate("old-main", [u1], 2);
    expect(selectCanonicalFlow([suggestion, main, oldMain]).canonicalId).toBe("main");
  });

  it("chains across cache_control movement and reminder injection", () => {
    const oldMain = candidate("old-main", [user("fix the bug", { cache_control: { type: "ephemeral" } })], 1);
    const main = candidate("main", [
      { role: "user", content: [{ type: "text", text: "fix the bug" }, { type: "text", text: "<system-reminder>x</system-reminder>" }] } as JsonValue,
      a1,
    ], 0);
    const selection = selectCanonicalFlow([main, oldMain]);
    expect(selection.canonicalId).toBe("main");
    expect(selection.chainIds).toEqual(["old-main", "main"]);
  });

  it("breaks singleton ties toward the newest flow and returns null with no candidates", () => {
    const older = candidate("older", [user("a")], 1);
    const newer = candidate("newer", [user("b")], 0);
    expect(selectCanonicalFlow([older, newer]).canonicalId).toBe("newer");
    expect(selectCanonicalFlow([]).canonicalId).toBeNull();
    expect(selectCanonicalFlow([candidate("s", [u1], 0, true)]).canonicalId).toBeNull();
  });
});
