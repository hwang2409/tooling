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
  it("ignores cache_control markers moving between requests (block top level)", () => {
    const earlier = normalizeMessage(user("hello", { cache_control: { type: "ephemeral" } }));
    const later = normalizeMessage(user("hello"));
    expect(messageMatches(earlier, later)).toBe(true);
    expect(messagesArePrefix([earlier], [later, normalizeMessage(assistant("hi"))])).toBe(true);
  });

  it("does NOT strip cache_control nested inside data such as tool inputs", () => {
    // A tool input legitimately containing a `cache_control` key is data;
    // collapsing it would fabricate false prefixes across differing calls.
    const withKey = normalizeMessage({
      role: "assistant",
      content: [{ type: "tool_use", name: "Bash", input: { command: "ls", cache_control: "keep-me" } }],
    } as JsonValue);
    const withoutKey = normalizeMessage({
      role: "assistant",
      content: [{ type: "tool_use", name: "Bash", input: { command: "ls" } }],
    } as JsonValue);
    expect(messageMatches(withKey, withoutKey)).toBe(false);
    expect(messageMatches(withoutKey, withKey)).toBe(false);
  });

  it("expands string content shorthand before comparing", () => {
    const shorthand = normalizeMessage({ role: "user", content: "hello" } as JsonValue);
    const expanded = normalizeMessage(user("hello"));
    expect(messageMatches(shorthand, expanded)).toBe(true);
  });

  it("tolerates appended <system-reminder> text blocks only", () => {
    const earlier = normalizeMessage(toolResult("t1", "ok"));
    const appended = (text: string) => normalizeMessage({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: "ok" },
        { type: "text", text },
      ],
    } as JsonValue);
    expect(messageMatches(earlier, appended("<system-reminder>injected</system-reminder>"))).toBe(true);
    expect(messageMatches(earlier, appended(
      "<system-reminder>one</system-reminder>\n<system-reminder>two</system-reminder>",
    ))).toBe(true);
    // Ordinary appended user text is a REAL edit, not an injection.
    expect(messageMatches(earlier, appended("please also check the logs"))).toBe(false);
  });

  it("rejects unbalanced nested reminder markup, accepts balanced nesting", () => {
    const earlier = normalizeMessage(toolResult("t1", "ok"));
    const appended = (text: string) => normalizeMessage({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: "ok" },
        { type: "text", text },
      ],
    } as JsonValue);
    // Reviewer probe: inner close must not satisfy the unclosed outer open.
    expect(messageMatches(earlier, appended(
      "<system-reminder>outer <system-reminder>inner</system-reminder>",
    ))).toBe(false);
    // Fully balanced nesting is complete markup.
    expect(messageMatches(earlier, appended(
      "<system-reminder>outer <system-reminder>inner</system-reminder> tail</system-reminder>",
    ))).toBe(true);
  });

  it("rejects incomplete or padded reminder markup as injection", () => {
    const earlier = normalizeMessage(toolResult("t1", "ok"));
    const appended = (text: string) => normalizeMessage({
      role: "user",
      content: [
        { type: "tool_result", tool_use_id: "t1", content: "ok" },
        { type: "text", text },
      ],
    } as JsonValue);
    // Unclosed reminder.
    expect(messageMatches(earlier, appended("<system-reminder>never closed"))).toBe(false);
    // Malformed markup.
    expect(messageMatches(earlier, appended("<system-reminder foo>x</system-reminder>"))).toBe(false);
    // Trailing text after the close tag.
    expect(messageMatches(earlier, appended(
      "<system-reminder>real</system-reminder> and my actual question",
    ))).toBe(false);
    // Leading text before the open tag.
    expect(messageMatches(earlier, appended(
      "my question <system-reminder>real</system-reminder>",
    ))).toBe(false);
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

  it("prefix dominance regression: an established chain beats a NEWER shared-root branch", () => {
    // Round-1 property, restored: within one shared-root component the chain
    // with more confirming members wins even though the auxiliary branch has
    // the most recent tip. The retired newest-tip-only policy selected `aux`.
    const root = candidate("root", [u1], 5);
    const main1 = candidate("main-1", [u1, a1], 3);
    const main2 = candidate("main-2", [u1, a1, chainEnd], 2);
    const aux = candidate("aux", [u1, assistant("side quest")], 0);
    const selection = selectCanonicalFlow([aux, main2, main1, root]);
    expect(selection.canonicalId).toBe("main-2");
    expect(selection.chainIds).toEqual(["root", "main-1", "main-2"]);
  });

  it("shared-root regression: an older equal-length branch must not steal the root from the newer thread", () => {
    // The retired greedy partition consumed the shared root into whichever
    // equal-length branch sorted first (the OLDER one), making that chain
    // dominant by member count. Non-consuming chains give the root to both
    // branches; the newer tip wins.
    const utilityOlder = candidate("utility-older", [u1, assistant("checking quota"), user("what quota remains?")], 2);
    const mainNewest = candidate("main-newest", [u1, a1, chainEnd], 0);
    const root = candidate("root", [u1], 4);
    const selection = selectCanonicalFlow([utilityOlder, mainNewest, root]);
    expect(selection.canonicalId).toBe("main-newest");
    expect(selection.chainIds).toEqual(["root", "main-newest"]);
  });

  it("compaction regression: a shorter post-compaction restart beats the stale longer chain", () => {
    // The retired member-count dominance kept rendering the pre-compaction
    // chain (3 members) after the client restarted with a compacted history.
    const stale1 = candidate("r1", [u1], 5);
    const stale2 = candidate("r2", [u1, a1, chainEnd], 4);
    const stale3 = candidate("r3", [u1, a1, chainEnd, assistant("more"), user("go on")], 3);
    const restartRoot = candidate("s1", [user("compacted summary of prior work")], 1);
    const restartTip = candidate("s2", [user("compacted summary of prior work"), assistant("resuming")], 0);
    const selection = selectCanonicalFlow([stale1, stale2, stale3, restartRoot, restartTip]);
    expect(selection.canonicalId).toBe("s2");
    expect(selection.chainIds).toEqual(["s1", "s2"]);
  });

  it("duplicate retransmits do not inflate branch evidence: distinct stages decide", () => {
    // The auxiliary branch has three MEMBERS (root + two identical
    // retransmits) but only two distinct history stages; the main chain has
    // three stages. Raw member counting would tie 3-3 and hand the win to
    // the newer auxiliary tip.
    const root = candidate("root", [u1], 6);
    const main1 = candidate("main-1", [u1, a1], 4);
    const main2 = candidate("main-2", [u1, a1, chainEnd], 3);
    const auxOld = candidate("aux-old", [u1, assistant("side quest")], 2);
    const auxNew = candidate("aux-new", [u1, assistant("side quest")], 0);
    const selection = selectCanonicalFlow([root, main1, main2, auxOld, auxNew]);
    expect(selection.canonicalId).toBe("main-2");
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

  it("breaks tie between divergent tips toward the newest and returns null with no candidates", () => {
    const older = candidate("older", [user("a")], 1);
    const newer = candidate("newer", [user("b")], 0);
    expect(selectCanonicalFlow([older, newer]).canonicalId).toBe("newer");
    expect(selectCanonicalFlow([]).canonicalId).toBeNull();
    expect(selectCanonicalFlow([candidate("s", [u1], 0, true)]).canonicalId).toBeNull();
  });
});
