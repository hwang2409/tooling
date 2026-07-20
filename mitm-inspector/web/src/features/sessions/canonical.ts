import type { JsonValue } from "../inspector/jsonTree";

/**
 * Canonical main-thread selection over fetched request bodies.
 *
 * Every main-thread request resends the full conversation history, so
 * consecutive main-thread requests form prefix chains once two benign — and
 * ONLY two — mutations are normalized away:
 *   - `cache_control` markers move between requests (stripped at the one
 *     place Anthropic puts them: the top level of a content block, never
 *     inside tool inputs or other nested data), and
 *   - earlier messages can gain appended `<system-reminder>` text blocks in
 *     later requests (any other appended block breaks the prefix).
 *
 * Chains are built non-consuming: a request extends every chain whose
 * messages are a prefix of its own, so a shared root belongs to all of its
 * branches rather than being stolen by whichever branch sorts first. The
 * conversation renders the tip of the maximal chain whose tip is most recent
 * by created order (grid order): a live branch beats a stale sibling, and a
 * post-compaction restart (shorter history, newer tip) beats the stale
 * pre-compaction chain. Suggestion-mode requests are excluded up front: they
 * duplicate the main history with an injected prompt and must never become
 * the rendered conversation.
 */

export interface CanonicalCandidate {
  readonly flowId: string;
  /** Raw `messages` array from the fetched request body. */
  readonly messages: readonly JsonValue[];
  /** Body-verified suggestion-mode request (see isSuggestionRequest). */
  readonly suggestion: boolean;
  /** Created order: grid position, 0 = newest. */
  readonly order: number;
}

export interface CanonicalSelection {
  readonly canonicalId: string | null;
  /** Members of the selected prefix chain, oldest (shortest) first. */
  readonly chainIds: readonly string[];
}

type JsonObject = { [key: string]: JsonValue };

function asObject(value: JsonValue | undefined): JsonObject | null {
  if (value === null || value === undefined || typeof value !== "object" || Array.isArray(value)) return null;
  return value;
}

/** Strip cache_control at a content block's top level only — nowhere else. */
function normalizeBlock(block: JsonValue): JsonValue {
  const object = asObject(block);
  if (object === null || !("cache_control" in object)) return block;
  const copy = { ...object };
  delete copy.cache_control;
  return copy;
}

/** Expand string content shorthand and normalize each content block. */
export function normalizeMessage(message: JsonValue): JsonValue {
  const object = asObject(message);
  if (object === null) return message;
  if (typeof object.content === "string") {
    return { ...object, content: [{ type: "text", text: object.content }] };
  }
  if (Array.isArray(object.content)) {
    return { ...object, content: object.content.map(normalizeBlock) };
  }
  return object;
}

function deepEqual(left: JsonValue, right: JsonValue): boolean {
  if (left === right) return true;
  if (left === null || right === null || typeof left !== "object" || typeof right !== "object") return false;
  const leftArray = Array.isArray(left);
  if (leftArray !== Array.isArray(right)) return false;
  if (leftArray) {
    const rightArray = right as readonly JsonValue[];
    if ((left as readonly JsonValue[]).length !== rightArray.length) return false;
    return (left as readonly JsonValue[]).every((item, index) => deepEqual(item, rightArray[index]));
  }
  const leftKeys = Object.keys(left);
  const rightObject = right as JsonObject;
  if (leftKeys.length !== Object.keys(rightObject).length) return false;
  return leftKeys.every((key) => key in rightObject && deepEqual((left as JsonObject)[key], rightObject[key]));
}

/** Exact shape of a harness-injected reminder: a text block starting with the marker. */
function isReminderInjection(value: JsonValue): boolean {
  const object = asObject(value);
  return object !== null && object.type === "text"
    && typeof object.text === "string" && object.text.startsWith("<system-reminder>");
}

/**
 * Whether an earlier request's message is the same turn as a later request's
 * message at the same index: equal after normalization, or the later version
 * gained appended system-reminder text blocks. Ordinary appended content —
 * user text, tool blocks, anything without the reminder marker — breaks the
 * match.
 */
export function messageMatches(earlier: JsonValue, later: JsonValue): boolean {
  if (deepEqual(earlier, later)) return true;
  const earlierObject = asObject(earlier);
  const laterObject = asObject(later);
  if (earlierObject === null || laterObject === null) return false;
  if (earlierObject.role !== laterObject.role) return false;
  const earlierContent = earlierObject.content;
  const laterContent = laterObject.content;
  if (!Array.isArray(earlierContent) || !Array.isArray(laterContent)) return false;
  if (laterContent.length < earlierContent.length) return false;
  if (!earlierContent.every((block, index) => deepEqual(block, laterContent[index]))) return false;
  if (!laterContent.slice(earlierContent.length).every(isReminderInjection)) return false;
  const rest = (object: JsonObject) => {
    const copy = { ...object };
    delete copy.content;
    return copy;
  };
  return deepEqual(rest(earlierObject), rest(laterObject));
}

/** Whether `earlier`'s normalized messages are a prefix of `later`'s. */
export function messagesArePrefix(earlier: readonly JsonValue[], later: readonly JsonValue[]): boolean {
  if (earlier.length > later.length) return false;
  return earlier.every((message, index) => messageMatches(message, later[index]));
}

interface Entry {
  readonly candidate: CanonicalCandidate;
  readonly normalized: readonly JsonValue[];
}

/** Whether `other` extends `entry`: strictly longer, or an equal-length newer retransmit. */
function supersedes(entry: Entry, other: Entry): boolean {
  if (!messagesArePrefix(entry.normalized, other.normalized)) return false;
  if (other.normalized.length > entry.normalized.length) return true;
  return other.candidate.order < entry.candidate.order;
}

export function selectCanonicalFlow(candidates: readonly CanonicalCandidate[]): CanonicalSelection {
  const entries: Entry[] = candidates
    .filter((candidate) => !candidate.suggestion)
    .map((candidate) => ({ candidate, normalized: candidate.messages.map(normalizeMessage) }));
  if (entries.length === 0) return { canonicalId: null, chainIds: [] };

  // Maximal tips: requests that no other request extends. Non-consuming — a
  // shared root is simply not a tip; it belongs to every branch's chain.
  const tips = entries.filter((entry) => !entries.some((other) => other !== entry && supersedes(entry, other)));

  // Policy: the live conversation is the maximal chain whose tip was created
  // most recently. Covers both a newer branch over a stale sibling and a
  // post-compaction restart over the longer pre-compaction chain.
  let selected = tips[0];
  for (const tip of tips.slice(1)) {
    if (tip.candidate.order < selected.candidate.order) selected = tip;
  }

  const members = entries
    .filter((entry) => messagesArePrefix(entry.normalized, selected.normalized))
    .sort((left, right) =>
      left.normalized.length - right.normalized.length || right.candidate.order - left.candidate.order);
  return {
    canonicalId: selected.candidate.flowId,
    chainIds: members.map((member) => member.candidate.flowId),
  };
}
