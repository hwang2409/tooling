import type { JsonValue } from "../inspector/jsonTree";

/**
 * Canonical main-thread selection over fetched request bodies.
 *
 * Every main-thread request resends the full conversation history, so
 * consecutive main-thread requests form a prefix chain once two benign
 * mutations are normalized away:
 *   - `cache_control` markers move between requests (content identical), and
 *   - earlier messages can GAIN appended system-reminder text blocks in
 *     later requests (the latest request's version is canonical).
 *
 * Utility side-calls (quota, topic detection, branched probes) share at most
 * a prefix root and then diverge, so they land in their own short chains.
 * The chat therefore renders the tip of the DOMINANT chain — the one with
 * the most members — never a newer-but-off-chain branch, even when that
 * branch matches the main thread's message count. Suggestion-mode requests
 * are excluded up front: they duplicate the main history with an injected
 * prompt and must never become the rendered conversation.
 */

export interface CanonicalCandidate {
  readonly flowId: string;
  /** Raw `messages` array from the fetched request body. */
  readonly messages: readonly JsonValue[];
  /** Body-verified suggestion-mode request (see isSuggestionRequest). */
  readonly suggestion: boolean;
  /** Grid position, 0 = newest. */
  readonly order: number;
}

export interface CanonicalSelection {
  readonly canonicalId: string | null;
  /** Members of the dominant prefix chain, oldest first. */
  readonly chainIds: readonly string[];
}

type JsonObject = { [key: string]: JsonValue };

function asObject(value: JsonValue | undefined): JsonObject | null {
  if (value === null || value === undefined || typeof value !== "object" || Array.isArray(value)) return null;
  return value;
}

function stripCacheControl(value: JsonValue): JsonValue {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stripCacheControl);
  const result: JsonObject = {};
  for (const [key, item] of Object.entries(value)) {
    if (key === "cache_control") continue;
    result[key] = stripCacheControl(item);
  }
  return result;
}

/** Strip cache_control everywhere and expand string content shorthand. */
export function normalizeMessage(message: JsonValue): JsonValue {
  const stripped = stripCacheControl(message);
  const object = asObject(stripped);
  if (object === null) return stripped;
  if (typeof object.content === "string") {
    return { ...object, content: [{ type: "text", text: object.content }] };
  }
  return stripped;
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

function isTextBlock(value: JsonValue): boolean {
  const object = asObject(value);
  return object !== null && object.type === "text";
}

/**
 * Whether an earlier request's message is the same turn as a later request's
 * message at the same index: equal after normalization, or the later version
 * gained appended text blocks (system-reminder injection).
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
  if (!laterContent.slice(earlierContent.length).every(isTextBlock)) return false;
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

interface ChainEntry {
  readonly candidate: CanonicalCandidate;
  readonly normalized: readonly JsonValue[];
}

interface Chain {
  members: ChainEntry[];
  tip: ChainEntry;
}

export function selectCanonicalFlow(candidates: readonly CanonicalCandidate[]): CanonicalSelection {
  const entries: ChainEntry[] = candidates
    .filter((candidate) => !candidate.suggestion)
    .map((candidate) => ({ candidate, normalized: candidate.messages.map(normalizeMessage) }));
  // Shortest first so each chain grows tip-by-tip; older first within equal
  // length so a retransmitted identical request advances the tip to the
  // newest copy.
  entries.sort((left, right) =>
    left.normalized.length - right.normalized.length || right.candidate.order - left.candidate.order);

  const chains: Chain[] = [];
  for (const entry of entries) {
    let best: Chain | null = null;
    for (const chain of chains) {
      if (!messagesArePrefix(chain.tip.normalized, entry.normalized)) continue;
      if (best === null || chain.tip.normalized.length > best.tip.normalized.length) best = chain;
    }
    if (best === null) chains.push({ members: [entry], tip: entry });
    else {
      best.members.push(entry);
      best.tip = entry;
    }
  }
  if (chains.length === 0) return { canonicalId: null, chainIds: [] };

  const newestOrder = (chain: Chain) => Math.min(...chain.members.map((member) => member.candidate.order));
  let dominant = chains[0];
  for (const chain of chains.slice(1)) {
    if (chain.members.length !== dominant.members.length) {
      if (chain.members.length > dominant.members.length) dominant = chain;
    } else if (chain.tip.normalized.length !== dominant.tip.normalized.length) {
      if (chain.tip.normalized.length > dominant.tip.normalized.length) dominant = chain;
    } else if (newestOrder(chain) < newestOrder(dominant)) {
      dominant = chain;
    }
  }
  return {
    canonicalId: dominant.tip.candidate.flowId,
    chainIds: dominant.members.map((member) => member.candidate.flowId),
  };
}
