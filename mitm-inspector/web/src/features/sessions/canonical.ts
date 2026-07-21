/* eslint-disable no-unused-vars */

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
  /**
   * Normalized request-context identity (model + system + tools) from
   * requestContextKey. A request that resends an identical message history
   * under a DIFFERENT context (a one-off haiku side-call borrowing the main
   * history) is a different conversation and must never supersede the main
   * thread by mere recency. Absent keys compare equal.
   */
  readonly contextKey?: string;
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

/** Emitted verbatim by the iterative stableStringify traversal. */
class Literal {
  constructor(readonly text: string) {}
}

/**
 * Key-order-independent serialization so contexts compare structurally.
 * Iterative (explicit stack): captured JSON can nest thousands of levels
 * deep (real tool schemas hit depth 10,000+), and a recursive traversal
 * overflows the call stack on valid input.
 */
function stableStringify(root: JsonValue): string {
  const out: string[] = [];
  const stack: Array<JsonValue | Literal> = [root];
  while (stack.length > 0) {
    const item = stack.pop() as JsonValue | Literal;
    if (item instanceof Literal) {
      out.push(item.text);
      continue;
    }
    if (item === null || typeof item !== "object") {
      out.push(JSON.stringify(item));
      continue;
    }
    if (Array.isArray(item)) {
      out.push("[");
      stack.push(new Literal("]"));
      for (let index = item.length - 1; index >= 0; index -= 1) {
        stack.push(item[index]);
        if (index > 0) stack.push(new Literal(","));
      }
      continue;
    }
    const object = item as JsonObject;
    const keys = Object.keys(object).sort();
    out.push("{");
    stack.push(new Literal("}"));
    for (let index = keys.length - 1; index >= 0; index -= 1) {
      stack.push(object[keys[index]]);
      stack.push(new Literal(`${JSON.stringify(keys[index])}:`));
      if (index > 0) stack.push(new Literal(","));
    }
  }
  return out.join("");
}

/**
 * Request-context identity for canonical selection: model + system + tools,
 * with the same benign normalizations as messages (string system shorthand
 * expanded, cache_control stripped at block/tool top level only) so a
 * cache-marker move never splits a context.
 */
export function requestContextKey(body: JsonValue): string {
  const object = asObject(body);
  if (object === null) return "";
  const context: JsonObject = {};
  if (object.model !== undefined) context.model = object.model;
  if (object.system !== undefined) {
    context.system = typeof object.system === "string"
      ? [{ type: "text", text: object.system }]
      : Array.isArray(object.system) ? object.system.map(normalizeBlock) : object.system;
  }
  if (object.tools !== undefined) {
    context.tools = Array.isArray(object.tools) ? object.tools.map(normalizeBlock) : object.tools;
  }
  return stableStringify(context);
}

/** Iterative for the same reason as stableStringify: deep captured JSON must not overflow the stack. */
function deepEqual(left: JsonValue, right: JsonValue): boolean {
  const stack: Array<readonly [JsonValue, JsonValue]> = [[left, right]];
  while (stack.length > 0) {
    const [l, r] = stack.pop()!;
    if (l === r) continue;
    if (l === null || r === null || typeof l !== "object" || typeof r !== "object") return false;
    const leftArray = Array.isArray(l);
    if (leftArray !== Array.isArray(r)) return false;
    if (leftArray) {
      const leftItems = l as readonly JsonValue[];
      const rightItems = r as readonly JsonValue[];
      if (leftItems.length !== rightItems.length) return false;
      for (let index = 0; index < leftItems.length; index += 1) stack.push([leftItems[index], rightItems[index]]);
      continue;
    }
    const leftObject = l as JsonObject;
    const rightObject = r as JsonObject;
    const leftKeys = Object.keys(leftObject);
    if (leftKeys.length !== Object.keys(rightObject).length) return false;
    for (const key of leftKeys) {
      if (!(key in rightObject)) return false;
      stack.push([leftObject[key], rightObject[key]]);
    }
  }
  return true;
}

const REMINDER_OPEN = "<system-reminder>";
const REMINDER_CLOSE = "</system-reminder>";

/**
 * Complete anchored reminder markup: one or more full
 * `<system-reminder>…</system-reminder>` elements with nothing but
 * whitespace around them. Unclosed tags, malformed markup, or any trailing
 * text disqualify the block — otherwise arbitrary user text could fabricate
 * a prefix match.
 */
function isWhitespaceOnly(text: string, from: number, to: number): boolean {
  for (let index = from; index < to; index += 1) {
    const code = text.charCodeAt(index);
    if (code !== 0x20 && code !== 0x09 && code !== 0x0a && code !== 0x0d && code !== 0x0b && code !== 0x0c) return false;
  }
  return true;
}

function isReminderMarkup(text: string): boolean {
  // Single linear pass: collect every tag occurrence once (each indexOf
  // resumes past the previous match), then walk the token stream with a
  // depth counter. The previous per-element slice+trim implementation was
  // quadratic — a 560 KB reminder block took ~750 ms per comparison.
  const tokens: Array<readonly [start: number, open: boolean]> = [];
  for (let at = text.indexOf(REMINDER_OPEN); at !== -1; at = text.indexOf(REMINDER_OPEN, at + REMINDER_OPEN.length)) {
    tokens.push([at, true]);
  }
  for (let at = text.indexOf(REMINDER_CLOSE); at !== -1; at = text.indexOf(REMINDER_CLOSE, at + REMINDER_CLOSE.length)) {
    tokens.push([at, false]);
  }
  if (tokens.length === 0) return false;
  tokens.sort((a, b) => a[0] - b[0]);
  // The open tag is a prefix of no other token and the close tag contains no
  // open tag, so sorted occurrences cannot overlap.
  let depth = 0;
  let cursor = 0;
  let sawElement = false;
  for (const [start, open] of tokens) {
    if (depth === 0) {
      // Between top-level elements (and before the first) only whitespace
      // may appear; a close tag here is malformed markup.
      if (!open || !isWhitespaceOnly(text, cursor, start)) return false;
      depth = 1;
      cursor = start + REMINDER_OPEN.length;
      continue;
    }
    depth += open ? 1 : -1;
    cursor = start + (open ? REMINDER_OPEN.length : REMINDER_CLOSE.length);
    if (depth === 0) sawElement = true;
  }
  // Every nested open tag must close before the element ends, otherwise
  // "<system-reminder>outer <system-reminder>inner </system-reminder>"
  // would count as complete despite the unclosed outer.
  if (depth !== 0) return false;
  return sawElement && isWhitespaceOnly(text, cursor, text.length);
}

/** Exact shape of a harness-injected reminder: a text block of pure reminder markup. */
function isReminderInjection(value: JsonValue): boolean {
  const object = asObject(value);
  return object !== null && object.type === "text"
    && typeof object.text === "string" && isReminderMarkup(object.text);
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

/**
 * Equal-length histories can differ only by the same reminder injection that
 * messageMatches permits while building prefix relations. This comparison is
 * deliberately symmetric: either history may be the version with the
 * appended harness reminder, while cache markers are already normalized.
 */
function historiesAreEquivalent(left: readonly JsonValue[], right: readonly JsonValue[]): boolean {
  if (left.length !== right.length) return false;
  return left.every((message, index) =>
    messageMatches(message, right[index]) || messageMatches(right[index], message));
}

/** Canonical key for terminal groups, removing only tolerated reminder suffixes. */
function historyGroupKey(history: readonly JsonValue[]): string {
  const stripReminderSuffix = (message: JsonValue): JsonValue => {
    const object = asObject(message);
    if (object === null || !Array.isArray(object.content)) return message;
    let end = object.content.length;
    while (end > 0 && isReminderInjection(object.content[end - 1])) end -= 1;
    return end === object.content.length ? message : { ...object, content: object.content.slice(0, end) };
  };
  return stableStringify(history.map(stripReminderSuffix));
}

interface IndexEntry {
  readonly id: string;
  /** Refreshed on every update — the grid order shifts as flows prepend. */
  candidate: CanonicalCandidate;
  readonly normalized: readonly JsonValue[];
  /** Interned history identity: equal normalized histories share a stamp. */
  readonly stamp: number;
}

interface HistoryGroup {
  readonly stamp: number;
  readonly history: readonly JsonValue[];
  readonly members: Set<string>;
  readonly contextCounts: Map<string, { count: number; oldestOrder: number }>;
  readonly newestByContext: Map<string, string>;
  readonly prefixGroupsInto: Set<number>;
  readonly equivalentGroups: Set<number>;
  representativeId: string;
}

export interface CanonicalIndexStats {
  /** Histories normalized — exactly one per new or changed candidate. */
  readonly normalizations: number;
  /**
   * messagesArePrefix evaluations. O(changed x candidates) per update;
   * unchanged candidates never re-evaluate (relations are kept incrementally).
   */
  readonly prefixEvaluations: number;
  /** Whole-history deep-equality evaluations for stage interning. */
  readonly historyEquals: number;
  readonly selections: number;
  /** Union-find lookups and rebuild edges used to group related chains. */
  readonly componentVisits: number;
  /** Non-empty history-length buckets retained by stage interning. */
  readonly stampBuckets: number;
  /** Entries visited only while materializing the winning lineage. */
  readonly memberVisits: number;
  /** Group-level stage evidence visits. */
  readonly evidenceVisits: number;
  /** Group-level ownership aggregate visits. */
  readonly ownershipVisits: number;
  /** Per-tip prefix-member arrays retained; lineage is materialized lazily. */
  readonly retainedMemberArrays: number;
}

export interface CanonicalIndex {
  readonly update: (candidates: readonly CanonicalCandidate[]) => CanonicalSelection;
  readonly stats: () => CanonicalIndexStats;
}

/**
 * Incremental canonical selection. Candidates are diffed by messages
 * reference + context key: unchanged candidates keep their normalized
 * history, stage stamp, and group-level prefix relations, so a delta touching
 * an open session costs O(changed x history-groups) prefix evaluations instead
 * of renormalizing and rescanning shared terminal members.
 *
 * ORDER INVARIANT: supersede contributions for a pair are applied when the
 * pair is (re)computed and reversed on removal using CURRENT orders. That is
 * sound because the grid contract preserves relative created order among
 * retained flows — a reordered flow (remove + reinsert) always arrives as a
 * changed candidate and gets its pairs recomputed.
 */
export function createCanonicalIndex(): CanonicalIndex {
  const entries = new Map<string, IndexEntry>();
  /** One aggregate per exact normalized history, never one prefix array per tip. */
  const historyGroups = new Map<number, HistoryGroup>();
  /** Reverse strict-prefix edges, used to update group supersession counts. */
  const longerGroupsByPrefix = new Map<number, Set<number>>();
  /** Number of members in longer groups that supersede each history group. */
  const strictSuperseders = new Map<number, number>();
  /** Dynamic connectivity over related entries. Removals trigger lazy rebuild. */
  const componentParent = new Map<string, string>();
  let componentsDirty = false;
  interface StampGroup { readonly stamp: number; readonly history: readonly JsonValue[]; holders: number }
  const stampsByLength = new Map<number, StampGroup[]>();
  let nextStamp = 0;
  let normalizations = 0;
  let prefixEvaluations = 0;
  let historyEquals = 0;
  let selections = 0;
  let componentVisits = 0;
  let memberVisits = 0;
  let evidenceVisits = 0;
  let ownershipVisits = 0;
  let lastSelection: CanonicalSelection | null = null;

  const makeComponent = (id: string): void => {
    componentParent.set(id, id);
  };

  const findComponent = (id: string): string => {
    componentVisits += 1;
    const parent = componentParent.get(id);
    if (parent === undefined || parent === id) return parent ?? id;
    const root = findComponent(parent);
    componentParent.set(id, root);
    return root;
  };

  const unionComponents = (left: string, right: string): void => {
    const leftRoot = findComponent(left);
    const rightRoot = findComponent(right);
    if (leftRoot !== rightRoot) componentParent.set(rightRoot, leftRoot);
  };

  const rebuildComponents = (): void => {
    componentParent.clear();
    for (const id of entries.keys()) makeComponent(id);
    for (const group of historyGroups.values()) {
      for (const member of group.members) unionComponents(group.representativeId, member);
      for (const shorter of group.prefixGroupsInto) {
        const shorterGroup = historyGroups.get(shorter);
        if (shorterGroup !== undefined) unionComponents(group.representativeId, shorterGroup.representativeId);
      }
      for (const equivalent of group.equivalentGroups) {
        const equivalentGroup = historyGroups.get(equivalent);
        if (equivalentGroup !== undefined) unionComponents(group.representativeId, equivalentGroup.representativeId);
      }
    }
    componentsDirty = false;
  };

  const intern = (normalized: readonly JsonValue[]): number => {
    const groups = stampsByLength.get(normalized.length) ?? [];
    for (const group of groups) {
      historyEquals += 1;
      if (normalized.every((message, index) => deepEqual(message, group.history[index]))) {
        group.holders += 1;
        return group.stamp;
      }
    }
    const group: StampGroup = { stamp: nextStamp, history: normalized, holders: 1 };
    nextStamp += 1;
    groups.push(group);
    stampsByLength.set(normalized.length, groups);
    return group.stamp;
  };

  const releaseStamp = (entry: IndexEntry): void => {
    const groups = stampsByLength.get(entry.normalized.length) ?? [];
    const position = groups.findIndex((group) => group.stamp === entry.stamp);
    if (position === -1) return;
    groups[position].holders -= 1;
    if (groups[position].holders === 0) {
      groups.splice(position, 1);
      if (groups.length === 0) stampsByLength.delete(entry.normalized.length);
    }
  };

  const refreshContextAggregate = (group: HistoryGroup, context: string): void => {
    let count = 0;
    let oldestOrder = Number.NEGATIVE_INFINITY;
    let newestId: string | undefined;
    let newestOrder = Number.POSITIVE_INFINITY;
    for (const id of group.members) {
      const member = entries.get(id);
      if (member === undefined || (member.candidate.contextKey ?? "") !== context) continue;
      count += 1;
      oldestOrder = Math.max(oldestOrder, member.candidate.order);
      if (member.candidate.order < newestOrder) {
        newestOrder = member.candidate.order;
        newestId = id;
      }
    }
    if (count === 0) {
      group.contextCounts.delete(context);
      group.newestByContext.delete(context);
    } else {
      group.contextCounts.set(context, { count, oldestOrder });
      group.newestByContext.set(context, newestId!);
    }
  };

  const refreshGroupAggregates = (group: HistoryGroup): void => {
    group.contextCounts.clear();
    group.newestByContext.clear();
    for (const id of group.members) {
      const entry = entries.get(id);
      if (entry === undefined) continue;
      const context = entry.candidate.contextKey ?? "";
      const current = group.contextCounts.get(context);
      group.contextCounts.set(context, {
        count: (current?.count ?? 0) + 1,
        oldestOrder: Math.max(current?.oldestOrder ?? Number.NEGATIVE_INFINITY, entry.candidate.order),
      });
      const newestId = group.newestByContext.get(context);
      const newest = newestId === undefined ? undefined : entries.get(newestId);
      if (newest === undefined || entry.candidate.order < newest.candidate.order) group.newestByContext.set(context, entry.id);
    }
  };

  const addGroupMember = (group: HistoryGroup, entry: IndexEntry): void => {
    const context = entry.candidate.contextKey ?? "";
    group.members.add(entry.id);
    for (const shorter of group.prefixGroupsInto) {
      strictSuperseders.set(shorter, (strictSuperseders.get(shorter) ?? 0) + 1);
    }
    const current = group.contextCounts.get(context);
    group.contextCounts.set(context, {
      count: (current?.count ?? 0) + 1,
      oldestOrder: Math.max(current?.oldestOrder ?? Number.NEGATIVE_INFINITY, entry.candidate.order),
    });
    const newestId = group.newestByContext.get(context);
    const newest = newestId === undefined ? undefined : entries.get(newestId);
    if (newest === undefined || entry.candidate.order < newest.candidate.order) group.newestByContext.set(context, entry.id);
    unionComponents(group.representativeId, entry.id);
  };

  const removeGroupMember = (group: HistoryGroup, entry: IndexEntry): void => {
    const context = entry.candidate.contextKey ?? "";
    for (const shorter of group.prefixGroupsInto) {
      strictSuperseders.set(shorter, (strictSuperseders.get(shorter) ?? 0) - 1);
    }
    const aggregate = group.contextCounts.get(context)!;
    const wasOldest = aggregate.oldestOrder === entry.candidate.order;
    const wasNewest = group.newestByContext.get(context) === entry.id;
    group.members.delete(entry.id);
    if (aggregate.count <= 1) {
      group.contextCounts.delete(context);
      group.newestByContext.delete(context);
    } else if (wasOldest || wasNewest) {
      refreshContextAggregate(group, context);
    } else {
      group.contextCounts.set(context, { ...aggregate, count: aggregate.count - 1 });
    }
  };

  const linkStrictGroups = (shorter: HistoryGroup, longer: HistoryGroup): void => {
    if (longer.prefixGroupsInto.has(shorter.stamp)) return;
    // The edge is stored on the longer group; its members supersede every
    // member of the shorter group, while the reverse index supports removal.
    longer.prefixGroupsInto.add(shorter.stamp);
    const longerGroups = longerGroupsByPrefix.get(shorter.stamp) ?? new Set<number>();
    longerGroups.add(longer.stamp);
    longerGroupsByPrefix.set(shorter.stamp, longerGroups);
    strictSuperseders.set(shorter.stamp, (strictSuperseders.get(shorter.stamp) ?? 0) + longer.members.size);
    unionComponents(shorter.representativeId, longer.representativeId);
  };

  const linkEquivalentGroups = (left: HistoryGroup, right: HistoryGroup): void => {
    left.equivalentGroups.add(right.stamp);
    right.equivalentGroups.add(left.stamp);
    unionComponents(left.representativeId, right.representativeId);
  };

  const removeEntry = (entry: IndexEntry): void => {
    const group = historyGroups.get(entry.stamp)!;
    removeGroupMember(group, entry);
    entries.delete(entry.id);
    componentsDirty = true;
    if (group.members.size === 0) {
      for (const shorter of group.prefixGroupsInto) {
        const longerGroups = longerGroupsByPrefix.get(shorter);
        longerGroups?.delete(group.stamp);
        if (longerGroups?.size === 0) longerGroupsByPrefix.delete(shorter);
      }
      for (const longer of longerGroupsByPrefix.get(group.stamp) ?? []) {
        const longerGroup = historyGroups.get(longer);
        longerGroup?.prefixGroupsInto.delete(group.stamp);
        strictSuperseders.set(group.stamp, (strictSuperseders.get(group.stamp) ?? 0) - (longerGroup?.members.size ?? 0));
      }
      longerGroupsByPrefix.delete(group.stamp);
      for (const equivalent of group.equivalentGroups) historyGroups.get(equivalent)?.equivalentGroups.delete(group.stamp);
      strictSuperseders.delete(group.stamp);
      historyGroups.delete(group.stamp);
    } else if (group.representativeId === entry.id) {
      group.representativeId = group.members.values().next().value as string;
    }
    releaseStamp(entry);
  };

  const addEntry = (candidate: CanonicalCandidate): void => {
    normalizations += 1;
    const normalized = candidate.messages.map(normalizeMessage);
    const entry: IndexEntry = { id: candidate.flowId, candidate, normalized, stamp: intern(normalized) };
    entries.set(entry.id, entry);
    makeComponent(entry.id);
    const existingGroup = historyGroups.get(entry.stamp);
    if (existingGroup !== undefined) {
      addGroupMember(existingGroup, entry);
      return;
    }
    const group: HistoryGroup = {
      stamp: entry.stamp,
      history: normalized,
      members: new Set([entry.id]),
      contextCounts: new Map(),
      newestByContext: new Map(),
      prefixGroupsInto: new Set(),
      equivalentGroups: new Set(),
      representativeId: entry.id,
    };
    historyGroups.set(group.stamp, group);
    strictSuperseders.set(group.stamp, 0);
    refreshContextAggregate(group, candidate.contextKey ?? "");
    for (const other of [...historyGroups.values()]) {
      if (other.stamp === group.stamp) continue;
      prefixEvaluations += 2;
      const out = messagesArePrefix(group.history, other.history);
      const into = messagesArePrefix(other.history, group.history);
      if (out && group.history.length < other.history.length) linkStrictGroups(group, other);
      else if (into && other.history.length < group.history.length) linkStrictGroups(other, group);
      else if ((out && into) || (group.history.length === other.history.length
        && historiesAreEquivalent(group.history, other.history))) linkEquivalentGroups(group, other);
    }
  };

  interface Chain {
    readonly tip: IndexEntry;
  }

  /**
   * TWO-LEVEL SELECTION POLICY — do not re-simplify to either half alone
   * (each half was shipped solo once and each was a review-verified bug):
   *
   *  (a) WITHIN a shared-root component, chain evidence dominates: the chain
   *      with more confirming members wins regardless of tip recency, so an
   *      established `root -> main-1 -> main-2` thread beats a newer
   *      `root -> auxiliary` branch. Equal evidence (equal member counts)
   *      falls back to the newer tip, so a live branch beats a stale sibling.
   *
   *  (b) ACROSS disjoint components — no shared root, i.e. a post-compaction
   *      restart — the newest tip wins: a shorter restarted history beats the
   *      stale pre-compaction chain that mere member counting would keep.
   */
  const select = (): CanonicalSelection => {
    selections += 1;
    if (entries.size === 0) return { canonicalId: null, chainIds: [] };

    // Maximal tips: requests that no other request extends. Equal-history
    // retransmits are collapsed by their group's newest-per-context aggregate;
    // no tip retains a prefix-member array.
    const isTip = (entry: IndexEntry): boolean => {
      const group = historyGroups.get(entry.stamp)!;
      return (strictSuperseders.get(entry.stamp) ?? 0) === 0
        && group.newestByContext.get(entry.candidate.contextKey ?? "") === entry.id;
    };
    const chains: Chain[] = [];
    for (const entry of entries.values()) if (isTip(entry)) chains.push({ tip: entry });

    // Group chains by dynamic connectivity. Adding an entry unions only its
    // history groups; removals rebuild lazily once, rather than rescanning
    // every tip.
    if (componentsDirty) rebuildComponents();
    const componentsByRoot = new Map<string, Chain[]>();
    for (const chain of chains) {
      const root = findComponent(chain.tip.id);
      const component = componentsByRoot.get(root) ?? [];
      component.push(chain);
      componentsByRoot.set(root, component);
    }
    const components = [...componentsByRoot.values()];

    // Chain evidence is aggregated once per exact history group. A shorter
    // history remains part of the lineage even when its request context
    // differs; equal-length different-context members are borrowing calls and
    // stay out of evidence and rendered chain ids.
    const stageCounts = new Map<number, number>();
    const distinctStages = (chain: Chain): number => {
      const cached = stageCounts.get(chain.tip.stamp);
      if (cached !== undefined) return cached;
      const group = historyGroups.get(chain.tip.stamp)!;
      const count = group.prefixGroupsInto.size + 1;
      evidenceVisits += count;
      stageCounts.set(chain.tip.stamp, count);
      return count;
    };

    const compareChains = (left: Chain, right: Chain): number => {
      const stageDelta = distinctStages(left) - distinctStages(right);
      if (stageDelta !== 0) return stageDelta;
      if (left.tip.candidate.order !== right.tip.candidate.order) {
        return right.tip.candidate.order - left.tip.candidate.order;
      }
      return left.tip.candidate.flowId < right.tip.candidate.flowId ? 1 : -1;
    };

    interface TerminalGroup {
      readonly chains: Chain[];
    }

    /**
     * Two-phase representative selection. First aggregate terminal ownership
     * for each reminder-equivalent history group, then compare those owning
     * representatives as branches. Stable history and flow keys make ties
     * independent of insertion order.
     */
    const representative = (component: readonly Chain[]): Chain => {
      const terminalGroups = new Map<string, TerminalGroup>();
      const ordered = [...component].sort((left, right) => {
        const leftKey = historyGroupKey(left.tip.normalized);
        const rightKey = historyGroupKey(right.tip.normalized);
        return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : left.tip.candidate.flowId.localeCompare(right.tip.candidate.flowId);
      });
      for (const chain of ordered) {
        const key = historyGroupKey(chain.tip.normalized);
        const group = terminalGroups.get(key);
        if (group === undefined) terminalGroups.set(key, { chains: [chain] });
        else group.chains.push(chain);
      }

      const representatives = [...terminalGroups.values()].map((terminal) => {
        for (const chain of terminal.chains) distinctStages(chain);
        const ownership = new Map<string, { count: number; oldestOrder: number }>();
        const seenStamps = new Set<number>();
        for (const chain of terminal.chains) {
          const group = historyGroups.get(chain.tip.stamp)!;
          if (seenStamps.has(group.stamp)) continue;
          seenStamps.add(group.stamp);
          for (const [context, aggregate] of group.contextCounts) {
            ownershipVisits += 1;
            const current = ownership.get(context);
            if (current === undefined) ownership.set(context, { ...aggregate });
            else ownership.set(context, {
              count: current.count + aggregate.count,
              oldestOrder: Math.max(current.oldestOrder, aggregate.oldestOrder),
            });
          }
        }
        const owner = [...ownership.entries()].sort((left, right) =>
          right[1].count - left[1].count
          || right[1].oldestOrder - left[1].oldestOrder
          || (left[0] < right[0] ? -1 : 1))[0]?.[0];
        const owningChains = owner === undefined
          ? terminal.chains
          : terminal.chains.filter((chain) => (chain.tip.candidate.contextKey ?? "") === owner);
        return owningChains.reduce((best, chain) => compareChains(chain, best) > 0 ? chain : best);
      });
      return representatives.reduce((best, chain) => compareChains(chain, best) > 0 ? chain : best);
    };

    // (b) newest tip across disjoint components.
    let selected = representative(components[0]);
    for (const component of components.slice(1)) {
      const contender = representative(component);
      if (contender.tip.candidate.order < selected.tip.candidate.order) selected = contender;
    }

    // Materialize members only after the winning chain is known. Prefix and
    // terminal-equivalent groups are aggregates; a borrowed equal-history
    // different-context member stays auxiliary.
    const selectedGroup = historyGroups.get(selected.tip.stamp)!;
    const selectedLength = selected.tip.normalized.length;
    const selectedKey = historyGroupKey(selected.tip.normalized);
    const lineageGroups = new Set<number>(selectedGroup.prefixGroupsInto);
    lineageGroups.add(selected.tip.stamp);
    for (const group of historyGroups.values()) {
      if (group.history.length === selectedLength && historyGroupKey(group.history) === selectedKey) lineageGroups.add(group.stamp);
    }
    const members = [...lineageGroups].flatMap((stamp) => {
      const group = historyGroups.get(stamp);
      if (group === undefined) return [];
      return [...group.members].flatMap((id) => {
        memberVisits += 1;
        const member = entries.get(id)!;
        if (member.normalized.length === selectedLength
          && (member.candidate.contextKey ?? "") !== (selected.tip.candidate.contextKey ?? "")) return [];
        return [member];
      });
    })
      .sort((left, right) =>
        left.normalized.length - right.normalized.length || right.candidate.order - left.candidate.order);
    return {
      canonicalId: selected.tip.candidate.flowId,
      chainIds: members.map((member) => member.candidate.flowId),
    };
  };

  return {
    update(candidates) {
      let structural = false;
      let orderChanged = false;
      const seen = new Set<string>();
      const incoming = new Map(candidates.filter((candidate) => !candidate.suggestion).map((candidate) => [candidate.flowId, candidate]));
      const touchedGroups = new Set<number>();

      // Refresh all existing order-derived aggregates before processing the
      // newest-first batch. Otherwise an added tip can be compared against an
      // old order snapshot and incremental selection diverges from fresh.
      for (const entry of entries.values()) {
        const candidate = incoming.get(entry.id);
        if (candidate === undefined
          || candidate.messages !== entry.candidate.messages
          || candidate.contextKey !== entry.candidate.contextKey) continue;
        if (candidate.order !== entry.candidate.order) {
          entry.candidate = candidate;
          touchedGroups.add(entry.stamp);
          orderChanged = true;
        }
      }
      for (const stamp of touchedGroups) refreshGroupAggregates(historyGroups.get(stamp)!);

      for (const candidate of candidates) {
        if (candidate.suggestion) continue;
        seen.add(candidate.flowId);
        const existing = entries.get(candidate.flowId);
        if (existing !== undefined
          && existing.candidate.messages === candidate.messages
          && existing.candidate.contextKey === candidate.contextKey) {
          if (existing.candidate.order !== candidate.order) orderChanged = true;
          existing.candidate = candidate;
          continue;
        }
        if (existing !== undefined) removeEntry(existing);
        addEntry(candidate);
        structural = true;
      }
      for (const entry of [...entries.values()]) {
        if (seen.has(entry.id)) continue;
        removeEntry(entry);
        structural = true;
      }
      if (!structural && !orderChanged && lastSelection !== null) return lastSelection;
      lastSelection = select();
      return lastSelection;
    },
    stats: () => ({
      normalizations,
      prefixEvaluations,
      historyEquals,
      selections,
      componentVisits,
      stampBuckets: stampsByLength.size,
      memberVisits,
      evidenceVisits,
      ownershipVisits,
      retainedMemberArrays: 0,
    }),
  };
}

/** One-shot canonical selection (tests, non-incremental callers). */
export function selectCanonicalFlow(candidates: readonly CanonicalCandidate[]): CanonicalSelection {
  return createCanonicalIndex().update(candidates);
}
