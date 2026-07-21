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

/** Key-order-independent serialization so contexts compare structurally. */
function stableStringify(value: JsonValue): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  const object = value as JsonObject;
  const keys = Object.keys(object).sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${stableStringify(object[key])}`).join(",")}}`;
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

const REMINDER_OPEN = "<system-reminder>";
const REMINDER_CLOSE = "</system-reminder>";

/**
 * Complete anchored reminder markup: one or more full
 * `<system-reminder>…</system-reminder>` elements with nothing but
 * whitespace around them. Unclosed tags, malformed markup, or any trailing
 * text disqualify the block — otherwise arbitrary user text could fabricate
 * a prefix match.
 */
function isReminderMarkup(text: string): boolean {
  let rest = text.trim();
  if (rest.length === 0) return false;
  while (rest.length > 0) {
    if (!rest.startsWith(REMINDER_OPEN)) return false;
    // Balanced scan: every nested open tag must close before the element
    // ends, otherwise "<system-reminder>outer <system-reminder>inner
    // </system-reminder>" would count as complete despite the unclosed outer.
    let depth = 1;
    let position = REMINDER_OPEN.length;
    while (depth > 0) {
      const nextOpen = rest.indexOf(REMINDER_OPEN, position);
      const nextClose = rest.indexOf(REMINDER_CLOSE, position);
      if (nextClose === -1) return false;
      if (nextOpen !== -1 && nextOpen < nextClose) {
        depth += 1;
        position = nextOpen + REMINDER_OPEN.length;
      } else {
        depth -= 1;
        position = nextClose + REMINDER_CLOSE.length;
      }
    }
    rest = rest.slice(position).trimStart();
  }
  return true;
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

interface IndexEntry {
  readonly id: string;
  /** Refreshed on every update — the grid order shifts as flows prepend. */
  candidate: CanonicalCandidate;
  readonly normalized: readonly JsonValue[];
  /** Interned history identity: equal normalized histories share a stamp. */
  readonly stamp: number;
}

/**
 * Whether `later` extends `earlier`: strictly longer, or an equal-length
 * newer retransmit UNDER THE SAME REQUEST CONTEXT. A different-context call
 * (other model/system/tools) resending an identical history is a borrowing
 * side-call, not a continuation — it must never displace the thread it
 * copied purely by being newer. Strictly growing histories remain one
 * lineage even when request context evolves between stages.
 */
function supersedes(earlier: IndexEntry, later: IndexEntry, earlierPrefixOfLater: boolean): boolean {
  if (!earlierPrefixOfLater) return false;
  if (later.normalized.length > earlier.normalized.length) return true;
  return later.candidate.contextKey === earlier.candidate.contextKey
    && later.candidate.order < earlier.candidate.order;
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
}

export interface CanonicalIndex {
  readonly update: (candidates: readonly CanonicalCandidate[]) => CanonicalSelection;
  readonly stats: () => CanonicalIndexStats;
}

/** Prefix directions of a related pair, from the owning entry's perspective. */
interface PairView {
  readonly out: boolean;
  readonly into: boolean;
}

/**
 * Incremental canonical selection. Candidates are diffed by messages
 * reference + context key: unchanged candidates keep their normalized
 * history, stage stamp, and pairwise prefix relations, so a delta touching
 * an open session costs O(changed x candidates) prefix evaluations instead
 * of renormalizing and rescanning the whole session (the shipped-and-reviewed
 * O(n^2) regression).
 *
 * ORDER INVARIANT: supersede contributions for a pair are applied when the
 * pair is (re)computed and reversed on removal using CURRENT orders. That is
 * sound because the grid contract preserves relative created order among
 * retained flows — a reordered flow (remove + reinsert) always arrives as a
 * changed candidate and gets its pairs recomputed.
 */
export function createCanonicalIndex(): CanonicalIndex {
  const entries = new Map<string, IndexEntry>();
  /** Sparse symmetric relation over related pairs only (some prefix holds). */
  const rel = new Map<string, Map<string, PairView>>();
  /** How many other entries supersede this one; tips have count 0. */
  const supersededBy = new Map<string, number>();
  /** Ids whose normalized messages are a prefix of this entry's (excl. self). */
  const prefixesInto = new Map<string, Set<string>>();
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
    for (const [id, neighbors] of rel) {
      for (const neighbor of neighbors.keys()) unionComponents(id, neighbor);
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

  const applyPair = (a: IndexEntry, b: IndexEntry, view: PairView, sign: 1 | -1): void => {
    const bump = (id: string) => supersededBy.set(id, (supersededBy.get(id) ?? 0) + sign);
    if (view.out) {
      const set = prefixesInto.get(b.id)!;
      if (sign === 1) set.add(a.id);
      else set.delete(a.id);
      if (supersedes(a, b, true)) bump(a.id);
    }
    if (view.into) {
      const set = prefixesInto.get(a.id)!;
      if (sign === 1) set.add(b.id);
      else set.delete(b.id);
      if (supersedes(b, a, true)) bump(b.id);
    }
  };

  const removeEntry = (entry: IndexEntry): void => {
    for (const [otherId, view] of rel.get(entry.id)!) {
      applyPair(entry, entries.get(otherId)!, view, -1);
      rel.get(otherId)!.delete(entry.id);
    }
    rel.delete(entry.id);
    prefixesInto.delete(entry.id);
    supersededBy.delete(entry.id);
    entries.delete(entry.id);
    componentsDirty = true;
    releaseStamp(entry);
  };

  const addEntry = (candidate: CanonicalCandidate): void => {
    normalizations += 1;
    const normalized = candidate.messages.map(normalizeMessage);
    const entry: IndexEntry = { id: candidate.flowId, candidate, normalized, stamp: intern(normalized) };
    const pairs = new Map<string, PairView>();
    rel.set(entry.id, pairs);
    prefixesInto.set(entry.id, new Set());
    supersededBy.set(entry.id, 0);
    makeComponent(entry.id);
    for (const other of entries.values()) {
      prefixEvaluations += 2;
      const out = messagesArePrefix(normalized, other.normalized);
      const into = messagesArePrefix(other.normalized, normalized);
      if (!out && !into) continue;
      const view: PairView = { out, into };
      pairs.set(other.id, view);
      rel.get(other.id)!.set(entry.id, { out: into, into: out });
      applyPair(entry, other, view, 1);
      unionComponents(entry.id, other.id);
    }
    entries.set(entry.id, entry);
  };

  interface Chain {
    readonly tip: IndexEntry;
    readonly members: readonly IndexEntry[];
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

    // Maximal tips: requests that no other request extends. Non-consuming — a
    // shared root is simply not a tip; it belongs to every branch's chain.
    const chains: Chain[] = [];
    for (const entry of entries.values()) {
      if ((supersededBy.get(entry.id) ?? 0) !== 0) continue;
      const members = [...prefixesInto.get(entry.id)!].map((id) => entries.get(id)!);
      members.push(entry);
      chains.push({ tip: entry, members });
    }

    // Group chains by dynamic connectivity. Adding an entry unions only its
    // related members; removals rebuild lazily once, rather than rescanning
    // every prior component for every chain on every selection.
    if (componentsDirty) rebuildComponents();
    const componentsByRoot = new Map<string, Chain[]>();
    for (const chain of chains) {
      const root = findComponent(chain.tip.id);
      const component = componentsByRoot.get(root) ?? [];
      component.push(chain);
      componentsByRoot.set(root, component);
    }
    const components = [...componentsByRoot.values()];

    // Chain evidence = DISTINCT history stages (interned stamps: deep
    // equality of the complete normalized history — duplicate retransmits
    // never inflate a branch, while reminder-only evolution still counts)
    // among lineage members. A shorter history remains part of the lineage
    // even when its request context differs: context may evolve while a
    // conversation grows. Equal-length different-context members are
    // borrowing side-calls and stay out of evidence and rendered chain ids.
    const stageCounts = new Map<Chain, number>();
    const lineageMembers = (chain: Chain): readonly IndexEntry[] => chain.members.filter((member) =>
      member.normalized.length < chain.tip.normalized.length
      || member.candidate.contextKey === chain.tip.candidate.contextKey);
    const distinctStages = (chain: Chain): number => {
      const cached = stageCounts.get(chain);
      if (cached !== undefined) return cached;
      const stamps = new Set<number>();
      for (const member of lineageMembers(chain)) {
        stamps.add(member.stamp);
      }
      stageCounts.set(chain, stamps.size);
      return stamps.size;
    };

    // (a) prefix dominance within a component, newer tip on ties.
    const representative = (component: readonly Chain[]): Chain =>
      component.reduce((best, chain) => {
        const stages = distinctStages(chain);
        const bestStages = distinctStages(best);
        if (stages !== bestStages) return stages > bestStages ? chain : best;
        if (chain.tip.stamp === best.tip.stamp
          && chain.tip.candidate.contextKey !== best.tip.candidate.contextKey) {
          // Equal-history different-context calls are not continuations. If
          // pruning leaves one established stage, retain older thread rather
          // than letting a newer borrowing clone win the tie.
          return chain.tip.candidate.order > best.tip.candidate.order ? chain : best;
        }
        return chain.tip.candidate.order < best.tip.candidate.order ? chain : best;
      });

    // (b) newest tip across disjoint components.
    let selected = representative(components[0]);
    for (const component of components.slice(1)) {
      const contender = representative(component);
      if (contender.tip.candidate.order < selected.tip.candidate.order) selected = contender;
    }

    // Render all growing lineage stages, including context evolution. A
    // borrowed equal-history different-context member stays auxiliary.
    const members = [...lineageMembers(selected)]
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
    }),
  };
}

/** One-shot canonical selection (tests, non-incremental callers). */
export function selectCanonicalFlow(candidates: readonly CanonicalCandidate[]): CanonicalSelection {
  return createCanonicalIndex().update(candidates);
}
