/* eslint-disable no-unused-vars */

import { useEffect, useState } from "react";

/**
 * Hard ceiling on remembered seen flow ids. Sized above the backend's
 * MAX_ACTIVE_FLOWS = 2000 retention window so a full working set fits
 * without silent FIFO rollover; the prune-to-retained pass below is the
 * real bound in practice.
 */
export const SEEN_FLOW_LIMIT = 4096;

export interface SeenFlowsController {
  readonly seen: ReadonlySet<string>;
  readonly mark: (flowId: string) => void;
}

/**
 * Track which flow rows the user has opened. The set survives transient
 * reconnects (empty retention windows are ignored) and is pruned against
 * the authoritative retained flow ids on every non-empty change so an
 * evicted flow's bit is dropped but every still-retained row keeps its
 * seen bit across arbitrary rollover.
 *
 * `retainedFlowIds` must be the ordered list from the browser reducer's
 * flow collection — the caller passes an id string so React can compare
 * by reference in the effect's dependency list.
 */
export function useSeenFlows(retainedFlowIds: readonly string[]): SeenFlowsController {
  const [seen, setSeen] = useState<ReadonlySet<string>>(() => new Set());
  const retainedIdsKey = retainedFlowIds.join(" ");

  useEffect(() => {
    if (retainedFlowIds.length === 0) return;
    setSeen((previous) => {
      if (previous.size === 0) return previous;
      const retained = new Set(retainedFlowIds);
      let changed = false;
      const next = new Set<string>();
      for (const flowId of previous) {
        if (retained.has(flowId)) next.add(flowId);
        else changed = true;
      }
      return changed ? next : previous;
    });
    // retainedIdsKey collapses the id list into a stable string so we
    // only prune when the retained window actually changes.
  }, [retainedIdsKey, retainedFlowIds]);

  const mark = (flowId: string) => {
    setSeen((previous) => {
      if (previous.has(flowId)) return previous;
      const next = new Set(previous);
      next.add(flowId);
      while (next.size > SEEN_FLOW_LIMIT) {
        const oldest = next.values().next().value;
        if (oldest === undefined) break;
        next.delete(oldest);
      }
      return next;
    });
  };

  return { seen, mark };
}
