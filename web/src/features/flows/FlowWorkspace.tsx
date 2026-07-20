import { useId, useMemo, useState } from "react";
import type { KeyboardEvent } from "react";

import type { BrowserState } from "../../state/browserState";
import { PairedInspector } from "../inspector/PairedInspector";
import { useFlowDetail } from "../inspector/flowDetail";
import type { FlowDetailLoader } from "../inspector/flowDetail";
import type { InspectorFlow } from "../inspector/models";
import { FlowGrid } from "./FlowGrid";
import { parseFilter } from "./filter";
import { buildFlowRow } from "./gridModel";
import type { FlowRow } from "./gridModel";
import "../../styles/flows.css";

const GROUP_BY_STORAGE_KEY = "mitm-inspector:flows:group-by";
type GroupBy = "none" | "session";

function storedGroupBy(): GroupBy {
  try {
    return globalThis.localStorage.getItem(GROUP_BY_STORAGE_KEY) === "session" ? "session" : "none";
  } catch {
    return "none";
  }
}

export interface FlowWorkspaceProps {
  browser: BrowserState;
  followLive: boolean;
  pauseLive: () => void;
  gridViewportHeight?: number;
  loadFlowDetail?: FlowDetailLoader;
  /**
   * Externally-owned bounded seen-flow set. Hoisted above this component so
   * a reconnect (which can unmount the workspace via the empty-state
   * branch) does not throw away the record of which rows were opened.
   */
  seenFlowIds?: ReadonlySet<string>;
  // eslint-disable-next-line no-unused-vars
  onFlowSeen?: (flowId: string) => void;
}

export function FlowWorkspace({ browser, followLive, pauseLive, gridViewportHeight, loadFlowDetail, seenFlowIds, onFlowSeen }: FlowWorkspaceProps) {
  const workspaceId = useId().replaceAll(":", "");
  const [query, setQuery] = useState("");
  const [groupBy, setGroupBy] = useState<GroupBy>(storedGroupBy);
  const [selectedFlowId, setSelectedFlowId] = useState<string | null>(null);
  const [localSeenFlowIds, setLocalSeenFlowIds] = useState<ReadonlySet<string>>(() => new Set());
  const effectiveSeenFlowIds = seenFlowIds ?? localSeenFlowIds;

  const rows = useMemo<readonly FlowRow[]>(
    () => browser.flows.entries.map((metadata) => buildFlowRow(metadata, browser.lifecycles.get(metadata.flow_id))),
    [browser.flows, browser.lifecycles],
  );
  const filter = useMemo(() => parseFilter(query), [query]);
  const filteredRows = useMemo(
    () => (filter.ok && !filter.empty ? rows.filter((row) => filter.matches(row.filterable)) : rows),
    [filter, rows],
  );

  const selectFlow = (flowId: string) => {
    setSelectedFlowId(flowId);
    if (onFlowSeen !== undefined) {
      onFlowSeen(flowId);
    } else {
      setLocalSeenFlowIds((previous) => {
        if (previous.has(flowId)) return previous;
        const next = new Set(previous);
        next.add(flowId);
        return next;
      });
    }
    if (followLive) pauseLive();
  };

  const closeInspector = () => setSelectedFlowId(null);

  const handleWorkspaceKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape" && selectedFlowId !== null) {
      event.preventDefault();
      closeInspector();
    }
  };

  const selectedMetadata = selectedFlowId === null ? undefined : browser.flows.get(selectedFlowId);
  const detail = useFlowDetail(selectedMetadata === undefined ? null : selectedMetadata.flow_id, loadFlowDetail);
  const selectedFlow: InspectorFlow | null = selectedMetadata === undefined
    ? null
    : {
        metadata: selectedMetadata,
        lifecycle: browser.lifecycles.get(selectedMetadata.flow_id),
        lifecycleTruncated: browser.lifecycles.isTruncated(selectedMetadata.flow_id),
        ...(detail?.status === "loaded" ? detail.overrides : {}),
      };

  return (
    <div className="flow-workspace" onKeyDown={handleWorkspaceKeyDown}>
      <div className="flow-toolbar">
        <label className="flow-filter-label" htmlFor={`${workspaceId}-filter`}>FILTER</label>
        <input
          id={`${workspaceId}-filter`}
          className={`flow-filter-input${filter.ok ? "" : " has-error"}`}
          type="text"
          spellCheck={false}
          autoComplete="off"
          placeholder='mitmproxy syntax: ~m post ~d api\. "literal" !~q'
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-describedby={filter.ok ? undefined : `${workspaceId}-filter-error`}
          aria-invalid={filter.ok ? undefined : true}
        />
        <span className="flow-filter-count" aria-label={`${filteredRows.length} of ${rows.length} flows shown`}>
          {filteredRows.length}/{rows.length}
        </span>
        <label className="flow-group-label" htmlFor={`${workspaceId}-group-by`}>GROUP BY</label>
        <select
          id={`${workspaceId}-group-by`}
          className="flow-group-select"
          aria-label="Group flows by"
          value={groupBy}
          onChange={(event) => {
            const value: GroupBy = event.target.value === "session" ? "session" : "none";
            setGroupBy(value);
            try {
              globalThis.localStorage.setItem(GROUP_BY_STORAGE_KEY, value);
            } catch {
              // Storage can be unavailable in private or restricted contexts.
            }
          }}
        >
          <option value="none">none</option>
          <option value="session">session</option>
        </select>
      </div>
      {!filter.ok && (
        <p id={`${workspaceId}-filter-error`} className="flow-filter-error" role="alert">
          {filter.error} Showing all flows until the filter parses.
        </p>
      )}

      <div className={`flow-main${selectedFlowId === null ? "" : " has-inspector"}`}>
        <div className="flow-main-grid">
          <FlowGrid
            rows={filteredRows}
            groupBy={groupBy}
            selectedFlowId={selectedFlowId}
            onSelectFlow={selectFlow}
            followLive={followLive}
            viewportHeight={gridViewportHeight}
            seenFlowIds={effectiveSeenFlowIds}
          />
          {selectedFlowId === null && (
            <p className="flow-inspector-hint">Select a flow row to open the paired request/response inspector. Selection pauses follow-live.</p>
          )}
        </div>

        {selectedFlowId !== null && (
          selectedFlow === null ? (
            <aside className="flow-inspector-side is-evicted" role="status">
              <p>The selected flow left the bounded retention window and can no longer be inspected.</p>
              <button className="flow-dock-action" type="button" onClick={closeInspector}>Dismiss</button>
            </aside>
          ) : (
            <aside className="flow-inspector-side" aria-label={`Inspector for ${selectedFlow.metadata.flow_id}`}>
              <PairedInspector flow={selectedFlow} compact />
              <button className="flow-inspector-close" type="button" onClick={closeInspector} aria-label="Close inspector" title="Close inspector">×</button>
            </aside>
          )
        )}
      </div>
    </div>
  );
}
