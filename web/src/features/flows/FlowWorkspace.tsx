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

export interface FlowWorkspaceProps {
  browser: BrowserState;
  followLive: boolean;
  pauseLive: () => void;
  gridViewportHeight?: number;
  loadFlowDetail?: FlowDetailLoader;
}

export function FlowWorkspace({ browser, followLive, pauseLive, gridViewportHeight, loadFlowDetail }: FlowWorkspaceProps) {
  const workspaceId = useId().replaceAll(":", "");
  const [query, setQuery] = useState("");
  const [selectedFlowId, setSelectedFlowId] = useState<string | null>(null);

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
      </div>
      {!filter.ok && (
        <p id={`${workspaceId}-filter-error`} className="flow-filter-error" role="alert">
          {filter.error} Showing all flows until the filter parses.
        </p>
      )}

      <FlowGrid
        rows={filteredRows}
        selectedFlowId={selectedFlowId}
        onSelectFlow={selectFlow}
        followLive={followLive}
        viewportHeight={gridViewportHeight}
      />

      {selectedFlowId === null ? (
        <p className="flow-inspector-hint">Select a flow row to open the paired request/response inspector. Selection pauses follow-live.</p>
      ) : selectedFlow === null ? (
        <div className="flow-inspector-dock is-evicted" role="status">
          <p>The selected flow left the bounded retention window and can no longer be inspected.</p>
          <button className="flow-dock-action" type="button" onClick={closeInspector}>Dismiss</button>
        </div>
      ) : (
        <div className="flow-inspector-dock">
          <div className="flow-dock-bar">
            <span className="flow-dock-title">INSPECTING <code>{selectedFlow.metadata.flow_id}</code></span>
            <button className="flow-dock-action" type="button" onClick={closeInspector}>Close inspector</button>
          </div>
          <PairedInspector flow={selectedFlow} compact />
        </div>
      )}
    </div>
  );
}
