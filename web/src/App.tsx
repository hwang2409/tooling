import { useMemo, useState } from "react";

import { useConnection } from "./features/connection/useConnection";
import type { TransportFactory } from "./features/connection/connectionClient";
import { webSocketTransportFactory } from "./features/connection/wsTransport";
import type { ConnectionViewModel } from "./features/connection/useConnection";
import { InspectorBodyPanel } from "./features/inspector/PairedInspector";
import { useFlowDetail } from "./features/inspector/flowDetail";
import type { InspectableBody } from "./features/inspector/models";
import { buildFlowRow } from "./features/flows/gridModel";
import type { FlowRow } from "./features/flows/gridModel";
import type { ImmutableFlowMetadata } from "./state/browserState";
import "./styles/shell.css";

export { formatBytes } from "./format";

export interface AppProps {
  transportFactory?: TransportFactory;
}

export function App({ transportFactory }: AppProps = {}) {
  return <Workbench view={useConnection(transportFactory ?? webSocketTransportFactory())} />;
}

export function Workbench({ view }: { view: ConnectionViewModel }) {
  const { browser } = view;
  const [expandedFlowId, setExpandedFlowId] = useState<string | null>(null);

  const rows = useMemo<readonly FlowRow[]>(
    () => browser.flows.entries.map((metadata) => buildFlowRow(metadata, browser.lifecycles.get(metadata.flow_id))),
    [browser.flows, browser.lifecycles],
  );

  const toggle = (flowId: string) => setExpandedFlowId((current) => (current === flowId ? null : flowId));

  return (
    <main className="app-shell">
      {rows.length === 0 ? (
        <p className="flow-empty">waiting for traffic</p>
      ) : (
        <ol className="flow-list" role="list">
          {rows.map((row) => {
            const isExpanded = row.flowId === expandedFlowId;
            const metadata = browser.flows.get(row.flowId);
            return (
              <li key={row.flowId} className={`flow-item${isExpanded ? " is-expanded" : ""}`}>
                <button
                  type="button"
                  className="flow-item-row"
                  onClick={() => toggle(row.flowId)}
                  aria-expanded={isExpanded}
                  title={row.url}
                >
                  <span className="flow-item-method">{row.method}</span>
                  <span className="flow-item-status">{row.statusLabel}</span>
                  <span className="flow-item-host">{row.host}</span>
                  <span className="flow-item-path">{row.path}</span>
                  <span className="flow-item-size">
                    {row.responseBody.compact}{row.responseBody.truncated ? "+" : ""}
                  </span>
                  <span className="flow-item-duration">{row.durationLabel}</span>
                </button>
                {isExpanded && metadata && <FlowDetail flowId={row.flowId} metadata={metadata} />}
              </li>
            );
          })}
        </ol>
      )}
    </main>
  );
}

function FlowDetail({ flowId, metadata }: { flowId: string; metadata: ImmutableFlowMetadata }) {
  const detail = useFlowDetail(flowId);
  const overrides = detail?.status === "loaded" ? detail.overrides : {};
  const requestBody: InspectableBody = overrides.request_body ?? metadata.request_body;
  const responseBody: InspectableBody =
    overrides.response_body ?? metadata.response_body ?? { state: "missing" };
  return (
    <div className="flow-detail">
      <section className="flow-detail-pane">
        <h3>request</h3>
        <InspectorBodyPanel body={requestBody} pane="request" />
      </section>
      <section className="flow-detail-pane">
        <h3>response</h3>
        <InspectorBodyPanel body={responseBody} pane="response" />
      </section>
    </div>
  );
}
