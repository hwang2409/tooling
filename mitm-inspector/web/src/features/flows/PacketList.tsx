import { useMemo, useState } from "react";

import type { BrowserState, ImmutableFlowMetadata } from "../../state/browserState";
import { bodyText } from "../inspector/decoders";
import type { BodyText } from "../inspector/decoders";
import { useFlowDetail } from "../inspector/flowDetail";
import type { FlowDetailLoader, FlowDetailResult } from "../inspector/flowDetail";
import { JsonTree, LARGE_TREE_COLLAPSE_THRESHOLD, safeParseJson } from "../inspector/jsonTree";
import type { InspectableBody } from "../inspector/models";
import "../../styles/shell.css";

export interface PacketListProps {
  browser: BrowserState;
  loadFlowDetail?: FlowDetailLoader;
}

export function PacketList({ browser, loadFlowDetail }: PacketListProps) {
  const [openFlowId, setOpenFlowId] = useState<string | null>(null);
  const flows = browser.flows.entries;
  const openFlow = openFlowId === null ? undefined : browser.flows.get(openFlowId);
  const detail = useFlowDetail(openFlow?.flow_id ?? null, loadFlowDetail);

  if (flows.length === 0) return <p className="packet-empty">no packets captured</p>;

  return (
    <ol className="packet-list" aria-label="Captured packets">
      {flows.map((metadata) => {
        const open = openFlow !== undefined && metadata.flow_id === openFlow.flow_id;
        return (
          <li key={metadata.flow_id} className="packet">
            <button
              type="button"
              className="packet-row"
              aria-expanded={open}
              onClick={() => setOpenFlowId(open ? null : metadata.flow_id)}
            >
              <span className="packet-method">{metadata.method}</span>
              <span className="packet-host">{metadata.host}</span>
              <span className="packet-path">{metadata.path}</span>
              <span className="packet-status">
                {typeof metadata.response_status === "string" ? metadata.response_status : "—"}
              </span>
            </button>
            {open && <PacketDetail metadata={metadata} detail={detail} />}
          </li>
        );
      })}
    </ol>
  );
}

function PacketDetail({ metadata, detail }: { metadata: ImmutableFlowMetadata; detail: FlowDetailResult | null }) {
  const overrides = detail?.status === "loaded" ? detail.overrides : undefined;
  return (
    <div className="packet-detail">
      <RequestBody body={overrides?.request_body ?? metadata.request_body} />
      <ResponseBody body={overrides?.response_body ?? metadata.response_body} />
    </div>
  );
}

function RequestBody({ body }: { body: InspectableBody | undefined }) {
  const decoded = useMemo(() => bodyText(body), [body]);
  if (decoded.kind === "absent") return <p className="packet-nobody">no request body</p>;
  return <DecodedBodyContent decoded={decoded} />;
}

function ResponseBody({ body }: { body: InspectableBody | undefined }) {
  const decoded = useMemo(() => bodyText(body), [body]);
  if (decoded.kind === "absent") return null;
  return (
    <div className="packet-response">
      <span className="packet-response-label">response</span>
      <DecodedBodyContent decoded={decoded} />
    </div>
  );
}

function DecodedBodyContent({ decoded }: { decoded: Extract<BodyText, { kind: "text" }> }) {
  const parsed = useMemo(() => safeParseJson(decoded.text), [decoded.text]);
  if (parsed.ok) return <JsonTree value={parsed.value} startCollapsed={decoded.byteLength > LARGE_TREE_COLLAPSE_THRESHOLD} />;
  return <pre className="packet-text">{decoded.text}</pre>;
}
