import type {
  BodyDescriptor,
  FlowLifecycle,
  FlowMetadata,
  Header,
  LifecycleState,
} from "../../protocol";

export type InspectorPane = "request" | "response" | "error";
export type BodyViewMode = "json" | "text" | "sse" | "hex";

export interface RedactedBody {
  state: "redacted";
  content_type?: string;
  reason?: string;
}

export type InspectableBody = BodyDescriptor | RedactedBody;

export interface InspectorHeader extends Header {
  redacted?: boolean;
}

export interface InspectorFlow {
  metadata: FlowMetadata;
  lifecycle?: readonly FlowLifecycle[];
  error?: string;
  request_headers?: readonly InspectorHeader[];
  response_headers?: readonly InspectorHeader[];
  request_body?: InspectableBody;
  response_body?: InspectableBody;
}

export interface LifecycleEntry {
  state: LifecycleState;
  sequence: string;
  occurredAt: string;
  eventId: string;
}

export interface InspectorProps {
  flow: InspectorFlow;
  className?: string;
  compact?: boolean;
  // eslint-disable-next-line no-unused-vars
  onPaneChange?: (pane: InspectorPane) => void;
}
