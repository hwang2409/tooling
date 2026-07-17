import type { DeepReadonly } from "../../immutable";
import type {
  BodyDescriptor,
  FlowLifecycle,
  FlowMetadata,
  Header,
  LifecycleState,
} from "../../protocol";

export type InspectorPane = "request" | "response" | "error";
export type BodyViewMode = "json" | "text" | "sse" | "hex";
export type BodyPane = Exclude<InspectorPane, "error">;

export interface BodySelection {
  flowId: string;
  pane: BodyPane;
}

export interface RedactedBody {
  state: "redacted";
  content_type?: string;
  reason?: string;
}

export type InspectableBody = BodyDescriptor | RedactedBody;

export interface InspectorHeader extends Header {
  redacted?: boolean;
}

export type InspectorLifecycleEvent = DeepReadonly<FlowLifecycle>;

export interface InspectorFlow {
  metadata: DeepReadonly<FlowMetadata>;
  lifecycle?: readonly InspectorLifecycleEvent[];
  lifecycleTruncated?: boolean;
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
  activePane?: InspectorPane;
  bodySelection?: BodySelection | null;
  // eslint-disable-next-line no-unused-vars
  onBodySelect?: (selection: BodySelection) => void;
  // eslint-disable-next-line no-unused-vars
  onPaneChange?: (pane: InspectorPane) => void;
}

export interface InspectorBodyPanelProps {
  body: InspectableBody;
  pane: BodyPane;
  /** F5 gating hook; F6 auto-renders the body so this is now advisory. */
  selected?: boolean;
  /** F5 gating hook; F6 auto-renders the body so this is now advisory. */
  onSelect?: () => void;
}
