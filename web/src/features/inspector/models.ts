import type { BodyDescriptor } from "../../protocol";

export interface RedactedBody {
  state: "redacted";
  content_type?: string;
  reason?: string;
}

export type InspectableBody = BodyDescriptor | RedactedBody;
