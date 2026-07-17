import type { FlowLifecycle } from "../../protocol";
import type { LifecycleEntry } from "./models";

const stateLabels: Record<FlowLifecycle["state"], string> = {
  request_started: "request opened",
  request_headers: "request headers",
  request_body: "request body",
  request_end: "request ended",
  response_started: "response opened",
  response_headers: "response headers",
  response_body: "response body",
  response_end: "response ended",
  error: "error observed",
  flow_completed: "flow completed",
};

export function lifecycleLabel(state: FlowLifecycle["state"]): string {
  return stateLabels[state];
}

export function orderLifecycle(events: readonly FlowLifecycle[] = []): LifecycleEntry[] {
  return events.map((event, index) => ({
    state: event.state,
    sequence: event.sequence,
    occurredAt: event.occurred_at,
    eventId: `${event.event_id}:${index}`,
  })).sort((left, right) => {
    const sequenceOrder = BigInt(left.sequence) - BigInt(right.sequence);
    if (sequenceOrder < 0n) return -1;
    if (sequenceOrder > 0n) return 1;
    return left.eventId.localeCompare(right.eventId);
  });
}

export function lifecyclePhase(entries: readonly LifecycleEntry[]): { requestEnded: boolean; responseStarted: boolean; completed: boolean; errored: boolean } {
  return {
    requestEnded: entries.some((entry) => entry.state === "request_end"),
    responseStarted: entries.some((entry) => entry.state === "response_started"),
    completed: entries.some((entry) => entry.state === "flow_completed"),
    errored: entries.some((entry) => entry.state === "error"),
  };
}
