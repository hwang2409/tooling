import { parseFlowExtras } from "../../protocol";
import type { FlowExtras } from "../../protocol";
import type { ImmutableFlowMetadata } from "../../state/browserState";

export const PLACEHOLDER = "—";

export type RowKind = "msg" | "cnt" | "generic";

export interface RowPreview {
  kind: "user" | "tool" | "muted" | "none";
  text: string;
}

export interface RowCells {
  time: string;
  badge: string;
  badgeKind: RowKind;
  model: string;
  messages: string;
  preview: RowPreview;
  sizes: string;
  duration: string;
  status: string;
  isError: boolean;
}

/** "475" / "6.1K" / "475K" / "1.2M" — compact byte counts for a dense row. */
export function formatBytes(decimal: string | undefined): string {
  if (decimal === undefined) return PLACEHOLDER;
  const value = Number(decimal);
  if (!Number.isFinite(value)) return PLACEHOLDER;
  if (value < 1000) return String(value);
  for (const [unit, scale] of [["K", 1e3], ["M", 1e6], ["G", 1e9]] as const) {
    const scaled = value / scale;
    if (scaled < 1000 || unit === "G") {
      return `${scaled < 10 ? scaled.toFixed(1).replace(/\.0$/, "") : String(Math.round(scaled))}${unit}`;
    }
  }
  return PLACEHOLDER;
}

/** "412ms" / "23.4s" / "2m05s". */
export function formatDuration(milliseconds: number): string {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) return PLACEHOLDER;
  if (milliseconds < 1000) return `${Math.round(milliseconds)}ms`;
  const seconds = milliseconds / 1000;
  const roundedTenths = Math.round(seconds * 10) / 10;
  if (roundedTenths < 60) return `${roundedTenths.toFixed(1)}s`;
  const totalSeconds = Math.round(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  return `${minutes}m${String(totalSeconds % 60).padStart(2, "0")}s`;
}

export function durationBetween(startedAt: string | undefined, endedAt: string | undefined): string {
  if (startedAt === undefined || endedAt === undefined) return PLACEHOLDER;
  const start = Date.parse(startedAt);
  const end = Date.parse(endedAt);
  if (Number.isNaN(start) || Number.isNaN(end)) return PLACEHOLDER;
  return formatDuration(end - start);
}

/** HH:MM:SS in the viewer's locale timezone. */
export function formatClockTime(startedAt: string | undefined): string {
  if (startedAt === undefined) return PLACEHOLDER;
  const parsed = new Date(startedAt);
  if (Number.isNaN(parsed.getTime())) return PLACEHOLDER;
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(parsed.getHours())}:${pad(parsed.getMinutes())}:${pad(parsed.getSeconds())}`;
}

export function shortModel(model: string | undefined): string {
  if (model === undefined || model.length === 0) return PLACEHOLDER;
  return model.startsWith("claude-") ? model.slice("claude-".length) : model;
}

function rowPreview(extras: FlowExtras): RowPreview {
  const preview = extras.summary?.preview;
  if (preview?.source === "user_text" && preview.text !== undefined) {
    return { kind: "user", text: `“${preview.text}”` };
  }
  if (preview?.source === "tool_result") {
    return { kind: "tool", text: `↳ tool_result: ${preview.tool_name ?? "?"}` };
  }
  if (extras.summary?.kind === "anthropic_count_tokens" && extras.summary.count_tokens_result !== undefined) {
    return { kind: "muted", text: `= ${extras.summary.count_tokens_result} tokens` };
  }
  return { kind: "none", text: PLACEHOLDER };
}

/** Project one flow's metadata onto the dense row cells. */
export function deriveRowCells(metadata: ImmutableFlowMetadata): RowCells {
  const extras = parseFlowExtras(metadata);
  const summary = extras.summary;
  const badgeKind: RowKind = summary?.kind === "anthropic_messages" ? "msg"
    : summary?.kind === "anthropic_count_tokens" ? "cnt"
      : "generic";

  const requestSize = extras.request_body_size
    ?? (metadata.request_body.state === "missing" || metadata.request_body.state === "empty" ? undefined : metadata.request_body.size_bytes);
  const responseSize = extras.response_body_size
    ?? (metadata.response_body === undefined || metadata.response_body.state === "missing" || metadata.response_body.state === "empty"
      ? undefined
      : metadata.response_body.size_bytes);
  const sizes = requestSize === undefined && responseSize === undefined
    ? PLACEHOLDER
    : `${formatBytes(requestSize)}→${formatBytes(responseSize)}`;

  const status = typeof metadata.response_status === "string" ? metadata.response_status : PLACEHOLDER;
  return {
    time: formatClockTime(extras.started_at),
    badge: badgeKind === "generic" ? `${metadata.method} ${metadata.host}${metadata.path}` : badgeKind,
    badgeKind,
    model: shortModel(summary?.model),
    messages: summary?.message_count === undefined ? PLACEHOLDER : `${summary.message_count}m`,
    preview: rowPreview(extras),
    sizes,
    duration: durationBetween(extras.started_at, extras.ended_at),
    status,
    isError: Number(status) >= 400,
  };
}
