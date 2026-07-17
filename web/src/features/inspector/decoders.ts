import { MAX_INGEST_BODY_PREFIX_BYTES } from "../../contracts/limits";
import type { BodyDescriptor } from "../../protocol";
import type { BodyViewMode, InspectableBody } from "./models";

/**
 * Decode ceiling equals the backend wire ceiling — the biggest body the
 * ingest listener will ever accept. Bodies smaller than this are surfaced
 * whole; larger ones cannot exist on the wire, so no additional client-side
 * clamp is meaningful. F5 shipped a 64 KiB cap that quietly truncated most
 * captured API responses; F6 uncaps it (Henry: "let's not cap anything").
 */
export const DEFAULT_BODY_LIMIT = MAX_INGEST_BODY_PREFIX_BYTES;
/**
 * Hex mode still shows a bounded prefix by default because the raw hex+ascii
 * grid is a scan surface, not a stream reader. Callers can pass a larger
 * limit up to DEFAULT_BODY_LIMIT for a deeper dump.
 */
export const DEFAULT_HEX_LIMIT = 4 * 1024;

export interface Base64DecodeResult {
  bytes: Uint8Array;
  invalid: boolean;
  truncated: boolean;
}

export interface TextDecodeResult {
  text: string;
  invalid: boolean;
}

export interface SseEvent {
  data: string;
  event?: string;
  id?: string;
  retry?: number;
  comments: string[];
  fields: Array<{ name: string; value: string }>;
}

export interface DecodedBody {
  mode: BodyViewMode;
  text: string;
  copyText: string;
  byteLength: number;
  invalidEncoding: boolean;
  truncated: boolean;
  fallback: "none" | "text" | "hex";
  events?: SseEvent[];
  /**
   * SSE payload bytes that arrived after the last complete `\n\n`-terminated
   * frame. Preserved verbatim so a truncated trailing frame can still be
   * rendered and copied — the parser itself never returns partial events.
   */
  pendingSseSuffix?: string;
}

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function bodyData(body: BodyDescriptor): string {
  return body.state === "captured" || body.state === "truncated" ? body.data : "";
}

function toHex(bytes: Uint8Array): string {
  const lines: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 16) {
    const row = bytes.subarray(offset, offset + 16);
    const hex = Array.from(row, (byte) => byte.toString(16).padStart(2, "0")).join(" ");
    const padded = hex.padEnd(47, " ");
    const ascii = Array.from(row, (byte) => byte >= 0x20 && byte <= 0x7e ? String.fromCharCode(byte) : ".").join("");
    lines.push(`${offset.toString(16).padStart(8, "0")}  ${padded}  |${ascii}|`);
  }
  return lines.join("\n");
}

export function decodeBase64Bounded(value: string, limit = DEFAULT_BODY_LIMIT): Base64DecodeResult {
  if (!Number.isSafeInteger(limit) || limit < 0) throw new RangeError("limit must be a non-negative safe integer");
  const boundedLimit = limit;
  const inputLimit = Math.ceil(boundedLimit / 3) * 4;
  const inputWasTruncated = value.length > inputLimit;
  const boundedValue = inputWasTruncated ? value.slice(0, inputLimit) : value;
  if (boundedValue.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(boundedValue)) {
    return { bytes: new Uint8Array(), invalid: true, truncated: false };
  }

  const decodedLength = boundedValue.length / 4 * 3 - (boundedValue.endsWith("==") ? 2 : boundedValue.endsWith("=") ? 1 : 0);
  const output = new Uint8Array(Math.min(decodedLength, boundedLimit));
  let write = 0;
  for (let index = 0; index < boundedValue.length; index += 4) {
    const a = BASE64_ALPHABET.indexOf(boundedValue[index]);
    const b = BASE64_ALPHABET.indexOf(boundedValue[index + 1]);
    const c = boundedValue[index + 2] === "=" ? 0 : BASE64_ALPHABET.indexOf(boundedValue[index + 2]);
    const d = boundedValue[index + 3] === "=" ? 0 : BASE64_ALPHABET.indexOf(boundedValue[index + 3]);
    const block = (a << 18) | (b << 12) | (c << 6) | d;
    if (write < output.length) output[write++] = (block >> 16) & 0xff;
    if (boundedValue[index + 2] !== "=" && write < output.length) output[write++] = (block >> 8) & 0xff;
    if (boundedValue[index + 3] !== "=" && write < output.length) output[write++] = block & 0xff;
  }
  return { bytes: output, invalid: false, truncated: inputWasTruncated || decodedLength > boundedLimit };
}

export function decodeUtf8(bytes: Uint8Array, preserveBom = false): TextDecodeResult {
  try {
    // Only the SSE path keeps U+FEFF; its parser owns the protocol's
    // one-leading-BOM rule. Other text modes retain TextDecoder defaults.
    return { text: new TextDecoder("utf-8", { fatal: true, ignoreBOM: preserveBom }).decode(bytes), invalid: false };
  } catch {
    return { text: toHex(bytes), invalid: true };
  }
}

export interface SseStreamParseResult {
  readonly events: SseEvent[];
  readonly pendingSuffix: string;
}

/**
 * Parse SSE text and also return the unconsumed suffix — the bytes after
 * the last complete `\n\n`-terminated frame. The suffix is preserved
 * verbatim so a truncated trailing frame can still be surfaced instead of
 * silently dropped.
 */
export function parseSseStream(text: string): SseStreamParseResult {
  const events = parseSseEvents(text);
  const withoutBom = text.startsWith("﻿") ? text.slice(1) : text;
  const normalized = withoutBom.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
  // The parser terminates a frame on the first blank line, so any text after
  // the last "\n\n" (or the whole payload when none exists) is unconsumed.
  const lastBoundary = normalized.lastIndexOf("\n\n");
  const suffix = lastBoundary === -1 ? normalized : normalized.slice(lastBoundary + 2);
  return { events, pendingSuffix: suffix.length === 0 ? "" : suffix };
}

export function parseSseEvents(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  let data: string[] = [];
  let event: string | undefined;
  let lastEventId: string | undefined;
  let lastRetry: number | undefined;
  let comments: string[] = [];
  let fields: Array<{ name: string; value: string }> = [];

  const dispatch = () => {
    if (data.length > 0) events.push({ data: data.join("\n"), event, id: lastEventId, retry: lastRetry, comments, fields });
    data = [];
    event = undefined;
    comments = [];
    fields = [];
  };

  const withoutBom = text.startsWith("\uFEFF") ? text.slice(1) : text;
  const normalized = withoutBom.replaceAll("\r\n", "\n").replaceAll("\r", "\n");
  const lines = normalized.split("\n");
  if (lines.at(-1) === "") lines.pop();
  for (const line of lines) {
    if (line === "") {
      dispatch();
      continue;
    }
    if (line.startsWith(":")) {
      comments.push(line.slice(1).startsWith(" ") ? line.slice(2) : line.slice(1));
      continue;
    }
    const separator = line.indexOf(":");
    const name = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? "" : line.slice(separator + 1).startsWith(" ") ? line.slice(separator + 2) : line.slice(separator + 1);
    fields.push({ name, value });
    if (name === "data") data.push(value);
    else if (name === "event") event = value;
    else if (name === "id" && !value.includes("\0")) lastEventId = value;
    else if (name === "retry" && /^[0-9]+$/.test(value)) {
      const parsedRetry = Number(value);
      if (Number.isSafeInteger(parsedRetry)) lastRetry = parsedRetry;
    }
  }
  return events;
}

function formatSse(events: SseEvent[]): string {
  return events.map((item, index) => {
    const heading = `event ${String(index + 1).padStart(2, "0")}${item.event ? ` · ${item.event}` : ""}`;
    const detail = [
      item.id === undefined ? "" : `id: ${item.id}`,
      item.retry === undefined ? "" : `retry: ${item.retry}`,
      ...item.comments.map((comment) => `: ${comment}`),
      ...item.data.split("\n").map((line) => `data: ${line}`),
    ].filter(Boolean);
    return [heading, ...detail].join("\n");
  }).join("\n\n");
}

export function bodyMetadata(body: InspectableBody | undefined): {
  state: string;
  size: string;
  captured: string;
  contentType: string;
} {
  if (!body) return { state: "missing", size: "—", captured: "—", contentType: "not provided" };
  if (body.state === "redacted") return { state: "redacted", size: "—", captured: "—", contentType: body.content_type || "content withheld" };
  if (body.state === "missing") return { state: "missing", size: "—", captured: "—", contentType: body.content_type || "not observed" };
  if (body.state === "empty") return { state: "empty", size: "0 B", captured: "0 B", contentType: body.content_type || "no content type" };
  if (body.state === "captured") return { state: "captured", size: formatBytes(body.size_bytes), captured: formatBytes(body.size_bytes), contentType: body.content_type || "unknown" };
  return { state: "truncated", size: formatBytes(body.size_bytes), captured: formatBytes(body.captured_bytes), contentType: body.content_type || "unknown" };
}

export function formatBytes(decimal: string): string {
  const bytes = BigInt(decimal);
  if (bytes < 1024n) return `${bytes} B`;
  if (bytes < 1024n * 1024n) return `${(Number(bytes) / 1024).toFixed(1)} KiB`;
  return `${(Number(bytes) / (1024 * 1024)).toFixed(1)} MiB`;
}

export function decodeBody(body: InspectableBody, mode: BodyViewMode, limit = DEFAULT_BODY_LIMIT): DecodedBody {
  if (body.state === "redacted" || body.state === "missing") return { mode, text: "Body is not available for inspection.", copyText: "", byteLength: 0, invalidEncoding: false, truncated: false, fallback: "none" };
  if (body.state === "empty") return { mode, text: "(empty body)", copyText: "", byteLength: 0, invalidEncoding: false, truncated: false, fallback: "none", events: mode === "sse" ? [] : undefined };

  const decodeLimit = mode === "hex" ? Math.min(limit, DEFAULT_HEX_LIMIT) : limit;
  const base64 = decodeBase64Bounded(bodyData(body), decodeLimit);
  if (base64.invalid) return { mode, text: "Invalid base64 payload; no bytes were decoded.", copyText: "", byteLength: 0, invalidEncoding: true, truncated: false, fallback: "hex" };
  const utf8 = decodeUtf8(base64.bytes, mode === "sse");
  const truncated = base64.truncated || body.state === "truncated";
  if (mode === "hex" || utf8.invalid) {
    const text = toHex(base64.bytes);
    return { mode, text: utf8.invalid && mode !== "hex" ? `${text}\n\nUTF-8 decoding failed; showing bounded hex.` : text, copyText: text, byteLength: base64.bytes.length, invalidEncoding: utf8.invalid, truncated, fallback: utf8.invalid && mode !== "hex" ? "hex" : "none" };
  }
  if (mode === "json") {
    try {
      const parsed: unknown = JSON.parse(utf8.text);
      const text = JSON.stringify(parsed, null, 2);
      return { mode, text, copyText: text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "none" };
    } catch {
      return { mode, text: `${utf8.text}\n\nNot valid JSON; showing decoded text.`, copyText: utf8.text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "text" };
    }
  }
  if (mode === "sse") {
    const parseResult = parseSseStream(utf8.text);
    const { events, pendingSuffix } = parseResult;
    if (events.length === 0) {
      // Preserve the raw payload verbatim so garbage or partial frames stay
      // inspectable; the caller renders this via the plain-text fallback.
      const text = utf8.text || "(empty text)";
      return { mode, text, copyText: utf8.text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "text", events: [], pendingSseSuffix: pendingSuffix };
    }
    const text = pendingSuffix.length > 0 ? `${formatSse(events)}\n\ntruncated frame\n${pendingSuffix}` : formatSse(events);
    return { mode, text, copyText: text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "none", events, pendingSseSuffix: pendingSuffix };
  }
  return { mode, text: utf8.text || "(empty text)", copyText: utf8.text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "none" };
}

export function gateBodyDecode(body: InspectableBody, selected: boolean, mode: BodyViewMode): { selected: false; metadata: ReturnType<typeof bodyMetadata> } | { selected: true; decoded: DecodedBody } {
  if (!selected) return { selected: false, metadata: bodyMetadata(body) };
  return { selected: true, decoded: decodeBody(body, mode) };
}

/**
 * Best-guess view mode for a captured body. Prefers the specialised parsers
 * (JSON / SSE) whenever the recorded content type hints at them so the
 * caller sees pretty output on first render, without a manual toggle.
 */
export function defaultBodyMode(body: InspectableBody): BodyViewMode {
  const contentType = (body as { content_type?: string }).content_type;
  if (typeof contentType !== "string") return "text";
  const bare = contentType.split(";", 1)[0].trim().toLowerCase();
  if (bare === "text/event-stream") return "sse";
  if (bare === "application/json" || bare.endsWith("+json")) return "json";
  return "text";
}

export function hexDump(bytes: Uint8Array): string {
  return toHex(bytes);
}
