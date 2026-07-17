import type { BodyDescriptor } from "../../protocol";
import type { BodyViewMode, InspectableBody } from "./models";

export const DEFAULT_BODY_LIMIT = 64 * 1024;
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
  if (value.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) {
    return { bytes: new Uint8Array(), invalid: true, truncated: false };
  }

  const decodedLength = value.length / 4 * 3 - (value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0);
  const output = new Uint8Array(Math.min(decodedLength, limit));
  let write = 0;
  for (let index = 0; index < value.length; index += 4) {
    const a = BASE64_ALPHABET.indexOf(value[index]);
    const b = BASE64_ALPHABET.indexOf(value[index + 1]);
    const c = value[index + 2] === "=" ? 0 : BASE64_ALPHABET.indexOf(value[index + 2]);
    const d = value[index + 3] === "=" ? 0 : BASE64_ALPHABET.indexOf(value[index + 3]);
    const block = (a << 18) | (b << 12) | (c << 6) | d;
    if (write < output.length) output[write++] = (block >> 16) & 0xff;
    if (value[index + 2] !== "=" && write < output.length) output[write++] = (block >> 8) & 0xff;
    if (value[index + 3] !== "=" && write < output.length) output[write++] = block & 0xff;
  }
  return { bytes: output, invalid: false, truncated: decodedLength > limit };
}

export function decodeUtf8(bytes: Uint8Array): TextDecodeResult {
  try {
    return { text: new TextDecoder("utf-8", { fatal: true }).decode(bytes), invalid: false };
  } catch {
    return { text: toHex(bytes), invalid: true };
  }
}

export function parseSseEvents(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  let data: string[] = [];
  let event: string | undefined;
  let id: string | undefined;
  let retry: number | undefined;
  let comments: string[] = [];
  let fields: Array<{ name: string; value: string }> = [];

  const dispatch = () => {
    if (data.length === 0 && event === undefined && id === undefined && retry === undefined && comments.length === 0 && fields.length === 0) return;
    events.push({ data: data.join("\n"), event, id, retry, comments, fields });
    data = [];
    event = undefined;
    id = undefined;
    retry = undefined;
    comments = [];
    fields = [];
  };

  for (const line of text.replaceAll("\r\n", "\n").replaceAll("\r", "\n").split("\n")) {
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
    else if (name === "id" && !value.includes("\0")) id = value;
    else if (name === "retry" && /^[0-9]+$/.test(value)) retry = Number(value);
  }
  dispatch();
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
  const utf8 = decodeUtf8(base64.bytes);
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
    const events = parseSseEvents(utf8.text);
    const text = events.length === 0 ? "No complete SSE event frames found." : formatSse(events);
    return { mode, text, copyText: text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "none", events };
  }
  return { mode, text: utf8.text || "(empty text)", copyText: utf8.text, byteLength: base64.bytes.length, invalidEncoding: false, truncated, fallback: "none" };
}

export function gateBodyDecode(body: InspectableBody, selected: boolean, mode: BodyViewMode): { selected: false; metadata: ReturnType<typeof bodyMetadata> } | { selected: true; decoded: DecodedBody } {
  if (!selected) return { selected: false, metadata: bodyMetadata(body) };
  return { selected: true, decoded: decodeBody(body, mode) };
}

export function hexDump(bytes: Uint8Array): string {
  return toHex(bytes);
}
