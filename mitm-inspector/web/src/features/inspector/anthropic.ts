import type { JsonValue } from "./jsonTree";
import { safeParseJson } from "./jsonTree";

/**
 * Conversation model for Anthropic /v1/messages payloads.
 *
 * Design invariant: NO INFORMATION LOSS. Parsing only re-arranges the JSON —
 * every key that is not lifted into a typed field lands in `extra`, and any
 * block whose shape is not recognised falls back to `kind: "unknown"` with
 * its raw value intact. The raw JSON tree remains one toggle away in the UI.
 */

export type ExtraFields = ReadonlyArray<readonly [string, JsonValue]>;

type JsonObject = { [key: string]: JsonValue };

function asObject(value: JsonValue | undefined): JsonObject | null {
  if (value === null || value === undefined || typeof value !== "object" || Array.isArray(value)) return null;
  return value;
}

/** Collect every key not claimed by the typed representation. */
function extraFields(object: JsonObject, claimed: readonly string[]): ExtraFields {
  const claimedSet = new Set(claimed);
  return Object.entries(object).filter(([key]) => !claimedSet.has(key));
}

function hasField(object: JsonObject, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(object, key);
}

export interface TextBlock {
  kind: "text";
  text: string;
  cacheControl?: JsonValue;
  extra: ExtraFields;
}

export interface ThinkingBlock {
  kind: "thinking";
  thinking: string;
  signature?: string;
  cacheControl?: JsonValue;
  extra: ExtraFields;
}

export interface RedactedThinkingBlock {
  kind: "redacted_thinking";
  data?: string;
  extra: ExtraFields;
}

export interface ToolUseBlock {
  kind: "tool_use";
  id?: string;
  name: string;
  input: JsonValue;
  /** Raw partial input JSON when an SSE stream ended before the input parsed. */
  inputRaw?: string;
  cacheControl?: JsonValue;
  extra: ExtraFields;
}

export interface ToolResultBlock {
  kind: "tool_result";
  toolUseId?: string;
  isError?: boolean;
  content: ContentBlock[];
  cacheControl?: JsonValue;
  extra: ExtraFields;
}

export interface ImageBlock {
  kind: "image";
  mediaType?: string;
  /** data: URI when the source carries inline base64 data. */
  dataUri?: string;
  sourceLabel: string;
  cacheControl?: JsonValue;
  extra: ExtraFields;
}

export interface UnknownBlock {
  kind: "unknown";
  raw: JsonValue;
}

export type ContentBlock =
  | TextBlock
  | ThinkingBlock
  | RedactedThinkingBlock
  | ToolUseBlock
  | ToolResultBlock
  | ImageBlock
  | UnknownBlock;

function imageBlock(object: JsonObject): ImageBlock {
  const source = asObject(object.source);
  let mediaType: string | undefined;
  let dataUri: string | undefined;
  let sourceLabel = "image";
  if (source !== null) {
    if (typeof source.media_type === "string") mediaType = source.media_type;
    if (source.type === "base64" && typeof source.data === "string" && mediaType !== undefined) {
      dataUri = `data:${mediaType};base64,${source.data}`;
      sourceLabel = "base64";
    } else if (source.type === "url" && typeof source.url === "string") {
      sourceLabel = source.url;
    } else if (typeof source.type === "string") {
      sourceLabel = source.type;
    }
  }
  const block: ImageBlock = { kind: "image", sourceLabel, extra: extraFields(object, ["type", "cache_control"]) };
  if (mediaType !== undefined) block.mediaType = mediaType;
  if (dataUri !== undefined) block.dataUri = dataUri;
  if (object.cache_control !== undefined) block.cacheControl = object.cache_control;
  return block;
}

/** Parse one content block; anything unrecognised falls back to `unknown`. */
export function parseContentBlock(value: JsonValue): ContentBlock {
  const object = asObject(value);
  if (object === null) return { kind: "unknown", raw: value };
  const type = object.type;

  if (type === "text" && typeof object.text === "string") {
    const block: TextBlock = { kind: "text", text: object.text, extra: extraFields(object, ["type", "text", "cache_control"]) };
    if (object.cache_control !== undefined) block.cacheControl = object.cache_control;
    return block;
  }
  if (type === "thinking" && typeof object.thinking === "string") {
    const claimed = ["type", "thinking", "cache_control"];
    const block: ThinkingBlock = { kind: "thinking", thinking: object.thinking, extra: [] };
    if (typeof object.signature === "string") {
      block.signature = object.signature;
      claimed.push("signature");
    }
    block.extra = extraFields(object, claimed);
    if (object.cache_control !== undefined) block.cacheControl = object.cache_control;
    return block;
  }
  if (type === "redacted_thinking") {
    const claimed = ["type"];
    const block: RedactedThinkingBlock = { kind: "redacted_thinking", extra: [] };
    if (typeof object.data === "string") {
      block.data = object.data;
      claimed.push("data");
    }
    block.extra = extraFields(object, claimed);
    return block;
  }
  if (type === "tool_use" && typeof object.name === "string") {
    const claimed = ["type", "name", "cache_control"];
    const block: ToolUseBlock = {
      kind: "tool_use",
      name: object.name,
      input: object.input ?? null,
      extra: [],
    };
    if (typeof object.id === "string") {
      block.id = object.id;
      claimed.push("id");
    }
    if (hasField(object, "input")) claimed.push("input");
    block.extra = extraFields(object, claimed);
    if (object.cache_control !== undefined) block.cacheControl = object.cache_control;
    return block;
  }
  if (type === "tool_result") {
    const claimed = ["type", "cache_control"];
    const block: ToolResultBlock = {
      kind: "tool_result",
      content: contentBlocks(object.content),
      extra: [],
    };
    if (typeof object.tool_use_id === "string") {
      block.toolUseId = object.tool_use_id;
      claimed.push("tool_use_id");
    }
    if (typeof object.is_error === "boolean") {
      block.isError = object.is_error;
      claimed.push("is_error");
    }
    if (typeof object.content === "string" || Array.isArray(object.content)) claimed.push("content");
    block.extra = extraFields(object, claimed);
    if (object.cache_control !== undefined) block.cacheControl = object.cache_control;
    return block;
  }
  if (type === "image") return imageBlock(object);
  return { kind: "unknown", raw: value };
}

/** Normalise message/tool_result content: string shorthand becomes one text block. */
export function contentBlocks(value: JsonValue | undefined): ContentBlock[] {
  if (value === undefined || value === null) return [];
  if (typeof value === "string") return [{ kind: "text", text: value, extra: [] }];
  if (!Array.isArray(value)) return [{ kind: "unknown", raw: value }];
  return value.map(parseContentBlock);
}

export interface MessageView {
  role: string;
  blocks: ContentBlock[];
  extra: ExtraFields;
}

export interface ToolView {
  name: string;
  description?: string;
  inputSchema?: JsonValue;
  raw: JsonValue;
}

export interface AnthropicRequest {
  model?: string;
  stream?: boolean;
  maxTokens?: number;
  effort?: string | number;
  /** Raw output_config, retained so structured-output/future children stay inspectable. */
  outputConfig?: JsonValue;
  thinking?: JsonValue;
  system: ContentBlock[];
  tools: ToolView[];
  messages: MessageView[];
  extra: ExtraFields;
}

function parseMessage(value: JsonValue): MessageView {
  const object = asObject(value);
  if (object === null) return { role: "?", blocks: [{ kind: "unknown", raw: value }], extra: [] };
  const claimed: string[] = [];
  const role = typeof object.role === "string" ? object.role : "?";
  if (typeof object.role === "string") claimed.push("role");
  if (typeof object.content === "string" || Array.isArray(object.content)) claimed.push("content");
  return {
    role,
    blocks: contentBlocks(object.content),
    extra: extraFields(object, claimed),
  };
}

function parseTool(value: JsonValue): ToolView {
  const object = asObject(value);
  if (object === null) return { name: "(unnamed tool)", raw: value };
  const tool: ToolView = { name: typeof object.name === "string" ? object.name : "(unnamed tool)", raw: value };
  if (typeof object.description === "string") tool.description = object.description;
  if (object.input_schema !== undefined) tool.inputSchema = object.input_schema;
  return tool;
}

/**
 * Parse a /v1/messages (or count_tokens) request body into the conversation
 * model. Returns null when the payload does not look like one, so callers
 * can fall back to the raw JSON tree.
 */
export function parseAnthropicRequest(value: JsonValue): AnthropicRequest | null {
  const object = asObject(value);
  if (object === null || !Array.isArray(object.messages)) return null;
  const request: AnthropicRequest = {
    system: contentBlocks(object.system),
    tools: Array.isArray(object.tools) ? object.tools.map(parseTool) : object.tools === undefined ? [] : [parseTool(object.tools)],
    messages: object.messages.map(parseMessage),
    extra: [],
  };
  const claimed = ["messages"];
  if (typeof object.model === "string") {
    request.model = object.model;
    claimed.push("model");
  }
  if (typeof object.stream === "boolean") {
    request.stream = object.stream;
    claimed.push("stream");
  }
  if (typeof object.max_tokens === "number") {
    request.maxTokens = object.max_tokens;
    claimed.push("max_tokens");
  }
  if (typeof object.effort === "string" || typeof object.effort === "number") {
    request.effort = object.effort;
    claimed.push("effort");
  }
  const outputConfig = asObject(object.output_config);
  if (request.effort === undefined && outputConfig !== null && (typeof outputConfig.effort === "string" || typeof outputConfig.effort === "number")) {
    request.effort = outputConfig.effort;
  }
  if (object.output_config !== undefined) {
    request.outputConfig = object.output_config;
    claimed.push("output_config");
  }
  if (object.thinking !== undefined) {
    request.thinking = object.thinking;
    claimed.push("thinking");
  }
  if (object.system !== undefined) claimed.push("system");
  if (object.tools !== undefined) claimed.push("tools");
  request.extra = extraFields(object, claimed);
  return request;
}

export interface AnthropicResponse {
  model?: string;
  role?: string;
  blocks: ContentBlock[];
  stopReason?: string;
  usage?: JsonValue;
  extra: ExtraFields;
}

/** Parse a plain (non-streaming) /v1/messages response body. */
export function parseAnthropicResponse(value: JsonValue): AnthropicResponse | null {
  const object = asObject(value);
  if (object === null || !Array.isArray(object.content)) return null;
  const claimed = ["content"];
  const response: AnthropicResponse = {
    blocks: contentBlocks(object.content),
    extra: [],
  };
  if (typeof object.model === "string") {
    response.model = object.model;
    claimed.push("model");
  }
  if (typeof object.role === "string") {
    response.role = object.role;
    claimed.push("role");
  }
  if (typeof object.stop_reason === "string") {
    response.stopReason = object.stop_reason;
    claimed.push("stop_reason");
  }
  if (object.usage !== undefined) response.usage = object.usage;
  response.extra = extraFields(object, claimed);
  return response;
}

export interface SseFrame {
  /** Exact frame text as captured, terminator excluded. */
  raw: string;
  event?: string;
  data?: string;
  /** False only for a trailing frame cut off mid-stream by capture limits. */
  complete: boolean;
}

/**
 * Split an SSE stream into frames, preserving the exact captured bytes —
 * including a truncated trailing frame, which is kept with complete=false.
 */
export function splitSseFrames(text: string): SseFrame[] {
  if (text.length === 0) return [];
  const frames: SseFrame[] = [];
  const parts = text.split(/\r?\n\r?\n/);
  const trailer = parts.pop() ?? "";
  for (const part of parts) frames.push(parseFrame(part, true));
  if (trailer.trim().length > 0) frames.push(parseFrame(trailer, false));
  return frames;
}

function parseFrame(raw: string, complete: boolean): SseFrame {
  const frame: SseFrame = { raw, complete };
  const dataLines: string[] = [];
  for (const line of raw.split(/\r?\n/)) {
    if (line.startsWith("event:")) frame.event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  if (dataLines.length > 0) frame.data = dataLines.join("\n");
  return frame;
}

export interface AssembledStream {
  model?: string;
  role?: string;
  blocks: ContentBlock[];
  stopReason?: string;
  usage?: JsonValue;
  frames: SseFrame[];
  /** Frames whose data payload did not parse as JSON (excluding the truncated trailer). */
  undecodedFrames: number;
}

interface StreamingBlock {
  block: ContentBlock;
  partialJson: string;
  signature: string;
}

function mergeUsage(current: JsonValue | undefined, incoming: JsonValue): JsonValue {
  const base = asObject(current ?? null);
  const update = asObject(incoming);
  if (base === null || update === null) return incoming;
  return { ...base, ...update };
}

/** Reassemble an SSE /v1/messages response stream into readable content blocks. */
export function assembleSse(text: string): AssembledStream {
  const frames = splitSseFrames(text);
  const assembled: AssembledStream = { blocks: [], frames, undecodedFrames: 0 };
  const open = new Map<number, StreamingBlock>();
  const ordered: StreamingBlock[] = [];

  for (const frame of frames) {
    if (frame.data === undefined) continue;
    const parsed = safeParseJson(frame.data);
    if (!parsed.ok) {
      if (frame.complete) assembled.undecodedFrames += 1;
      continue;
    }
    const payload = asObject(parsed.value);
    if (payload === null) continue;

    if (payload.type === "message_start") {
      const message = asObject(payload.message);
      if (message !== null) {
        if (typeof message.model === "string") assembled.model = message.model;
        if (typeof message.role === "string") assembled.role = message.role;
        if (message.usage !== undefined) assembled.usage = mergeUsage(assembled.usage, message.usage);
      }
    } else if (payload.type === "content_block_start" && typeof payload.index === "number") {
      const streaming: StreamingBlock = {
        block: parseContentBlock(payload.content_block ?? null),
        partialJson: "",
        signature: "",
      };
      open.set(payload.index, streaming);
      ordered.push(streaming);
    } else if (payload.type === "content_block_delta" && typeof payload.index === "number") {
      const streaming = open.get(payload.index);
      const delta = asObject(payload.delta);
      if (streaming !== undefined && delta !== null) applyDelta(streaming, delta);
    } else if (payload.type === "message_delta") {
      const delta = asObject(payload.delta);
      if (delta !== null && typeof delta.stop_reason === "string") assembled.stopReason = delta.stop_reason;
      if (payload.usage !== undefined) assembled.usage = mergeUsage(assembled.usage, payload.usage);
    }
  }

  for (const streaming of ordered) {
    if (streaming.block.kind === "thinking" && streaming.signature.length > 0) {
      streaming.block.signature = streaming.signature;
    }
    if (streaming.block.kind === "tool_use" && streaming.partialJson.length > 0) {
      const parsed = safeParseJson(streaming.partialJson);
      if (parsed.ok) streaming.block.input = parsed.value;
      else streaming.block.inputRaw = streaming.partialJson;
    }
  }
  assembled.blocks = ordered.map((streaming) => streaming.block);
  return assembled;
}

function applyDelta(streaming: StreamingBlock, delta: JsonObject): void {
  const { block } = streaming;
  if (delta.type === "text_delta" && typeof delta.text === "string" && block.kind === "text") {
    block.text += delta.text;
  } else if (delta.type === "thinking_delta" && typeof delta.thinking === "string" && block.kind === "thinking") {
    block.thinking += delta.thinking;
  } else if (delta.type === "signature_delta" && typeof delta.signature === "string" && block.kind === "thinking") {
    streaming.signature += delta.signature;
  } else if (delta.type === "input_json_delta" && typeof delta.partial_json === "string") {
    streaming.partialJson += delta.partial_json;
  }
}

/** Whether a response body looks like an SSE event stream. */
export function looksLikeSse(text: string, contentType?: string): boolean {
  if (contentType !== undefined && contentType.toLowerCase().includes("text/event-stream")) return true;
  const head = text.slice(0, 512).trimStart();
  return head.startsWith("event:") || head.startsWith("data:");
}
