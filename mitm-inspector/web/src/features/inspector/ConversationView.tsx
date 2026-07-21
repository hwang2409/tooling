import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { FlowExtras } from "../../protocol";
import type {
  AnthropicRequest,
  AnthropicResponse,
  AssembledStream,
  ContentBlock,
  ExtraFields,
  ToolView,
} from "./anthropic";
import { assembleSse, looksLikeSse, parseAnthropicResponse } from "./anthropic";
import { MarkdownProse } from "./Markdown";
import { JsonTree, safeParseJson } from "./jsonTree";
import type { JsonValue } from "./jsonTree";
import { durationBetween, formatBytes } from "../flows/rowSummary";

export type ResponseView =
  | { kind: "absent" }
  | { kind: "sse"; assembled: AssembledStream; text: string }
  | { kind: "json"; parsed: AnthropicResponse }
  | { kind: "json-raw"; value: JsonValue }
  | { kind: "text"; text: string };

/** Classify the decoded response body for conversation-style rendering. */
export function deriveResponseView(text: string | undefined, contentType: string | undefined): ResponseView {
  if (text === undefined) return { kind: "absent" };
  if (looksLikeSse(text, contentType)) return { kind: "sse", assembled: assembleSse(text), text };
  const parsed = safeParseJson(text);
  if (!parsed.ok) return { kind: "text", text };
  const conversation = parseAnthropicResponse(parsed.value);
  if (conversation !== null) return { kind: "json", parsed: conversation };
  return { kind: "json-raw", value: parsed.value };
}

function utf8Length(text: string): number {
  return new TextEncoder().encode(text).length;
}

function firstLine(text: string): string {
  const line = text.split("\n", 1)[0];
  return line.length > 160 ? `${line.slice(0, 160)}…` : line;
}

export function Collapse({ label, meta, children, defaultOpen = false, className }: {
  label: string;
  meta?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`conv-collapse${className ? ` ${className}` : ""}`}>
      <button type="button" className="conv-collapse-head" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <span className="conv-collapse-marker">{open ? "−" : "+"}</span>
        <span className="conv-collapse-label">{label}</span>
        {meta}
      </button>
      {open ? <div className="conv-collapse-body">{children}</div> : null}
    </div>
  );
}

function CacheBadge({ value }: { value: JsonValue | undefined }) {
  if (value === undefined) return null;
  const type = typeof value === "object" && value !== null && !Array.isArray(value) && typeof value.type === "string"
    ? value.type
    : null;
  return <span className="conv-badge" title={JSON.stringify(value)}>cache{type !== null && type !== "ephemeral" ? `:${type}` : ""}</span>;
}

function MoreFields({ extra }: { extra: ExtraFields }) {
  if (extra.length === 0) return null;
  const value = Object.fromEntries(extra) as JsonValue;
  return (
    <Collapse className="conv-more" label={`${extra.length} more ${extra.length === 1 ? "field" : "fields"}`}>
      <JsonTree value={value} />
    </Collapse>
  );
}

function BlockView({ block, toolNames, markdown }: { block: ContentBlock; toolNames: ReadonlyMap<string, string>; markdown: boolean }) {
  if (block.kind === "text") {
    return (
      <div className="conv-block conv-block-text">
        {markdown ? <MarkdownProse text={block.text} /> : <span className="conv-prose">{block.text}</span>}
        <CacheBadge value={block.cacheControl} />
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  if (block.kind === "thinking") {
    return (
      <div className="conv-block conv-block-thinking">
        <span className="conv-thinking-label">thinking</span>
        <span className="conv-prose conv-thinking-text">{block.thinking}</span>
        {block.signature !== undefined ? (
          <Collapse className="conv-more" label="signature">
            <pre className="conv-pre">{block.signature}</pre>
          </Collapse>
        ) : null}
        <CacheBadge value={block.cacheControl} />
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  if (block.kind === "redacted_thinking") {
    return (
      <div className="conv-block conv-block-thinking">
        <span className="conv-thinking-label">redacted thinking</span>
        {block.data !== undefined ? (
          <Collapse className="conv-more" label="data">
            <pre className="conv-pre">{block.data}</pre>
          </Collapse>
        ) : null}
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  if (block.kind === "tool_use") {
    // Payload always renders as a labeled code block: use the captured raw
    // string when present so the wire representation is preserved verbatim;
    // otherwise pretty-print the parsed input with a stable 2-space indent
    // (never as a JsonTree — the kickoff wants a real <pre><code>).
    const body = block.inputRaw ?? JSON.stringify(block.input, null, 2);
    return (
      <div className="conv-block conv-card conv-card-tool-use">
        <div className="conv-card-head">
          <span className="conv-card-kind">tool_use</span>
          <span className="conv-card-name">{block.name}</span>
          {block.id !== undefined ? <span className="conv-card-id">{block.id}</span> : null}
          <CacheBadge value={block.cacheControl} />
        </div>
        <pre className="conv-pre"><code>{body}</code></pre>
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  if (block.kind === "tool_result") {
    const linkedName = block.toolUseId !== undefined ? toolNames.get(block.toolUseId) : undefined;
    return (
      <div className="conv-block conv-card conv-card-tool-result">
        <div className="conv-card-head">
          <span className="conv-card-kind">↳ tool_result</span>
          <span className="conv-card-name">{linkedName ?? block.toolUseId ?? "?"}</span>
          {block.isError === true ? <span className="conv-card-error">error</span> : null}
          <CacheBadge value={block.cacheControl} />
        </div>
        {/* Tool output is data, not prose — every child renders as a labeled
            code block. Text/thinking children go straight into <pre>; nested
            tool_use payloads pretty-print; images / unknown blocks fall
            through to their normal card renderer with markdown disabled. */}
        {block.content.map((child, index) => {
          if (child.kind === "text") {
            return <pre key={index} className="conv-pre"><code>{child.text}</code></pre>;
          }
          if (child.kind === "thinking") {
            return <pre key={index} className="conv-pre"><code>{child.thinking}</code></pre>;
          }
          return <BlockView key={index} block={child} toolNames={toolNames} markdown={false} />;
        })}
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  if (block.kind === "image") {
    return (
      <div className="conv-block conv-card conv-card-image">
        <div className="conv-card-head">
          <span className="conv-card-kind">image</span>
          <span className="conv-card-name">{block.mediaType ?? block.sourceLabel}</span>
          <CacheBadge value={block.cacheControl} />
        </div>
        {block.dataUri !== undefined
          ? <img className="conv-image" src={block.dataUri} alt={block.mediaType ?? "image block"} />
          : <span className="conv-muted">({block.sourceLabel})</span>}
        <MoreFields extra={block.extra} />
      </div>
    );
  }
  return (
    <div className="conv-block conv-block-unknown">
      <JsonTree value={block.raw} />
    </div>
  );
}

function collectToolNames(request: AnthropicRequest, response: ResponseView): ReadonlyMap<string, string> {
  const names = new Map<string, string>();
  const visit = (blocks: readonly ContentBlock[]) => {
    for (const block of blocks) {
      if (block.kind === "tool_use" && block.id !== undefined) names.set(block.id, block.name);
      if (block.kind === "tool_result") visit(block.content);
    }
  };
  for (const message of request.messages) visit(message.blocks);
  if (response.kind === "sse") visit(response.assembled.blocks);
  if (response.kind === "json") visit(response.parsed.blocks);
  return names;
}

function usageEntries(usage: JsonValue | undefined): Array<[string, string]> {
  if (usage === undefined || usage === null || typeof usage !== "object" || Array.isArray(usage)) return [];
  const entries: Array<[string, string]> = [];
  for (const [key, value] of Object.entries(usage)) {
    if (typeof value === "number" || typeof value === "string") entries.push([key, String(value)]);
  }
  return entries;
}

function thinkingLabel(thinking: JsonValue): string {
  if (typeof thinking === "object" && thinking !== null && !Array.isArray(thinking)) {
    const type = typeof thinking.type === "string" ? thinking.type : "?";
    const budget = typeof thinking.budget_tokens === "number" ? `·${thinking.budget_tokens}` : "";
    return `${type}${budget}`;
  }
  return JSON.stringify(thinking);
}

function HeaderStrip({ request, extras, response }: { request: AnthropicRequest; extras: FlowExtras; response: ResponseView }) {
  const summary = extras.summary;
  const usage = response.kind === "sse" ? response.assembled.usage : response.kind === "json" ? response.parsed.usage : undefined;
  const usagePairs = new Map(usageEntries(usage));
  const token = (usageKey: string, summaryValue: string | undefined) => usagePairs.get(usageKey) ?? summaryValue;
  const stopReason = (response.kind === "sse" ? response.assembled.stopReason : response.kind === "json" ? response.parsed.stopReason : undefined)
    ?? summary?.stop_reason;

  const items: Array<[string, string]> = [];
  const model = request.model ?? summary?.model;
  if (model !== undefined) items.push(["model", model]);
  if (request.stream !== undefined || summary?.stream !== undefined) items.push(["stream", String(request.stream ?? summary?.stream)]);
  if (request.maxTokens !== undefined) items.push(["max_tokens", String(request.maxTokens)]);
  if (request.thinking !== undefined) items.push(["thinking", thinkingLabel(request.thinking)]);
  if (request.effort !== undefined) items.push(["effort", String(request.effort)]);
  const tokens = [
    ["in", token("input_tokens", summary?.input_tokens)],
    ["cache", token("cache_read_input_tokens", summary?.cache_read_input_tokens)],
    ["out", token("output_tokens", summary?.output_tokens)],
    ["think", token("thinking_tokens", summary?.thinking_tokens)],
  ].filter((pair): pair is [string, string] => pair[1] !== undefined);
  if (tokens.length > 0) items.push(["tokens", tokens.map(([label, value]) => `${label} ${value}`).join(" · ")]);
  if (stopReason !== undefined) items.push(["stop", stopReason]);
  const duration = durationBetween(extras.started_at, extras.ended_at);
  if (duration !== "—") items.push(["duration", duration]);

  return (
    <div className="conv-header">
      {items.map(([label, value]) => (
        <span key={label} className="conv-header-item">
          <span className="conv-header-label">{label}</span>
          <span className="conv-header-value">{value}</span>
        </span>
      ))}
      {request.outputConfig !== undefined ? (
        <Collapse className="conv-header-raw" label="output_config">
          <JsonTree value={request.outputConfig} startCollapsed />
        </Collapse>
      ) : null}
      {request.thinking !== undefined ? (
        <Collapse className="conv-header-raw" label="thinking details">
          <JsonTree value={request.thinking} />
        </Collapse>
      ) : null}
      <MoreFields extra={request.extra} />
    </div>
  );
}

function SystemSection({ blocks, markdown }: { blocks: readonly ContentBlock[]; markdown: boolean }) {
  if (blocks.length === 0) return null;
  return (
    <section className="conv-section">
      <Collapse
        label={`system (${blocks.length})`}
        meta={
          <span className="conv-section-meta">
            {blocks.map((block, index) => (
              block.kind === "text" ? <CacheBadge key={index} value={block.cacheControl} /> : null
            ))}
          </span>
        }
      >
        {blocks.map((block, index) => {
          if (block.kind !== "text") return <BlockView key={index} block={block} toolNames={new Map()} markdown={markdown} />;
          return (
            <Collapse
              key={index}
              className="conv-system-block"
              label={firstLine(block.text) || "(empty)"}
              meta={
                <span className="conv-section-meta">
                  <span className="conv-size">{formatBytes(String(utf8Length(block.text)))}</span>
                  <CacheBadge value={block.cacheControl} />
                </span>
              }
            >
              {markdown ? <MarkdownProse text={block.text} /> : <span className="conv-prose">{block.text}</span>}
              <MoreFields extra={block.extra} />
            </Collapse>
          );
        })}
      </Collapse>
    </section>
  );
}

function ToolsSection({ tools }: { tools: readonly ToolView[] }) {
  if (tools.length === 0) return null;
  return (
    <section className="conv-section">
      <Collapse label={`tools (${tools.length})`} meta={<span className="conv-section-meta">{tools.map((tool) => tool.name).join(" · ")}</span>}>
        {tools.map((tool, index) => (
          <Collapse key={index} className="conv-tool" label={tool.name} meta={tool.description !== undefined ? <span className="conv-section-meta">{firstLine(tool.description)}</span> : undefined}>
            <JsonTree value={tool.raw} startCollapsed />
          </Collapse>
        ))}
      </Collapse>
    </section>
  );
}

function UsageLine({ usage }: { usage: JsonValue | undefined }) {
  const entries = usageEntries(usage);
  if (entries.length === 0) return null;
  return (
    <div className="conv-usage">
      {entries.map(([key, value]) => `${key} ${value}`).join(" · ")}
    </div>
  );
}

function SseFrames({ assembled }: { assembled: AssembledStream }) {
  return (
    <Collapse className="conv-more" label={`raw frames (${assembled.frames.length})`}>
      <ol className="conv-frames">
        {assembled.frames.map((frame, index) => (
          <li key={index} className="conv-frame">
            <pre className="conv-pre">{frame.raw}</pre>
            {!frame.complete ? <span className="conv-muted">(truncated frame)</span> : null}
          </li>
        ))}
      </ol>
    </Collapse>
  );
}

function ResponseMeta({ model, role }: { model?: string; role?: string }) {
  return (
    <div className="conv-role conv-role-response">
      response
      {role !== undefined ? <span className="conv-response-meta">role {role}</span> : null}
      {model !== undefined ? <span className="conv-response-meta">model {model}</span> : null}
    </div>
  );
}

function ResponseSection({ response, toolNames, markdown }: { response: ResponseView; toolNames: ReadonlyMap<string, string>; markdown: boolean }) {
  if (response.kind === "absent") return null;
  return (
    <section className="conv-section conv-response">
      {response.kind === "sse" ? <ResponseMeta model={response.assembled.model} role={response.assembled.role} /> : null}
      {response.kind === "json" ? <ResponseMeta model={response.parsed.model} role={response.parsed.role} /> : null}
      {response.kind !== "sse" && response.kind !== "json" ? <ResponseMeta /> : null}
      {response.kind === "sse" ? (
        <>
          {response.assembled.blocks.map((block, index) => (
            <BlockView key={index} block={block} toolNames={toolNames} markdown={markdown} />
          ))}
          {response.assembled.undecodedFrames > 0
            ? <span className="conv-muted">({response.assembled.undecodedFrames} undecoded frames)</span>
            : null}
          <UsageLine usage={response.assembled.usage} />
          <SseFrames assembled={response.assembled} />
        </>
      ) : null}
      {response.kind === "json" ? (
        <>
          {response.parsed.blocks.map((block, index) => (
            <BlockView key={index} block={block} toolNames={toolNames} markdown={markdown} />
          ))}
          <UsageLine usage={response.parsed.usage} />
          <MoreFields extra={response.parsed.extra} />
        </>
      ) : null}
      {response.kind === "json-raw" ? <JsonTree value={response.value} /> : null}
      {response.kind === "text" ? <pre className="packet-text">{response.text}</pre> : null}
    </section>
  );
}

export interface ConversationViewProps {
  request: AnthropicRequest;
  extras: FlowExtras;
  responseText?: string;
  responseContentType?: string;
  /**
   * When true, suppress the request/response header meta strip
   * (model/stop/tokens). The chat surface owns the meta rendering itself
   * so the transcript stays as chat-only chrome.
   */
  chromeless?: boolean;
}

export function ConversationView({ request, extras, responseText, responseContentType, chromeless = false }: ConversationViewProps) {
  const [plainText, setPlainText] = useState(false);
  const markdown = !plainText;
  const response = useMemo(
    () => deriveResponseView(responseText, responseContentType ?? extras.response_content_type),
    [responseText, responseContentType, extras],
  );
  const toolNames = useMemo(() => collectToolNames(request, response), [request, response]);
  return (
    <div className={`conv${chromeless ? " conv-chromeless" : ""}`} data-testid="conversation-view">
      <div className="conv-text-modes">
        <button type="button" className="packet-mode" aria-pressed={markdown} onClick={() => setPlainText(false)}>md</button>
        <button type="button" className="packet-mode" aria-pressed={plainText} onClick={() => setPlainText(true)}>plain</button>
      </div>
      {chromeless ? null : <HeaderStrip request={request} extras={extras} response={response} />}
      <SystemSection blocks={request.system} markdown={markdown} />
      <ToolsSection tools={request.tools} />
      <section className="conv-section conv-transcript">
        {request.messages.map((message, index) => (
          <div key={index} className={`conv-message conv-message-${message.role}`}>
            <div className="conv-role">{message.role}</div>
            {message.blocks.map((block, blockIndex) => (
              <BlockView key={blockIndex} block={block} toolNames={toolNames} markdown={markdown} />
            ))}
            <MoreFields extra={message.extra} />
          </div>
        ))}
      </section>
      <ResponseSection response={response} toolNames={toolNames} markdown={markdown} />
    </div>
  );
}
