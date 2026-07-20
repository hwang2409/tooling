import { useState } from "react";

export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };

/** Parse a JSON string, or return the sentinel error result. */
export function safeParseJson(text: string): { ok: true; value: JsonValue } | { ok: false; error: string } {
  try {
    return { ok: true, value: JSON.parse(text) as JsonValue };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : "invalid JSON" };
  }
}

/**
 * Whether every array/object descendant is small enough to keep expanded on
 * first render without turning into a wall of text.
 */
export function shouldStartCollapsed(value: JsonValue, depth = 0): boolean {
  if (depth >= 3) return true;
  if (Array.isArray(value)) {
    if (value.length > 8) return true;
    return value.some((item) => shouldStartCollapsed(item, depth + 1));
  }
  if (value !== null && typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length > 12) return true;
    return entries.some(([, item]) => shouldStartCollapsed(item, depth + 1));
  }
  return false;
}

/** Compact summary for a collapsed node, e.g. "{ 3 keys }" or "[ 42 ]". */
export function collapsedSummary(value: JsonValue): string {
  if (Array.isArray(value)) return `[ ${value.length} ${value.length === 1 ? "item" : "items"} ]`;
  const entries = Object.entries(value as { [key: string]: JsonValue });
  return `{ ${entries.length} ${entries.length === 1 ? "key" : "keys"} }`;
}

interface JsonNodeProps {
  value: JsonValue;
  depth: number;
  keyLabel?: string;
  trailingComma?: boolean;
  forceCollapsed?: boolean;
}

/**
 * Threshold above which the tree root defaults to collapsed regardless of
 * shape. Uncapped body prefixes (see decoders.DEFAULT_BODY_LIMIT) can be
 * multi-megabyte; mounting every node up-front on a 2 MiB payload freezes
 * the DOM for hundreds of ms. Collapse-first keeps the initial render cheap
 * — the user opens the branches they care about.
 */
export const LARGE_TREE_COLLAPSE_THRESHOLD = 256 * 1024;

function JsonScalar({ value }: { value: JsonValue }) {
  if (value === null) return <span className="json-null">null</span>;
  if (typeof value === "boolean") return <span className="json-bool">{String(value)}</span>;
  if (typeof value === "number") return <span className="json-number">{Number.isFinite(value) ? String(value) : "null"}</span>;
  return <span className="json-string">&quot;{escapeJsonString(value as string)}&quot;</span>;
}

/**
 * A string value that itself parses as a JSON object or array. Common in
 * gateway payloads where structured content gets stringified (`"user":"{...}"`).
 * We surface these as nested trees so the escaped `\"` never reaches the user.
 * Only triggers when the trimmed string starts with `{` or `[` — a cheap
 * heuristic that keeps plain strings from being probed on every render.
 */
function embeddedJson(value: string): JsonValue | null {
  const trimmed = value.trim();
  if (trimmed.length < 2) return null;
  const first = trimmed.charCodeAt(0);
  if (first !== 0x7b /* { */ && first !== 0x5b /* [ */) return null;
  const parsed = safeParseJson(trimmed);
  if (!parsed.ok) return null;
  const inner = parsed.value;
  if (inner === null || typeof inner !== "object") return null;
  return inner;
}

// eslint-disable-next-line no-control-regex -- JSON string escaping intentionally targets the ASCII control range.
const JSON_ESCAPE_PATTERN = /[\\"\b\f\n\r\t\x00-\x1f]/g;

function escapeJsonString(value: string): string {
  return value.replace(JSON_ESCAPE_PATTERN, (character) => {
    switch (character) {
      case "\\": return "\\\\";
      case '"': return '\\"';
      case "\b": return "\\b";
      case "\f": return "\\f";
      case "\n": return "\\n";
      case "\r": return "\\r";
      case "\t": return "\\t";
      default: return `\\u${character.charCodeAt(0).toString(16).padStart(4, "0")}`;
    }
  });
}

function JsonNode({ value, depth, keyLabel, trailingComma, forceCollapsed }: JsonNodeProps) {
  const isObject = value !== null && typeof value === "object";
  const isArray = Array.isArray(value);
  const [collapsed, setCollapsed] = useState<boolean>(() => isObject ? (forceCollapsed === true ? true : shouldStartCollapsed(value, depth)) : false);

  const prefix = keyLabel !== undefined
    ? <span className="json-key">&quot;{escapeJsonString(keyLabel)}&quot;</span>
    : null;

  if (!isObject) {
    if (typeof value === "string") {
      const inner = embeddedJson(value);
      if (inner !== null) {
        return (
          <JsonNode
            value={inner}
            depth={depth}
            keyLabel={keyLabel}
            trailingComma={trailingComma}
            forceCollapsed={forceCollapsed}
          />
        );
      }
    }
    return (
      <div className="json-line" style={{ paddingLeft: depth * 12 }}>
        {prefix}{prefix ? <span className="json-punct">: </span> : null}
        <JsonScalar value={value} />
        {trailingComma ? <span className="json-punct">,</span> : null}
      </div>
    );
  }

  const open = isArray ? "[" : "{";
  const close = isArray ? "]" : "}";
  const entries = isArray
    ? (value as JsonValue[]).map((item, index) => ({ key: String(index), value: item, showKey: false }))
    : Object.entries(value as { [key: string]: JsonValue }).map(([key, item]) => ({ key, value: item, showKey: true }));

  if (entries.length === 0) {
    return (
      <div className="json-line" style={{ paddingLeft: depth * 12 }}>
        {prefix}{prefix ? <span className="json-punct">: </span> : null}
        <span className="json-punct">{open}{close}</span>
        {trailingComma ? <span className="json-punct">,</span> : null}
      </div>
    );
  }

  return (
    <div className="json-block">
      <div className="json-line json-header" style={{ paddingLeft: depth * 12 }}>
        <button
          type="button"
          className="json-toggle"
          aria-expanded={!collapsed}
          aria-label={collapsed ? "expand" : "collapse"}
          onClick={() => setCollapsed((previous) => !previous)}
        >{collapsed ? "+" : "−"}</button>
        {prefix}{prefix ? <span className="json-punct">: </span> : null}
        <span className="json-punct">{open}</span>
        {collapsed ? (
          <>
            <span className="json-collapsed"> {collapsedSummary(value)} </span>
            <span className="json-punct">{close}</span>
            {trailingComma ? <span className="json-punct">,</span> : null}
          </>
        ) : null}
      </div>
      {!collapsed ? (
        <>
          <div className="json-children">
            {entries.map((entry, index) => (
              <JsonNode
                key={entry.key}
                value={entry.value}
                depth={depth + 1}
                keyLabel={entry.showKey ? entry.key : undefined}
                trailingComma={index < entries.length - 1}
              />
            ))}
          </div>
          <div className="json-line" style={{ paddingLeft: depth * 12 }}>
            <span className="json-punct">{close}</span>
            {trailingComma ? <span className="json-punct">,</span> : null}
          </div>
        </>
      ) : null}
    </div>
  );
}

export function JsonTree({ value, startCollapsed = false }: { value: JsonValue; startCollapsed?: boolean }) {
  return (
    <div className="json-tree" role="tree" aria-label="JSON body">
      <JsonNode value={value} depth={0} forceCollapsed={startCollapsed} />
    </div>
  );
}
