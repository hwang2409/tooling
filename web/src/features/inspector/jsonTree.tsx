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
}

function JsonScalar({ value }: { value: JsonValue }) {
  if (value === null) return <span className="json-null">null</span>;
  if (typeof value === "boolean") return <span className="json-bool">{String(value)}</span>;
  if (typeof value === "number") return <span className="json-number">{Number.isFinite(value) ? String(value) : "null"}</span>;
  return <span className="json-string">&quot;{escapeJsonString(value as string)}&quot;</span>;
}

function escapeJsonString(value: string): string {
  return value.replace(/[\\"\b\f\n\r\t\x00-\x1f]/g, (character) => {
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

function JsonNode({ value, depth, keyLabel, trailingComma }: JsonNodeProps) {
  const isObject = value !== null && typeof value === "object";
  const isArray = Array.isArray(value);
  const [collapsed, setCollapsed] = useState<boolean>(() => isObject ? shouldStartCollapsed(value, depth) : false);

  const prefix = keyLabel !== undefined
    ? <span className="json-key">&quot;{escapeJsonString(keyLabel)}&quot;</span>
    : null;

  if (!isObject) {
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

export function JsonTree({ value }: { value: JsonValue }) {
  return (
    <div className="json-tree" role="tree" aria-label="JSON body">
      <JsonNode value={value} depth={0} />
    </div>
  );
}
