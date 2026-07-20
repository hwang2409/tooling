/* eslint-disable no-unused-vars */

import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { KeyboardEvent, UIEvent } from "react";

import { formatDurationMs } from "../../format";
import type { FlowRow } from "./gridModel";

export const FLOW_ROW_HEIGHT = 28;
export const DEFAULT_GRID_VIEWPORT_HEIGHT = FLOW_ROW_HEIGHT * 14;
const DEFAULT_OVERSCAN = 6;
const COLUMN_COUNT = 10;

export interface FlowGridProps {
  rows: readonly FlowRow[];
  groupBy?: "none" | "session";
  selectedFlowId: string | null;
  onSelectFlow: (flowId: string) => void;
  followLive: boolean;
  viewportHeight?: number;
  overscan?: number;
  /**
   * Flow IDs the user has already opened. Rows not in this set render an
   * "unseen" marker so a caller can spot fresh traffic at a glance.
   */
  seenFlowIds?: ReadonlySet<string>;
}

export interface GridWindow {
  readonly start: number;
  readonly end: number;
}

type GridItem =
  | { readonly kind: "flow"; readonly row: FlowRow; readonly originalIndex: number }
  | {
      readonly kind: "session";
      readonly key: string;
      readonly sessionId: string | null;
      readonly rows: readonly FlowRow[];
      readonly firstSeenAtMs: number | null;
      readonly lastSeenAtMs: number | null;
      readonly collapsed: boolean;
    };

const UNASSIGNED_KEY = "__unassigned__";

function sessionLabel(sessionId: string): string {
  return sessionId.slice(0, 8);
}

function clockLabel(timestamp: number | null): string {
  if (timestamp === null) return "—";
  const date = new Date(timestamp);
  if (!Number.isFinite(date.getTime())) return "—";
  return [date.getUTCHours(), date.getUTCMinutes(), date.getUTCSeconds()]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function sessionItems(
  rows: readonly FlowRow[],
  collapsedKeys: ReadonlySet<string>,
): GridItem[] {
  const groups = new Map<string, { sessionId: string | null; rows: Array<{ row: FlowRow; originalIndex: number }>; order: number }>();
  rows.forEach((row, originalIndex) => {
    const key = row.sessionId ?? UNASSIGNED_KEY;
    const existing = groups.get(key);
    if (existing === undefined) {
      groups.set(key, { sessionId: row.sessionId, rows: [{ row, originalIndex }], order: originalIndex });
    } else {
      existing.rows.push({ row, originalIndex });
    }
  });
  const sortedGroups = [...groups.values()].sort((left, right) => {
    if (left.sessionId === null && right.sessionId === null) return 0;
    if (left.sessionId === null) return 1;
    if (right.sessionId === null) return -1;
    const leftFirst = left.rows.reduce<number | null>((value, item) => value === null || (item.row.firstSeenAtMs !== null && item.row.firstSeenAtMs < value) ? item.row.firstSeenAtMs : value, null);
    const rightFirst = right.rows.reduce<number | null>((value, item) => value === null || (item.row.firstSeenAtMs !== null && item.row.firstSeenAtMs < value) ? item.row.firstSeenAtMs : value, null);
    const leftSort = leftFirst ?? Number.POSITIVE_INFINITY;
    const rightSort = rightFirst ?? Number.POSITIVE_INFINITY;
    return leftSort - rightSort || left.order - right.order || left.sessionId.localeCompare(right.sessionId);
  });

  const items: GridItem[] = [];
  for (const group of sortedGroups) {
    const firstSeenAtMs = group.rows.reduce<number | null>((value, item) => value === null || (item.row.firstSeenAtMs !== null && item.row.firstSeenAtMs < value) ? item.row.firstSeenAtMs : value, null);
    const lastSeenAtMs = group.rows.reduce<number | null>((value, item) => value === null || (item.row.lastSeenAtMs !== null && item.row.lastSeenAtMs > value) ? item.row.lastSeenAtMs : value, null);
    const key = group.sessionId ?? UNASSIGNED_KEY;
    const collapsed = collapsedKeys.has(key);
    items.push({
      kind: "session",
      key,
      sessionId: group.sessionId,
      rows: group.rows.map((item) => item.row),
      firstSeenAtMs,
      lastSeenAtMs,
      collapsed,
    });
    if (!collapsed) {
      items.push(...group.rows.map(({ row, originalIndex }) => ({ kind: "flow" as const, row, originalIndex })));
    }
  }
  return items;
}

export function computeGridWindow(
  rowCount: number,
  scrollTop: number,
  viewportHeight: number,
  overscan: number,
): GridWindow {
  const total = rowCount * FLOW_ROW_HEIGHT;
  const maxScroll = Math.max(0, total - viewportHeight);
  const clamped = Math.min(Math.max(0, scrollTop), maxScroll);
  const firstVisible = Math.floor(clamped / FLOW_ROW_HEIGHT);
  const visibleCount = Math.ceil(viewportHeight / FLOW_ROW_HEIGHT) + 1;
  return {
    start: Math.max(0, firstVisible - overscan),
    end: Math.min(rowCount, firstVisible + visibleCount + overscan),
  };
}

export function FlowGrid({
  rows,
  groupBy = "none",
  selectedFlowId,
  onSelectFlow,
  followLive,
  viewportHeight = DEFAULT_GRID_VIEWPORT_HEIGHT,
  overscan = DEFAULT_OVERSCAN,
  seenFlowIds,
}: FlowGridProps) {
  const gridId = useId().replaceAll(":", "");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [collapsedSessionKeys, setCollapsedSessionKeys] = useState<ReadonlySet<string>>(() => new Set());
  const [scrollTop, setScrollTop] = useState(0);
  const items = useMemo(
    () => groupBy === "session"
      ? sessionItems(rows, collapsedSessionKeys)
      : rows.map((row, originalIndex): GridItem => ({ kind: "flow", row, originalIndex })),
    [collapsedSessionKeys, groupBy, rows],
  );
  const totalHeight = items.length * FLOW_ROW_HEIGHT;

  useEffect(() => {
    if (!followLive) return;
    const container = containerRef.current;
    if (container === null) return;
    container.scrollTop = totalHeight;
    setScrollTop(container.scrollTop);
  }, [followLive, totalHeight]);

  const { start, end } = computeGridWindow(items.length, scrollTop, viewportHeight, overscan);
  const selectedIndex = selectedFlowId === null ? -1 : items.findIndex((item) => item.kind === "flow" && item.row.flowId === selectedFlowId);
  const activeDescendant = selectedIndex >= start && selectedIndex < end ? `${gridId}-row-${selectedIndex}` : undefined;

  const handleScroll = (event: UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  };

  const revealRow = (index: number) => {
    const container = containerRef.current;
    if (container === null) return;
    const rowTop = index * FLOW_ROW_HEIGHT;
    const rowBottom = rowTop + FLOW_ROW_HEIGHT;
    let nextScrollTop = container.scrollTop;
    if (rowTop < nextScrollTop) nextScrollTop = rowTop;
    else if (rowBottom > nextScrollTop + viewportHeight) nextScrollTop = rowBottom - viewportHeight;
    container.scrollTop = nextScrollTop;
    setScrollTop(container.scrollTop);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const flowIndexes = items.flatMap((item, index) => item.kind === "flow" ? [index] : []);
    if (flowIndexes.length === 0) return;
    const selectedFlowIndex = flowIndexes.indexOf(selectedIndex);
    let nextIndex: number | undefined;
    if (event.key === "ArrowDown") nextIndex = flowIndexes[selectedFlowIndex < 0 ? 0 : Math.min(flowIndexes.length - 1, selectedFlowIndex + 1)];
    else if (event.key === "ArrowUp") nextIndex = flowIndexes[selectedFlowIndex < 0 ? flowIndexes.length - 1 : Math.max(0, selectedFlowIndex - 1)];
    else if (event.key === "Home") nextIndex = flowIndexes[0];
    else if (event.key === "End") nextIndex = flowIndexes[flowIndexes.length - 1];
    if (nextIndex === undefined) return;
    event.preventDefault();
    revealRow(nextIndex);
    const item = items[nextIndex];
    if (item.kind === "flow") onSelectFlow(item.row.flowId);
  };

  const isSeen = (flowId: string) => seenFlowIds === undefined ? true : seenFlowIds.has(flowId);

  return (
    <div
      ref={containerRef}
      id={`${gridId}-grid`}
      className="flow-grid"
      role="grid"
      aria-label="Captured flows"
      aria-rowcount={items.length + 1}
      aria-colcount={COLUMN_COUNT}
      aria-activedescendant={activeDescendant}
      tabIndex={0}
      style={{ height: viewportHeight + FLOW_ROW_HEIGHT }}
      onScroll={handleScroll}
      onKeyDown={handleKeyDown}
    >
      <div className="flow-grid-head" role="row" aria-rowindex={1}>
        <span role="columnheader" className="flow-cell flow-cell-marker" aria-label="new" />
        <span role="columnheader" className="flow-cell flow-cell-index">#</span>
        <span role="columnheader" className="flow-cell flow-cell-method">method</span>
        <span role="columnheader" className="flow-cell flow-cell-host">host</span>
        <span role="columnheader" className="flow-cell flow-cell-path">path</span>
        <span role="columnheader" className="flow-cell flow-cell-phase">phase</span>
        <span role="columnheader" className="flow-cell flow-cell-status">status</span>
        <span role="columnheader" className="flow-cell flow-cell-size">req</span>
        <span role="columnheader" className="flow-cell flow-cell-size">resp</span>
        <span role="columnheader" className="flow-cell flow-cell-duration">dur</span>
      </div>
      {rows.length === 0 ? (
        <p className="flow-grid-empty" role="note">No flows to list.</p>
      ) : (
        <div className="flow-grid-canvas" role="presentation" style={{ height: totalHeight }}>
          <div className="flow-grid-window" role="presentation" style={{ transform: `translateY(${start * FLOW_ROW_HEIGHT}px)` }}>
            {items.slice(start, end).map((item, offset) => {
              const index = start + offset;
              if (item.kind === "session") {
                const elapsed = item.firstSeenAtMs !== null && item.lastSeenAtMs !== null
                  ? formatDurationMs(Math.max(0, item.lastSeenAtMs - item.firstSeenAtMs))
                  : "—";
                const label = item.sessionId === null
                  ? `unassigned · ${item.rows.length} flows`
                  : `session ${sessionLabel(item.sessionId)} · ${item.rows.length} flows · ${clockLabel(item.firstSeenAtMs)} · ${elapsed}`;
                return (
                  <div key={`session-${item.key}`} className="flow-grid-session" role="row" aria-rowindex={index + 2} aria-expanded={!item.collapsed}>
                    <button type="button" className="flow-grid-session-toggle" aria-expanded={!item.collapsed} onClick={() => setCollapsedSessionKeys((previous) => {
                      const next = new Set(previous);
                      if (next.has(item.key)) next.delete(item.key); else next.add(item.key);
                      return next;
                    })}>
                      <span aria-hidden="true">{item.collapsed ? "▸" : "▾"}</span> {label}
                    </button>
                  </div>
                );
              }
              const row = item.row;
              const selected = row.flowId === selectedFlowId;
              const seen = isSeen(row.flowId);
              const rowClasses = [
                "flow-grid-row",
                `is-${row.phase}`,
                selected ? "is-selected" : "",
                seen ? "" : "is-unseen",
                row.isStreaming ? "is-streaming" : "",
              ].filter(Boolean).join(" ");
              return (
                <div
                  key={row.flowId}
                  id={`${gridId}-row-${index}`}
                  className={rowClasses}
                  role="row"
                  aria-rowindex={index + 2}
                  aria-selected={selected}
                  title={row.url}
                  onClick={() => onSelectFlow(row.flowId)}
                >
                  <span role="gridcell" className="flow-cell flow-cell-marker" aria-label={seen ? "" : "unseen"}>
                    {seen ? "" : <span className="flow-marker-dot" aria-hidden="true" />}
                  </span>
                  <span role="gridcell" className="flow-cell flow-cell-index">{item.originalIndex + 1}</span>
                  <span role="gridcell" className="flow-cell flow-cell-method">{row.method}</span>
                  <span role="gridcell" className="flow-cell flow-cell-host">{row.scheme === "https" ? "" : "http "}{row.host}:{row.port}</span>
                  <span role="gridcell" className="flow-cell flow-cell-path">{row.path}</span>
                  <span role="gridcell" className="flow-cell flow-cell-phase"><span className={`flow-phase is-${row.phase}`}>{row.phaseLabel}</span></span>
                  <span role="gridcell" className={`flow-cell flow-cell-status is-${row.phase}`}>{row.statusLabel}</span>
                  <span role="gridcell" className={`flow-cell flow-cell-size${row.requestBody.truncated ? " is-truncated" : ""}`}>{row.requestBody.compact}{row.requestBody.truncated ? "+" : ""}</span>
                  <span role="gridcell" className={`flow-cell flow-cell-size${row.responseBody.truncated ? " is-truncated" : ""}`}>
                    {row.responseBody.compact}{row.responseBody.truncated ? "+" : ""}
                    {row.isStreaming && <span className="flow-sse-flag" aria-label="server-sent events">{" SSE ▸"}</span>}
                  </span>
                  <span role="gridcell" className="flow-cell flow-cell-duration">{row.durationLabel}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
