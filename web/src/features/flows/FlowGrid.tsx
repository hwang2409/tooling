/* eslint-disable no-unused-vars */

import { useEffect, useId, useRef, useState } from "react";
import type { KeyboardEvent, UIEvent } from "react";

import type { FlowRow } from "./gridModel";

export const FLOW_ROW_HEIGHT = 28;
export const DEFAULT_GRID_VIEWPORT_HEIGHT = FLOW_ROW_HEIGHT * 14;
const DEFAULT_OVERSCAN = 6;
const COLUMN_COUNT = 8;

export interface FlowGridProps {
  rows: readonly FlowRow[];
  selectedFlowId: string | null;
  onSelectFlow: (flowId: string) => void;
  followLive: boolean;
  viewportHeight?: number;
  overscan?: number;
}

export interface GridWindow {
  readonly start: number;
  readonly end: number;
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
  selectedFlowId,
  onSelectFlow,
  followLive,
  viewportHeight = DEFAULT_GRID_VIEWPORT_HEIGHT,
  overscan = DEFAULT_OVERSCAN,
}: FlowGridProps) {
  const gridId = useId().replaceAll(":", "");
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const totalHeight = rows.length * FLOW_ROW_HEIGHT;

  useEffect(() => {
    if (!followLive) return;
    const container = containerRef.current;
    if (container === null) return;
    container.scrollTop = totalHeight;
    setScrollTop(container.scrollTop);
  }, [followLive, totalHeight]);

  const { start, end } = computeGridWindow(rows.length, scrollTop, viewportHeight, overscan);
  const selectedIndex = selectedFlowId === null ? -1 : rows.findIndex((row) => row.flowId === selectedFlowId);
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
    if (rows.length === 0) return;
    let nextIndex: number | undefined;
    if (event.key === "ArrowDown") nextIndex = selectedIndex < 0 ? 0 : Math.min(rows.length - 1, selectedIndex + 1);
    else if (event.key === "ArrowUp") nextIndex = selectedIndex < 0 ? rows.length - 1 : Math.max(0, selectedIndex - 1);
    else if (event.key === "Home") nextIndex = 0;
    else if (event.key === "End") nextIndex = rows.length - 1;
    if (nextIndex === undefined) return;
    event.preventDefault();
    revealRow(nextIndex);
    onSelectFlow(rows[nextIndex].flowId);
  };

  return (
    <div
      ref={containerRef}
      id={`${gridId}-grid`}
      className="flow-grid"
      role="grid"
      aria-label="Captured flows"
      aria-rowcount={rows.length + 1}
      aria-colcount={COLUMN_COUNT}
      aria-activedescendant={activeDescendant}
      tabIndex={0}
      style={{ height: viewportHeight + FLOW_ROW_HEIGHT }}
      onScroll={handleScroll}
      onKeyDown={handleKeyDown}
    >
      <div className="flow-grid-head" role="row" aria-rowindex={1}>
        <span role="columnheader" className="flow-cell flow-cell-index">#</span>
        <span role="columnheader" className="flow-cell flow-cell-method">method</span>
        <span role="columnheader" className="flow-cell flow-cell-host">host</span>
        <span role="columnheader" className="flow-cell flow-cell-path">path</span>
        <span role="columnheader" className="flow-cell flow-cell-phase">phase</span>
        <span role="columnheader" className="flow-cell flow-cell-size">req</span>
        <span role="columnheader" className="flow-cell flow-cell-size">resp</span>
        <span role="columnheader" className="flow-cell flow-cell-type">type</span>
      </div>
      {rows.length === 0 ? (
        <p className="flow-grid-empty" role="note">No flows to list.</p>
      ) : (
        <div className="flow-grid-canvas" role="presentation" style={{ height: totalHeight }}>
          <div className="flow-grid-window" role="presentation" style={{ transform: `translateY(${start * FLOW_ROW_HEIGHT}px)` }}>
            {rows.slice(start, end).map((row, offset) => {
              const index = start + offset;
              const selected = row.flowId === selectedFlowId;
              return (
                <div
                  key={row.flowId}
                  id={`${gridId}-row-${index}`}
                  className={`flow-grid-row is-${row.phase}${selected ? " is-selected" : ""}`}
                  role="row"
                  aria-rowindex={index + 2}
                  aria-selected={selected}
                  title={row.url}
                  onClick={() => onSelectFlow(row.flowId)}
                >
                  <span role="gridcell" className="flow-cell flow-cell-index">{index + 1}</span>
                  <span role="gridcell" className="flow-cell flow-cell-method">{row.method}</span>
                  <span role="gridcell" className="flow-cell flow-cell-host">{row.scheme === "https" ? "" : "http "}{row.host}:{row.port}</span>
                  <span role="gridcell" className="flow-cell flow-cell-path">{row.path}</span>
                  <span role="gridcell" className="flow-cell flow-cell-phase"><span className={`flow-phase is-${row.phase}`}>{row.phaseLabel}</span></span>
                  <span role="gridcell" className={`flow-cell flow-cell-size${row.requestBody.truncated ? " is-truncated" : ""}`}>{row.requestBody.text}{row.requestBody.truncated ? "+" : ""}</span>
                  <span role="gridcell" className={`flow-cell flow-cell-size${row.responseBody.truncated ? " is-truncated" : ""}`}>{row.responseBody.text}{row.responseBody.truncated ? "+" : ""}</span>
                  <span role="gridcell" className="flow-cell flow-cell-type">{row.contentType}</span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
