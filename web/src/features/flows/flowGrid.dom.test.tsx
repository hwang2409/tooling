// @vitest-environment jsdom

import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImmutableFlowMetadata } from "../../state/browserState";
import { computeGridWindow, FLOW_ROW_HEIGHT, FlowGrid } from "./FlowGrid";
import { buildFlowRow } from "./gridModel";
import type { FlowRow } from "./gridModel";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function metadata(index: number): ImmutableFlowMetadata {
  return {
    flow_id: `flow-${index}`,
    method: "GET",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: `/v1/item/${index}`,
    request_headers: [],
    request_body: { state: "missing" },
  };
}

function makeRows(count: number): FlowRow[] {
  return Array.from({ length: count }, (_, index) => buildFlowRow(metadata(index)));
}

interface Mounted {
  root: ReturnType<typeof createRoot>;
  container: HTMLDivElement;
  grid: () => HTMLElement;
  bodyRows: () => HTMLElement[];
}

const mounts: Mounted[] = [];

async function mount(element: ReactNode): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(element));
  const mounted: Mounted = {
    root,
    container,
    grid: () => {
      const grid = container.querySelector<HTMLElement>('[role="grid"]');
      if (!grid) throw new Error("grid not mounted");
      return grid;
    },
    bodyRows: () => Array.from(container.querySelectorAll<HTMLElement>(".flow-grid-row")),
  };
  mounts.push(mounted);
  return mounted;
}

async function scrollTo(grid: HTMLElement, top: number): Promise<void> {
  await act(async () => {
    grid.scrollTop = top;
    grid.dispatchEvent(new Event("scroll", { bubbles: true }));
  });
}

async function pressKey(grid: HTMLElement, key: string): Promise<void> {
  await act(async () => {
    grid.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }));
  });
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("computeGridWindow", () => {
  it("clamps the window to the row range with overscan", () => {
    expect(computeGridWindow(0, 0, 280, 6)).toEqual({ start: 0, end: 0 });
    expect(computeGridWindow(200, 0, 280, 6)).toEqual({ start: 0, end: 17 });
    expect(computeGridWindow(200, 200 * FLOW_ROW_HEIGHT, 280, 6)).toEqual({ start: 184, end: 200 });
    const middle = computeGridWindow(200, 100 * FLOW_ROW_HEIGHT, 280, 6);
    expect(middle.start).toBe(94);
    expect(middle.end).toBe(117);
  });
});

describe("mounted FlowGrid", () => {
  it("virtualizes rows: renders only the visible window out of the full row count", async () => {
    const rows = makeRows(500);
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId={null} onSelectFlow={() => {}} followLive={false} viewportHeight={280} />,
    );
    const grid = mounted.grid();
    expect(grid.getAttribute("aria-rowcount")).toBe("501");
    const rendered = mounted.bodyRows();
    expect(rendered.length).toBe(17);
    expect(rendered[0].textContent).toContain("/v1/item/0");
    expect(mounted.container.textContent).not.toContain("/v1/item/499");
  });

  it("re-windows on scroll so distant rows become reachable", async () => {
    const rows = makeRows(500);
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId={null} onSelectFlow={() => {}} followLive={false} viewportHeight={280} />,
    );
    await scrollTo(mounted.grid(), 400 * FLOW_ROW_HEIGHT);
    expect(mounted.container.textContent).toContain("/v1/item/400");
    expect(mounted.container.textContent).not.toContain("/v1/item/0 ");
    expect(mounted.bodyRows()[0].getAttribute("aria-rowindex")).toBe(String(394 + 2));
  });

  it("pins the window to the newest rows while following live", async () => {
    const rows = makeRows(300);
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId={null} onSelectFlow={() => {}} followLive viewportHeight={280} />,
    );
    expect(mounted.container.textContent).toContain("/v1/item/299");
    expect(mounted.container.textContent).not.toContain("/v1/item/100");
  });

  it("selects rows by click and reflects selection state", async () => {
    const rows = makeRows(20);
    const onSelect = vi.fn();
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId="flow-3" onSelectFlow={onSelect} followLive={false} viewportHeight={280} />,
    );
    const selected = mounted.bodyRows().find((row) => row.getAttribute("aria-selected") === "true");
    expect(selected?.textContent).toContain("/v1/item/3");
    expect(mounted.grid().getAttribute("aria-activedescendant")).toBe(selected?.id);
    await act(async () => mounted.bodyRows()[5].click());
    expect(onSelect).toHaveBeenCalledWith("flow-5");
  });

  it("supports keyboard row navigation with arrows, Home, and End", async () => {
    const rows = makeRows(50);
    const onSelect = vi.fn();
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId={null} onSelectFlow={onSelect} followLive={false} viewportHeight={280} />,
    );
    await pressKey(mounted.grid(), "ArrowDown");
    expect(onSelect).toHaveBeenLastCalledWith("flow-0");
    await pressKey(mounted.grid(), "End");
    expect(onSelect).toHaveBeenLastCalledWith("flow-49");
    await pressKey(mounted.grid(), "Home");
    expect(onSelect).toHaveBeenLastCalledWith("flow-0");
  });

  it("steps from the current selection and scrolls the target row into view", async () => {
    const rows = makeRows(200);
    const onSelect = vi.fn();
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId="flow-100" onSelectFlow={onSelect} followLive={false} viewportHeight={280} />,
    );
    await pressKey(mounted.grid(), "ArrowDown");
    expect(onSelect).toHaveBeenLastCalledWith("flow-101");
    expect(mounted.container.textContent).toContain("/v1/item/101");
    await pressKey(mounted.grid(), "ArrowUp");
    expect(onSelect).toHaveBeenLastCalledWith("flow-99");
  });

  it("marks streaming responses with the SSE triangle glyph", async () => {
    const streamingMetadata = {
      ...metadata(0),
      response_body: { state: "captured" as const, size_bytes: "0", encoding: "base64" as const, data: "", content_type: "text/event-stream" },
    } as ImmutableFlowMetadata;
    const rows: FlowRow[] = [buildFlowRow(streamingMetadata)];
    const mounted = await mount(
      <FlowGrid rows={rows} selectedFlowId={null} onSelectFlow={() => {}} followLive={false} viewportHeight={280} />,
    );
    const rendered = mounted.container.querySelector<HTMLElement>(".flow-sse-flag");
    expect(rendered?.textContent).toBe(" SSE ▸");
  });

  it("renders an explicit empty note when no rows are given", async () => {
    const mounted = await mount(
      <FlowGrid rows={[]} selectedFlowId={null} onSelectFlow={() => {}} followLive={false} viewportHeight={280} />,
    );
    expect(mounted.container.textContent).toContain("No flows to list.");
    expect(mounted.grid().getAttribute("aria-rowcount")).toBe("1");
  });
});
