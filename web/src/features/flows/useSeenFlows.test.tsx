/* eslint-disable no-unused-vars */
// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { SEEN_FLOW_LIMIT, useSeenFlows } from "./useSeenFlows";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface Harness {
  root: ReturnType<typeof createRoot>;
  container: HTMLDivElement;
  latest: () => { seen: readonly string[]; mark: (flowId: string) => void };
  render: (retained: readonly string[]) => Promise<void>;
}

const mounts: Harness[] = [];

async function mount(initial: readonly string[]): Promise<Harness> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let snapshot: { seen: readonly string[]; mark: (flowId: string) => void } = { seen: [], mark: () => {} };
  function Probe({ retained }: { retained: readonly string[] }) {
    const controller = useSeenFlows(retained);
    snapshot = { seen: [...controller.seen], mark: controller.mark };
    return null;
  }
  const render = async (retained: readonly string[]) => {
    await act(async () => root.render(<Probe retained={retained} />));
  };
  await render(initial);
  const harness: Harness = { root, container, latest: () => snapshot, render };
  mounts.push(harness);
  return harness;
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

describe("useSeenFlows", () => {
  it("hard cap is sized above the backend retention window", () => {
    expect(SEEN_FLOW_LIMIT).toBeGreaterThanOrEqual(2000);
  });

  it("preserves every seen bit across a full 2000-flow retention window", async () => {
    const ids = Array.from({ length: 2000 }, (_, index) => `flow-${index}`);
    const harness = await mount(ids);
    for (const flowId of ids) {
      await act(async () => harness.latest().mark(flowId));
    }
    expect(harness.latest().seen).toHaveLength(2000);
    for (const flowId of ids) {
      expect(new Set(harness.latest().seen).has(flowId), `should retain ${flowId}`).toBe(true);
    }
  });

  it("keeps seen bits when retention transiently drops to empty (reconnect)", async () => {
    const harness = await mount(["a", "b", "c"]);
    for (const flowId of ["a", "b", "c"]) {
      await act(async () => harness.latest().mark(flowId));
    }
    expect(new Set(harness.latest().seen)).toEqual(new Set(["a", "b", "c"]));
    await harness.render([]);
    expect(new Set(harness.latest().seen)).toEqual(new Set(["a", "b", "c"]));
    await harness.render(["a", "b", "c"]);
    expect(new Set(harness.latest().seen)).toEqual(new Set(["a", "b", "c"]));
  });

  it("drops seen bits for ids that leave the retained window on non-empty updates", async () => {
    const harness = await mount(["a", "b"]);
    await act(async () => harness.latest().mark("a"));
    await act(async () => harness.latest().mark("b"));
    await harness.render(["b", "c"]);
    expect(new Set(harness.latest().seen)).toEqual(new Set(["b"]));
  });
});
