// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { ImmutableFlowMetadata } from "../../state/browserState";
import type { FlowDetailLoader } from "./flowDetail";
import { useFlowDetails } from "./flowDetail";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const flow = (flowId: string): ImmutableFlowMetadata => ({
  flow_id: flowId,
  method: "POST",
  scheme: "https",
  host: "api.example.test",
  port: "443",
  path: "/v1/messages",
  request_headers: [],
  request_body: { state: "missing" },
} as ImmutableFlowMetadata);

function Probe({ flows, loader }: { flows: readonly ImmutableFlowMetadata[]; loader: FlowDetailLoader }) {
  const details = useFlowDetails(flows, loader);
  return <output data-testid="detail-ids">{[...details.keys()].join(",")}</output>;
}

describe("useFlowDetails", () => {
  const mounts: Array<{ root: ReturnType<typeof createRoot>; container: HTMLDivElement }> = [];

  afterEach(async () => {
    for (const mounted of mounts.splice(0)) {
      await act(async () => mounted.root.unmount());
      mounted.container.remove();
    }
  });

  it("keeps unchanged in-flight requests when a candidate is added", async () => {
    const pending = new Map<string, () => void>();
    const requested: string[] = [];
    const loader: FlowDetailLoader = (flowId) => {
      requested.push(flowId);
      return new Promise((resolve) => pending.set(flowId, () => resolve({ status: "unavailable" })));
    };
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    const render = async (flows: readonly ImmutableFlowMetadata[]) => {
      await act(async () => root.render(<Probe flows={flows} loader={loader} />));
    };
    await render([flow("a"), flow("b")]);
    expect(requested).toEqual(["a", "b"]);
    await render([flow("a"), flow("b"), flow("c")]);
    expect(requested).toEqual(["a", "b", "c"]);

    await act(async () => {
      pending.get("a")?.();
      pending.get("b")?.();
      pending.get("c")?.();
    });
    expect(container.querySelector("[data-testid='detail-ids']")?.textContent).toBe("a,b,c");
  });
});
