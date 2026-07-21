// @vitest-environment jsdom

import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { ImmutableFlowMetadata } from "../../state/browserState";
import type { FlowDetailLoader } from "./flowDetail";
import { useFlowDetail, useFlowDetails } from "./flowDetail";

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

function SingleProbe({ flowId, version, sourceEpoch, loader, onRender }: {
  flowId: string;
  version: string;
  sourceEpoch: number;
  loader: FlowDetailLoader;
  onRender?: Array<string>["push"];
}) {
  const detail = useFlowDetail(flowId, loader, version, sourceEpoch);
  const body = detail?.status === "loaded" ? detail.overrides.request_body : undefined;
  const marker = body !== undefined && "data" in body ? body.data : "pending";
  onRender?.(marker);
  return <output data-testid="single-detail" data-marker={marker}>{marker}</output>;
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

  it("does not return a prior body during a same-id version or epoch replacement", async () => {
    const resolutions: Array<() => void> = [];
    const renderedMarkers: string[] = [];
    let call = 0;
    const loader: FlowDetailLoader = async () => new Promise((resolve) => {
      const marker = call === 0 ? "old-body" : call === 1 ? "new-body" : "epoch-body";
      call += 1;
      resolutions.push(() => resolve({
        status: "loaded",
        overrides: { request_body: { state: "captured", size_bytes: "1", encoding: "base64", data: marker } },
      }));
    });
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    mounts.push({ root, container });
    const render = async (version: string, sourceEpoch: number) => {
      await act(async () => root.render(
        <SingleProbe
          flowId="reused"
          version={version}
          sourceEpoch={sourceEpoch}
          loader={loader}
          onRender={(marker) => renderedMarkers.push(marker)}
        />,
      ));
    };

    await render("v1", 1);
    await act(async () => resolutions.shift()?.());
    expect(container.querySelector("[data-testid='single-detail']")?.textContent).toBe("old-body");

    const beforeVersionReplacement = renderedMarkers.length;
    await render("v2", 1);
    expect(renderedMarkers.slice(beforeVersionReplacement)).not.toContain("old-body");
    expect(renderedMarkers.at(-1)).toBe("pending");
    expect(container.querySelector("[data-testid='single-detail']")?.textContent).toBe("pending");
    await act(async () => resolutions.shift()?.());
    expect(container.querySelector("[data-testid='single-detail']")?.textContent).toBe("new-body");

    const beforeEpochReplacement = renderedMarkers.length;
    await render("v2", 2);
    expect(renderedMarkers.slice(beforeEpochReplacement)).not.toContain("new-body");
    expect(renderedMarkers.at(-1)).toBe("pending");
    expect(container.querySelector("[data-testid='single-detail']")?.textContent).toBe("pending");
    await act(async () => resolutions.shift()?.());
    expect(container.querySelector("[data-testid='single-detail']")?.textContent).toBe("epoch-body");
  });
});
