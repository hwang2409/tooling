// @vitest-environment jsdom

import { Buffer } from "node:buffer";
import { performance } from "node:perf_hooks";
import { act } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import { LARGE_TREE_COLLAPSE_THRESHOLD } from "./jsonTree";
import { PairedInspector } from "./PairedInspector";
import type { InspectorFlow } from "./models";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

interface Mounted {
  root: ReturnType<typeof createRoot>;
  container: HTMLDivElement;
}

const mounts: Mounted[] = [];

async function mount(node: ReactNode): Promise<Mounted> {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  await act(async () => root.render(node));
  const mounted = { root, container };
  mounts.push(mounted);
  return mounted;
}

afterEach(async () => {
  for (const mounted of mounts.splice(0)) {
    await act(async () => mounted.root.unmount());
    mounted.container.remove();
  }
});

function buildLargeJsonBody(targetBytes: number): { data: string; sizeBytes: number } {
  // Build a JSON blob with lots of nested branches so `shouldStartCollapsed`
  // AND the LARGE_TREE_COLLAPSE_THRESHOLD guard both matter. We stop as soon
  // as we cross the target size — tests keep total work bounded.
  const items: Record<string, unknown>[] = [];
  const filler = "x".repeat(96);
  let size = 2; // account for the surrounding `[]`
  while (size < targetBytes) {
    const item = {
      id: items.length,
      label: filler,
      nested: { a: filler, b: [filler, filler, filler] },
    };
    items.push(item);
    size += JSON.stringify(item).length + 1;
  }
  const text = JSON.stringify(items);
  const data = Buffer.from(text, "utf8").toString("base64");
  return { data, sizeBytes: text.length };
}

describe("F6 large-body render guardrail", () => {
  it("mounts a 2 MiB JSON body without freezing the main thread", async () => {
    const { data, sizeBytes } = buildLargeJsonBody(2 * 1024 * 1024);
    expect(sizeBytes).toBeGreaterThan(LARGE_TREE_COLLAPSE_THRESHOLD);
    const flow: InspectorFlow = {
      metadata: {
        flow_id: "large-json",
        method: "POST",
        scheme: "https",
        host: "api.example.test",
        port: "443",
        path: "/v1/echo",
        request_headers: [],
        request_body: {
          state: "captured",
          size_bytes: String(sizeBytes),
          encoding: "base64",
          data,
          content_type: "application/json",
        },
      },
      lifecycle: [],
    };

    const start = performance.now();
    const mounted = await mount(<PairedInspector flow={flow} />);
    const elapsed = performance.now() - start;

    // Budget: full 2 MiB render (base64 decode + JSON.parse + collapsed root)
    // must complete under a wall-clock budget so real captures never lock
    // the UI. The budget is generous to keep the test stable in CI, but
    // still tight enough that a regression (e.g. mounting every child of a
    // 20k-node tree) is caught immediately.
    expect(elapsed).toBeLessThan(2000);

    // The root JsonTree mounts; the recursive children stay collapsed so
    // the DOM node count is bounded to a small constant even for a huge
    // input.
    const tree = mounted.container.querySelector(".json-tree");
    expect(tree).not.toBeNull();
    const lines = mounted.container.querySelectorAll(".json-line");
    expect(lines.length).toBeLessThan(64);
  });
});
