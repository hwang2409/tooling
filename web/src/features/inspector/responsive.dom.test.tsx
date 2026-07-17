// @vitest-environment jsdom

import { act } from "react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";

import type { InspectorFlow } from "./models";
import { PairedInspector } from "./PairedInspector";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const flow: InspectorFlow = {
  metadata: {
    flow_id: "responsive-flow",
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: "/stream",
    request_headers: [],
    request_body: { state: "empty", size_bytes: "0" },
  },
  lifecycle: [],
};

describe("paired inspector responsive DOM contract", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("fits the 800px viewport by stacking the exchange columns", async () => {
    const viewport = globalThis as typeof globalThis & { innerWidth: number };
    Object.defineProperty(viewport, "innerWidth", { configurable: true, value: 800 });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    await act(async () => root.render(<PairedInspector flow={flow} />));

    const inspector = document.querySelector('[data-testid="paired-inspector"]');
    expect(inspector).not.toBeNull();
    expect(inspector?.getBoundingClientRect().right ?? Infinity).toBeLessThanOrEqual(viewport.innerWidth);

    const inspectorCss = readFileSync(resolve(process.cwd(), "src/styles/inspector.css"), "utf8");
    expect(inspectorCss).toMatch(/@media \(max-width: 900px\) \{[\s\S]*?\.inspector-pane-layout \{ display: block; \}/);
  });
});
