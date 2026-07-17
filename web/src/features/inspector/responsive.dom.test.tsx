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

describe("F6 layout contracts pinned in CSS", () => {
  let container: HTMLDivElement;
  let root: ReturnType<typeof createRoot>;

  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
  });

  it("pane layout stacks below 900px and inspector dock switches to a single column", async () => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);

    await act(async () => root.render(<PairedInspector flow={flow} />));

    const inspector = document.querySelector('[data-testid="paired-inspector"]');
    expect(inspector).not.toBeNull();

    const inspectorCss = readFileSync(resolve(process.cwd(), "src/styles/inspector.css"), "utf8");
    // Narrow viewport: two-column body/metadata splits into a stack so the
    // body reader stays the full width.
    expect(inspectorCss).toMatch(/@media \(max-width: 900px\) \{[\s\S]*?\.inspector-pane-layout \{ display: block; \}/);
  });

  it("workspace docks the inspector on the right by default and stacks below 1024px", () => {
    const flowsCss = readFileSync(resolve(process.cwd(), "src/styles/flows.css"), "utf8");
    // Default layout: flow-main is a flexbox row that keeps the grid on the
    // left and the inspector docked on the right.
    expect(flowsCss).toMatch(/\.flow-main \{[^}]*display:\s*flex[^}]*\}/);
    expect(flowsCss).toMatch(/\.flow-inspector-side \{[^}]*flex:\s*0 0 clamp\(/);
    // Stacking breakpoint: below 1024px the flexbox row flips to a column
    // and the inspector loses its sticky pin.
    expect(flowsCss).toMatch(/@media \(max-width: 1023px\) \{[\s\S]*?\.flow-main \{[^}]*flex-direction:\s*column/);
    expect(flowsCss).toMatch(/@media \(max-width: 1023px\) \{[\s\S]*?\.flow-inspector-side \{[^}]*position:\s*static/);
  });
});
