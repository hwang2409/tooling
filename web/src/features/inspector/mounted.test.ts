import { Buffer } from "node:buffer";
import { act, createElement, useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { FlowLifecycle } from "../../protocol";
import type { InspectorFlow, InspectorPane } from "./models";

// eslint-disable-next-line no-unused-vars
type Listener = (event: MiniEvent) => void;

class MiniEvent {
  readonly type: string;
  readonly bubbles: boolean;
  readonly cancelable: boolean;
  readonly key?: string;
  target: MiniNode | null = null;
  currentTarget: MiniNode | null = null;
  defaultPrevented = false;
  cancelBubble = false;

  constructor(type: string, init: { bubbles?: boolean; cancelable?: boolean; key?: string } = {}) {
    this.type = type;
    this.bubbles = init.bubbles ?? false;
    this.cancelable = init.cancelable ?? false;
    this.key = init.key;
  }

  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
  stopPropagation() { this.cancelBubble = true; }
}

class MiniNode {
  readonly nodeType: number;
  readonly nodeName: string;
  readonly ownerDocument: MiniDocument;
  parentNode: MiniNode | null = null;
  childNodes: MiniNode[] = [];
  private readonly listeners = new Map<string, Array<{ listener: Listener; capture: boolean }>>();

  constructor(nodeType: number, nodeName: string, ownerDocument: MiniDocument) {
    this.nodeType = nodeType;
    this.nodeName = nodeName;
    this.ownerDocument = ownerDocument;
  }

  get parentElement(): MiniElement | null { return this.parentNode instanceof MiniElement ? this.parentNode : null; }
  get children(): MiniElement[] { return this.childNodes.filter((child): child is MiniElement => child instanceof MiniElement); }
  get childElementCount(): number { return this.children.length; }
  get firstChild(): MiniNode | null { return this.childNodes[0] ?? null; }
  get nextSibling(): MiniNode | null {
    if (!this.parentNode) return null;
    const index = this.parentNode.childNodes.indexOf(this);
    return this.parentNode.childNodes[index + 1] ?? null;
  }
  get textContent(): string { return this.childNodes.map((child) => child.textContent).join(""); }
  set textContent(value: string) {
    this.childNodes = value === "" ? [] : [new MiniText(value, this.ownerDocument)];
    for (const child of this.childNodes) child.parentNode = this;
  }

  appendChild<T extends MiniNode>(child: T): T {
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  insertBefore<T extends MiniNode>(child: T, before: MiniNode | null): T {
    if (child.parentNode) child.parentNode.removeChild(child);
    const index = before === null ? this.childNodes.length : this.childNodes.indexOf(before);
    child.parentNode = this;
    this.childNodes.splice(index < 0 ? this.childNodes.length : index, 0, child);
    return child;
  }

  removeChild<T extends MiniNode>(child: T): T {
    const index = this.childNodes.indexOf(child);
    if (index < 0) throw new Error("child not found");
    this.childNodes.splice(index, 1);
    child.parentNode = null;
    return child;
  }

  contains(node: MiniNode | null): boolean {
    if (node === this) return true;
    return node !== null && this.childNodes.some((child) => child.contains(node));
  }

  addEventListener(type: string, listener: Listener, options?: boolean | { capture?: boolean }) {
    const capture = typeof options === "boolean" ? options : options?.capture ?? false;
    const entries = this.listeners.get(type) ?? [];
    entries.push({ listener, capture });
    this.listeners.set(type, entries);
  }

  removeEventListener(type: string, listener: Listener, options?: boolean | { capture?: boolean }) {
    const capture = typeof options === "boolean" ? options : options?.capture ?? false;
    const entries = this.listeners.get(type) ?? [];
    this.listeners.set(type, entries.filter((entry) => entry.listener !== listener || entry.capture !== capture));
  }

  dispatchEvent(event: MiniEvent): boolean {
    event.target = this;
    const path: MiniNode[] = [];
    let current: MiniNode | null = this;
    while (current) {
      path.push(current);
      current = current.parentNode;
    }
    const defaultView = this.ownerDocument.defaultView;
    if (defaultView) path.push(defaultView);
    const capturePath = [...path].reverse();
    for (const node of capturePath) this.invokeListeners(node, event, true);
    if (!event.cancelBubble) for (const node of path) {
      this.invokeListeners(node, event, false);
      if (event.cancelBubble) break;
    }
    return !event.defaultPrevented;
  }

  private invokeListeners(node: MiniNode, event: MiniEvent, capture: boolean) {
    for (const entry of node.listeners.get(event.type) ?? []) {
      if (entry.capture !== capture) continue;
      event.currentTarget = node;
      entry.listener(event);
      if (event.cancelBubble) break;
    }
  }
}

class MiniText extends MiniNode {
  nodeValue: string;
  constructor(value: string, ownerDocument: MiniDocument) {
    super(3, "#text", ownerDocument);
    this.nodeValue = value;
  }
  override get textContent(): string { return this.nodeValue; }
  override set textContent(value: string) { this.nodeValue = value; }
}

class MiniComment extends MiniText {
  constructor(value: string, ownerDocument: MiniDocument) { super(value, ownerDocument); }
}

class MiniAttributes extends Map<string, string> {
  get length(): number { return this.size; }
}

class MiniElement extends MiniNode {
  readonly tagName: string;
  readonly nodeName: string;
  readonly namespaceURI = "http://www.w3.org/1999/xhtml";
  readonly attributes = new MiniAttributes();
  readonly style: Record<string, string> & { cssText?: string } = {};
  className = "";
  id = "";
  tabIndex = 0;
  disabled = false;

  constructor(tagName: string, ownerDocument: MiniDocument) {
    super(1, tagName.toUpperCase(), ownerDocument);
    this.tagName = tagName.toUpperCase();
    this.nodeName = this.tagName;
  }

  setAttribute(name: string, value: string) {
    const normalized = name.toLowerCase();
    this.attributes.set(normalized, String(value));
    if (normalized === "id") this.id = String(value);
    if (normalized === "class") this.className = String(value);
    if (normalized === "tabindex") this.tabIndex = Number(value);
    if (normalized === "disabled") this.disabled = true;
  }

  getAttribute(name: string): string | null { return this.attributes.get(name.toLowerCase()) ?? null; }
  getAttributeNames(): string[] { return [...this.attributes.keys()]; }
  hasAttribute(name: string): boolean { return this.attributes.has(name.toLowerCase()); }
  removeAttribute(name: string) { this.attributes.delete(name.toLowerCase()); }

  focus() { this.ownerDocument.activeElement = this; }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = this.ownerDocument.body; }
  click() { if (!this.disabled) this.dispatchEvent(new MiniEvent("click", { bubbles: true, cancelable: true })); }

  matches(selector: string): boolean {
    let candidate = selector.trim();
    const tag = candidate.match(/^[a-zA-Z][\w-]*/)?.[0];
    if (tag) {
      if (this.tagName.toLowerCase() !== tag.toLowerCase()) return false;
      candidate = candidate.slice(tag.length);
    }
    for (const id of candidate.matchAll(/#([\w-]+)/g)) if (this.id !== id[1]) return false;
    for (const className of candidate.matchAll(/\.([\w-]+)/g)) if (!this.className.split(/\s+/).includes(className[1])) return false;
    for (const attribute of candidate.matchAll(/\[([\w:-]+)(?:="([^"]*)")?\]/g)) {
      const actual = this.getAttribute(attribute[1]);
      if (actual === null || (attribute[2] !== undefined && actual !== attribute[2])) return false;
    }
    return true;
  }

  querySelectorAll(selector: string): MiniElement[] {
    const found: MiniElement[] = [];
    const visit = (node: MiniNode) => {
      for (const child of node.childNodes) {
        if (child instanceof MiniElement) {
          if (child.matches(selector)) found.push(child);
          visit(child);
        }
      }
    };
    visit(this);
    return found;
  }

  querySelector(selector: string): MiniElement | null { return this.querySelectorAll(selector)[0] ?? null; }
  get innerHTML(): string { return this.textContent; }
  set innerHTML(value: string) { this.textContent = value; }
}

class MiniIFrameElement extends MiniElement {}

class MiniDocument extends MiniNode {
  readonly documentElement: MiniElement;
  readonly body: MiniElement;
  defaultView: MiniWindow | null = null;
  activeElement: MiniElement;

  constructor() {
    super(9, "#document", null as unknown as MiniDocument);
    this.documentElement = new MiniElement("html", this);
    this.body = new MiniElement("body", this);
    this.appendChild(this.documentElement);
    this.documentElement.appendChild(this.body);
    this.activeElement = this.body;
  }

  createElement(tagName: string): MiniElement { return new MiniElement(tagName, this); }
  createElementNS(_namespace: string, tagName: string): MiniElement { return this.createElement(tagName); }
  createTextNode(value: string): MiniText { return new MiniText(value, this); }
  createComment(value: string): MiniComment { return new MiniComment(value, this); }
  querySelectorAll(selector: string): MiniElement[] { return this.documentElement.querySelectorAll(selector); }
  querySelector(selector: string): MiniElement | null { return this.documentElement.querySelector(selector); }
  getElementById(id: string): MiniElement | null { return this.querySelector(`#${id}`); }
}

class MiniWindow extends MiniNode {
  readonly document: MiniDocument;
  readonly navigator = { userAgent: "mini-dom" };
  readonly location = { protocol: "http:" };
  parent = this;
  self = this;
  event: MiniEvent | undefined = undefined;

  constructor(document: MiniDocument) {
    super(0, "window", document);
    this.document = document;
    Object.assign(this as unknown as Record<string, unknown>, {
      Node: MiniNode,
      Element: MiniElement,
      HTMLElement: MiniElement,
      HTMLIFrameElement: MiniIFrameElement,
      SVGElement: MiniElement,
      ShadowRoot: MiniNode,
    });
  }

  getSelection() { return null; }
  getComputedStyle() { return {}; }
  requestAnimationFrame(callback: FrameRequestCallback): number { callback(0); return 0; }
  cancelAnimationFrame() { /* no-op */ }
}

function installMiniDom() {
  const document = new MiniDocument();
  const window = new MiniWindow(document);
  document.defaultView = window;
  const globals = globalThis as unknown as Record<string, unknown>;
  Object.assign(globals, {
    window,
    document,
    Node: MiniNode,
    Element: MiniElement,
    HTMLElement: MiniElement,
    HTMLButtonElement: MiniElement,
    HTMLIFrameElement: MiniIFrameElement,
    SVGElement: MiniElement,
    Text: MiniText,
    Comment: MiniComment,
    Event: MiniEvent,
    KeyboardEvent: MiniEvent,
    MouseEvent: MiniEvent,
    IS_REACT_ACT_ENVIRONMENT: true,
  });
  Object.defineProperty(globals, "navigator", { configurable: true, value: window.navigator, writable: true });
  return { document, window };
}

const { document } = installMiniDom();
const { PairedInspector } = await import("./PairedInspector");
const { createRoot } = await import("react-dom/client");

function ControlledInspector({ flow }: { flow: InspectorFlow }) {
  const [activePane, setActivePane] = useState<InspectorPane>("request");
  return createElement(PairedInspector, { flow, activePane, onPaneChange: setActivePane });
}

function makeFlow(flowId: string, bodyData = Buffer.from("hello", "utf8").toString("base64"), error = "<img src=x onerror=alert(1)>"): InspectorFlow {
  const lifecycle = (state: FlowLifecycle["state"], sequence: string): FlowLifecycle => ({ protocol_version: "1", type: "flow.lifecycle", source_id: "source", flow_id: flowId, event_id: `${flowId}-${sequence}`, occurred_at: "2026-07-17T12:00:00.000Z", sequence, state });
  return {
    metadata: {
      flow_id: flowId,
      method: "POST",
      scheme: "https",
      host: "api.example.test",
      port: "443",
      path: "/stream",
      request_headers: [{ name: "x-trace", value: "first" }, { name: "x-trace", value: "second" }],
      request_body: { state: "captured", size_bytes: "5", encoding: "base64", data: bodyData },
    },
    lifecycle: [lifecycle("request_end", "9"), lifecycle("response_started", "8"), lifecycle("response_end", "10")],
    error,
  };
}

function find(container: MiniElement, selector: string): MiniElement {
  const element = container.querySelector(selector);
  if (!element) throw new Error(`missing ${selector}`);
  return element;
}

function dispatchKey(element: MiniElement, key: string) {
  element.dispatchEvent(new MiniEvent("keydown", { key, bubbles: true, cancelable: true }));
}

describe("mounted PairedInspector DOM behavior", () => {
  let container: MiniElement;
  let root: ReturnType<typeof createRoot>;
  let extraRoots: Array<{ container: MiniElement; root: ReturnType<typeof createRoot> }>;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container as unknown as Element);
    extraRoots = [];
  });

  afterEach(async () => {
    await act(async () => root.unmount());
    document.body.removeChild(container);
    for (const extra of extraRoots) {
      await act(async () => extra.root.unmount());
      document.body.removeChild(extra.container);
    }
    document.activeElement = document.body;
  });

  it("runs mounted clicks and renders the gated-to-decoded transition", async () => {
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-a") })));
    expect(container.querySelector(".inspector-action")).not.toBeNull();
    expect(container.querySelector(".inspector-output")).toBeNull();
    const inspect = find(container, ".inspector-action");
    inspect.focus();
    await act(async () => inspect.click());
    expect(container.querySelector(".inspector-output")?.textContent).toContain("hello");
    expect(document.activeElement?.getAttribute("role")).toBe("tab");
  });

  it("runs body keyboard navigation with horizontal-only arrows and roving tabindex", async () => {
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-body") })));
    await act(async () => find(container, ".inspector-action").click());
    const bodyTablist = find(container, '[aria-label="request body view mode"]');
    const tabs = bodyTablist.querySelectorAll('[role="tab"]');
    expect(tabs[0].getAttribute("tabindex")).toBe("-1");
    expect(tabs[1].getAttribute("tabindex")).toBe("0");
    tabs[1].focus();
    await act(async () => dispatchKey(tabs[1], "ArrowRight"));
    expect(tabs[2].getAttribute("aria-selected")).toBe("true");
    expect(document.activeElement).toBe(tabs[2]);
    await act(async () => dispatchKey(tabs[2], "ArrowDown"));
    expect(document.activeElement).toBe(tabs[2]);
    const bodyPanelId = tabs[2].getAttribute("aria-controls");
    expect(bodyPanelId).not.toBeNull();
    expect(container.querySelectorAll('[role="tabpanel"]').some((panel) => panel.getAttribute("id") === bodyPanelId)).toBe(true);
  });

  it("updates uncontrolled pane state and reports click and keyboard requests", async () => {
    const onPaneChange = vi.fn();
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-uncontrolled"), onPaneChange })));
    const tabs = find(container, '[aria-label="Exchange panes"]').querySelectorAll('[role="tab"]');
    await act(async () => tabs[1].click());
    expect(onPaneChange).toHaveBeenLastCalledWith("response");
    expect(tabs[1].getAttribute("aria-selected")).toBe("true");
    await act(async () => dispatchKey(tabs[1], "ArrowRight"));
    expect(onPaneChange).toHaveBeenLastCalledWith("error");
    expect(tabs[2].getAttribute("aria-selected")).toBe("true");
  });

  it("preserves body selection for same-pane requests and preserves committed panes across handoffs", async () => {
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection") })));
    await act(async () => find(container, ".inspector-action").click());
    expect(container.querySelector(".inspector-output")?.textContent).toContain("hello");
    await act(async () => outerTab(container, "request").click());
    expect(container.querySelector(".inspector-output")?.textContent).toContain("hello");

    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection"), activePane: "request" })));
    expect(container.querySelector(".inspector-output")?.textContent).toContain("hello");
    await act(async () => outerTab(container, "request").click());
    expect(container.querySelector(".inspector-output")?.textContent).toContain("hello");

    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection"), activePane: "response" })));
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection"), activePane: "error" })));
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection") })));
    expect(outerTab(container, "error").getAttribute("aria-selected")).toBe("true");

    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection") })));
    await act(async () => outerTab(container, "response").click());
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection"), activePane: "response" })));
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-selection") })));
    expect(outerTab(container, "response").getAttribute("aria-selected")).toBe("true");
  });

  it("does not steal focus from outer tabs, but recovers when the body control owns focus", async () => {
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-a") })));
    const outerTabs = find(container, '[aria-label="Exchange panes"]').querySelectorAll('[role="tab"]');
    outerTabs[0].focus();
    await act(async () => dispatchKey(outerTabs[0], "ArrowDown"));
    expect(document.activeElement).toBe(outerTabs[0]);
    expect(outerTabs[0].getAttribute("aria-selected")).toBe("true");
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-b") })));
    expect(document.activeElement).toBe(outerTabs[0]);

    const inspect = find(container, ".inspector-action");
    await act(async () => inspect.click());
    const bodyTab = find(container, '[aria-label="request body view mode"]').querySelector('[role="tab"]');
    if (!bodyTab) throw new Error("missing body tab");
    bodyTab.focus();
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-c") })));
    expect(document.activeElement?.className).toContain("inspector-action");
  });

  it("rerenders controlled panes through click and keyboard, preserves focus, and resolves every tab panel", async () => {
    await act(async () => root.render(createElement(ControlledInspector, { flow: makeFlow("flow-error") })));
    const tablist = find(container, '[aria-label="Exchange panes"]');
    const tabs = tablist.querySelectorAll('[role="tab"]');
    const panels = container.querySelectorAll('[role="tabpanel"]');
    expect(panels).toHaveLength(3);
    for (const tab of tabs) {
      const controls = tab.getAttribute("aria-controls");
      expect(controls).not.toBeNull();
      expect(panels.some((panel) => panel.getAttribute("id") === controls)).toBe(true);
    }

    await act(async () => tabs[1].click());
    expect(tabs[1].getAttribute("aria-selected")).toBe("true");
    expect(tabs[0].getAttribute("aria-selected")).toBe("false");
    const responsePanel = panels.find((panel) => panel.getAttribute("id") === tabs[1].getAttribute("aria-controls"));
    expect(responsePanel?.hasAttribute("hidden")).toBe(false);
    const requestPanel = panels.find((panel) => panel.getAttribute("id") === tabs[0].getAttribute("aria-controls"));
    expect(requestPanel?.hasAttribute("hidden")).toBe(true);
    expect(requestPanel?.querySelector(".inspector-body-panel")).toBeNull();
    expect(requestPanel?.querySelector(".inspector-output")).toBeNull();

    tabs[1].focus();
    await act(async () => dispatchKey(tabs[1], "ArrowRight"));
    expect(tabs[2].getAttribute("aria-selected")).toBe("true");
    await act(async () => dispatchKey(tabs[2], "ArrowDown"));
    expect(tabs[2].getAttribute("aria-selected")).toBe("true");
    const errorPanel = panels.find((panel) => panel.getAttribute("id") === tabs[2].getAttribute("aria-controls"));
    expect(errorPanel?.getAttribute("aria-labelledby")).toBe(tabs[2].getAttribute("id"));
    expect(errorPanel?.textContent).toContain("<img src=x onerror=alert(1)>");
    expect(container.querySelector("script")).toBeNull();
  });

  it("keeps accessibility IDs unique and locally resolvable for two inspectors", async () => {
    const otherContainer = document.createElement("div");
    document.body.appendChild(otherContainer);
    const otherRoot = createRoot(otherContainer as unknown as Element);
    extraRoots.push({ container: otherContainer, root: otherRoot });
    await act(async () => {
      root.render(createElement(PairedInspector, { flow: makeFlow("flow-one") }));
      otherRoot.render(createElement(PairedInspector, { flow: makeFlow("flow-two") }));
    });

    const instanceIds = (instance: MiniElement) => instance.querySelectorAll("[id]").map((element) => element.id);
    const firstIds = instanceIds(container);
    const secondIds = instanceIds(otherContainer);
    expect(new Set(firstIds).size).toBe(firstIds.length);
    expect(new Set(secondIds).size).toBe(secondIds.length);
    expect(firstIds.every((id) => !secondIds.includes(id))).toBe(true);
    for (const instance of [container, otherContainer]) {
      const ids = new Set(instanceIds(instance));
      for (const element of instance.querySelectorAll("[aria-controls]")) expect(ids.has(element.getAttribute("aria-controls") ?? "")).toBe(true);
      for (const element of instance.querySelectorAll("[aria-labelledby]")) expect(ids.has(element.getAttribute("aria-labelledby") ?? "")).toBe(true);
    }
  });

  it("renders lifecycle observations in sequence and bounds a large invalid base64 payload", async () => {
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-sequence") })));
    const trace = container.querySelectorAll(".inspector-trace-label");
    expect(trace.map((item) => item.textContent)).toEqual(["response opened", "request ended", "response ended"]);
    await act(async () => root.render(createElement(PairedInspector, { flow: makeFlow("flow-invalid", `!${"A".repeat(16 * 1024 * 1024)}`) })));
    await act(async () => find(container, ".inspector-action").click());
    expect(find(container, ".inspector-output").textContent).toContain("Invalid base64 payload");
  });
});

function outerTab(container: MiniElement, name: "request" | "response" | "error"): MiniElement {
  const tabs = find(container, '[aria-label="Exchange panes"]').querySelectorAll('[role="tab"]');
  const tab = tabs[name === "request" ? 0 : name === "response" ? 1 : 2];
  if (!tab) throw new Error(`missing ${name} exchange tab`);
  return tab;
}
