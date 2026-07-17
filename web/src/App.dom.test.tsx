/* eslint-disable no-unused-vars */

import { createRoot } from "react-dom/client";
import { act, StrictMode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";
import type { TransportFactory, TransportHandlers } from "./features/connection/connectionClient";

type Listener = (event: FakeEvent) => void;

class FakeEvent {
  public readonly type: string;
  public readonly bubbles: boolean;
  public target: FakeNode | null = null;
  public currentTarget: FakeNode | null = null;

  public constructor(type: string, options: { bubbles?: boolean } = {}) {
    this.type = type;
    this.bubbles = options.bubbles ?? false;
  }
}

class FakeNode {
  public readonly nodeType: number;
  public ownerDocument: FakeDocument;
  public parentNode: FakeNode | null = null;
  public childNodes: FakeNode[] = [];
  private readonly listeners = new Map<string, Set<Listener>>();

  public constructor(nodeType: number, ownerDocument: FakeDocument) {
    this.nodeType = nodeType;
    this.ownerDocument = ownerDocument;
  }

  public appendChild<T extends FakeNode>(child: T): T {
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  public insertBefore<T extends FakeNode>(child: T, before: FakeNode | null): T {
    child.parentNode = this;
    const index = before === null ? -1 : this.childNodes.indexOf(before);
    if (index === -1) this.childNodes.push(child);
    else this.childNodes.splice(index, 0, child);
    return child;
  }

  public removeChild<T extends FakeNode>(child: T): T {
    const index = this.childNodes.indexOf(child);
    if (index !== -1) this.childNodes.splice(index, 1);
    child.parentNode = null;
    return child;
  }

  public addEventListener(type: string, listener: Listener): void {
    const listeners = this.listeners.get(type) ?? new Set<Listener>();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  public removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }

  public dispatchEvent(event: FakeEvent): boolean {
    if (event.target === null) event.target = this;
    event.currentTarget = this;
    for (const listener of [...(this.listeners.get(event.type) ?? [])]) listener(event);
    if (event.bubbles && this.parentNode !== null) this.parentNode.dispatchEvent(event);
    return true;
  }

  public get firstChild(): FakeNode | null {
    return this.childNodes[0] ?? null;
  }

  public get textContent(): string {
    return this.childNodes.map((child) => child.textContent).join("");
  }

  public set textContent(value: string) {
    this.childNodes = value === "" ? [] : [new FakeText(value, this.ownerDocument)];
  }
}

class FakeText extends FakeNode {
  private value: string;

  public constructor(value: string, ownerDocument: FakeDocument) {
    super(3, ownerDocument);
    this.value = value;
  }

  public override get textContent(): string {
    return this.value;
  }

  public override set textContent(value: string) {
    this.value = value;
  }

  public get nodeValue(): string {
    return this.value;
  }

  public set nodeValue(value: string) {
    this.value = value;
  }
}

class FakeElement extends FakeNode {
  public readonly tagName: string;
  public readonly nodeName: string;
  public readonly style: Record<string, string> = {};
  public disabled = false;
  public value = "";
  public tabIndex = 0;
  private readonly attributes = new Map<string, string>();

  public constructor(tagName: string, ownerDocument: FakeDocument) {
    super(1, ownerDocument);
    this.tagName = tagName.toUpperCase();
    this.nodeName = this.tagName;
  }

  public setAttribute(name: string, value: string): void {
    this.attributes.set(name, value);
    if (name === "disabled") this.disabled = true;
  }

  public getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  public removeAttribute(name: string): void {
    this.attributes.delete(name);
    if (name === "disabled") this.disabled = false;
  }

  public querySelectorAll(selector: string): FakeElement[] {
    const matches: FakeElement[] = [];
    const visit = (node: FakeNode) => {
      for (const child of node.childNodes) {
        if (child instanceof FakeElement && matchesSelector(child, selector)) matches.push(child);
        visit(child);
      }
    };
    visit(this);
    return matches;
  }

  public click(): void {
    this.dispatchEvent(new FakeEvent("click", { bubbles: true }));
  }

  public focus(): void {
    this.ownerDocument.activeElement = this;
  }
}

class FakeDocument extends FakeNode {
  public readonly documentElement: FakeElement;
  public readonly body: FakeElement;
  public activeElement: FakeElement | null = null;
  public defaultView: FakeWindow | null = null;

  public constructor() {
    super(9, null as unknown as FakeDocument);
    this.documentElement = new FakeElement("html", this);
    this.body = new FakeElement("body", this);
    this.documentElement.appendChild(this.body);
  }

  public createElement(tagName: string): FakeElement {
    return new FakeElement(tagName, this);
  }

  public createElementNS(_namespace: string, tagName: string): FakeElement {
    return this.createElement(tagName);
  }

  public createTextNode(value: string): FakeText {
    return new FakeText(value, this);
  }

}

class FakeWindow extends FakeNode {
  public readonly document: FakeDocument;
  public readonly navigator = { userAgent: "fake-browser" };
  public readonly HTMLIFrameElement = FakeElement;

  public constructor(document: FakeDocument) {
    super(0, document);
    this.document = document;
  }
}

function matchesSelector(element: FakeElement, selector: string): boolean {
  const role = selector.match(/^\[role="([^"]+)"\]$/)?.[1];
  if (role !== undefined) return element.getAttribute("role") === role;
  const attribute = selector.match(/^\[aria-label="([^"]+)"\]$/)?.[1];
  if (attribute !== undefined) return element.getAttribute("aria-label") === attribute;
  return selector === element.tagName.toLowerCase();
}

function installDom(): { document: FakeDocument; restore: () => void } {
  const document = new FakeDocument();
  const window = new FakeWindow(document);
  document.defaultView = window;
  const previous = {
    document: globalThis.document,
    window: globalThis.window,
    navigator: globalThis.navigator,
    Node: globalThis.Node,
    HTMLElement: globalThis.HTMLElement,
    Event: globalThis.Event,
    actEnvironment: (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT,
  };
  Object.assign(globalThis, { document, window, Node: FakeNode, HTMLElement: FakeElement, Event: FakeEvent });
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: window.navigator, writable: true });
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  return {
    document,
    restore: () => {
      Object.assign(globalThis, { document: previous.document, window: previous.window, Node: previous.Node, HTMLElement: previous.HTMLElement, Event: previous.Event });
      Object.defineProperty(globalThis, "navigator", { configurable: true, value: previous.navigator, writable: true });
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = previous.actEnvironment;
    },
  };
}

function flow(flowId: string) {
  return {
    flow_id: flowId,
    method: "POST",
    scheme: "https",
    host: "api.example.test",
    port: "443",
    path: "/v1/messages",
    request_headers: [],
    request_body: { state: "empty", size_bytes: "0" },
  };
}

function hello(sourceId: string) {
  return {
    protocol_version: "1", type: "source.hello", source_id: sourceId,
    occurred_at: "2026-01-01T00:00:00Z",
    capabilities: { body_chunks: true, redaction: "headers-and-query" },
    limits: { max_body_prefix_bytes: "1048576", max_in_memory_bytes: "134217728" },
  };
}

function snapshot(cursor: string, snapshotId: string, flows: ReturnType<typeof flow>[]) {
  return { protocol_version: "1", type: "browser.snapshot", snapshot_id: snapshotId, cursor, flows };
}

function gap(cursor: string) {
  return { protocol_version: "1", type: "browser.delta", cursor, changes: [] };
}

interface MountedHarness {
  readonly handlers: TransportHandlers[];
  readonly connections: Array<{ close: ReturnType<typeof vi.fn>; requestResync?: ReturnType<typeof vi.fn> }>;
  readonly factory: TransportFactory;
}

function mountedHarness(withResync = true): MountedHarness {
  const handlers: TransportHandlers[] = [];
  const connections: MountedHarness["connections"] = [];
  const factory: TransportFactory = (nextHandlers) => {
    handlers.push(nextHandlers);
    const connection = { close: vi.fn(), requestResync: vi.fn() };
    connections.push(connection);
    return withResync ? connection : { close: connection.close };
  };
  return { handlers, connections, factory };
}

function buttons(container: FakeElement): FakeElement[] {
  return container.querySelectorAll("button");
}

function buttonWithText(container: FakeElement, text: string): FakeElement {
  const button = buttons(container).find((candidate) => candidate.textContent.includes(text));
  if (!button) throw new Error(`button not found: ${text}`);
  return button;
}

function buttonWithAttribute(container: FakeElement, name: string, value: string): FakeElement {
  const button = buttons(container).find((candidate) => candidate.getAttribute(name) === value);
  if (!button) throw new Error(`button not found: ${name}=${value}`);
  return button;
}

describe("mounted App connection behavior", () => {
  let restore: (() => void) | undefined;

  afterEach(() => {
    vi.useRealTimers();
    restore?.();
    restore = undefined;
  });

  it("mounts under StrictMode, pauses A, reconnects to B, and resyncs only B", async () => {
    vi.useFakeTimers();
    const installed = installDom();
    restore = installed.restore;
    const harness = mountedHarness();
    const container = installed.document.createElement("div");
    installed.document.body.appendChild(container);
    const root = createRoot(container as unknown as Element);

    await act(async () => {
      root.render(<StrictMode><App transportFactory={harness.factory} /></StrictMode>);
    });
    expect(harness.handlers).toHaveLength(2);
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
    const connectionA = harness.connections[1];
    await act(async () => {
      harness.handlers[1].onOpen();
      harness.handlers[1].onMessage(hello("source-a"));
      harness.handlers[1].onMessage(snapshot("1", "a", [flow("old-flow")]));
      harness.handlers[1].onMessage(gap("3"));
    });

    await act(async () => buttonWithAttribute(container, "aria-pressed", "true").click());
    expect(container.textContent).toContain("Live paused");
    expect(container.textContent).toContain("paused at cursor 1");
    expect(container.textContent).toContain("source-a");

    await act(async () => harness.handlers[1].onClose("source A closed"));
    expect(container.textContent).toContain("Reconnecting");
    expect(container.textContent).toContain("Last failure: source A closed");
    expect(container.textContent).toContain("Stop reconnecting");
    await act(async () => vi.advanceTimersByTime(5_000));
    expect(harness.handlers).toHaveLength(3);
    const connectionB = harness.connections[2];
    await act(async () => {
      harness.handlers[2].onOpen();
      harness.handlers[2].onMessage(hello("source-b"));
      harness.handlers[2].onMessage(snapshot("0", "b", [flow("new-flow")]));
    });

    expect(container.textContent).toContain("source-a");
    expect(container.textContent).not.toContain("source-b");
    const staleRequest = buttonWithText(container, "Snapshot request unavailable");
    expect(staleRequest.disabled).toBe(true);
    await act(async () => staleRequest.click());
    expect(connectionB.requestResync).not.toHaveBeenCalled();

    await act(async () => harness.handlers[2].onMessage(gap("2")));
    const request = buttonWithText(container, "Request snapshot");
    await act(async () => request.click());
    expect(connectionB.requestResync).toHaveBeenCalledWith(expect.objectContaining({ requested_cursor: "0" }));
    expect(connectionA.requestResync).not.toHaveBeenCalled();

    await act(async () => buttonWithText(container, "Live paused").click());
    expect(container.textContent).toContain("source-b");
    expect(container.textContent).toContain("cursor 2");

    await act(async () => root.unmount());
    expect(connectionB.close).toHaveBeenCalledOnce();
  });

  it("mounts an unsupported transport as an explicit unavailable resync action", async () => {
    const installed = installDom();
    restore = installed.restore;
    const harness = mountedHarness(false);
    const container = installed.document.createElement("div");
    installed.document.body.appendChild(container);
    const root = createRoot(container as unknown as Element);

    await act(async () => root.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(snapshot("1", "one", []));
      harness.handlers[0].onMessage(gap("3"));
    });
    const unavailable = buttonWithText(container, "Snapshot request unavailable");
    expect(unavailable.disabled).toBe(true);

    await act(async () => root.unmount());
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
  });

  it("renders resync availability, transport errors, and cleans up on unmount", async () => {
    const installed = installDom();
    restore = installed.restore;
    const harness = mountedHarness();
    const container = installed.document.createElement("div");
    installed.document.body.appendChild(container);
    const root = createRoot(container as unknown as Element);

    await act(async () => root.render(<App transportFactory={harness.factory} />));
    await act(async () => {
      harness.handlers[0].onOpen();
      harness.handlers[0].onMessage(snapshot("1", "one", []));
      harness.handlers[0].onMessage(gap("3"));
    });
    expect(buttonWithText(container, "Request snapshot").disabled).toBe(false);
    harness.connections[0].requestResync?.mockImplementation(() => { throw new Error("send failed"); });
    await act(async () => buttonWithText(container, "Request snapshot").click());
    expect(container.textContent).toContain("Snapshot request could not be sent.");

    await act(async () => root.unmount());
    expect(harness.connections[0].close).toHaveBeenCalledOnce();
  });
});
