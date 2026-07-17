/* eslint-disable no-unused-vars */

import { describe, expect, it, vi } from "vitest";

import {
  ConnectionClient,
  exponentialBackoff,
  type ConnectionTimer,
  type TransportFactory,
} from "./connectionClient";

class TestTimer implements ConnectionTimer {
  private nextId = 0;
  private callbacks = new Map<number, () => void>();

  public set(callback: () => void): number {
    const id = ++this.nextId;
    this.callbacks.set(id, callback);
    return id;
  }

  public clear(handle: unknown): void {
    this.callbacks.delete(handle as number);
  }

  public runAll(): void {
    const callbacks = [...this.callbacks.values()];
    this.callbacks.clear();
    for (const callback of callbacks) callback();
  }

  public get size(): number {
    return this.callbacks.size;
  }
}

function factoryHarness(withResync = true) {
  const handlers: Array<Parameters<TransportFactory>[0]> = [];
  const closes: Array<() => void> = [];
  const resyncs: Array<(...args: [{ requested_cursor: string }]) => void> = [];
  const factory: TransportFactory = (nextHandlers) => {
    handlers.push(nextHandlers);
    const close = vi.fn();
    const requestResync = vi.fn();
    closes.push(close);
    resyncs.push(requestResync);
    return withResync ? { close, requestResync } : { close };
  };
  return { factory, handlers, closes, resyncs };
}

describe("connection client", () => {
  it("revalidates after an attempt listener disconnects before factory invocation", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, autoReconnect: false });
    client.subscribe((event) => { if (event.type === "attempt") client.disconnect(); });

    client.connect();

    expect(client.getSnapshot().state).toBe("disconnected");
    expect(client.getSnapshot().requestResyncAvailable).toBe(false);
    expect(harness.handlers).toHaveLength(0);
    expect(timer.size).toBe(0);
  });

  it("reconnects with an injected backoff seam after an unexpected close", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, retryDelayMs: () => 1 });
    client.connect();
    harness.handlers[0].onOpen();
    harness.handlers[0].onClose("gone");

    expect(client.getSnapshot().state).toBe("reconnecting");
    expect(client.getSnapshot().attempt).toBe(1);
    expect(timer.size).toBe(1);
    timer.runAll();
    expect(harness.handlers).toHaveLength(2);
    expect(client.getSnapshot().state).toBe("reconnecting");
  });

  it("ignores late error and close callbacks from an obsolete transport", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, retryDelayMs: () => 1 });
    const attempts: number[] = [];
    client.subscribe((event) => { if (event.type === "attempt") attempts.push(event.id); });
    client.connect();
    harness.handlers[0].onOpen();
    harness.handlers[0].onClose("retry");
    timer.runAll();
    harness.handlers[1].onOpen();

    harness.handlers[0].onError("late error");
    harness.handlers[0].onClose("late close");

    expect(client.getSnapshot().state).toBe("live");
    expect(client.getSnapshot().error).toBeNull();
    expect(timer.size).toBe(1);
    expect(attempts).toEqual([1, 2]);
  });

  it("closes a connection returned after synchronous obsolete callbacks", () => {
    const timer = new TestTimer();
    const staleClose = vi.fn();
    const factory: TransportFactory = (handlers) => {
      handlers.onError("synchronous failure");
      return { close: staleClose };
    };
    const client = new ConnectionClient({ transportFactory: factory, timer, autoReconnect: false });
    client.connect();

    expect(staleClose).toHaveBeenCalledOnce();
    expect(client.getSnapshot().state).toBe("error");
    expect(client.requestResync("1")).toEqual({ ok: false, reason: "not-connected" });
  });

  it("also closes a connection returned after a synchronous close callback", () => {
    const timer = new TestTimer();
    const staleClose = vi.fn();
    const factory: TransportFactory = (handlers) => {
      handlers.onClose("synchronous close");
      return { close: staleClose };
    };
    const client = new ConnectionClient({ transportFactory: factory, timer, autoReconnect: false });
    client.connect();

    expect(staleClose).toHaveBeenCalledOnce();
    expect(client.getSnapshot().state).toBe("error");
  });

  it("marks a quiet live source stale and returns to live on the next event", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, staleAfterMs: 1 });
    client.connect();
    harness.handlers[0].onOpen();
    timer.runAll();
    expect(client.getSnapshot().state).toBe("stale");
    harness.handlers[0].onMessage({ protocol_version: "1", type: "future.message" });
    expect(client.getSnapshot().state).toBe("live");
  });

  it("cancels pending retry and stale timers on teardown", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, retryDelayMs: () => 1 });
    client.connect();
    harness.handlers[0].onClose();
    expect(timer.size).toBe(1);
    client.disconnect();
    timer.runAll();
    expect(harness.handlers).toHaveLength(1);
    expect(client.getSnapshot().state).toBe("disconnected");
    expect(harness.closes[0]).not.toHaveBeenCalled();
  });

  it("does not duplicate a listener across StrictMode-style subscribe cycles", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, autoReconnect: false });
    const listener = vi.fn();
    const unsubscribe = client.subscribe(listener);
    client.connect();
    client.disconnect();
    unsubscribe();
    client.subscribe(listener);
    client.connect();
    harness.handlers[1].onOpen();

    expect(listener.mock.calls.filter(([event]) => event.type === "status")).toHaveLength(6);
    expect(harness.handlers).toHaveLength(2);
    client.destroy();
    expect(timer.size).toBe(0);
  });

  it("keeps unknown protocol messages observable and malformed messages actionable", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const events: string[] = [];
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, autoReconnect: false });
    client.subscribe((event) => events.push(event.type));
    client.connect();
    harness.handlers[0].onOpen();
    harness.handlers[0].onMessage({ protocol_version: "1", type: "future.message" });
    harness.handlers[0].onMessage({ protocol_version: "1", type: "browser.delta", cursor: "bad", changes: [] });

    expect(events).toContain("message");
    expect(events).toContain("protocol-error");
    expect(client.getSnapshot().state).toBe("error");
  });

  it("isolates an observer that throws while handling a valid message", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const observed: string[] = [];
    const listenerErrors: unknown[] = [];
    const client = new ConnectionClient({
      transportFactory: harness.factory,
      timer,
      autoReconnect: false,
      onListenerError: (error) => listenerErrors.push(error),
    });
    client.subscribe((event) => {
      if (event.type === "message") throw new Error("observer failed");
    });
    client.subscribe((event) => observed.push(event.type));
    client.connect();
    harness.handlers[0].onOpen();
    harness.handlers[0].onMessage({ protocol_version: "1", type: "future.message" });

    expect(client.getSnapshot().state).toBe("live");
    expect(harness.closes[0]).not.toHaveBeenCalled();
    expect(observed).toContain("message");
    expect(observed).not.toContain("protocol-error");
    expect(listenerErrors).toHaveLength(1);
  });

  it("publishes frozen status, message, and protocol-error events to hostile observers", () => {
    const timer = new TestTimer();
    const harness = factoryHarness();
    const seen: string[] = [];
    const client = new ConnectionClient({ transportFactory: harness.factory, timer, autoReconnect: false });
    client.subscribe((event) => {
      try { (event as { type: string }).type = "corrupted"; } catch { /* frozen publication */ }
      if (event.type === "status") {
        try { (event.status as { state: string }).state = "corrupted"; } catch { /* frozen publication */ }
      }
      if (event.type === "message") {
        try { (event.envelope as { kind: string }).kind = "corrupted"; } catch { /* frozen publication */ }
      }
      if (event.type === "protocol-error") {
        try { event.error.message = "corrupted"; } catch { /* frozen publication */ }
      }
    });
    client.subscribe((event) => {
      if (event.type === "status") seen.push(`status:${event.status.state}`);
      if (event.type === "message") seen.push(`message:${event.envelope.kind}`);
      if (event.type === "protocol-error") seen.push(`error:${event.error.message}`);
    });

    client.connect();
    harness.handlers[0].onOpen();
    harness.handlers[0].onMessage({ protocol_version: "1", type: "future.message" });
    harness.handlers[0].onMessage({ protocol_version: "1", type: "browser.delta", cursor: "bad", changes: [] });

    expect(seen).toContain("status:connecting");
    expect(seen).toContain("status:live");
    expect(seen).toContain("message:unknown");
    expect(seen.some((event) => event.startsWith("error:") && event !== "error:corrupted")).toBe(true);
    expect(client.getSnapshot().state).toBe("error");
  });

  it("exposes explicit resync capability and request results", () => {
    const timer = new TestTimer();
    const supported = factoryHarness(true);
    const client = new ConnectionClient({ transportFactory: supported.factory, timer, autoReconnect: false });
    client.connect();
    expect(client.getSnapshot().requestResyncAvailable).toBe(true);
    const publicSnapshot = client.getSnapshot();
    expect(Object.isFrozen(publicSnapshot)).toBe(true);
    expect(() => { (publicSnapshot as { attempt: number }).attempt = 99; }).toThrow(TypeError);
    expect(client.requestResync("01")).toEqual({ ok: false, reason: "invalid-cursor" });
    expect(client.requestResync("18446744073709551616")).toEqual({ ok: false, reason: "invalid-cursor" });
    expect(client.requestResync("41")).toEqual({ ok: true });
    expect(supported.resyncs[0]).toHaveBeenCalledWith({
      protocol_version: "1", type: "browser.resync", reason: "cursor_gap", requested_cursor: "41",
    });

    const unsupported = factoryHarness(false);
    const unsupportedClient = new ConnectionClient({ transportFactory: unsupported.factory, timer, autoReconnect: false });
    unsupportedClient.connect();
    expect(unsupportedClient.getSnapshot().requestResyncAvailable).toBe(false);
    expect(unsupportedClient.requestResync("41")).toEqual({ ok: false, reason: "unsupported" });
  });

  it("provides bounded jittered exponential delays", () => {
    expect(exponentialBackoff(1, { baseMs: 100, maxMs: 150, jitter: 0, random: () => 0 })).toBe(100);
    expect(exponentialBackoff(3, { baseMs: 100, maxMs: 150, jitter: 0, random: () => 0 })).toBe(150);
  });
});
