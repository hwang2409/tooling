/* eslint-disable no-unused-vars */

import { describe, expect, it } from "vitest";

import type { TransportHandlers } from "./connectionClient";
import { defaultStreamUrl, STREAM_PATH, webSocketTransportFactory } from "./wsTransport";

class FakeWebSocket {
  public static instances: FakeWebSocket[] = [];
  public onopen: (() => void) | null = null;
  public onmessage: ((event: { data: unknown }) => void) | null = null;
  public onerror: (() => void) | null = null;
  public onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
  public readyState = 0;
  public sent: string[] = [];
  public closedWith: number | undefined;

  public constructor(public readonly url: string) {
    FakeWebSocket.instances.push(this);
  }

  public send(data: string): void {
    this.sent.push(data);
  }

  public close(code?: number): void {
    this.closedWith = code;
  }
}

function makeHandlers() {
  const events: Array<{ kind: string; value?: unknown }> = [];
  const handlers: TransportHandlers = {
    onOpen: () => events.push({ kind: "open" }),
    onMessage: (value) => events.push({ kind: "message", value }),
    onError: (error) => events.push({ kind: "error", value: error }),
    onClose: (reason) => events.push({ kind: "close", value: reason }),
  };
  return { events, handlers };
}

describe("webSocketTransportFactory", () => {
  it("builds a loopback stream URL from the page location", () => {
    expect(defaultStreamUrl({ protocol: "http:", host: "127.0.0.1:5173" })).toBe(
      `ws://127.0.0.1:5173${STREAM_PATH}`,
    );
    expect(defaultStreamUrl({ protocol: "https:", host: "localhost:8443" })).toBe(
      `wss://localhost:8443${STREAM_PATH}`,
    );
  });

  it("parses incoming text frames as JSON values and forwards resync requests", () => {
    FakeWebSocket.instances = [];
    const { events, handlers } = makeHandlers();
    const factory = webSocketTransportFactory("ws://127.0.0.1:1/api/v1/stream", FakeWebSocket);
    const connection = factory(handlers);
    const socket = FakeWebSocket.instances[0];
    socket.onopen?.();
    socket.onmessage?.({ data: '{"protocol_version":"1","type":"browser.resync","reason":"initial_connect","requested_cursor":"0"}' });
    connection.requestResync?.({ protocol_version: "1", type: "browser.resync", reason: "cursor_gap", requested_cursor: "4" });

    expect(events[0]).toEqual({ kind: "open" });
    expect(events[1].kind).toBe("message");
    expect((events[1].value as { type: string }).type).toBe("browser.resync");
    expect(JSON.parse(socket.sent[0]).requested_cursor).toBe("4");
  });

  it("reports malformed JSON as a transport error and closes with 1002", () => {
    FakeWebSocket.instances = [];
    const { events, handlers } = makeHandlers();
    factoryConnect(handlers);
    const socket = FakeWebSocket.instances[0];
    socket.onmessage?.({ data: "{not json" });
    socket.onclose?.({ code: 1002, reason: "", wasClean: false });

    expect(events).toHaveLength(1);
    expect(events[0].kind).toBe("error");
    expect(socket.closedWith).toBe(1002);
  });

  it("reports non-text frames as a transport error", () => {
    FakeWebSocket.instances = [];
    const { events, handlers } = makeHandlers();
    factoryConnect(handlers);
    const socket = FakeWebSocket.instances[0];
    socket.onmessage?.({ data: new ArrayBuffer(4) });

    expect(events[0].kind).toBe("error");
    expect(socket.closedWith).toBe(1002);
  });

  it("reports a close reason exactly once and mutes callbacks after close()", () => {
    FakeWebSocket.instances = [];
    const { events, handlers } = makeHandlers();
    factoryConnect(handlers);
    const socket = FakeWebSocket.instances[0];
    socket.onclose?.({ code: 1001, reason: "going away", wasClean: true });
    expect(events).toEqual([{ kind: "close", value: "going away" }]);

    const second = makeHandlers();
    const secondConnection = factoryConnect(second.handlers);
    const secondSocket = FakeWebSocket.instances[1];
    secondConnection.close();
    expect(secondSocket.closedWith).toBe(1000);
    expect(secondSocket.onclose).toBeNull();
    expect(second.events).toHaveLength(0);
  });

  it("throws when no WebSocket implementation exists", () => {
    const factory = webSocketTransportFactory("ws://127.0.0.1:1/api/v1/stream", undefined);
    const globalSocket = Reflect.get(globalThis, "WebSocket");
    if (globalSocket !== undefined) return; // browser-like environment provides one
    const { handlers } = makeHandlers();
    expect(() => factory(handlers)).toThrowError(/WebSocket/);
  });
});

function factoryConnect(handlers: TransportHandlers) {
  const factory = webSocketTransportFactory("ws://127.0.0.1:1/api/v1/stream", FakeWebSocket);
  return factory(handlers);
}
