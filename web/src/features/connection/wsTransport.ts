/* eslint-disable no-unused-vars */

import type { ResyncRequest, TransportConnection, TransportFactory, TransportHandlers } from "./connectionClient";

export const STREAM_PATH = "/api/v1/stream";

export function defaultStreamUrl(location?: { protocol: string; host: string }): string {
  const page = location ?? (globalThis as { location?: { protocol: string; host: string } }).location;
  if (page === undefined) throw new Error("This environment has no page location for the local stream.");
  const scheme = page.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${page.host}${STREAM_PATH}`;
}

interface WebSocketLike {
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onerror: (() => void) | null;
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null;
  readonly readyState: number;
  send(data: string): void;
  close(code?: number): void;
}

export type WebSocketConstructor = new (url: string) => WebSocketLike;

/** Build the production transport speaking protocol-v1 over the loopback WebSocket. */
export function webSocketTransportFactory(
  url?: string,
  WebSocketImpl?: WebSocketConstructor,
): TransportFactory {
  return (handlers: TransportHandlers): TransportConnection => {
    const Ctor = WebSocketImpl ?? (globalThis.WebSocket as WebSocketConstructor | undefined);
    if (Ctor === undefined) {
      throw new Error("This environment has no WebSocket support for the local stream.");
    }
    const socket = new Ctor(url ?? defaultStreamUrl());
    let settled = false;
    socket.onopen = () => handlers.onOpen();
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") {
        settled = true;
        handlers.onError(new Error("The stream sent a non-text frame."));
        socket.close(1002);
        return;
      }
      let value: unknown;
      try {
        value = JSON.parse(event.data);
      } catch {
        settled = true;
        handlers.onError(new Error("The stream sent malformed JSON."));
        socket.close(1002);
        return;
      }
      handlers.onMessage(value);
    };
    socket.onerror = () => {
      if (settled) return;
      settled = true;
      handlers.onError(new Error("The local stream connection failed."));
    };
    socket.onclose = (event) => {
      if (settled) return;
      settled = true;
      handlers.onClose(event.reason || `Stream closed (code ${event.code})`);
    };
    return {
      close: () => {
        settled = true;
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close(1000);
      },
      requestResync: (message: ResyncRequest) => {
        socket.send(JSON.stringify(message));
      },
    };
  };
}
