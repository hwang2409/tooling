/* eslint-disable no-unused-vars */

import { parseProtocolMessage } from "../../protocol";
import type { ParsedMessage, ResyncReason } from "../../protocol";

export type ConnectionStatusName = "disconnected" | "connecting" | "live" | "reconnecting" | "stale" | "error";

export interface ConnectionStatus {
  state: ConnectionStatusName;
  attempt: number;
  error: string | null;
  lastMessageAt: number | null;
}

export interface TransportHandlers {
  onOpen: () => void;
  onMessage: (value: unknown) => void;
  onError: (error: unknown) => void;
  onClose: (reason?: unknown) => void;
}

export interface TransportConnection {
  close: () => void;
  requestResync?: (message: {
    protocol_version: "1";
    type: "browser.resync";
    reason: ResyncReason;
    requested_cursor: string;
  }) => void;
}

export type TransportFactory = (handlers: TransportHandlers) => TransportConnection;

export interface ConnectionTimer {
  set: (callback: () => void, delayMs: number) => unknown;
  clear: (handle: unknown) => void;
}

export interface ConnectionClientOptions {
  transportFactory: TransportFactory;
  timer?: ConnectionTimer;
  now?: () => number;
  retryDelayMs?: (attempt: number) => number;
  staleAfterMs?: number;
  autoReconnect?: boolean;
}

export type ConnectionEvent =
  | { type: "status"; status: ConnectionStatus }
  | { type: "message"; envelope: ParsedMessage }
  | { type: "protocol-error"; error: Error };

const defaultTimer: ConnectionTimer = {
  set: (callback, delayMs) => globalThis.setTimeout(callback, delayMs),
  clear: (handle) => globalThis.clearTimeout(handle as number),
};

export function exponentialBackoff(
  attempt: number,
  options: { baseMs?: number; maxMs?: number; jitter?: number; random?: () => number } = {},
): number {
  const baseMs = options.baseMs ?? 500;
  const maxMs = options.maxMs ?? 30_000;
  const jitter = options.jitter ?? 0.2;
  const random = options.random ?? Math.random;
  const exponential = Math.min(maxMs, baseMs * 2 ** Math.max(0, attempt - 1));
  const spread = exponential * jitter;
  return Math.round(exponential - spread + random() * spread * 2);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error || "Unknown connection error");
}

export class ConnectionClient {
  private readonly transportFactory: TransportFactory;
  private readonly timer: ConnectionTimer;
  private readonly now: () => number;
  private readonly retryDelayMs: (attempt: number) => number;
  private readonly staleAfterMs: number;
  private readonly autoReconnect: boolean;
  private readonly listeners = new Set<(event: ConnectionEvent) => void>();
  private connection: TransportConnection | null = null;
  private reconnectTimer: unknown = null;
  private staleTimer: unknown = null;
  private generation = 0;
  private ignoredCloseGeneration: number | null = null;
  private destroyed = false;
  private userDisconnected = true;
  private status: ConnectionStatus = { state: "disconnected", attempt: 0, error: null, lastMessageAt: null };

  public constructor(options: ConnectionClientOptions) {
    this.transportFactory = options.transportFactory;
    this.timer = options.timer ?? defaultTimer;
    this.now = options.now ?? Date.now;
    this.retryDelayMs = options.retryDelayMs ?? ((attempt) => exponentialBackoff(attempt));
    this.staleAfterMs = options.staleAfterMs ?? 15_000;
    this.autoReconnect = options.autoReconnect ?? true;
  }

  public getSnapshot(): ConnectionStatus {
    return this.status;
  }

  public subscribe(listener: (event: ConnectionEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public connect(): void {
    if (this.destroyed || !this.userDisconnected && (this.status.state === "connecting" || this.status.state === "live" || this.status.state === "stale" || this.status.state === "reconnecting")) return;
    this.userDisconnected = false;
    this.clearReconnectTimer();
    this.open(++this.generation);
  }

  public disconnect(): void {
    if (this.destroyed) return;
    this.userDisconnected = true;
    this.generation += 1;
    this.clearReconnectTimer();
    this.clearStaleTimer();
    this.closeConnection();
    this.updateStatus({ state: "disconnected", attempt: 0, error: null, lastMessageAt: null });
  }

  public destroy(): void {
    if (this.destroyed) return;
    this.disconnect();
    this.destroyed = true;
    this.listeners.clear();
  }

  public requestResync(cursor: string, reason: ResyncReason = "cursor_gap"): void {
    this.connection?.requestResync?.({
      protocol_version: "1",
      type: "browser.resync",
      reason,
      requested_cursor: cursor,
    });
  }

  private open(generation: number): void {
    this.updateStatus({ state: this.status.attempt > 0 ? "reconnecting" : "connecting", error: null });
    try {
      this.connection = this.transportFactory({
        onOpen: () => {
          if (generation !== this.generation || this.userDisconnected) return;
          this.updateStatus({ state: "live", attempt: 0, error: null, lastMessageAt: this.now() });
          this.scheduleStale(generation);
        },
        onMessage: (value) => this.receive(value, generation),
        onError: (error) => this.fail(error, generation),
        onClose: (reason) => {
          if (this.ignoredCloseGeneration === generation) {
            this.ignoredCloseGeneration = null;
            return;
          }
          this.closed(reason, generation);
        },
      });
    } catch (error) {
      this.fail(error, generation);
    }
  }

  private receive(value: unknown, generation: number): void {
    if (generation !== this.generation || this.userDisconnected) return;
    try {
      const envelope = parseProtocolMessage(value);
      this.updateStatus({ state: "live", error: null, lastMessageAt: this.now() });
      this.scheduleStale(generation);
      this.emit({ type: "message", envelope });
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error(errorMessage(error));
      this.emit({ type: "protocol-error", error: normalized });
      this.fail(normalized, generation);
    }
  }

  private fail(error: unknown, generation: number): void {
    if (generation !== this.generation || this.userDisconnected) return;
    this.clearStaleTimer();
    this.closeConnection();
    this.updateStatus({ state: "error", error: errorMessage(error) });
    if (this.autoReconnect) this.scheduleReconnect(generation);
  }

  private closed(reason: unknown, generation: number): void {
    if (generation !== this.generation || this.userDisconnected) return;
    this.connection = null;
    this.clearStaleTimer();
    this.updateStatus({ state: "error", error: reason ? errorMessage(reason) : "Source closed the stream" });
    if (this.autoReconnect) this.scheduleReconnect(generation);
  }

  private scheduleReconnect(generation: number): void {
    this.clearReconnectTimer();
    const attempt = this.status.attempt + 1;
    this.updateStatus({ state: "reconnecting", attempt });
    this.reconnectTimer = this.timer.set(() => {
      this.reconnectTimer = null;
      if (generation === this.generation && !this.userDisconnected) this.open(generation);
    }, this.retryDelayMs(attempt));
  }

  private scheduleStale(generation: number): void {
    this.clearStaleTimer();
    this.staleTimer = this.timer.set(() => {
      if (generation === this.generation && !this.userDisconnected && this.status.state === "live") {
        this.updateStatus({ state: "stale" });
      }
    }, this.staleAfterMs);
  }

  private updateStatus(patch: Partial<ConnectionStatus>): void {
    this.status = { ...this.status, ...patch };
    this.emit({ type: "status", status: this.status });
  }

  private emit(event: ConnectionEvent): void {
    for (const listener of [...this.listeners]) listener(event);
  }

  private closeConnection(): void {
    const connection = this.connection;
    this.connection = null;
    if (connection) {
      this.ignoredCloseGeneration = this.generation;
      connection.close();
    }
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer !== null) {
      this.timer.clear(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private clearStaleTimer(): void {
    if (this.staleTimer !== null) {
      this.timer.clear(this.staleTimer);
      this.staleTimer = null;
    }
  }
}

export function unavailableTransportFactory(): TransportConnection {
  throw new Error("No local source adapter is configured yet. Start the B3 API to connect.");
}
