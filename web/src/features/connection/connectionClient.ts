/* eslint-disable no-unused-vars */

import { MAX_U64, parseProtocolMessage } from "../../protocol";
import type { ParsedMessage, ResyncReason } from "../../protocol";

export type ConnectionStatusName = "disconnected" | "connecting" | "live" | "reconnecting" | "stale" | "error";

export interface ConnectionStatus {
  readonly state: ConnectionStatusName;
  readonly attempt: number;
  readonly error: string | null;
  readonly lastMessageAt: number | null;
  readonly requestResyncAvailable: boolean;
}

export interface TransportHandlers {
  onOpen: () => void;
  onMessage: (value: unknown) => void;
  onError: (error: unknown) => void;
  onClose: (reason?: unknown) => void;
}

export interface ResyncRequest {
  protocol_version: "1";
  type: "browser.resync";
  reason: ResyncReason;
  requested_cursor: string;
}

export interface TransportConnection {
  close: () => void;
  requestResync?: (message: ResyncRequest) => void;
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
  onListenerError?: (error: unknown, event: ConnectionEvent) => void;
}

export type ResyncRequestResult =
  | { ok: true }
  | { ok: false; reason: "invalid-cursor" | "not-connected" | "unsupported" | "transport-error" | "stale-source"; error?: string };

export type ConnectionEvent =
  | { type: "attempt"; id: number }
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

function immutableError(error: Error): Error {
  const copy = new Error(error.message);
  copy.name = error.name;
  if (error.stack !== undefined) copy.stack = error.stack;
  return Object.freeze(copy);
}

function immutableEvent(event: ConnectionEvent): ConnectionEvent {
  if (event.type === "protocol-error") {
    return Object.freeze({ ...event, error: immutableError(event.error) });
  }
  return Object.freeze({ ...event });
}

function isCanonicalCursor(value: unknown): value is string {
  if (typeof value !== "string" || !/^(0|[1-9][0-9]*)$/.test(value)) return false;
  try {
    return BigInt(value) <= MAX_U64;
  } catch {
    return false;
  }
}

export class ConnectionClient {
  private readonly transportFactory: TransportFactory;
  private readonly timer: ConnectionTimer;
  private readonly now: () => number;
  private readonly retryDelayMs: (attempt: number) => number;
  private readonly staleAfterMs: number;
  private readonly autoReconnect: boolean;
  private readonly onListenerError: (error: unknown, event: ConnectionEvent) => void;
  private readonly listeners = new Set<(event: ConnectionEvent) => void>();
  private connection: TransportConnection | null = null;
  private connectionAttemptId: number | null = null;
  private reconnectTimer: unknown = null;
  private staleTimer: unknown = null;
  private generation = 0;
  private nextAttemptId = 0;
  private activeAttemptId: number | null = null;
  private destroyed = false;
  private userDisconnected = true;
  private status: ConnectionStatus = Object.freeze({
    state: "disconnected",
    attempt: 0,
    error: null,
    lastMessageAt: null,
    requestResyncAvailable: false,
  });

  public constructor(options: ConnectionClientOptions) {
    this.transportFactory = options.transportFactory;
    this.timer = options.timer ?? defaultTimer;
    this.now = options.now ?? Date.now;
    this.retryDelayMs = options.retryDelayMs ?? ((attempt) => exponentialBackoff(attempt));
    this.staleAfterMs = options.staleAfterMs ?? 15_000;
    this.autoReconnect = options.autoReconnect ?? true;
    this.onListenerError = options.onListenerError ?? (() => undefined);
  }

  public getSnapshot(): ConnectionStatus {
    return this.status;
  }

  public subscribe(listener: (event: ConnectionEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  public connect(): void {
    const stateCanOpen = this.status.state === "error" || this.status.state === "disconnected";
    if (this.destroyed || !this.userDisconnected && !stateCanOpen) return;
    this.userDisconnected = false;
    this.clearReconnectTimer();
    this.open(++this.generation);
  }

  public disconnect(): void {
    if (this.destroyed) return;
    this.userDisconnected = true;
    this.generation += 1;
    this.activeAttemptId = null;
    this.clearReconnectTimer();
    this.clearStaleTimer();
    this.closeCurrentConnection();
    this.updateStatus({ state: "disconnected", attempt: 0, error: null, lastMessageAt: null, requestResyncAvailable: false });
  }

  public destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.userDisconnected = true;
    this.generation += 1;
    this.activeAttemptId = null;
    this.clearReconnectTimer();
    this.clearStaleTimer();
    this.closeCurrentConnection();
    this.updateStatus({ state: "disconnected", attempt: 0, error: null, lastMessageAt: null, requestResyncAvailable: false });
    this.listeners.clear();
  }

  public requestResync(cursor: string, reason: ResyncReason = "cursor_gap"): ResyncRequestResult {
    if (!isCanonicalCursor(cursor)) return { ok: false, reason: "invalid-cursor" };
    if (this.connection === null || this.activeAttemptId === null) return { ok: false, reason: "not-connected" };
    if (this.connection.requestResync === undefined) return { ok: false, reason: "unsupported" };
    try {
      this.connection.requestResync({ protocol_version: "1", type: "browser.resync", reason, requested_cursor: cursor });
      return { ok: true };
    } catch (error) {
      return { ok: false, reason: "transport-error", error: errorMessage(error) };
    }
  }

  private open(generation: number): void {
    const attemptId = ++this.nextAttemptId;
    this.activeAttemptId = attemptId;
    this.emit({ type: "attempt", id: attemptId });
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.updateStatus({
      state: this.status.attempt > 0 ? "reconnecting" : "connecting",
      requestResyncAvailable: false,
    });
    if (!this.isCurrentAttempt(generation, attemptId)) return;

    let candidate: TransportConnection;
    try {
      candidate = this.transportFactory({
        onOpen: () => this.opened(generation, attemptId),
        onMessage: (value) => this.receive(value, generation, attemptId),
        onError: (error) => this.failed(error, generation, attemptId),
        onClose: (reason) => this.closed(reason, generation, attemptId),
      });
    } catch (error) {
      if (this.isCurrentAttempt(generation, attemptId)) this.failed(error, generation, attemptId);
      return;
    }

    if (!this.isCurrentAttempt(generation, attemptId)) {
      this.safeClose(candidate);
      return;
    }
    this.connection = candidate;
    this.connectionAttemptId = attemptId;
    this.updateStatus({ requestResyncAvailable: candidate.requestResync !== undefined });
  }

  private opened(generation: number, attemptId: number): void {
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.updateStatus({ state: "live", attempt: 0, error: null, lastMessageAt: this.now() });
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.scheduleStale(generation, attemptId);
  }

  private receive(value: unknown, generation: number, attemptId: number): void {
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    let envelope: ParsedMessage;
    try {
      envelope = parseProtocolMessage(value);
    } catch (error) {
      const normalized = error instanceof Error ? error : new Error(errorMessage(error));
      this.emit({ type: "protocol-error", error: normalized });
      this.failed(normalized, generation, attemptId);
      return;
    }
    this.updateStatus({ state: "live", error: null, lastMessageAt: this.now() });
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.scheduleStale(generation, attemptId);
    this.emit({ type: "message", envelope });
  }

  private failed(error: unknown, generation: number, attemptId: number): void {
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.activeAttemptId = null;
    this.clearStaleTimer();
    this.closeCurrentConnection(attemptId);
    this.updateStatus({ state: "error", error: errorMessage(error), requestResyncAvailable: false });
    if (this.canContinueAfterBoundary(generation) && this.autoReconnect) this.scheduleReconnect(generation);
  }

  private closed(reason: unknown, generation: number, attemptId: number): void {
    if (!this.isCurrentAttempt(generation, attemptId)) return;
    this.activeAttemptId = null;
    this.connection = null;
    this.connectionAttemptId = null;
    this.clearStaleTimer();
    this.updateStatus({ state: "error", error: reason ? errorMessage(reason) : "Source closed the stream", requestResyncAvailable: false });
    if (this.canContinueAfterBoundary(generation) && this.autoReconnect) this.scheduleReconnect(generation);
  }

  private scheduleReconnect(generation: number): void {
    this.clearReconnectTimer();
    const attempt = this.status.attempt + 1;
    this.updateStatus({ state: "reconnecting", attempt });
    if (!this.canContinueAfterBoundary(generation)) return;
    this.reconnectTimer = this.timer.set(() => {
      this.reconnectTimer = null;
      if (generation === this.generation && !this.userDisconnected) this.open(generation);
    }, this.retryDelayMs(attempt));
  }

  private scheduleStale(generation: number, attemptId: number): void {
    this.clearStaleTimer();
    this.staleTimer = this.timer.set(() => {
      if (this.isCurrentAttempt(generation, attemptId) && this.status.state === "live") {
        this.updateStatus({ state: "stale" });
      }
    }, this.staleAfterMs);
  }

  private isCurrentAttempt(generation: number, attemptId: number): boolean {
    return !this.destroyed && !this.userDisconnected && generation === this.generation && attemptId === this.activeAttemptId;
  }

  private canContinueAfterBoundary(generation: number): boolean {
    return !this.destroyed && !this.userDisconnected && generation === this.generation && this.activeAttemptId === null;
  }

  private updateStatus(patch: Partial<ConnectionStatus>): void {
    this.status = Object.freeze({ ...this.status, ...patch });
    this.emit({ type: "status", status: this.status });
  }

  private emit(event: ConnectionEvent): void {
    const published = immutableEvent(event);
    for (const listener of [...this.listeners]) {
      try {
        listener(published);
      } catch (error) {
        try {
          this.onListenerError(error, published);
        } catch {
          // Observer diagnostics must not become a transport lifecycle failure.
        }
      }
    }
  }

  private closeCurrentConnection(attemptId?: number): void {
    if (attemptId !== undefined && this.connectionAttemptId !== attemptId) return;
    const connection = this.connection;
    this.connection = null;
    this.connectionAttemptId = null;
    if (connection) this.safeClose(connection);
  }

  private safeClose(connection: TransportConnection): void {
    try {
      connection.close();
    } catch {
      // Teardown must not turn an obsolete transport's close failure into a live error.
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
