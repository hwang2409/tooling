/* eslint-disable no-unused-vars */

import { useEffect, useReducer, useRef, useState } from "react";

import { browserViewReducer, initialBrowserViewState } from "../../state/browserState";
import type { BrowserState } from "../../state/browserState";
import { ConnectionClient, unavailableTransportFactory } from "./connectionClient";
import type { ConnectionStatus, ResyncRequestResult, TransportFactory } from "./connectionClient";

export interface ConnectionViewModel {
  browser: BrowserState;
  status: ConnectionStatus;
  latestBrowser: BrowserState;
  followLive: boolean;
  connect: () => void;
  disconnect: () => void;
  retry: () => void;
  pauseLive: () => void;
  resumeLive: () => void;
  requestResync: (...args: [string]) => ResyncRequestResult;
}

export function useConnection(factory: TransportFactory = unavailableTransportFactory): ConnectionViewModel {
  const clientRef = useRef<ConnectionClient | null>(null);
  const [browserView, dispatch] = useReducer(browserViewReducer, initialBrowserViewState);
  const [, refresh] = useState(0);

  if (clientRef.current === null) {
    // Kept in a ref so StrictMode's effect probe cannot create duplicate transports.
    clientRef.current = new ConnectionClient({ transportFactory: factory, autoReconnect: true });
  }
  const client = clientRef.current;

  useEffect(() => {
    const unsubscribe = client.subscribe((event) => {
      if (event.type === "attempt") dispatch({ type: "reset" });
      if (event.type === "message") dispatch({ type: "protocol", envelope: event.envelope });
      refresh((value) => value + 1);
    });
    client.connect();
    return () => {
      unsubscribe();
      client.disconnect();
    };
  }, [client]);

  return {
    browser: browserView.displayed,
    latestBrowser: browserView.latest,
    status: client.getSnapshot(),
    followLive: browserView.followLive,
    connect: () => client.connect(),
    disconnect: () => client.disconnect(),
    retry: () => client.connect(),
    pauseLive: () => dispatch({ type: "pause" }),
    resumeLive: () => dispatch({ type: "resume" }),
    requestResync: (...args) => client.requestResync(args[0]),
  };
}
