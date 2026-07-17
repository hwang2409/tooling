/* eslint-disable no-unused-vars */

import { useEffect, useReducer, useRef, useState } from "react";

import { browserReducer, initialBrowserState } from "../../state/browserState";
import type { BrowserState } from "../../state/browserState";
import { ConnectionClient, unavailableTransportFactory } from "./connectionClient";
import type { ConnectionStatus, TransportFactory } from "./connectionClient";

export interface ConnectionViewModel {
  browser: BrowserState;
  status: ConnectionStatus;
  connect: () => void;
  disconnect: () => void;
  retry: () => void;
  requestResync: (...args: [string]) => void;
}

export function useConnection(factory: TransportFactory = unavailableTransportFactory): ConnectionViewModel {
  const clientRef = useRef<ConnectionClient | null>(null);
  const [browser, dispatch] = useReducer(browserReducer, initialBrowserState);
  const [, refresh] = useState(0);

  if (clientRef.current === null) {
    // Kept in a ref so StrictMode's effect probe cannot create duplicate transports.
    clientRef.current = new ConnectionClient({ transportFactory: factory, autoReconnect: false });
  }
  const client = clientRef.current;

  useEffect(() => {
    const unsubscribe = client.subscribe((event) => {
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
    browser,
    status: client.getSnapshot(),
    connect: () => client.connect(),
    disconnect: () => client.disconnect(),
    retry: () => client.connect(),
    requestResync: (...args) => client.requestResync(args[0]),
  };
}
