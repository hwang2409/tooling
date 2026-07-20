import { useEffect, useState } from "react";

import { useConnection } from "./features/connection/useConnection";
import type { TransportFactory } from "./features/connection/connectionClient";
import { webSocketTransportFactory } from "./features/connection/wsTransport";
import { PacketList } from "./features/flows/PacketList";
import "./styles/shell.css";

export interface AppProps {
  transportFactory?: TransportFactory;
}

export function App({ transportFactory }: AppProps = {}) {
  const view = useConnection(transportFactory ?? webSocketTransportFactory());
  const [searchActive, setSearchActive] = useState(false);
  // Freeze the displayed flow list while a search is being read so live
  // traffic cannot reshuffle the results out from under the user.
  useEffect(() => {
    if (searchActive) view.pauseLive();
    else view.resumeLive();
  }, [searchActive]);
  return (
    <main className="app-shell">
      <PacketList browser={view.browser} onSearchActiveChange={setSearchActive} />
    </main>
  );
}
