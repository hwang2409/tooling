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
  return (
    <main className="app-shell">
      <PacketList browser={view.browser} />
    </main>
  );
}
