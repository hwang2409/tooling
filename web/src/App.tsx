import { useConnection } from "./features/connection/useConnection";
import type { TransportFactory } from "./features/connection/connectionClient";
import { webSocketTransportFactory } from "./features/connection/wsTransport";
import type { ConnectionViewModel } from "./features/connection/useConnection";
import { FlowWorkspace } from "./features/flows/FlowWorkspace";
import "./styles/shell.css";

export { formatBytes } from "./format";

export interface AppProps {
  transportFactory?: TransportFactory;
}

export function App({ transportFactory }: AppProps = {}) {
  return <Workbench view={useConnection(transportFactory ?? webSocketTransportFactory())} />;
}

export function Workbench({ view }: { view: ConnectionViewModel }) {
  const { browser, followLive, pauseLive } = view;
  return (
    <main className="app-shell">
      <FlowWorkspace browser={browser} followLive={followLive} pauseLive={pauseLive} />
    </main>
  );
}
