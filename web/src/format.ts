export function formatBytes(value: string | undefined): string {
  if (!value) return "—";
  let bytes: bigint;
  try {
    bytes = BigInt(value);
  } catch {
    return "—";
  }
  const mib = 1024n * 1024n;
  const kib = 1024n;
  if (bytes >= mib) return `${bytes / mib} MiB`;
  if (bytes >= kib) return `${bytes / kib} KiB`;
  return `${bytes} B`;
}
