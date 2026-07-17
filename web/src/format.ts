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

/** Terse byte-size for dense grid cells: 128B, 4.2K, 1.1M, 2.4G. */
export function formatBytesCompact(value: string | undefined | null): string {
  if (value === undefined || value === null || value === "") return "—";
  let bytes: bigint;
  try {
    bytes = BigInt(value);
  } catch {
    return "—";
  }
  if (bytes < 0n) return "—";
  const kib = 1024n;
  const mib = kib * 1024n;
  const gib = mib * 1024n;
  if (bytes < kib) return `${bytes}B`;
  if (bytes < mib) return `${trimDecimal(Number(bytes) / 1024)}K`;
  if (bytes < gib) return `${trimDecimal(Number(bytes) / 1024 / 1024)}M`;
  return `${trimDecimal(Number(bytes) / 1024 / 1024 / 1024)}G`;
}

function trimDecimal(value: number): string {
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(1);
}

/** Compact duration for grid rows: 12ms, 940ms, 1.4s, 62s, 3m. */
export function formatDurationMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return "—";
  if (value < 1000) return `${Math.round(value)}ms`;
  const seconds = value / 1000;
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : seconds.toFixed(0)}s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${minutes < 10 ? minutes.toFixed(1) : minutes.toFixed(0)}m`;
  return `${(minutes / 60).toFixed(1)}h`;
}
