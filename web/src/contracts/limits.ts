/**
 * Wire ceiling constants mirrored from the backend contract at
 * `src/mitm_inspector/api/limits.py` + `src/mitm_inspector/protocol.py`.
 *
 * Any drift is caught by `limits.contract.test.ts`, which reads both Python
 * modules and asserts numeric equality with the values below. Update this
 * file and the backend in the same commit — never one without the other.
 */

// From src/mitm_inspector/protocol.py
export const MAX_METADATA_HEADER_BYTES = 64 * 1024;
export const MAX_INGEST_LINE_BYTES = 8 * 1024 * 1024;

// From src/mitm_inspector/api/limits.py
export const INGEST_FIXED_ENVELOPE_BYTES = 1024;
export const INGEST_ENVELOPE_ALLOWANCE_BYTES = 2 * MAX_METADATA_HEADER_BYTES + INGEST_FIXED_ENVELOPE_BYTES;
export const MAX_INGEST_BODY_PREFIX_BYTES = Math.floor(
  ((MAX_INGEST_LINE_BYTES - INGEST_ENVELOPE_ALLOWANCE_BYTES) * 3) / 8,
);
