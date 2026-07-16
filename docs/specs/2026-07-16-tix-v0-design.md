# tix v0 — Design

Personal Linear-clone ticket tracker CLI. Motivation (Henry 2026-07-16): Linear is work-only; personal projects (pufferclone, wiki, misc) allocate ticket IDs by hand across vault notes ("next free ID" watchouts). tix owns ticket identity, state, and listing for all personal projects, standalone — the wiki vault keeps prose/notes, tix keeps tickets. CLI-first, no web UI, no daemon.

Out of scope for v0: sync/remote, multi-user, Linear import/export, comments-as-threads, attachments, due dates, TUI.

## Shape

- Rust, single binary `tix` (clap, derive API). Same toolchain as pufferclone.
- Storage: plain JSON files under `TIX_DATA_DIR` (default `~/.tix`), git-friendly, no database:
  - `projects.json` — prefix registry: `{"PUF": {"name": "pufferclone", "created": "..."}, ...}`
  - `tickets/<PREFIX>/<PREFIX>-<n>.json` — one file per ticket.
- Ticket schema: id, title, description (markdown string), state (`todo|in-progress|done|canceled`), priority (`P1|P2|P3`, optional), labels (string array, optional), notes (append-only array of {ts, text} — agent handoff breadcrumbs), created/updated timestamps.
- Atomic writes everywhere (tmp+rename). ID allocation must be concurrency-safe WITHOUT a lock file: allocate by `O_CREAT|O_EXCL` creation of the ticket file at the next free number, retry upward on EEXIST — two agents allocating simultaneously must never collide or skip-lock.

## Commands

- `tix init <prefix> <name>` — register project (rejects duplicate prefix; prefix = uppercase letters).
- `tix projects` — list prefixes, names, ticket counts by state.
- `tix add <prefix> <title>` — create ticket; `-d/--desc`, `-p/--priority`, `-l/--label` (repeatable). Prints new ID.
- `tix ls [prefix]` — table: id, state, priority, title. Filters: `--state <s>` (repeatable), `--priority`, `--label`, `--all` (include done/canceled; default hides them).
- `tix show <id>` — full ticket incl. notes.
- `tix start <id>` / `tix done <id>` / `tix cancel <id>` / `tix reopen <id>` — state transitions (validate legal moves: done/canceled only from todo/in-progress; reopen only from done/canceled).
- `tix edit <id>` — `--title`, `--desc`, `--priority`, `--add-label`, `--rm-label`.
- `tix note <id> <text>` — append note.
- `tix next <prefix>` — print next free ID without allocating (informational; replaces hot.md "next free ticket ID" watchouts).
- Global `--json` on every read command; stable machine shape. Human tables default.
- Errors: unknown id/prefix → non-zero exit, message names the thing not found and the data dir searched.

## Tests

- CLI integration via compiled binary (assert_cmd), `TIX_DATA_DIR` pointed at tempdir per test — never touches `~/.tix`.
- Coverage: init/add/ls/show round-trip, every legal + illegal state transition, filters, `--json` shape stability, concurrent allocation (N threads/processes adding to one prefix → N distinct sequential IDs, no gaps beyond races' natural ordering, no collisions), atomicity (kill mid-write leaves no corrupt ticket — write to temp then rename means partial files never bear final names).
- `cargo clippy --all-targets` clean, `cargo fmt` applied.

## Delivery

Single ticket TIX-1 builds all of v0. Future (unticketed): `tix migrate` importer for vault todo.md lines; wiki CLI / orchestrator integration so agents allocate IDs through tix.
