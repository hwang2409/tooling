# tix

`tix` is a small personal Linear-style ticket tracker. It stores plain JSON files, one ticket per file, so the data is easy to inspect and version.

## Install

```sh
cargo install --path .
```

The default data directory is `~/.tix`. Set `TIX_DATA_DIR` to use another directory (tests and scripts should always use a temporary directory).

## Commands

```text
tix init <PREFIX> <NAME>
tix projects
tix add <PREFIX> <TITLE> [-d DESCRIPTION] [-p P1|P2|P3] [-l LABEL ...]
tix ls [PREFIX] [--state STATE] [--priority P1|P2|P3] [--label LABEL] [--all]
tix show <ID>
tix start|done|cancel|reopen <ID>
tix edit <ID> [--title TITLE] [--desc DESCRIPTION] [--priority P1|P2|P3]
tix edit <ID> [--add-label LABEL] [--rm-label LABEL]
tix note <ID> <TEXT>
tix next <PREFIX>
```

Pass global `--json` to read commands (`projects`, `ls`, `show`, and `next`) for stable machine-readable output. `ls` hides done and canceled tickets unless `--all` or an explicit `--state` filter is supplied.

Storage uses `projects.json` and `tickets/<PREFIX>/<PREFIX>-<n>.json` below the data directory. Writes are atomic, and concurrent `add` commands reserve IDs with exclusive file creation.
