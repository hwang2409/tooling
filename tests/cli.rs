use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use std::process::Stdio;
use std::sync::Arc;
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use tempfile::TempDir;

fn tix(data_dir: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_tix"));
    command.env("TIX_DATA_DIR", data_dir);
    command
}

fn setup() -> TempDir {
    let dir = tempfile::tempdir().unwrap();
    tix(dir.path())
        .args(["init", "ABC", "Example"])
        .assert()
        .success();
    dir
}

fn add(dir: &Path, title: &str) -> String {
    let output = tix(dir).args(["add", "ABC", title]).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_string()
}

fn stdout(dir: &Path, args: &[&str]) -> String {
    let output = tix(dir).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn round_trip_and_json_shapes() {
    let dir = setup();
    let id = tix(dir.path())
        .args([
            "add",
            "ABC",
            "Ship it",
            "--desc",
            "markdown",
            "--priority",
            "P1",
            "--label",
            "work",
            "--label",
            "urgent",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8(id).unwrap().trim(), "ABC-1");

    tix(dir.path())
        .args(["note", "ABC-1", "handoff"])
        .assert()
        .success();
    let show = tix(dir.path())
        .args(["--json", "show", "ABC-1"])
        .output()
        .unwrap();
    let ticket: Value = serde_json::from_slice(&show.stdout).unwrap();
    assert_eq!(ticket["id"], "ABC-1");
    assert_eq!(ticket["notes"][0]["text"], "handoff");

    let listed = tix(dir.path()).args(["ls", "--json"]).output().unwrap();
    let tickets: Value = serde_json::from_slice(&listed.stdout).unwrap();
    assert_eq!(tickets[0]["priority"], "P1");
    assert_eq!(tickets[0]["labels"][1], "urgent");
}

#[test]
fn transitions_accept_legal_moves_and_reject_illegal_moves() {
    let dir = setup();
    add(dir.path(), "done from todo");
    add(dir.path(), "cancel from todo");
    add(dir.path(), "done from progress");
    add(dir.path(), "cancel from progress");
    add(dir.path(), "stays todo");

    tix(dir.path()).args(["done", "ABC-1"]).assert().success();
    tix(dir.path()).args(["cancel", "ABC-2"]).assert().success();
    tix(dir.path()).args(["start", "ABC-3"]).assert().success();
    tix(dir.path()).args(["done", "ABC-3"]).assert().success();
    tix(dir.path()).args(["start", "ABC-4"]).assert().success();
    tix(dir.path()).args(["cancel", "ABC-4"]).assert().success();
    tix(dir.path()).args(["reopen", "ABC-1"]).assert().success();
    tix(dir.path()).args(["reopen", "ABC-2"]).assert().success();
    tix(dir.path()).args(["cancel", "ABC-1"]).assert().success();
    tix(dir.path()).args(["start", "ABC-2"]).assert().success();
    tix(dir.path()).args(["done", "ABC-2"]).assert().success();

    for (id, command) in [
        ("ABC-1", "start"),
        ("ABC-1", "done"),
        ("ABC-2", "cancel"),
        ("ABC-3", "start"),
        ("ABC-4", "start"),
        ("ABC-4", "done"),
        ("ABC-5", "reopen"),
    ] {
        let illegal = tix(dir.path()).args([command, id]).output().unwrap();
        assert!(!illegal.status.success());
        let error = String::from_utf8_lossy(&illegal.stderr);
        assert!(error.contains(id) && error.contains("cannot move"));
    }
}

#[test]
fn filters_hide_terminal_by_default() {
    let dir = setup();
    tix(dir.path())
        .args(["add", "ABC", "open", "--priority", "P1", "--label", "focus"])
        .assert()
        .success();
    add(dir.path(), "closed");
    tix(dir.path()).args(["done", "ABC-2"]).assert().success();
    let default_list = stdout(dir.path(), &["ls"]);
    assert!(default_list.contains("ABC-1"));
    assert!(default_list.contains("open"));
    assert!(!default_list.contains("closed"));
    let done_list = stdout(dir.path(), &["ls", "--all", "--state", "done"]);
    assert!(done_list.contains("ABC-2"));
    let filtered = stdout(dir.path(), &["ls", "--priority", "P1", "--label", "focus"]);
    assert!(filtered.contains("ABC-1"));
    assert!(!filtered.contains("ABC-2"));
}

#[test]
fn concurrent_adds_allocate_distinct_sequential_ids() {
    let dir = setup();
    let path = Arc::new(dir.path().to_path_buf());
    let mut handles = Vec::new();
    for index in 0..16 {
        let path = Arc::clone(&path);
        handles.push(thread::spawn(move || {
            let output = StdCommand::new(env!("CARGO_BIN_EXE_tix"))
                .env("TIX_DATA_DIR", &*path)
                .args(["add", "ABC", &format!("ticket-{index}")])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        }));
    }
    let mut ids: Vec<String> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    ids.sort_by_key(|id| id.split('-').nth(1).unwrap().parse::<u64>().unwrap());
    assert_eq!(
        ids,
        (1..=16).map(|n| format!("ABC-{n}")).collect::<Vec<_>>()
    );
}

#[test]
fn writes_leave_only_parseable_final_ticket_files() {
    let dir = setup();
    add(dir.path(), "ticket");
    tix(dir.path())
        .args(["edit", "ABC-1", "--desc", "updated"])
        .assert()
        .success();
    let ticket_path = dir.path().join("tickets/ABC/ABC-1.json");
    let _: Value = serde_json::from_slice(&fs::read(ticket_path).unwrap()).unwrap();
    let files: Vec<_> = fs::read_dir(dir.path().join("tickets/ABC"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(files.len(), 1);
}

#[test]
fn concurrent_notes_preserve_every_append() {
    let dir = setup();
    add(dir.path(), "notes");
    let marker = dir.path().join("notes.start");
    let mut children = Vec::new();
    for index in 0..32 {
        children.push(
            StdCommand::new(env!("CARGO_BIN_EXE_tix"))
                .env("TIX_DATA_DIR", dir.path())
                .env("TIX_TEST_START_MARKER", &marker)
                .env("TIX_TEST_HOLD_AFTER_READ", "1")
                .args(["note", "ABC-1", &format!("note-{index}")])
                .spawn()
                .unwrap(),
        );
    }
    fs::write(&marker, b"go").unwrap();
    for child in children {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let output = tix(dir.path())
        .args(["show", "ABC-1", "--json"])
        .output()
        .unwrap();
    let ticket: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(ticket["notes"].as_array().unwrap().len(), 32);
}

#[test]
fn concurrent_init_preserves_every_project() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("init.start");
    let mut children = Vec::new();
    for (index, prefix) in ('A'..='P').enumerate() {
        let prefix = prefix.to_string();
        children.push(
            StdCommand::new(env!("CARGO_BIN_EXE_tix"))
                .env("TIX_DATA_DIR", dir.path())
                .env("TIX_TEST_START_MARKER", &marker)
                .env("TIX_TEST_HOLD_AFTER_READ", "1")
                .stdout(Stdio::null())
                .args(["init", &prefix, &format!("Project {index}")])
                .spawn()
                .unwrap(),
        );
    }
    fs::write(&marker, b"go").unwrap();
    for child in children {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let output = tix(dir.path())
        .args(["projects", "--json"])
        .output()
        .unwrap();
    let projects: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(projects.as_array().unwrap().len(), 16);
}

#[test]
fn interrupted_add_never_leaves_partial_final_ticket() {
    let dir = setup();
    let mut child = StdCommand::new(env!("CARGO_BIN_EXE_tix"))
        .env("TIX_DATA_DIR", dir.path())
        .env("TIX_TEST_PAUSE_AFTER_TEMP", "1")
        .args(["add", "ABC", "interrupted"])
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let ticket_dir = dir.path().join("tickets/ABC");
    let mut temp_seen = false;
    for _ in 0..100 {
        temp_seen = fs::read_dir(&ticket_dir).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".tmp-")
        });
        if temp_seen {
            break;
        }
        sleep(Duration::from_millis(10));
    }
    assert!(temp_seen, "add did not reach its fsynced temp file");
    child.kill().unwrap();
    child.wait().unwrap();
    assert!(!ticket_dir.join("ABC-1.json").exists());
    add(dir.path(), "after interruption");
    let _: Value =
        serde_json::from_slice(&fs::read(ticket_dir.join("ABC-1.json")).unwrap()).unwrap();
}

#[test]
fn add_and_edit_reject_blank_or_multiline_titles() {
    let dir = setup();
    for title in ["", "  ", "line\nbreak", "line\rbreak"] {
        let output = tix(dir.path())
            .args(["add", "ABC", title])
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "title should be rejected: {title:?}"
        );
    }
    add(dir.path(), "valid");
    for title in ["", "\t", "bad\nline", "bad\rline"] {
        let output = tix(dir.path())
            .args(["edit", "ABC-1", "--title", title])
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "title should be rejected: {title:?}"
        );
    }
}

#[test]
fn relative_data_dir_supports_init_and_add() {
    let cwd = tempfile::tempdir().unwrap();
    let mut init = Command::new(env!("CARGO_BIN_EXE_tix"));
    init.current_dir(cwd.path())
        .env("TIX_DATA_DIR", "data")
        .args(["init", "REL", "Relative"])
        .assert()
        .success();

    let mut add = Command::new(env!("CARGO_BIN_EXE_tix"));
    add.current_dir(cwd.path())
        .env("TIX_DATA_DIR", "data")
        .args(["add", "REL", "works"])
        .assert()
        .success();

    let ticket = cwd.path().join("data/tickets/REL/REL-1.json");
    let ticket: Value = serde_json::from_slice(&fs::read(ticket).unwrap()).unwrap();
    assert_eq!(ticket["id"], "REL-1");
}
