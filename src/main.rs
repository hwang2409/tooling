use clap::{Args, Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::env;
use std::fmt::{self, Display};
use std::fs::{self, DirEntry, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

type AppResult<T> = Result<T, AppError>;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct AppError(String);

impl AppError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<io::Error> for AppError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for AppError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum State {
    Todo,
    #[value(name = "in-progress")]
    InProgress,
    Done,
    Canceled,
}

impl Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Todo => "todo",
            Self::InProgress => "in-progress",
            Self::Done => "done",
            Self::Canceled => "canceled",
        };
        f.write_str(value)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
enum Priority {
    #[value(name = "P1")]
    P1,
    #[value(name = "P2")]
    P2,
    #[value(name = "P3")]
    P3,
}

impl Display for Priority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::P1 => "P1",
            Self::P2 => "P2",
            Self::P3 => "P3",
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct Project {
    name: String,
    created: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Note {
    ts: String,
    text: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Ticket {
    id: String,
    title: String,
    description: String,
    state: State,
    priority: Option<Priority>,
    labels: Vec<String>,
    notes: Vec<Note>,
    created: String,
    updated: String,
}

#[derive(Debug, Serialize)]
struct ProjectJson {
    prefix: String,
    name: String,
    counts: Counts,
}

#[derive(Clone, Debug, Default, Serialize)]
struct Counts {
    todo: usize,
    #[serde(rename = "in-progress")]
    in_progress: usize,
    done: usize,
    canceled: usize,
}

impl Counts {
    fn add(&mut self, state: State) {
        match state {
            State::Todo => self.todo += 1,
            State::InProgress => self.in_progress += 1,
            State::Done => self.done += 1,
            State::Canceled => self.canceled += 1,
        }
    }
}

#[derive(Parser)]
#[command(name = "tix", version, about = "A personal ticket tracker")]
struct Cli {
    #[arg(long, global = true, help = "Emit stable JSON on read commands")]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Init { prefix: String, name: String },
    Projects,
    Add(AddArgs),
    Ls(ListArgs),
    Show { id: String },
    Start { id: String },
    Done { id: String },
    Cancel { id: String },
    Reopen { id: String },
    Edit(EditArgs),
    Note { id: String, text: String },
    Next { prefix: String },
}

#[derive(Args)]
struct AddArgs {
    prefix: String,
    title: String,
    #[arg(short = 'd', long = "desc", default_value = "")]
    description: String,
    #[arg(short, long)]
    priority: Option<Priority>,
    #[arg(short, long = "label")]
    labels: Vec<String>,
}

#[derive(Args)]
struct ListArgs {
    prefix: Option<String>,
    #[arg(long = "state")]
    states: Vec<State>,
    #[arg(long)]
    priority: Option<Priority>,
    #[arg(long)]
    label: Option<String>,
    #[arg(long, help = "Include done and canceled tickets")]
    all: bool,
}

#[derive(Args)]
struct EditArgs {
    id: String,
    #[arg(long)]
    title: Option<String>,
    #[arg(long)]
    desc: Option<String>,
    #[arg(long)]
    priority: Option<Priority>,
    #[arg(long = "add-label")]
    add_labels: Vec<String>,
    #[arg(long = "rm-label")]
    remove_labels: Vec<String>,
}

struct Storage {
    root: PathBuf,
}

struct FileLock<'a> {
    file: &'a File,
}

impl<'a> FileLock<'a> {
    fn exclusive(file: &'a File) -> AppResult<Self> {
        // SAFETY: file is an open descriptor and flock only changes its advisory lock state.
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if result == -1 {
            return Err(io::Error::last_os_error().into());
        }
        Ok(Self { file })
    }
}

impl Drop for FileLock<'_> {
    fn drop(&mut self) {
        // SAFETY: file is still open while the guard is being dropped.
        unsafe {
            libc::flock(self.file.as_raw_fd(), libc::LOCK_UN);
        }
    }
}

fn same_file(file: &File, path: &Path) -> AppResult<bool> {
    let locked = file.metadata()?;
    let current = fs::metadata(path)?;
    Ok(locked.dev() == current.dev() && locked.ino() == current.ino())
}

fn wait_for_test_marker() {
    let Some(marker) = env::var_os("TIX_TEST_START_MARKER") else {
        return;
    };
    let marker = PathBuf::from(marker);
    while !marker.exists() {
        thread::sleep(Duration::from_millis(5));
    }
}

fn hold_for_test_race() {
    if env::var_os("TIX_TEST_HOLD_AFTER_READ").is_some() {
        thread::sleep(Duration::from_millis(100));
    }
}

fn decode_projects(bytes: &[u8]) -> AppResult<BTreeMap<String, Project>> {
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return Ok(BTreeMap::new());
    }
    Ok(serde_json::from_slice(bytes)?)
}

fn read_projects(file: &File) -> AppResult<BTreeMap<String, Project>> {
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(0))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    decode_projects(&bytes)
}

impl Storage {
    fn from_env() -> AppResult<Self> {
        let root = match env::var_os("TIX_DATA_DIR") {
            Some(path) => PathBuf::from(path),
            None => {
                let home = env::var_os("HOME")
                    .ok_or_else(|| AppError::new("HOME is not set; set TIX_DATA_DIR"))?;
                PathBuf::from(home).join(".tix")
            }
        };
        let root = if root.is_relative() {
            env::current_dir()?.join(root)
        } else {
            root
        };
        Ok(Self { root })
    }

    fn projects_path(&self) -> PathBuf {
        self.root.join("projects.json")
    }

    fn project_dir(&self, prefix: &str) -> PathBuf {
        self.root.join("tickets").join(prefix)
    }

    fn ticket_path(&self, id: &str) -> AppResult<PathBuf> {
        let (prefix, number) = parse_id(id)?;
        Ok(self
            .project_dir(prefix)
            .join(format!("{prefix}-{number}.json")))
    }

    fn projects(&self) -> AppResult<BTreeMap<String, Project>> {
        let path = self.projects_path();
        if !path.exists() {
            return Ok(BTreeMap::new());
        }
        decode_projects(&fs::read(path)?)
    }

    fn require_project<'a>(
        &self,
        projects: &'a BTreeMap<String, Project>,
        prefix: &str,
    ) -> AppResult<&'a Project> {
        projects.get(prefix).ok_or_else(|| {
            AppError::new(format!(
                "project '{prefix}' not found in data dir {}",
                self.root.display()
            ))
        })
    }

    fn update_projects<F>(&self, update: F) -> AppResult<()>
    where
        F: Fn(&mut BTreeMap<String, Project>) -> AppResult<()>,
    {
        create_dir_all_durable(&self.root)?;
        wait_for_test_marker();
        loop {
            let path = self.projects_path();
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&path)?;
            let _lock = FileLock::exclusive(&file)?;
            if !same_file(&file, &path)? {
                continue;
            }
            let mut projects = read_projects(&file)?;
            hold_for_test_race();
            update(&mut projects)?;
            atomic_write(&path, &serde_json::to_vec_pretty(&projects)?)?;
            return Ok(());
        }
    }

    fn read_ticket(&self, id: &str) -> AppResult<Ticket> {
        let path = self.ticket_path(id)?;
        if !path.exists() {
            return Err(self.unknown_ticket(id));
        }
        serde_json::from_slice(&fs::read(path)?)
            .map_err(|error| AppError::new(format!("could not read ticket '{id}': {error}")))
    }

    fn unknown_ticket(&self, id: &str) -> AppError {
        AppError::new(format!(
            "ticket '{id}' not found in data dir {}",
            self.root.display()
        ))
    }

    fn mutate_ticket<F>(&self, id: &str, update: F) -> AppResult<()>
    where
        F: Fn(&mut Ticket) -> AppResult<()>,
    {
        let path = self.ticket_path(id)?;
        wait_for_test_marker();
        loop {
            let file = match OpenOptions::new().read(true).write(true).open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::NotFound => {
                    return Err(self.unknown_ticket(id))
                }
                Err(error) => return Err(error.into()),
            };
            let _lock = FileLock::exclusive(&file)?;
            if !same_file(&file, &path)? {
                continue;
            }
            let mut reader = file.try_clone()?;
            reader.seek(SeekFrom::Start(0))?;
            let mut bytes = Vec::new();
            reader.read_to_end(&mut bytes)?;
            let mut ticket: Ticket = serde_json::from_slice(&bytes)
                .map_err(|error| AppError::new(format!("could not read ticket '{id}': {error}")))?;
            hold_for_test_race();
            update(&mut ticket)?;
            atomic_write(&path, &serde_json::to_vec_pretty(&ticket)?)?;
            return Ok(());
        }
    }

    fn all_tickets(&self, prefix: Option<&str>) -> AppResult<Vec<Ticket>> {
        let projects = self.projects()?;
        let prefixes: Vec<String> = match prefix {
            Some(prefix) => {
                self.require_project(&projects, prefix)?;
                vec![prefix.to_string()]
            }
            None => projects.keys().cloned().collect(),
        };

        let mut tickets = Vec::new();
        for prefix in prefixes {
            let dir = self.project_dir(&prefix);
            if !dir.exists() {
                continue;
            }
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                if is_ticket_file(&entry) {
                    let path = entry.path();
                    match serde_json::from_slice::<Ticket>(&fs::read(&path)?) {
                        Ok(ticket) => tickets.push(ticket),
                        Err(error) => {
                            return Err(AppError::new(format!(
                                "could not read ticket file {}: {error}",
                                path.display()
                            )))
                        }
                    }
                }
            }
        }
        tickets.sort_by_key(|ticket| ticket_sort_key(&ticket.id));
        Ok(tickets)
    }

    fn allocate_ticket(&self, prefix: &str, mut ticket: Ticket) -> AppResult<String> {
        create_dir_all_durable(&self.project_dir(prefix))?;
        let mut number = 1_u64;
        loop {
            let id = format!("{prefix}-{number}");
            let path = self.project_dir(prefix).join(format!("{id}.json"));
            ticket.id = id.clone();
            let bytes = serde_json::to_vec_pretty(&ticket)?;
            match publish_new(&path, &bytes)? {
                true => return Ok(id),
                false => number += 1,
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let result = Storage::from_env().and_then(|storage| run(&storage, cli));
    if let Err(error) = result {
        eprintln!("error: {error}");
        process::exit(1);
    }
}

fn run(storage: &Storage, cli: Cli) -> AppResult<()> {
    match cli.command {
        Command::Init { prefix, name } => init(storage, &prefix, &name),
        Command::Projects => list_projects(storage, cli.json),
        Command::Add(args) => add_ticket(storage, args),
        Command::Ls(args) => list_tickets(storage, args, cli.json),
        Command::Show { id } => show_ticket(storage, &id, cli.json),
        Command::Start { id } => transition(storage, &id, State::InProgress),
        Command::Done { id } => transition(storage, &id, State::Done),
        Command::Cancel { id } => transition(storage, &id, State::Canceled),
        Command::Reopen { id } => transition(storage, &id, State::Todo),
        Command::Edit(args) => edit_ticket(storage, args),
        Command::Note { id, text } => add_note(storage, &id, &text),
        Command::Next { prefix } => next_ticket(storage, &prefix, cli.json),
    }
}

fn init(storage: &Storage, prefix: &str, name: &str) -> AppResult<()> {
    validate_prefix(prefix)?;
    if name.trim().is_empty() {
        return Err(AppError::new("project name must not be empty"));
    }
    storage.update_projects(|projects| {
        if projects.contains_key(prefix) {
            return Err(AppError::new(format!(
                "project '{prefix}' already exists in data dir {}",
                storage.root.display()
            )));
        }
        projects.insert(
            prefix.to_string(),
            Project {
                name: name.to_string(),
                created: now(),
            },
        );
        Ok(())
    })?;
    create_dir_all_durable(&storage.project_dir(prefix))?;
    println!("initialized {prefix} ({name})");
    Ok(())
}

fn list_projects(storage: &Storage, json: bool) -> AppResult<()> {
    let projects = storage.projects()?;
    let mut rows = Vec::new();
    for (prefix, project) in &projects {
        let counts = count_tickets(storage, prefix)?;
        rows.push(ProjectJson {
            prefix: prefix.clone(),
            name: project.name.clone(),
            counts,
        });
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
    } else {
        println!("PREFIX  NAME                 TODO  IN-PROGRESS  DONE  CANCELED");
        for row in rows {
            println!(
                "{:<7} {:<20} {:>4}  {:>11}  {:>4}  {:>8}",
                row.prefix,
                row.name,
                row.counts.todo,
                row.counts.in_progress,
                row.counts.done,
                row.counts.canceled
            );
        }
    }
    Ok(())
}

fn add_ticket(storage: &Storage, args: AddArgs) -> AppResult<()> {
    validate_title(&args.title)?;
    let projects = storage.projects()?;
    storage.require_project(&projects, &args.prefix)?;
    let timestamp = now();
    let ticket = Ticket {
        id: String::new(),
        title: args.title,
        description: args.description,
        state: State::Todo,
        priority: args.priority,
        labels: unique(args.labels),
        notes: Vec::new(),
        created: timestamp.clone(),
        updated: timestamp,
    };
    let id = storage.allocate_ticket(&args.prefix, ticket)?;
    println!("{id}");
    Ok(())
}

fn list_tickets(storage: &Storage, args: ListArgs, json: bool) -> AppResult<()> {
    let tickets = storage.all_tickets(args.prefix.as_deref())?;
    let filtered: Vec<Ticket> = tickets
        .into_iter()
        .filter(|ticket| {
            let state_matches = args.states.is_empty() || args.states.contains(&ticket.state);
            let default_state_matches = args.all
                || !args.states.is_empty()
                || matches!(ticket.state, State::Todo | State::InProgress);
            let priority_matches = args.priority.is_none() || args.priority == ticket.priority;
            let label_matches = args
                .label
                .as_ref()
                .is_none_or(|label| ticket.labels.iter().any(|item| item == label));
            state_matches && default_state_matches && priority_matches && label_matches
        })
        .collect();

    if json {
        println!("{}", serde_json::to_string_pretty(&filtered)?);
    } else {
        println!("ID       STATE        PRIORITY  TITLE");
        for ticket in filtered {
            println!(
                "{:<8} {:<12} {:<8}  {}",
                ticket.id,
                ticket.state,
                ticket
                    .priority
                    .map_or_else(|| "-".to_string(), |priority| priority.to_string()),
                ticket.title
            );
        }
    }
    Ok(())
}

fn show_ticket(storage: &Storage, id: &str, json: bool) -> AppResult<()> {
    let ticket = storage.read_ticket(id)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&ticket)?);
    } else {
        println!("{}", ticket.id);
        println!("title: {}", ticket.title);
        println!("state: {}", ticket.state);
        println!(
            "priority: {}",
            ticket
                .priority
                .map_or_else(|| "-".to_string(), |priority| priority.to_string())
        );
        println!("labels: {}", ticket.labels.join(", "));
        println!("created: {}", ticket.created);
        println!("updated: {}", ticket.updated);
        println!("\n{}", ticket.description);
        if !ticket.notes.is_empty() {
            println!("\nnotes:");
            for note in ticket.notes {
                println!("- [{}] {}", note.ts, note.text);
            }
        }
    }
    Ok(())
}

fn transition(storage: &Storage, id: &str, target: State) -> AppResult<()> {
    storage.mutate_ticket(id, |ticket| {
        let legal = match target {
            State::InProgress => ticket.state == State::Todo,
            State::Done | State::Canceled => {
                matches!(ticket.state, State::Todo | State::InProgress)
            }
            State::Todo => matches!(ticket.state, State::Done | State::Canceled),
        };
        if !legal {
            return Err(AppError::new(format!(
                "cannot move ticket '{}' from {} to {}",
                ticket.id, ticket.state, target
            )));
        }
        ticket.state = target;
        ticket.updated = now();
        Ok(())
    })
}

fn edit_ticket(storage: &Storage, args: EditArgs) -> AppResult<()> {
    if args.title.is_none()
        && args.desc.is_none()
        && args.priority.is_none()
        && args.add_labels.is_empty()
        && args.remove_labels.is_empty()
    {
        return Err(AppError::new("edit requires at least one field"));
    }
    let id = args.id;
    let title = args.title;
    let description = args.desc;
    let priority = args.priority;
    let add_labels = args.add_labels;
    let remove_labels = args.remove_labels;
    storage.mutate_ticket(&id, |ticket| {
        if let Some(title) = title.as_ref() {
            validate_title(title)?;
            ticket.title = title.clone();
        }
        if let Some(description) = description.as_ref() {
            ticket.description = description.clone();
        }
        if let Some(priority) = priority {
            ticket.priority = Some(priority);
        }
        for label in &add_labels {
            if !ticket.labels.contains(label) {
                ticket.labels.push(label.clone());
            }
        }
        for label in &remove_labels {
            ticket.labels.retain(|item| item != label);
        }
        ticket.updated = now();
        Ok(())
    })
}

fn add_note(storage: &Storage, id: &str, text: &str) -> AppResult<()> {
    if text.is_empty() {
        return Err(AppError::new("note text must not be empty"));
    }
    storage.mutate_ticket(id, |ticket| {
        let timestamp = now();
        ticket.notes.push(Note {
            ts: timestamp.clone(),
            text: text.to_string(),
        });
        ticket.updated = timestamp;
        Ok(())
    })
}

fn next_ticket(storage: &Storage, prefix: &str, json: bool) -> AppResult<()> {
    let projects = storage.projects()?;
    storage.require_project(&projects, prefix)?;
    let mut number = 1_u64;
    while storage
        .project_dir(prefix)
        .join(format!("{prefix}-{number}.json"))
        .exists()
    {
        number += 1;
    }
    let id = format!("{prefix}-{number}");
    if json {
        println!("{}", serde_json::json!({ "prefix": prefix, "next_id": id }));
    } else {
        println!("{id}");
    }
    Ok(())
}

fn count_tickets(storage: &Storage, prefix: &str) -> AppResult<Counts> {
    let mut counts = Counts::default();
    for ticket in storage.all_tickets(Some(prefix))? {
        counts.add(ticket.state);
    }
    Ok(counts)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> AppResult<()> {
    let temp = write_temp(path, bytes)?;
    let result = (|| {
        fs::rename(&temp, path)?;
        sync_parent_dir(path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn publish_new(path: &Path, bytes: &[u8]) -> AppResult<bool> {
    let temp = write_temp(path, bytes)?;
    if env::var_os("TIX_TEST_PAUSE_AFTER_TEMP").is_some() {
        println!("temp-ready");
        io::stdout().flush()?;
        loop {
            thread::sleep(Duration::from_millis(25));
        }
    }
    let result = match fs::hard_link(&temp, path) {
        Ok(()) => sync_parent_dir(path).map(|()| true),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(AppError::from(error)),
    };
    let _ = fs::remove_file(&temp);
    result
}

fn write_temp(path: &Path, bytes: &[u8]) -> AppResult<PathBuf> {
    if let Some(parent) = path.parent() {
        create_dir_all_durable(parent)?;
    }
    let file_name = path
        .file_name()
        .ok_or_else(|| AppError::new(format!("invalid path {}", path.display())))?
        .to_string_lossy();
    let temp = path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}-{}",
        process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    if let Err(error) = write_and_sync(&mut file, bytes) {
        let _ = fs::remove_file(&temp);
        return Err(error);
    }
    Ok(temp)
}

fn sync_parent_dir(path: &Path) -> AppResult<()> {
    let parent = path
        .parent()
        .ok_or_else(|| AppError::new(format!("invalid path {}", path.display())))?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn create_dir_all_durable(path: &Path) -> AppResult<()> {
    let mut missing = Vec::new();
    let mut current = path;
    loop {
        match fs::metadata(current) {
            Ok(metadata) => {
                if !metadata.is_dir() {
                    return Err(AppError::new(format!(
                        "{} is not a directory",
                        current.display()
                    )));
                }
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                missing.push(current.to_path_buf());
                current = current.parent().ok_or_else(|| {
                    AppError::new(format!(
                        "cannot find parent directory for {}",
                        path.display()
                    ))
                })?;
            }
            Err(error) => return Err(error.into()),
        }
    }

    for directory in missing.iter().rev() {
        match fs::create_dir(directory) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        File::open(directory)?.sync_all()?;
        sync_parent_dir(directory)?;
    }
    Ok(())
}

fn write_and_sync(file: &mut File, bytes: &[u8]) -> AppResult<()> {
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn is_ticket_file(entry: &DirEntry) -> bool {
    entry.file_type().is_ok_and(|file_type| file_type.is_file())
        && entry
            .path()
            .extension()
            .is_some_and(|extension| extension == "json")
}

fn ticket_sort_key(id: &str) -> (String, u64) {
    let (prefix, number) = id.rsplit_once('-').unwrap_or((id, "0"));
    (prefix.to_string(), number.parse().unwrap_or(0))
}

fn parse_id(id: &str) -> AppResult<(&str, &str)> {
    let (prefix, number) = id
        .rsplit_once('-')
        .ok_or_else(|| AppError::new(format!("invalid ticket id '{id}'")))?;
    validate_prefix(prefix)?;
    if number.is_empty() || !number.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(AppError::new(format!("invalid ticket id '{id}'")));
    }
    Ok((prefix, number))
}

fn validate_prefix(prefix: &str) -> AppResult<()> {
    if prefix.is_empty() || !prefix.bytes().all(|byte| byte.is_ascii_uppercase()) {
        return Err(AppError::new(format!(
            "invalid project prefix '{prefix}': use uppercase letters"
        )));
    }
    Ok(())
}

fn validate_title(title: &str) -> AppResult<()> {
    if title.trim().is_empty() {
        return Err(AppError::new("title must not be empty or whitespace-only"));
    }
    if title.contains(['\r', '\n']) {
        return Err(AppError::new("title must not contain CR or LF"));
    }
    Ok(())
}

fn unique(values: Vec<String>) -> Vec<String> {
    let mut result = Vec::new();
    for value in values {
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn now() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::Mutex;

    static CWD_LOCK: Mutex<()> = Mutex::new(());

    struct CwdGuard {
        original_cwd: PathBuf,
        original_data_dir: Option<OsString>,
    }

    impl Drop for CwdGuard {
        fn drop(&mut self) {
            env::set_current_dir(&self.original_cwd).unwrap();
            match &self.original_data_dir {
                Some(value) => env::set_var("TIX_DATA_DIR", value),
                None => env::remove_var("TIX_DATA_DIR"),
            }
        }
    }

    #[test]
    fn relative_storage_root_does_not_follow_cwd_changes() {
        let _cwd_lock = CWD_LOCK.lock().unwrap();
        let original_cwd = env::current_dir().unwrap();
        let original_data_dir = env::var_os("TIX_DATA_DIR");
        let first_cwd = tempfile::tempdir().unwrap();
        let second_cwd = tempfile::tempdir().unwrap();
        env::set_current_dir(first_cwd.path()).unwrap();
        env::set_var("TIX_DATA_DIR", "data");
        let _cwd_guard = CwdGuard {
            original_cwd,
            original_data_dir,
        };

        let storage = Storage::from_env().unwrap();
        let expected_root = env::current_dir().unwrap().join("data");
        env::set_current_dir(second_cwd.path()).unwrap();
        init(&storage, "CWD", "Cwd test").unwrap();
        add_ticket(
            &storage,
            AddArgs {
                prefix: "CWD".to_string(),
                title: "stable root".to_string(),
                description: String::new(),
                priority: None,
                labels: Vec::new(),
            },
        )
        .unwrap();
        add_note(&storage, "CWD-1", "still here").unwrap();

        assert_eq!(storage.root, expected_root);
        assert!(expected_root.join("tickets/CWD/CWD-1.json").exists());
        assert!(!second_cwd.path().join("data").exists());
    }
}
