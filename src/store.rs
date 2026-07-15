//! Object storage abstractions and the local filesystem implementation.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(unix))]
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{Error, Result};

/// A minimal object store used by the WAL and manifest layers.
pub trait ObjectStore {
    /// Create an object, failing if it already exists except for manifest keys.
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()>;

    /// Read an object by key.
    fn get(&self, key: &str) -> Result<Vec<u8>>;

    /// List object keys that start with `prefix`.
    fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Delete an object by key. Missing objects are reported as [`Error::NotFound`].
    fn delete(&self, key: &str) -> Result<()>;
}

/// An [`ObjectStore`] backed by files below a local root directory.
pub struct LocalDirStore {
    root: PathBuf,
}

const TEMP_DIR: &str = ".tmp";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

impl LocalDirStore {
    /// Create a store rooted at `root`, creating the directory if needed.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        ensure_directory_tree(&root)?;
        ensure_directory_tree(&root.join(TEMP_DIR))?;
        Ok(Self { root })
    }

    fn path_for_key(&self, key: &str) -> Result<PathBuf> {
        validate_key(key)?;
        Ok(self.root.join(key))
    }
}

impl ObjectStore for LocalDirStore {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        let path = self.path_for_key(key)?;
        let parent = path
            .parent()
            .ok_or_else(|| Error::Store(format!("object has no parent: {key}")))?;
        ensure_parent_dirs(&self.root, parent)?;
        let temp_dir = self.root.join(TEMP_DIR);
        let temp_path = temporary_path(&temp_dir)?;

        let write_result = (|| {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp_path)?;
            file.write_all(bytes)?;
            file.sync_all()?;

            if is_manifest_key(key) {
                fs::rename(&temp_path, &path)?;
            } else {
                // hard_link creates the destination atomically and fails if another
                // writer published this write-once key first.
                fs::hard_link(&temp_path, &path)?;
                fs::remove_file(&temp_path)?;
            }

            sync_directory(&temp_dir)?;
            sync_ancestors(parent, &self.root)?;
            Ok::<(), std::io::Error>(())
        })();

        if write_result.is_err() {
            let _ = fs::remove_file(&temp_path);
        }
        write_result.map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists && !is_manifest_key(key) {
                Error::AlreadyExists(key.to_owned())
            } else {
                Error::Io(error)
            }
        })
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        let path = self.path_for_key(key)?;
        match fs::read(path) {
            Ok(bytes) => Ok(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(Error::NotFound(key.to_owned()))
            }
            Err(error) => Err(Error::Io(error)),
        }
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        validate_prefix(prefix)?;
        let mut keys = Vec::new();
        collect_files(&self.root, &self.root, prefix, &mut keys)?;
        keys.sort();
        Ok(keys)
    }

    fn delete(&self, key: &str) -> Result<()> {
        let path = self.path_for_key(key)?;
        let parent = path
            .parent()
            .ok_or_else(|| Error::Store(format!("object has no parent: {key}")))?;
        match fs::remove_file(&path) {
            Ok(()) => {
                sync_ancestors(parent, &self.root)?;
                Ok(())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Err(Error::NotFound(key.to_owned()))
            }
            Err(error) => Err(Error::Io(error)),
        }
    }
}

fn validate_key(key: &str) -> Result<()> {
    if key.is_empty() || key.contains('\\') || key.contains("..") {
        return Err(Error::InvalidKey(key.to_owned()));
    }

    // Path::components() normalizes `.` segments, so reject aliases in the raw
    // key before handing it to path resolution.
    if key
        .split('/')
        .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return Err(Error::InvalidKey(key.to_owned()));
    }

    let mut first_component = true;
    for component in Path::new(key).components() {
        let Component::Normal(name) = component else {
            return Err(Error::InvalidKey(key.to_owned()));
        };
        if first_component && name == TEMP_DIR {
            return Err(Error::InvalidKey(key.to_owned()));
        }
        first_component = false;
    }

    if first_component {
        return Err(Error::InvalidKey(key.to_owned()));
    }
    Ok(())
}

fn validate_prefix(prefix: &str) -> Result<()> {
    if prefix.is_empty() {
        return Ok(());
    }
    validate_key(prefix.strip_suffix('/').unwrap_or(prefix))
}

fn is_manifest_key(key: &str) -> bool {
    Path::new(key).file_name().and_then(|name| name.to_str()) == Some("MANIFEST.json")
}

fn temporary_path(temp_dir: &Path) -> Result<PathBuf> {
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(temp_dir.join(format!("{:016x}-{:032x}", counter, random_suffix()?)))
}

#[cfg(unix)]
fn random_suffix() -> Result<u128> {
    let mut bytes = [0; 16];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(u128::from_ne_bytes(bytes))
}

#[cfg(not(unix))]
fn random_suffix() -> Result<u128> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::Store(format!("system clock before Unix epoch: {error}")))?
        .as_nanos();
    Ok(timestamp ^ u128::from(std::process::id()))
}

fn ensure_parent_dirs(root: &Path, parent: &Path) -> Result<()> {
    let relative = parent
        .strip_prefix(root)
        .map_err(|error| Error::Store(format!("object parent escaped root: {error}")))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err(Error::Store(format!(
                "invalid object parent: {}",
                parent.display()
            )));
        };
        current.push(name);
        match fs::create_dir(&current) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !current.is_dir() {
                    return Err(Error::Io(std::io::Error::new(
                        std::io::ErrorKind::NotADirectory,
                        format!("object parent is not a directory: {}", current.display()),
                    )));
                }
            }
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Ok(())
}

fn ensure_directory_tree(path: &Path) -> Result<()> {
    let mut missing = Vec::new();
    let mut current = path.to_path_buf();
    while !current.exists() {
        missing.push(current.clone());
        let parent = current
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        if parent == current {
            break;
        }
        current = parent;
    }

    fs::create_dir_all(path)?;
    for directory in missing {
        let parent = directory
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        sync_directory(parent)?;
    }
    Ok(())
}

fn sync_ancestors(parent: &Path, root: &Path) -> std::io::Result<()> {
    let mut current = parent.to_path_buf();
    loop {
        sync_directory(&current)?;
        if current == root {
            return Ok(());
        }
        current = current
            .parent()
            .ok_or_else(|| std::io::Error::other("store root is not an ancestor"))?
            .to_path_buf();
    }
}

fn sync_directory(path: &Path) -> std::io::Result<()> {
    File::open(path)?.sync_all()
}

fn collect_files(
    root: &Path,
    directory: &Path,
    prefix: &str,
    keys: &mut Vec<String>,
) -> Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if directory == root && entry.file_name() == TEMP_DIR {
            continue;
        }
        if path.is_dir() {
            collect_files(root, &path, prefix, keys)?;
        } else if path.is_file() {
            let relative = path
                .strip_prefix(root)
                .map_err(|_| Error::InvalidKey(path.display().to_string()))?;
            let key = relative
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/");
            if key.starts_with(prefix) {
                keys.push(key);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Arc;
    use std::thread;

    use tempfile::tempdir;

    use super::{LocalDirStore, ObjectStore, TEMP_DIR};
    use crate::Error;

    #[test]
    fn local_store_supports_put_get_list_delete_and_write_once() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");

        store.put("ns/demo/a", b"alpha").expect("put");
        store.put("ns/demo/b", b"beta").expect("put");
        assert_eq!(store.get("ns/demo/a").expect("get"), b"alpha");
        assert_eq!(
            store.list("ns/demo/").expect("list"),
            vec!["ns/demo/a", "ns/demo/b"]
        );
        assert!(matches!(
            store.put("ns/demo/a", b"new"),
            Err(Error::AlreadyExists(key)) if key == "ns/demo/a"
        ));

        store.delete("ns/demo/a").expect("delete");
        assert!(matches!(store.get("ns/demo/a"), Err(Error::NotFound(key)) if key == "ns/demo/a"));
    }

    #[test]
    fn manifest_key_may_be_overwritten() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");

        store.put("ns/demo/MANIFEST.json", b"one").expect("put");
        store
            .put("ns/demo/MANIFEST.json", b"two")
            .expect("overwrite");
        assert_eq!(store.get("ns/demo/MANIFEST.json").expect("get"), b"two");
    }

    #[test]
    fn concurrent_puts_to_one_key_publish_only_one_value() {
        let root = tempdir().expect("tempdir");
        let store = Arc::new(LocalDirStore::new(root.path()).expect("store"));
        let workers = (0..8)
            .map(|index| {
                let store = Arc::clone(&store);
                thread::spawn(move || store.put("ns/demo/concurrent", &[index]))
            })
            .collect::<Vec<_>>();
        let results = workers
            .into_iter()
            .map(|worker| worker.join().expect("worker"))
            .collect::<Vec<_>>();

        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, Err(Error::AlreadyExists(_))))
                .count(),
            7
        );
        assert_eq!(store.list("ns/demo/").expect("list").len(), 1);
    }

    #[test]
    fn list_hides_stray_temp_files() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        store.put("ns/demo/live", b"live").expect("put");
        fs::write(root.path().join(TEMP_DIR).join("stray-temp"), b"partial").expect("stray");

        assert_eq!(store.list("").expect("list"), vec!["ns/demo/live"]);
    }

    #[test]
    fn invalid_keys_are_rejected_at_the_store_boundary() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        for key in [
            "../x",
            "/absolute",
            "ns//x",
            "ns/./x",
            "ns/x/.",
            ".tmp/hidden",
            "./.tmp/x",
            "./ns/x",
        ] {
            assert!(
                matches!(store.put(key, b"x"), Err(Error::InvalidKey(_))),
                "{key}"
            );
            assert!(matches!(store.get(key), Err(Error::InvalidKey(_))), "{key}");
            assert!(
                matches!(store.delete(key), Err(Error::InvalidKey(_))),
                "{key}"
            );
        }
    }

    #[test]
    fn only_the_exact_manifest_filename_is_overwritable() {
        let root = tempdir().expect("tempdir");
        let store = LocalDirStore::new(root.path()).expect("store");
        store.put("ns/demo/fooMANIFEST.json", b"one").expect("put");

        assert!(matches!(
            store.put("ns/demo/fooMANIFEST.json", b"two"),
            Err(Error::AlreadyExists(key)) if key == "ns/demo/fooMANIFEST.json"
        ));
    }
}
