//! Foundation types and durable storage for the pufferclone v0 engine.

pub mod segment;
pub mod store;
pub mod types;
pub mod wal;

pub use types::{AttrValue, Doc};
pub use wal::{Manifest, SegmentMeta, WalBatch};

/// The result type used by pufferclone's public APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by pufferclone's storage and serialization layers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An underlying filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON encoding or decoding failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Bincode encoding or decoding failed.
    #[error("bincode error: {0}")]
    Bincode(#[from] Box<bincode::ErrorKind>),

    /// A key is not valid for an object store.
    #[error("invalid object-store key: {0}")]
    InvalidKey(String),

    /// An object already exists and is write-once.
    #[error("object already exists: {0}")]
    AlreadyExists(String),

    /// An object does not exist.
    #[error("object not found: {0}")]
    NotFound(String),

    /// An internal store invariant or synchronization operation failed.
    #[error("internal store error: {0}")]
    Store(String),
}
