//! S3-compatible object storage for pufferclone.

use std::future::Future;
use std::sync::Arc;
use std::thread;

use futures_util::TryStreamExt;
use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};
use object_store::path::Path as ObjectPath;
use object_store::{Error as ObjectStoreError, ObjectStore as CloudObjectStore};
use object_store::{ObjectStoreExt, PutMode, PutOptions};
use tokio::runtime::RuntimeFlavor;

use crate::store::{is_manifest_key, validate_key, validate_prefix};
use crate::{Error, Result};

type Client = Arc<dyn CloudObjectStore>;

/// An [`crate::store::ObjectStore`] backed by an S3-compatible service.
///
/// The synchronous pufferclone trait is bridged to object_store's async API.
/// Calls use `block_in_place` when the caller is already on a multi-threaded
/// Tokio runtime. Current-thread runtimes and non-Tokio callers use a short-
/// lived dedicated runtime thread instead, avoiding a nested `block_on`.
///
/// Unlike [`crate::store::LocalDirStore`], this store has no temporary-file or
/// directory-fsync ceremony: the object service is the durability domain, and
/// its atomic put operation is the publication boundary.
pub struct S3Store {
    client: Client,
}

impl S3Store {
    fn configured_builder(
        builder: AmazonS3Builder,
        url: &str,
        bucket: impl Into<String>,
    ) -> AmazonS3Builder {
        builder
            // AWS_ENDPOINT_URL_S3 has higher precedence than the generic
            // endpoint, so explicitly replace that field too.
            .with_config(AmazonS3ConfigKey::S3Endpoint, url)
            .with_bucket_name(bucket)
            .with_virtual_hosted_style_request(false)
            .with_allow_http(url.starts_with("http://"))
    }

    /// Build a store for an S3 endpoint and bucket.
    ///
    /// `url` is the service endpoint, such as `https://s3.example.test` or
    /// `http://127.0.0.1:9000` for MinIO. AWS credential and region settings
    /// are read from the standard `AWS_*` environment variables.
    pub fn new(url: impl Into<String>, bucket: impl Into<String>) -> Result<Self> {
        let url = url.into();
        let client = Self::configured_builder(AmazonS3Builder::from_env(), &url, bucket)
            .build()
            .map_err(|error| Error::Store(format!("failed to build S3 client: {error}")))?;
        Ok(Self {
            client: Arc::new(client),
        })
    }

    fn run<T, F>(
        &self,
        operation: impl FnOnce(Client) -> F + Send + 'static,
    ) -> object_store::Result<T>
    where
        T: Send + 'static,
        F: Future<Output = object_store::Result<T>> + Send + 'static,
    {
        let client = Arc::clone(&self.client);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if matches!(handle.runtime_flavor(), RuntimeFlavor::MultiThread) => {
                tokio::task::block_in_place(|| handle.block_on(operation(client)))
            }
            _ => thread::spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| ObjectStoreError::Generic {
                        store: "pufferclone-s3-runtime",
                        source: Box::new(error),
                    })?;
                runtime.block_on(operation(client))
            })
            .join()
            .map_err(|_| ObjectStoreError::Generic {
                store: "pufferclone-s3-runtime",
                source: Box::new(std::io::Error::other("S3 runtime thread panicked")),
            })?,
        }
    }

    fn map_error(key: &str, error: ObjectStoreError) -> Error {
        match error {
            ObjectStoreError::NotFound { .. } => Error::NotFound(key.to_owned()),
            other => Error::Store(format!("S3 object {key}: {other}")),
        }
    }
}

impl crate::store::ObjectStore for S3Store {
    fn put(&self, key: &str, bytes: &[u8]) -> Result<()> {
        validate_key(key)?;
        let manifest = is_manifest_key(key);
        let object_key = key.to_owned();
        let payload = bytes.to_vec();
        let result = self.run(move |client| async move {
            let mode = if manifest {
                PutMode::Overwrite
            } else {
                PutMode::Create
            };
            client
                .put_opts(
                    &ObjectPath::from(object_key),
                    payload.into(),
                    PutOptions {
                        mode,
                        ..Default::default()
                    },
                )
                .await
                .map(|_| ())
        });
        result.map_err(|error| match error {
            ObjectStoreError::AlreadyExists { .. } | ObjectStoreError::Precondition { .. }
                if !manifest =>
            {
                Error::AlreadyExists(key.to_owned())
            }
            other => Self::map_error(key, other),
        })
    }

    fn get(&self, key: &str) -> Result<Vec<u8>> {
        validate_key(key)?;
        let object_key = key.to_owned();
        self.run(move |client| async move {
            client
                .get(&ObjectPath::from(object_key))
                .await?
                .bytes()
                .await
        })
        .map(|bytes| bytes.to_vec())
        .map_err(|error| Self::map_error(key, error))
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        validate_prefix(prefix)?;
        let filter_prefix = prefix.to_owned();
        self.run(move |client| async move {
            // object_store's prefix is path-segment based, while this trait
            // deliberately promises lexical starts_with semantics. Listing
            // from the root keeps exact keys and partial components visible;
            // the filter below is the contract boundary.
            client.list(None).try_collect::<Vec<_>>().await
        })
        .map(|objects| {
            let mut keys = objects
                .into_iter()
                .map(|object| object.location.to_string())
                .filter(|key| key.starts_with(&filter_prefix))
                .collect::<Vec<_>>();
            keys.sort();
            keys
        })
        .map_err(|error| Self::map_error(prefix, error))
    }

    fn delete(&self, key: &str) -> Result<()> {
        validate_key(key)?;
        let object_key = key.to_owned();
        self.run(move |client| async move {
            let location = ObjectPath::from(object_key);
            // S3 DELETE is idempotent and normally succeeds for a missing key;
            // HEAD first to preserve pufferclone's NotFound contract.
            client.head(&location).await?;
            client.delete(&location).await
        })
        .map_err(|error| Self::map_error(key, error))
    }
}

#[cfg(test)]
mod tests {
    use super::S3Store;
    use object_store::aws::{AmazonS3Builder, AmazonS3ConfigKey};

    #[test]
    fn configured_endpoint_overrides_s3_specific_environment_endpoint() {
        let builder = AmazonS3Builder::new().with_config(
            AmazonS3ConfigKey::S3Endpoint,
            "http://environment.example.test",
        );
        let configured =
            S3Store::configured_builder(builder, "http://pufferclone.example.test", "bucket");
        let debug = format!("{configured:?}");

        assert!(debug.contains("s3_endpoint: Some(\"http://pufferclone.example.test\")"));
        assert!(!debug.contains("environment.example.test"));
    }
}
