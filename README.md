# pufferclone

## MinIO end-to-end checks

The repository includes a single-service MinIO fixture. It uses API port
`9000`, console port `9001`, credentials `minioadmin`/`minioadmin`, and the
named volume `pufferclone-minio-data`.

Start MinIO and create the test bucket with:

```sh
make minio-up
```

Run the S3 engine integration tests with:

```sh
PUFFERCLONE_TEST_S3_URL=http://127.0.0.1:9000 \
PUFFERCLONE_TEST_S3_BUCKET=pufferclone-test \
AWS_ACCESS_KEY_ID=minioadmin \
AWS_SECRET_ACCESS_KEY=minioadmin \
cargo test --test s3_e2e
```

`PUFFERCLONE_TEST_S3_URL` is the gate: without it, the S3 integration tests
print a skip message and pass. `make minio-test` runs the same setup and test
command. Stop the fixture with `make minio-down`; remove its retained data
only when desired with `docker volume rm pufferclone-minio-data`.

The real-binary smoke check starts MinIO when necessary, creates its bucket,
drives the HTTP API, and removes its namespace on exit. It refuses to run if
the fixed API port `8666` is already occupied.

```sh
scripts/s3_smoke.sh
```

## S3 and LocalDirStore behavior

The S3 adapter preserves the object-store contract while its mechanics differ
from `LocalDirStore`:

- S3 listing is a paginated service operation. The adapter asks the service
  for a root listing, filters lexical prefixes locally, and sorts the result;
  `LocalDirStore` recursively walks the filesystem. Listing a large bucket is
  therefore broader and slower for S3.
- Write-once objects use S3 conditional create and map `AlreadyExists` or
  precondition failures to `Error::AlreadyExists`; manifest objects use
  overwrite. The local store uses hard-link publication for write-once files
  and rename for manifests.
- Every S3 operation crosses the synchronous/async bridge and the network.
  It has materially higher latency than local filesystem access, so engine
  correctness does not depend on local-store timing.
