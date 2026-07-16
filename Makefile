.PHONY: minio-up minio-down minio-test

MINIO_MC_IMAGE ?= minio/mc:latest
MINIO_BUCKET ?= pufferclone-test

minio-up:
	docker compose up -d minio
	until curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null; do sleep 1; done
	docker run --rm --name pufferclone-mc --network pufferclone-minio-net --entrypoint /bin/sh $(MINIO_MC_IMAGE) -c 'mc alias set local http://minio:9000 minioadmin minioadmin && mc mb --ignore-existing local/$(MINIO_BUCKET)'

minio-down:
	docker compose down

minio-test: minio-up
	PUFFERCLONE_TEST_S3_URL=http://127.0.0.1:9000 \
	PUFFERCLONE_TEST_S3_BUCKET=$(MINIO_BUCKET) \
	AWS_ACCESS_KEY_ID=minioadmin \
	AWS_SECRET_ACCESS_KEY=minioadmin \
	cargo test --test s3_e2e
