# Deployment Reference

> **Note:** The public CI (`.github/workflows/ci.yml`) runs lint and tests only and does not deploy. The commands below are how the two functions were deployed in production, genericized with placeholder values. This document is a reference rather than an active pipeline.

Copy `config.example.yml` to `config.yml` (git-ignored) and fill in your real values before running these commands.

---

## Pose cropper (Cloud Functions Gen2, storage-triggered)

```bash
gcloud functions deploy pose-cropper \
  --source posecropper --runtime python310 --gen2 \
  --project=$PROJECT_ID --region=$GCP_REGION \
  --entry-point pose_cropper_gcf --memory 12GB --timeout=540 \
  --env-vars-file ./config.yml \
  --trigger-event-filters="type=google.cloud.storage.object.v1.finalized" \
  --trigger-event-filters="bucket=$SOURCE_BUCKET"
```

Triggers on every image finalized in `$SOURCE_BUCKET`. The function parses the filename for pose and model codes, resolves the garment category from the catalog, runs pose detection, produces aspect-ratio-correct crops, and writes them to the destination bucket — routing ambiguous cases to `/review/` and failures to `/error/`.

---

## Folder syncer (HTTP-triggered)

```bash
gcloud functions deploy folder-syncer \
  --source posecropper --runtime python310 --gen2 \
  --project=$PROJECT_ID --region=$GCP_REGION \
  --entry-point sync_folders --memory 12GB --timeout=295 \
  --env-vars-file ./config.yml --trigger-http
```

Mirrors the destination bucket into a Google Drive folder tree via recursive diff-and-sync (100-worker thread pool, idempotent folder creation).

---

Secrets (the Drive service-account JSON) are read from Secret Manager at runtime; see `config.example.yml`.
