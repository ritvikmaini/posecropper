"""Production Cloud Functions Gen2 entry point (storage.object.v1.finalized trigger).
Thin GCP adapter around posecropper.pipeline — the routing/crop logic lives in the core.
Imports google.* lazily here so the offline demo never pulls in GCP."""
import os

from google.cloud import storage

from posecropper.catalog.bigquery import BigQueryCategoryProvider
from posecropper.io.gcs import GcsImageSource, GcsOutputSink
from posecropper.metadata import classify_category, parse_filename
from posecropper.pipeline import process_image


def _ensure_triage_folders(bucket, folder):
    for sub in ("", "error", "pose", "review"):
        path = f"{folder}/{sub}/" if sub else f"{folder}/"
        blob = bucket.blob(path)
        if not blob.exists():
            blob.upload_from_string("")


def pose_cropper_gcf(event, context):
    client = storage.Client()
    file_path = event["name"]
    folder = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base = file_name[:-4] if file_name.endswith(".jpg") else file_name

    source = GcsImageSource(client.bucket(event["bucket"]))
    dest_bucket = client.bucket(os.environ["DESTINATION_BUCKET"])
    sink = GcsOutputSink(dest_bucket)

    try:
        _ensure_triage_folders(dest_bucket, folder)
        image_rgb = source.read(file_path)

        meta = parse_filename(file_path)
        item_group = BigQueryCategoryProvider().get_item_group(meta["model"])
        category = classify_category(item_group)

        decision, crops, overlay = process_image(image_rgb, meta["pose"], category)

        if overlay is not None:
            sink.write(overlay, f"{folder}/pose/landmarks_{base}.jpg")

        for name, img in crops.items():
            suffix = f"_{name}" if decision.review else ""
            sub = "review/" if decision.review else ""
            sink.write(img, f"{folder}/{sub}{base}{suffix}.jpg")
        return "Success"
    except Exception as e:  # noqa: BLE001 — copy original to /error/ as the prod system did
        print(f"Unexpected error in pose_cropper_gcf: {e}")
        try:
            original = source.read(file_path)
            sink.write(original, f"{folder}/error/{file_name}")
        except Exception as inner:  # noqa: BLE001
            print(f"Error writing to error folder: {inner}")
        return "ERROR"
