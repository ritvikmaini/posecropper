"""GCS-to-Google-Drive folder syncer (cloud path only).
Recursively diffs bucket folders against Drive folders and uploads any missing files
using a 100-worker ThreadPoolExecutor. Race-tolerant folder creation with retries.
All infra values (project ID, bucket name, Drive folder ID, secret name) are read
from environment variables — no real identifiers anywhere in source.
Imports google.cloud / pydrive2 / oauth2client lazily inside this module only."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import secretmanager, storage
from oauth2client.service_account import ServiceAccountCredentials
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ---------------- Helper Functions ----------------

def list_drive_folders(drive_service, parent_id):
    """List only immediate subfolders in the Drive folder specified by parent_id."""
    query = (
        f"'{parent_id}' in parents and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    folder_list = drive_service.ListFile({'q': query}).GetList()
    return {folder['title']: folder['id'] for folder in folder_list}


def list_bucket_subfolders(bucket, prefix):
    """
    List immediate subfolders in the bucket "folder" specified by prefix.
    Returns a dict mapping subfolder name to its full prefix.
    For example, if prefix='' and a blob has prefix "folder1/", then it returns {'folder1': 'folder1/'}.
    """
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    subfolders = {}
    for page in blobs.pages:
        if page.prefixes:
            for sub_prefix in page.prefixes:
                # Remove the current prefix from sub_prefix to get just the immediate folder name.
                folder_name = sub_prefix[len(prefix):].rstrip('/')
                subfolders[folder_name] = sub_prefix
    return subfolders


def get_existing_folder_id(drive_service, folder_name, parent_id):
    """Check if a folder exists in Drive under the parent folder and return its ID."""
    query = (
        f"'{parent_id}' in parents and "
        f"title = '{folder_name}' and "
        f"mimeType = 'application/vnd.google-apps.folder' and "
        f"trashed = false"
    )
    folder_list = drive_service.ListFile({'q': query}).GetList()
    return folder_list[0]['id'] if folder_list else None


def create_folder_in_drive(drive_service, folder_name, parent_id, max_retries=3, delay_seconds=1):
    """Create a folder in Drive under the specified parent folder."""
    # Check several times to see if another process created it already.
    for attempt in range(max_retries):
        folder_id = get_existing_folder_id(drive_service, folder_name, parent_id)
        if folder_id:
            return folder_id
        if attempt < max_retries - 1:
            time.sleep(delay_seconds)

    folder_metadata = {
        'title': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [{'id': parent_id}]
    }
    try:
        folder = drive_service.CreateFile(folder_metadata)
        folder.Upload()
    except Exception as e:
        time.sleep(delay_seconds)
        folder_id = get_existing_folder_id(drive_service, folder_name, parent_id)
        if folder_id:
            return folder_id
        raise e

    # Double-check after creation.
    time.sleep(delay_seconds)
    check_folder_id = get_existing_folder_id(drive_service, folder_name, parent_id)
    if check_folder_id and check_folder_id != folder['id']:
        return check_folder_id

    return folder['id']


def upload_file_to_drive(drive_service, file_name, folder_id, file_content):
    """Upload a file to Google Drive."""
    temp_file_path = f'/tmp/{file_name}'
    with open(temp_file_path, 'wb') as f:
        f.write(file_content)
    file = drive_service.CreateFile({'parents': [{"id": folder_id}], 'title': file_name})
    file.SetContentFile(temp_file_path)
    file.Upload()
    os.remove(temp_file_path)


def sync_files_in_folder(drive_service, bucket, bucket_prefix, drive_folder_id, files_to_upload=None):
    """
    Sync only the files (non-recursive) from the bucket "folder" (prefix) to a Drive folder.
    If files_to_upload is provided, only the matching files (by name) are processed.
    """
    blobs = bucket.list_blobs(prefix=bucket_prefix, delimiter='/')
    files_to_process = [
        blob for blob in blobs
        if blob.name.endswith('.jpg') and (files_to_upload is None or blob.name.split('/')[-1] in files_to_upload)
    ]

    def process_blob(blob):
        try:
            local_storage_client = storage.Client()
            blob_data = blob.download_as_bytes(client=local_storage_client)
            file_name = blob.name.split('/')[-1]
            upload_file_to_drive(drive_service, file_name, drive_folder_id, blob_data)
        except Exception as e:
            print(f"Error processing file '{blob.name}': {str(e)}")

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(process_blob, blob): blob for blob in files_to_process}
        for future in as_completed(futures):
            if future.exception() is not None:
                print(f"Exception during file processing: {future.exception()}")


def get_drive_files(drive_service, folder_id):
    """Get a set of file names (titles) in a Drive folder."""
    query = (
        f"'{folder_id}' in parents and "
        f"mimeType != 'application/vnd.google-apps.folder' and "
        f"title contains '.jpg' and "
        f"trashed = false"
    )
    param = {'q': query, 'fields': 'items(title)'}
    file_list = drive_service.ListFile(param).GetList()
    return {file['title'] for file in file_list}


def get_bucket_files(bucket, folder_prefix):
    """Get a set of file names in a bucket folder (only .jpg files)."""
    blobs = bucket.list_blobs(prefix=folder_prefix, delimiter='/')
    return {blob.name.split('/')[-1] for blob in blobs if blob.name.endswith('.jpg')}


def compare_and_sync_folder(drive_service, bucket, drive_folder_id, bucket_prefix):
    """
    Recursively compare file counts and names between a Drive folder and the bucket folder (prefix).
    Any files present in the bucket but missing in Drive are uploaded.
    Then, the function recurses over each subfolder.
    """
    # Sync files in the current folder.
    drive_files = get_drive_files(drive_service, drive_folder_id)
    bucket_files = get_bucket_files(bucket, bucket_prefix)
    missing_in_drive = bucket_files - drive_files
    print(f"Comparing folder '{bucket_prefix}': Drive files = {len(drive_files)}, Bucket files = {len(bucket_files)}")
    if missing_in_drive:
        print(f"Syncing missing files in '{bucket_prefix}': {missing_in_drive}")
        sync_files_in_folder(drive_service, bucket, bucket_prefix, drive_folder_id, files_to_upload=missing_in_drive)

    # Recurse into subfolders.
    drive_subfolders = list_drive_folders(drive_service, drive_folder_id)
    bucket_subfolders = list_bucket_subfolders(bucket, bucket_prefix)
    for subfolder_name, subfolder_bucket_prefix in bucket_subfolders.items():
        if subfolder_name in drive_subfolders:
            compare_and_sync_folder(drive_service, bucket, drive_subfolders[subfolder_name], subfolder_bucket_prefix)
        else:
            print(f"Subfolder '{subfolder_name}' under '{bucket_prefix}' initial sync.")
            new_drive_subfolder_id = create_folder_in_drive(drive_service, subfolder_name, drive_folder_id)
            # Removed the extra sync call here.
            compare_and_sync_folder(drive_service, bucket, new_drive_subfolder_id, subfolder_bucket_prefix)


def get_folder_file_counts_and_compare(drive_service, base_folder_id, bucket):
    """
    Compare file counts between top-level Drive and bucket folders and sync differences.
    This function uses list_bucket_subfolders with an empty prefix ('') to get the top-level folders in the bucket.
    """
    drive_folders = list_drive_folders(drive_service, base_folder_id)
    bucket_folders = list_bucket_subfolders(bucket, '')

    # For folders that exist in the bucket:
    for folder_name, bucket_prefix in bucket_folders.items():
        if folder_name in drive_folders:
            compare_and_sync_folder(drive_service, bucket, drive_folders[folder_name], bucket_prefix)
        else:
            print(f"Folder '{folder_name}' initial sync.")
            new_drive_folder_id = create_folder_in_drive(drive_service, folder_name, base_folder_id)
            sync_files_in_folder(drive_service, bucket, bucket_prefix, new_drive_folder_id)
            compare_and_sync_folder(drive_service, bucket, new_drive_folder_id, bucket_prefix)


# ---------------- Cloud Service Preparation ----------------

def get_secret_by_name(project_id: str, secret_name: str) -> str:
    """Retrieve the secret value from Google Secret Manager."""
    sc = secretmanager.SecretManagerServiceClient()
    request = {"name": f"projects/{project_id}/secrets/{secret_name}/versions/latest"}
    token_resp = sc.access_secret_version(request)
    return token_resp.payload.data.decode("UTF-8")


def prepare_drive():
    """Prepare the Google Drive service."""
    try:
        gauth = GoogleAuth()
        project_id = os.environ.get('PROJECT_ID')
        secret_name = os.environ.get('GCP_SECRET_NAME')
        service_acc_dict = json.loads(get_secret_by_name(project_id, secret_name))
        SCOPES = ['https://www.googleapis.com/auth/drive']
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_dict(service_acc_dict, SCOPES)
        drive_service = GoogleDrive(gauth)
        base_folder_id = os.environ.get("DESTINATION_DRIVE_FOLDER_ID")
        if not base_folder_id:
            raise ValueError("DESTINATION_DRIVE_FOLDER_ID environment variable is not set")
        return drive_service, base_folder_id
    except Exception as e:
        print(f"Error initializing Google Drive service: {str(e)}")
        return None, None


def prepare_bucket():
    """Prepare the Google Cloud Storage bucket."""
    destination_bucket_name = os.environ.get('DESTINATION_BUCKET')
    if not destination_bucket_name:
        raise ValueError("DESTINATION_BUCKET environment variable is not set")
    storage_client = storage.Client()
    return storage_client.bucket(destination_bucket_name)


# ---------------- Main Sync Function ----------------

def sync_folders(event):
    """Main function to sync folders from bucket to Drive."""
    try:
        drive_service, base_folder_id = prepare_drive()
        if not drive_service or not base_folder_id:
            raise ValueError("Failed to initialize Google Drive service")
        bucket = prepare_bucket()
        get_folder_file_counts_and_compare(drive_service, base_folder_id, bucket)
        return {"status": "success"}
    except Exception as e:
        print(f"Error in sync_folders: {str(e)}")
        return {"status": "error", "message": str(e)}
