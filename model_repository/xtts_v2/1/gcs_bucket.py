import re
import os
from google.cloud import storage

# Initialize the GCS client
storage_client = storage.Client.from_service_account_json('/opt/tritonserver/model_repository/xtts_v2/service_account.json')

def parse_gcs_path(gcs_path):
    """Parse a GCS path into bucket name and blob prefix."""
    match = re.match(r'gs://([^/]+)/(.+)', gcs_path)
    if not match:
        raise Exception(f"Invalid GCS path: {gcs_path}")
    return match.group(1), match.group(2)

def download_all_files_in_folder(gcs_path, destination_folder):
    """Downloads all files from a GCS folder to a local destination folder."""
    # Parse the GCS path
    bucket_name, prefix = parse_gcs_path(gcs_path)

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List blobs in the folder
    blobs = bucket.list_blobs(prefix=prefix)

    # Download each blob
    for blob in blobs:
        # Skip "directory" blobs (names ending in `/`)
        if blob.name.endswith('/'):
            continue

        # Construct the local file path
        relative_path = os.path.relpath(blob.name, prefix)
        destination_file_name = os.path.join(destination_folder, relative_path)

        # Ensure the local directory exists
        destination_dir = os.path.dirname(destination_file_name)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        # Download the blob to the local file
        print(f"Downloading {blob.name} to {destination_file_name}")
        blob.download_to_filename(destination_file_name)

    print("Download complete.")