import os, io, json
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# Load credentials from GitHub Secret
creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
creds = service_account.Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/drive"]
)

drive_service = build("drive", "v3", credentials=creds)

# File ID of your CSV in Drive
FILE_ID = "1i7L4oGAUodH1nVmLyyLtckiuZXcGA0N5"  # replace with your Drive file ID

# --- Download CSV ---
def download_file(file_id, local_path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded {local_path}")

# --- Upload CSV ---
def upload_file(local_path, file_id):
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    updated = drive_service.files().update(
        fileId=file_id, media_body=media
    ).execute()
    print(f"Updated file {updated['name']} in Drive")

# --- Main workflow ---
local_csv = "data.csv"
download_file(FILE_ID, local_csv)

# Example: edit CSV
df = pd.read_csv(local_csv)
df["new_col"] = df[df.columns[0]] * 2  # simple modification
df.to_csv(local_csv, index=False)

upload_file(local_csv, FILE_ID)
