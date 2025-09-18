#!/usr/bin/env python3
# drive_script.py

import os, io, json
from pathlib import Path
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import argparse

# -------------------------
# Google Drive utilities
# -------------------------

def get_drive_service():
    creds_dict = json.loads(os.environ["GDRIVE_CREDENTIALS"])
    creds = service_account.Credentials.from_service_account_info(
        creds_dict,
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    return build("drive", "v3", credentials=creds)

def download_file(service, file_id, local_path):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(local_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    print(f"Downloaded {local_path}")

def upload_file(service, local_path, file_id):
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)
    updated = service.files().update(fileId=file_id, media_body=media).execute()
    print(f"Uploaded {updated['name']} to Drive")

# -------------------------
# Flatten CSV logic
# -------------------------

def find_col(df, target, fallbacks):
    cols = {c.lower(): c for c in df.columns}
    for name in [target, *fallbacks]:
        c = cols.get(name.lower())
        if c:
            return c
    raise ValueError(f"Required column '{target}' not found. Available: {list(df.columns)}")

def flatten_file(input_path: Path, output_path: Path, chunk_size: int = 30000, window: int = 30):
    if chunk_size % window != 0:
        raise ValueError(f"chunk_size ({chunk_size}) must be a multiple of window ({window}).")

    flattened_rows = []

    for chunk in pd.read_csv(input_path, chunksize=chunk_size):
        x_col = find_col(chunk, "X", ["x", "acc_x"])
        y_col = find_col(chunk, "Y", ["y", "acc_y"])
        z_col = find_col(chunk, "Z", ["z", "acc_z"])
        t_col = find_col(chunk, "time", ["Time", "timestamp", "datetime"])

        chunk = chunk[[x_col, y_col, z_col, t_col]].dropna()
        n = len(chunk)
        if n < window:
            continue

        x = chunk[x_col].to_numpy()
        y = chunk[y_col].to_numpy()
        z = chunk[z_col].to_numpy()
        t = chunk[t_col].to_numpy()

        usable = (n // window) * window
        x = x[:usable]; y = y[:usable]; z = z[:usable]; t = t[:usable]

        xw = x.reshape(-1, window)
        yw = y.reshape(-1, window)
        zw = z.reshape(-1, window)
        tw = t.reshape(-1, window)

        for i in range(xw.shape[0]):
            row = {}
            for j in range(window):
                row[f"x_{j+1}"] = xw[i, j]
                row[f"y_{j+1}"] = yw[i, j]
                row[f"z_{j+1}"] = zw[i, j]
            row["Time"] = tw[i, -1]
            flattened_rows.append(row)

    if not flattened_rows:
        cols = [*(f"x_{i}" for i in range(1, window+1)),
                *(f"y_{i}" for i in range(1, window+1)),
                *(f"z_{i}" for i in range(1, window+1)),
                "Time"]
        pd.DataFrame(columns=cols).to_csv(output_path, index=False)
        return

    pd.DataFrame(flattened_rows).to_csv(output_path, index=False)

# -------------------------
# Main workflow
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Download CSV from Drive, flatten, and upload back.")
    ap.add_argument("--file-id", required=True, help="Google Drive file ID of CSV")
    ap.add_argument("--chunk-size", type=int, default=30000, help="Rows per chunk (multiple of window)")
    ap.add_argument("--window", type=int, default=30, help="Window size")
    args = ap.parse_args()

    service = get_drive_service()
    local_in = Path("data.csv")
    local_out = Path("data_flattened.csv")

    # Download CSV
    download_file(service, args.file_id, local_in)

    # Flatten CSV
    flatten_file(local_in, local_out, chunk_size=args.chunk_size, window=args.window)
