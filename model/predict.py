import os, io, re, json, base64
import numpy as np
import pandas as pd

# Google Drive
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

# ML
from tensorflow.keras.models import load_model

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def drive_client(sa_path: str = "sa.json"):
    """
    Prefer GOOGLE_SERVICE_ACCOUNT_JSON_B64 (base64-encoded key) for CI.
    Fall back to reading sa.json from disk for local/dev.
    """
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        info = json.loads(base64.b64decode(b64))
        creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


def find_csv_in_folder(service, folder_id: str, filename: str = ""):
    """
    Return (file_id, file_name). If filename empty, pick most recently modified CSV in the folder.
    """
    if filename:
        q = f"'{folder_id}' in parents and name = '{filename}' and mimeType = 'text/csv' and trashed = false"
    else:
        q = f"'{folder_id}' in parents and mimeType = 'text/csv' and trashed = false"

    resp = service.files().list(
        q=q, orderBy="modifiedTime desc",
        fields="files(id,name,modifiedTime,size,mimeType)"
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError("No matching CSV files found in the specified folder.")
    return files[0]["id"], files[0]["name"]


def download_file(service, file_id: str, local_path: str):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, mode="wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_file(service, folder_id: str, local_path: str, upload_name: str):
    meta = {"name": upload_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    return service.files().create(body=meta, media_body=media, fields="id,name,webViewLink").execute()


def detect_xyz_columns(columns):
    """
    Detect contiguous triplets x_i,y_i,z_i for i starting at 1.
    Returns (T, xs, ys, zs).
    """
    if not any(re.fullmatch(r"[xyz]_\d+", c) for c in columns):
        return 0, [], [], []
    T, xs, ys, zs = 0, [], [], []
    i = 1
    while True:
        x, y, z = f"x_{i}", f"y_{i}", f"z_{i}"
        if x in columns and y in columns and z in columns:
            xs.append(x); ys.append(y); zs.append(z)
            T += 1; i += 1
        else:
            break
    return T, xs, ys, zs


def build_input_tensor(df: pd.DataFrame):
    """
    If a wide format (x_1..z_T) is found, return (X, T) with shape [N, T, 3].
    Drops obvious metadata columns if present.
    """
    # drop metadata-like columns if present (adjust if your CSV differs)
    for c in ["sheep number", "real Time"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    cols = list(df.columns)
    T, xs, ys, zs = detect_xyz_columns(cols)
    if T < 3:
        raise ValueError("Could not auto-detect x_i,y_i,z_i columns like x_1..z_T.")

    N = len(df)
    Xx = df[xs].to_numpy(dtype=np.float32).reshape(N, T)
    Xy = df[ys].to_numpy(dtype=np.float32).reshape(N, T)
    Xz = df[zs].to_numpy(dtype=np.float32).reshape(N, T)
    X = np.stack([Xx, Xy, Xz], axis=2)  # [N, T, 3]
    return X, T, df  # return df (potentially column-dropped) for consistency


def main():
    folder_id     = os.getenv("INPUT_DRIVE_FOLDER_ID")
    csv_filename  = os.getenv("INPUT_CSV_FILENAME", "").strip()
    model_path    = os.getenv("INPUT_MODEL_PATH")
    class_labels  = [c.strip() for c in os.getenv("INPUT_CLASS_LABELS", "").split(",") if c.strip()]

    if not folder_id:
        raise RuntimeError("INPUT_DRIVE_FOLDER_ID is required.")
    if not model_path:
        raise RuntimeError("INPUT_MODEL_PATH is required (e.g., sheepApp/model/ram_blstm_model.h5).")

    service = drive_client()
    file_id, file_name = find_csv_in_folder(service, folder_id, csv_filename)
    print(f"Using CSV: {file_name} (id={file_id})")

    # Download CSV
    in_csv = "input.csv"
    download_file(service, file_id, in_csv)

    # Load data
    df_raw = pd.read_csv(in_csv)
    df_out = df_raw.copy()  # keep original columns for output

    # Build model input
    X, T, _ = build_input_tensor(df_raw)
    print(f"Detected wide xyz format with T={T}; input tensor shape: {X.shape}")

    # Load model and predict
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    raw = model.predict(X, verbose=0)

    # Map predictions to labels
    if raw.ndim == 2 and raw.shape[1] > 1:
        n_classes = raw.shape[1]
        idx = np.argmax(raw, axis=1)
        if class_labels and len(class_labels) == n_classes:
            labels = [class_labels[int(i)] for i in idx]
        else:
            if class_labels and len(class_labels) != n_classes:
                print(f"Provided class_labels ({len(class_labels)}) != model outputs ({n_classes}); using class_0..")
            labels = [f"class_{int(i)}" for i in idx]
    else:
        idx = (raw.ravel() >= 0.5).astype(int)
        if class_labels and len(class_labels) >= 2:
            labels = [class_labels[int(i)] for i in idx]
        else:
            labels = [f"class_{int(i)}" for i in idx]

    # Save + upload
    df_out["predict"] = labels
    base, ext = os.path.splitext(file_name)
    out_name = f"{base}_predicted.csv"
    df_out.to_csv(out_name, index=False)
    print(f"Saved predictions to {out_name}")

    uploaded = upload_file(service, folder_id, out_name, out_name)
    print(f"Uploaded: {uploaded.get('name')} (id={uploaded.get('id')})")
    print(f"Web link: {uploaded.get('webViewLink')}")


if __name__ == "__main__":
    main()
