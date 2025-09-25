import os, io, re, json, base64
import numpy as np
import pandas as pd

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.errors import HttpError

from tensorflow.keras.models import load_model

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive"]


def drive_client(sa_path: str = "sa.json"):
    b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if b64:
        info = json.loads(base64.b64decode(b64))
        creds = service_account.Credentials.from_service_account_info(info, scopes=DRIVE_SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file(sa_path, scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


def find_csv_in_folder(service, folder_id: str, filename: str = ""):
    if filename:
        q = f"'{folder_id}' in parents and name = '{filename}' and mimeType = 'text/csv' and trashed = false"
    else:
        q = f"'{folder_id}' in parents and mimeType = 'text/csv' and trashed = false"
    resp = service.files().list(
        q=q, orderBy="modifiedTime desc",
        fields="files(id,name,modifiedTime,size,mimeType,parents)"
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError("No matching CSV files found in the specified folder.")
    f = files[0]
    return f["id"], f["name"]


def download_file(service, file_id: str, local_path: str):
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, mode="wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def upload_create(service, folder_id: str, local_path: str, upload_name: str):
    meta = {"name": upload_name, "parents": [folder_id]}
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    # supportsAllDrives helps when folder is a Shared Drive
    return service.files().create(
        body=meta, media_body=media, fields="id,name,webViewLink", supportsAllDrives=True
    ).execute()


def upload_update(service, file_id: str, local_path: str, new_name: str | None = None):
    meta = {"name": new_name} if new_name else None
    media = MediaIoBaseUpload(io.FileIO(local_path, "rb"), mimetype="text/csv", resumable=True)
    return service.files().update(
        fileId=file_id, body=meta, media_body=media,
        fields="id,name,webViewLink"
    ).execute()


def detect_xyz_columns(columns):
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

    # Build 5 channels: x, y, z, magnitude, delta_magnitude
    mag = np.sqrt(Xx**2 + Xy**2 + Xz**2)
    dmag = np.concatenate([np.zeros((N, 1), np.float32), np.diff(mag, axis=1).astype(np.float32)], axis=1)

    X = np.stack([Xx, Xy, Xz, mag, dmag], axis=2).astype(np.float32)  # [N, T, 5]
    return X, T


def main():
    folder_id     = os.getenv("INPUT_DRIVE_FOLDER_ID")
    csv_filename  = os.getenv("INPUT_CSV_FILENAME", "").strip()
    model_path    = os.getenv("INPUT_MODEL_PATH")
    class_labels  = [c.strip() for c in os.getenv("INPUT_CLASS_LABELS", "").split(",") if c.strip()]
    output_mode   = os.getenv("OUTPUT_MODE", "update").lower()  # 'update' (default) or 'create'

    if not folder_id:
        raise RuntimeError("INPUT_DRIVE_FOLDER_ID is required.")
    if not model_path:
        raise RuntimeError("INPUT_MODEL_PATH is required (e.g., model/ram_blstm_model.h5).")

    service = drive_client()
    file_id, file_name = find_csv_in_folder(service, folder_id, csv_filename)
    print(f"Using CSV: {file_name} (id={file_id})")

    # Download CSV
    in_csv = "input.csv"
    download_file(service, file_id, in_csv)

    # Load data
    df = pd.read_csv(in_csv)
    df_out = df.copy()

    # Build model input
    X, T = build_input_tensor(df)
    print(f"Detected wide xyz format with T={T}; input tensor shape: {X.shape}")

    # Load model and predict
    print(f"Loading model from {model_path}")
    if not os.path.exists(model_path):
        print("Current working dir:", os.getcwd())
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Check workflow input 'model_path'.")
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

    # Save output locally
    base, ext = os.path.splitext(file_name)
    out_name = f"{base}_predicted.csv"
    df_out["predict"] = labels
    df_out.to_csv(out_name, index=False)
    print(f"Saved predictions to {out_name}")

    # Upload logic
    try:
        if output_mode == "create":
            print("OUTPUT_MODE=create → attempting to create a new file...")
            created = upload_create(service, folder_id, out_name, out_name)
            print(f"Uploaded (create): {created.get('name')} (id={created.get('id')})")
            print(f"Web link: {created.get('webViewLink')}")
        else:
            print("OUTPUT_MODE=update → overwriting the original file content...")
            updated = upload_update(service, file_id, out_name, None)  # keep same name
            print(f"Uploaded (update): {updated.get('name')} (id={updated.get('id')})")
            print(f"Web link: {updated.get('webViewLink')}")
    except HttpError as e:
        # Fallback: if quota error on create, update the original instead
        if output_mode == "create" and e.resp.status == 403 and b"storageQuotaExceeded" in e.content:
            print("Create failed due to service account quota. Falling back to update-in-place...")
            updated = upload_update(service, file_id, out_name, None)
            print(f"Uploaded (update fallback): {updated.get('name')} (id={updated.get('id')})")
            print(f"Web link: {updated.get('webViewLink')}")
        else:
            raise


if __name__ == "__main__":
    main()
