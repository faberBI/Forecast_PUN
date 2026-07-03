# ============================================================
# gdrive_io.py — accesso Google Drive "per percorso" (nomi cartelle)
# ============================================================
# Google Drive usa ID, non path. Questo layer traduce un percorso a nomi
# (es. "PUN Forecast/forecast_pun/forecast_mi/calabria/dataset.parquet")
# in ID, navigando l'albero, e offre download/upload/exists/read_parquet.
#
# AUTENTICazione: service account.
#   - crea un service account su Google Cloud, scarica il JSON
#   - CONDIVIDI la cartella "PUN Forecast" (o una sua antenata) con l'email
#     del service account (client_email), permesso Editor
#   - passa il JSON: get_service_from_info(dict) (da st.secrets) oppure
#     get_service_from_file("service_account.json") (per gli script locali)
#
# Dipendenze: google-api-python-client, google-auth
# (import "pigri" dentro le funzioni, così il modulo si importa anche senza
#  le librerie per testare la sola logica di navigazione.)
# ============================================================

import os
import re
import tempfile
from typing import Optional

FOLDER_MIME = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]


# ------------------------------------------------------------
# AUTH
# ------------------------------------------------------------
def get_service_from_info(info: dict):
    """Costruisce il client Drive da un dict di credenziali service account."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_info(dict(info), scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_service_from_file(path: str):
    """Costruisce il client Drive da un file JSON service account (script locali)."""
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_file(path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_service_from_oauth(client_id: str, client_secret: str, refresh_token: str):
    """Costruisce il client Drive con credenziali UTENTE (OAuth).
    L'app agisce come l'utente: i file sono di sua proprietà (usa la sua quota),
    quindi CREA e MODIFICA funzionano su Drive personale."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_service_from_secrets(secrets):
    """Sceglie l'auth dai secrets: preferisce OAuth utente (gcp_oauth),
    altrimenti service account (gcp_service_account).
    NB: su Drive personale il service account NON può creare file (no quota);
    usa gcp_oauth."""
    if "gcp_oauth" in secrets:
        o = secrets["gcp_oauth"]
        return get_service_from_oauth(o["client_id"], o["client_secret"], o["refresh_token"])
    if "gcp_service_account" in secrets:
        return get_service_from_info(dict(secrets["gcp_service_account"]))
    raise RuntimeError(
        "Credenziali Google mancanti: aggiungi [gcp_oauth] (consigliato) "
        "oppure [gcp_service_account] nei secrets."
    )


# ------------------------------------------------------------
# NAVIGAZIONE PATH -> ID
# ------------------------------------------------------------
def _esc(name: str) -> str:
    return name.replace("\\", "\\\\").replace("'", "\\'")


def _find_child(service, name: str, parent_id: Optional[str], is_folder: Optional[bool]):
    """Trova un figlio per nome sotto parent_id (None = ricerca globale, per la radice condivisa)."""
    q = [f"name = '{_esc(name)}'", "trashed = false"]
    if parent_id:
        q.append(f"'{parent_id}' in parents")
    if is_folder is True:
        q.append(f"mimeType = '{FOLDER_MIME}'")
    elif is_folder is False:
        q.append(f"mimeType != '{FOLDER_MIME}'")

    res = service.files().list(
        q=" and ".join(q),
        spaces="drive",
        fields="files(id, name, mimeType)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        pageSize=10,
    ).execute()
    files = res.get("files", [])
    return files[0]["id"] if files else None


def _create_folder(service, name: str, parent_id: Optional[str]) -> str:
    meta = {"name": name, "mimeType": FOLDER_MIME}
    if parent_id:
        meta["parents"] = [parent_id]
    f = service.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return f["id"]


def resolve_folder(service, path: str, create: bool = False) -> Optional[str]:
    """Percorso di sole cartelle -> ID dell'ultima cartella. create=True le crea se mancanti."""
    parts = [p for p in path.strip("/").split("/") if p]
    parent = None
    for part in parts:
        fid = _find_child(service, part, parent_id=parent, is_folder=True)
        if fid is None:
            if not create:
                return None
            fid = _create_folder(service, part, parent)
        parent = fid
    return parent


def _split(path: str):
    parts = [p for p in path.strip("/").split("/") if p]
    return "/".join(parts[:-1]), parts[-1]


def resolve_file(service, path: str) -> Optional[str]:
    """Percorso completo (con nome file) -> ID del file, oppure None."""
    folder, fname = _split(path)
    parent = resolve_folder(service, folder, create=False) if folder else None
    if folder and parent is None:
        return None
    return _find_child(service, fname, parent_id=parent, is_folder=False)


def path_exists(service, path: str) -> bool:
    return resolve_file(service, path) is not None


# ------------------------------------------------------------
# DOWNLOAD / UPLOAD
# ------------------------------------------------------------
def download_file(service, path: str, local_path: str) -> str:
    from googleapiclient.http import MediaIoBaseDownload
    fid = resolve_file(service, path)
    if fid is None:
        raise FileNotFoundError(f"Drive: file non trovato: {path}")
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    req = service.files().get_media(fileId=fid, supportsAllDrives=True)
    with open(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return local_path


def upload_file(service, local_path: str, drive_path: str, overwrite: bool = True) -> str:
    """Carica local_path in drive_path (crea le cartelle mancanti). Sovrascrive se esiste."""
    from googleapiclient.http import MediaFileUpload
    folder, fname = _split(drive_path)
    parent = resolve_folder(service, folder, create=True)
    existing = _find_child(service, fname, parent_id=parent, is_folder=False)

    media = MediaFileUpload(local_path, resumable=True)
    if existing and overwrite:
        service.files().update(fileId=existing, media_body=media, supportsAllDrives=True).execute()
        return existing
    if existing and not overwrite:
        return existing
    meta = {"name": fname, "parents": [parent]}
    f = service.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    return f["id"]


# ------------------------------------------------------------
# COMODITÀ PANDAS
# ------------------------------------------------------------
def read_parquet(service, path: str):
    import pandas as pd
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    try:
        download_file(service, path, tmp.name)
        return pd.read_parquet(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
