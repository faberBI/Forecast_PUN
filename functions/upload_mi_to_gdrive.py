# ============================================================
# upload_mi_to_gdrive.py — carica dataset + modelli MI su GOOGLE DRIVE
# ============================================================
# Mette i file LOCALI ai path che la sezione MI dell'app si aspetta su Drive:
#   dataset : PUN Forecast/forecast_pun/forecast_mi/<zona>/dataset.parquet
#   modello : PUN Forecast/forecast_pun/forecast_mi/models_mi/<zona>/<file>
# (le cartelle mancanti su Drive vengono create automaticamente)
#
# AUTENTICAZIONE: service account JSON. Condividi "PUN Forecast" con la
# client_email del service account (Editor).
#
# USO:
#   python upload_mi_to_gdrive.py --creds service_account.json --dry-run
#   python upload_mi_to_gdrive.py --creds service_account.json
#   python upload_mi_to_gdrive.py --creds service_account.json --only nord sud
#
# Dipendenze: google-api-python-client, google-auth
# ============================================================

import os
import sys
import argparse

import gdrive_io as gdrive

# ---- CONFIG (allinea a mi_section.py / run_local.py) ----
GDRIVE_ROOT = "PUN Forecast/forecast_pun/forecast_mi"
LOCAL_MODELS_ROOT = "models_mi"     # cartelle prodotte da run_local.py
LOCAL_DATA_FOLDER = "MI"            # cartella con i MI_<zona>.parquet
DATASET_NAME = "dataset.parquet"

MODEL_FILES = ["mi_direct_lgbm_quantiles.joblib", "mi_direct_metadata.json"]
ALSO_UPLOAD_EVAL_CSV = False
EVAL_CSV = ["mi_direct_eval_pinball_by_hour.csv", "mi_direct_eval_coverage_by_hour.csv"]

ZONE_KEYS = [
    "italia_senza_vincoli", "calabria", "centro_nord", "centro_sud",
    "nord", "sardegna", "sicilia", "sud", "italia_coupling",
]


def zone_drive(zone_key: str) -> dict:
    base = f"{GDRIVE_ROOT}/{zone_key}"
    return {
        "dataset": f"{base}/{DATASET_NAME}",
        "model_dir": f"{GDRIVE_ROOT}/models_mi/{zone_key}",
    }


def plan_uploads(only=None):
    zones = ZONE_KEYS if not only else [z for z in ZONE_KEYS if z in only]
    jobs, missing = [], []
    for z in zones:
        d = zone_drive(z)

        ds_local = os.path.join(LOCAL_DATA_FOLDER, f"MI_{z}.parquet")
        (jobs if os.path.exists(ds_local) else missing).append((ds_local, d["dataset"]))

        files = MODEL_FILES + (EVAL_CSV if ALSO_UPLOAD_EVAL_CSV else [])
        for fn in files:
            m_local = os.path.join(LOCAL_MODELS_ROOT, z, fn)
            (jobs if os.path.exists(m_local) else missing).append((m_local, f"{d['model_dir']}/{fn}"))
    return jobs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creds", required=True, help="path al JSON del service account")
    ap.add_argument("--dry-run", action="store_true", help="elenca soltanto, non carica")
    ap.add_argument("--only", nargs="*", default=None, help="solo alcune zone (es. --only nord sud)")
    args = ap.parse_args()

    jobs, missing = plan_uploads(only=args.only)

    print("=" * 70)
    print("PIANO UPLOAD MI -> Google Drive")
    print(f"Root: {GDRIVE_ROOT}")
    print("=" * 70)
    for local, drive_path in jobs:
        mb = os.path.getsize(local) / 1e6
        print(f"  [OK] {local}  ->  {drive_path}   ({mb:.1f} MB)")
    if missing:
        print("\nFILE LOCALI MANCANTI (saltati):")
        for local, drive_path in missing:
            print(f"  [--] {local}   (atteso per {drive_path})")

    if args.dry_run:
        print("\n--dry-run: niente caricato.")
        return
    if not jobs:
        print("\nNessun file da caricare. Controlla LOCAL_MODELS_ROOT / LOCAL_DATA_FOLDER.")
        return
    if not os.path.exists(args.creds):
        print(f"\n❌ Credenziali non trovate: {args.creds}")
        sys.exit(1)

    svc = gdrive.get_service_from_file(args.creds)

    print("\nCarico su Drive...")
    ok = 0
    for local, drive_path in jobs:
        try:
            gdrive.upload_file(svc, local, drive_path, overwrite=True)
            print(f"  ✅ {drive_path}")
            ok += 1
        except Exception as e:
            print(f"  ❌ {drive_path} -> {type(e).__name__}: {e}")

    print(f"\nFatto: {ok}/{len(jobs)} file caricati.")


if __name__ == "__main__":
    main()
