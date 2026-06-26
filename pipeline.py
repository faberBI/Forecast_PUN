
"""
pipeline.py — PUN Forecasting Pipeline BASE ONLY

- legge dataset_history.parquet da Dropbox
- training su TUTTO il database disponibile
- LightGBM + ForecasterDirect con iper-parametri fissi
- salva artefatti locali compatibili con app.py/Gradio
- upload artefatti su Dropbox
- scheduler opzionale: retrain automatico ogni giorno a mezzanotte Europe/Rome

Env Dropbox:
1) consigliato refresh token:
   DROPBOX_REFRESH_TOKEN, DROPBOX_APP_KEY, DROPBOX_APP_SECRET
2) oppure token classico:
   DROPBOX_TOKEN

Env opzionali:
- DROPBOX_DATASET_PATH=/forecast_pun/dataset_history.parquet
- DROPBOX_MODEL_DIR=/forecast_pun/models
- LOCAL_MODEL_DIR=models
- PIPELINE_ENABLE_SCHEDULER=1
- PIPELINE_RUN_NOW=1
"""

import io
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import dropbox
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pun_pipeline_base_only")

STEPS = 96
RANDOM_STATE = 42
DATA_FREQ = "15min"
TZ_NAME = "Europe/Rome"

DROPBOX_DATASET_PATH = os.environ.get(
    "DROPBOX_DATASET_PATH",
    "/forecast_pun/dataset_history.parquet",
)
DROPBOX_MODEL_DIR = os.environ.get(
    "DROPBOX_MODEL_DIR",
    "/forecast_pun/models",
).rstrip("/")
MODEL_DIR = Path(os.environ.get("LOCAL_MODEL_DIR", "models"))

BEST_PARAMS = {
            "learning_rate": 0.03561019439085495,
            "num_leaves": 22,
            "max_depth": 8,
            "n_estimators": 533,
            "subsample": 0.6849356442713105,
            "colsample_bytree": 0.6727299868828402,
            "lambda_l1": 0.9170225492671691,
            "lambda_l2": 6.0848448591907545,
            "min_child_samples": 47
        }

BEST_LAGS = list(range(1, 97))
BEST_SELECTED_EXOG = [
                     "minute",
                    "lag_2d",
                    "lag_7d",
                    "pun_ret_1h",
                    "pun_ret_1d",
                    "pun_ret_7d",
                    "momentum_4h",
                    "momentum_1d",
                    "bologna_temperature_2m",
                    "bari_wind_speed_80m",
                    "cloud_cover_mean",
                    "forecast_total_load_MW",
                    "actual_generation_GWh_hydro",
                    "load_ramp_1h",
                    "load_forecast_error",
                    "CALA_B16",
                    "CNOR_B16",
                    "CSUD_B16",
                    "NORD_B16",
                    "SARD_B16",
                    "SICI_B16",
                    "SUD_B16"
        ]  

RECENCY_MIN_WEIGHT = 0.5
RECENCY_MAX_WEIGHT = 1.5
PEAK_WINDOW_EXTRA = 1.20
MIDDAY_WINDOW_EXTRA = 1.35
EVENING_WINDOW_EXTRA = 1.50
MIDDAY_QOD = set(range(48, 69))
EVENING_QOD = set(range(76, 87))

def weight_func(index: pd.DatetimeIndex) -> np.ndarray:
    return np.ones(len(index))
   
def ensure_datetime_freq(obj, freq: str = DATA_FREQ, fill: bool = True):
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("L'oggetto deve avere DatetimeIndex.")
    out.index = pd.to_datetime(out.index)
    out = out.sort_index().asfreq(freq)
    if fill:
        out = out.ffill()
    return out


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def validate_dataframe(df: pd.DataFrame) -> None:
    if "PUN" not in df.columns:
        raise ValueError("Colonna 'PUN' mancante nel DataFrame.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("L'indice deve essere DatetimeIndex.")
    if len(df) < STEPS * 10:
        raise ValueError(f"Dataset troppo corto: {len(df)} righe. Minimo richiesto: {STEPS * 10}.")

    inferred = pd.infer_freq(df.index)
    if inferred not in ("15min", "15T", "900S"):
        logger.warning("Frequenza inferita=%s. Verrà forzata a %s.", inferred, DATA_FREQ)

    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        logger.warning("NaN presenti nelle colonne:\n%s", nan_cols.to_string())

    logger.info(
        "Validazione OK — righe=%d colonne=%d periodo=%s → %s",
        len(df), len(df.columns), df.index.min(), df.index.max()
    )


def resolve_selected_exog(df: pd.DataFrame) -> list[str]:
    if BEST_SELECTED_EXOG is not None:
        missing = [c for c in BEST_SELECTED_EXOG if c not in df.columns]
        if missing:
            raise ValueError(f"BEST_SELECTED_EXOG contiene colonne mancanti: {missing}")
        return list(BEST_SELECTED_EXOG)

    selected = [c for c in df.columns if c != "PUN"]
    if not selected:
        raise ValueError("Nessuna esogena trovata: servono colonne oltre a PUN.")
    return selected


def production_weight_func(index):
    index = pd.DatetimeIndex(index)
    qod = index.hour * 4 + (index.minute // 15)

    recency = np.linspace(RECENCY_MIN_WEIGHT, RECENCY_MAX_WEIGHT, len(index))
    hour_weight = np.ones(len(index), dtype=float)

    peak_mask = (index.hour >= 8) & (index.hour < 21)
    midday_mask = np.isin(qod, list(MIDDAY_QOD))
    evening_mask = np.isin(qod, list(EVENING_QOD))

    hour_weight[peak_mask] *= PEAK_WINDOW_EXTRA
    hour_weight[midday_mask] *= MIDDAY_WINDOW_EXTRA
    hour_weight[evening_mask] *= EVENING_WINDOW_EXTRA

    return (recency * hour_weight).astype(float)


def build_forecaster() -> ForecasterDirect:
    estimator = LGBMRegressor(**BEST_PARAMS)
    return ForecasterDirect(
        estimator=estimator,
        lags=BEST_LAGS,
        steps=STEPS,
        weight_func=production_weight_func,
    )


def make_dropbox_client(token: str | None = None) -> dropbox.Dropbox:
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")

    if refresh_token and app_key and app_secret:
        logger.info("Dropbox auth: refresh token OAuth")
        return dropbox.Dropbox(
            oauth2_refresh_token=refresh_token,
            app_key=app_key,
            app_secret=app_secret,
        )

    if token is None:
        token = os.environ.get("DROPBOX_TOKEN")

    if not token:
        raise RuntimeError(
            "Token Dropbox non trovato. Imposta DROPBOX_TOKEN oppure "
            "DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET."
        )

    logger.info("Dropbox auth: access token classico")
    return dropbox.Dropbox(oauth2_access_token=token)


def load_from_dropbox(dropbox_path: str = DROPBOX_DATASET_PATH,
                      token: str | None = None) -> pd.DataFrame:
    dbx = make_dropbox_client(token)

    try:
        dbx.files_get_metadata(dropbox_path)
    except Exception as e:
        raise RuntimeError(f"files_get_metadata FALLITA ({dropbox_path}): {e}") from e

    try:
        _, res = dbx.files_download(dropbox_path)
    except Exception as e:
        raise RuntimeError(f"files_download FALLITA ({dropbox_path}): {e}") from e

    try:
        df = pd.read_parquet(io.BytesIO(res.content))
    except Exception as e:
        raise RuntimeError(f"read_parquet FALLITA ({dropbox_path}): {e}") from e

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Il parquet Dropbox non contiene DatetimeIndex valido né colonna Datetime.")

    df.index = pd.to_datetime(df.index)
    df = ensure_datetime_freq(df)

    logger.info(
        "DataFrame caricato da Dropbox: righe=%d colonne=%d path=%s periodo=%s → %s",
        df.shape[0], df.shape[1], dropbox_path, df.index.min(), df.index.max()
    )
    return df


def fit_production_model(df: pd.DataFrame) -> tuple[ForecasterDirect, list[str]]:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = ensure_datetime_freq(df)
    validate_dataframe(df)

    selected_exog = resolve_selected_exog(df)
    y_full = pd.to_numeric(df["PUN"], errors="coerce").astype(float).ffill()
    exog_full = coerce_numeric(df[selected_exog]).ffill()

    if y_full.isna().any():
        raise ValueError("PUN contiene ancora NaN dopo ffill: controlla inizio serie/storico.")
    if exog_full.isna().any().any():
        bad = exog_full.columns[exog_full.isna().any()].tolist()
        raise ValueError(f"Exog contiene ancora NaN dopo ffill nelle colonne: {bad}")

    model = build_forecaster()
    model.fit(y=y_full, exog=exog_full)

    logger.info(
        "Production model allenato su TUTTO il dataset: righe=%d exog=%d periodo=%s → %s",
        len(y_full), len(selected_exog), y_full.index.min(), y_full.index.max()
    )
    return model, selected_exog


def save_models(model_prod,
                selected_exog: list[str],
                df: pd.DataFrame,
                save_dir: Path | str = MODEL_DIR) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_prod, save_dir / "model_prod.pkl")
    joblib.dump({}, save_dir / "local_cfg_prod.pkl")
    joblib.dump(selected_exog, save_dir / "selected_exog.pkl")
    joblib.dump([], save_dir / "residual_feature_cols.pkl")

    metadata = {
        "trained_at": datetime.now(ZoneInfo(TZ_NAME)).isoformat(),
        "timezone": TZ_NAME,
        "mode": "base_only_full_dataset_fixed_params",
        "dataset_rows": int(len(df)),
        "dataset_start": df.index.min().isoformat() if len(df) else None,
        "dataset_end": df.index.max().isoformat() if len(df) else None,
        "data_freq": DATA_FREQ,
        "steps": STEPS,
        "best_params": BEST_PARAMS,
        "best_lags": list(BEST_LAGS),
        "n_selected_exog": len(selected_exog),
        "selected_exog": selected_exog,
        "residual_model_used": False,
        "optuna_used": False,
        "dropbox_dataset_path": DROPBOX_DATASET_PATH,
        "dropbox_model_dir": DROPBOX_MODEL_DIR,
    }

    with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Artefatti salvati localmente in: %s", save_dir.resolve())
    return save_dir


def upload_models_to_dropbox(local_model_dir: str | Path = MODEL_DIR,
                             dropbox_model_dir: str = DROPBOX_MODEL_DIR) -> None:
    dbx = make_dropbox_client()
    local_model_dir = Path(local_model_dir)

    required = [
        "model_prod.pkl",
        "local_cfg_prod.pkl",
        "selected_exog.pkl",
        "residual_feature_cols.pkl",
        "metadata.json",
    ]

    for name in required:
        local_file = local_model_dir / name
        if not local_file.exists():
            raise FileNotFoundError(f"Artefatto mancante: {local_file}")

        dropbox_path = f"{dropbox_model_dir}/{name}"
        with open(local_file, "rb") as f:
            dbx.files_upload(
                f.read(),
                dropbox_path,
                mode=dropbox.files.WriteMode.overwrite,
            )
        logger.info("Upload %s → %s", name, dropbox_path)


def load_models(save_dir: Path | str = MODEL_DIR) -> dict:
    save_dir = Path(save_dir)
    metadata_path = save_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "model_prod": joblib.load(save_dir / "model_prod.pkl"),
        "local_cfg_prod": joblib.load(save_dir / "local_cfg_prod.pkl"),
        "selected_exog": joblib.load(save_dir / "selected_exog.pkl"),
        "residual_feature_cols": joblib.load(save_dir / "residual_feature_cols.pkl"),
        "metadata": metadata,
    }


def run_retrain_once() -> dict:
    logger.info("========== START RETRAIN FULL DATASET ==========")
    logger.info("Dataset Dropbox: %s", DROPBOX_DATASET_PATH)
    logger.info("Model Dropbox dir: %s", DROPBOX_MODEL_DIR)

    df = load_from_dropbox(DROPBOX_DATASET_PATH)
    model_prod, selected_exog = fit_production_model(df)
    local_dir = save_models(model_prod, selected_exog, df, MODEL_DIR)
    upload_models_to_dropbox(local_dir, DROPBOX_MODEL_DIR)

    result = {
        "status": "ok",
        "trained_at": datetime.now(ZoneInfo(TZ_NAME)).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_start": df.index.min().isoformat(),
        "dataset_end": df.index.max().isoformat(),
        "n_selected_exog": len(selected_exog),
        "local_model_dir": str(Path(local_dir).resolve()),
        "dropbox_model_dir": DROPBOX_MODEL_DIR,
    }
    logger.info("Retrain completato: %s", json.dumps(result, ensure_ascii=False))
    logger.info("========== END RETRAIN FULL DATASET ==========")
    return result


def seconds_until_next_midnight(tz_name: str = TZ_NAME) -> float:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    next_midnight = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return max(0.0, (next_midnight - now).total_seconds())


def run_daily_midnight_scheduler(run_immediately: bool = False) -> None:
    logger.info("Scheduler attivo: retrain ogni giorno a mezzanotte timezone=%s", TZ_NAME)

    if run_immediately:
        try:
            run_retrain_once()
        except Exception:
            logger.exception("Retrain immediato fallito")

    while True:
        sleep_s = seconds_until_next_midnight(TZ_NAME)
        logger.info("Prossimo retrain a mezzanotte Europe/Rome. Sleep %.0f secondi.", sleep_s)
        time.sleep(sleep_s)

        try:
            run_retrain_once()
        except Exception:
            logger.exception("Retrain schedulato fallito")

        time.sleep(60)


# Compatibilità con eventuale app.py esistente
def run_pipeline(df: pd.DataFrame | None = None) -> dict:
    if df is None:
        return run_retrain_once()

    model_prod, selected_exog = fit_production_model(df)
    local_dir = save_models(model_prod, selected_exog, df, MODEL_DIR)
    upload_models_to_dropbox(local_dir, DROPBOX_MODEL_DIR)

    return {
        "model_final": model_prod,
        "model_prod": model_prod,
        "local_cfg": {},
        "local_cfg_prod": {},
        "df_resid": pd.DataFrame(),
        "residual_feature_cols": [],
        "best_params": BEST_PARAMS.copy(),
        "best_lags": BEST_LAGS.copy(),
        "selected_lags": BEST_LAGS.copy(),
        "selected_exog": selected_exog,
        "optuna_study": None,
        "test_results": pd.DataFrame(),
        "metrics_base": pd.Series(dtype=float),
        "metrics_corr": pd.Series(dtype=float),
        "diagnostics": {},
        "train_predictions": pd.DataFrame(),
        "local_model_dir": str(Path(local_dir).resolve()),
        "dropbox_model_dir": DROPBOX_MODEL_DIR,
    }


if __name__ == "__main__":
    enable_scheduler = os.environ.get("PIPELINE_ENABLE_SCHEDULER", "0").strip() == "1"
    run_now = os.environ.get("PIPELINE_RUN_NOW", "1").strip() == "1"

    if enable_scheduler:
        run_daily_midnight_scheduler(run_immediately=run_now)
    else:
        run_retrain_once()
