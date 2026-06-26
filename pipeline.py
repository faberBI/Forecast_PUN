import io
import json
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

import dropbox
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect


# =========================================================
# CONFIG
# =========================================================
STEPS = 96
DATA_FREQ = "15min"

DROPBOX_DATASET_PATH = "/forecast_pun/dataset_history.parquet"
DROPBOX_MODEL_DIR = "/forecast_pun/models"

DROPBOX_METADATA_PATH = f"{DROPBOX_MODEL_DIR}/metadata.json"
DROPBOX_MODEL_PATH = f"{DROPBOX_MODEL_DIR}/model_prod.pkl"

LOCAL_MODEL_DIR = Path("models")
LOCAL_MODEL_PATH = LOCAL_MODEL_DIR / "model_prod.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pun_pipeline_streamlit")


# =========================================================
# EXTRA FEATURES (COME HF)
# =========================================================
EXTRA_SPIKE_FEATURES = [
    "hour",
    "minute_of_day",
    "is_evening",
    "is_peak_hour",
    "lag_1d_same_hour",
    "lag_2d_same_hour",
    "lag_7d_same_hour",
    "delta_1h_lag_1d",
    "delta_3h_lag_1d",
    "delta_6h_lag_1d",
    "rolling_mean_6h_lag_1d",
    "rolling_std_6h_lag_1d",
    "rolling_max_1d_lag_1d",
    "load_ratio_1d",
    "load_ramp_1h_safe",
    "load_ramp_3h_safe",
    "generation_ratio_1d",
    "hydro_generation_ratio_1d",
]


# =========================================================
# WEIGHT (UGUALE HF)
# =========================================================
def weight_func(index: pd.DatetimeIndex):
    index = pd.DatetimeIndex(index)
    w = np.ones(len(index))

    hours = index.hour

    w[hours >= 18] = 2
    w[np.isin(hours, [18, 19, 20, 21])] = 4

    return w


# =========================================================
# DROPBOX
# =========================================================
def get_dbx():
    token = os.environ.get("DROPBOX_TOKEN")

    if not token:
        raise RuntimeError("DROPBOX_TOKEN mancante")

    return dropbox.Dropbox(token)


def upload_bytes(content: bytes, path: str):
    dbx = get_dbx()
    dbx.files_upload(content, path, mode=dropbox.files.WriteMode.overwrite)


# =========================================================
# DATETIME
# =========================================================
def normalize_datetime_index(df):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
        else:
            raise ValueError("Datetime mancante")

    df = df.sort_index()
    df = df.asfreq(DATA_FREQ)

    return df


# =========================================================
# SPIKE FEATURES (UGUALE HF)
# =========================================================
def add_pun_spike_features(df):

    df = df.copy()

    df["PUN"] = pd.to_numeric(df["PUN"], errors="coerce")

    df["hour"] = df.index.hour
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["is_evening"] = (df.index.hour >= 18).astype(int)
    df["is_peak_hour"] = df.index.hour.isin([18, 19, 20, 21]).astype(int)

    df["minute"] = df.index.minute

    df["lag_1d_same_hour"] = df["PUN"].shift(96)
    df["lag_2d_same_hour"] = df["PUN"].shift(96 * 2)
    df["lag_7d_same_hour"] = df["PUN"].shift(96 * 7)

    df["delta_1h_lag_1d"] = df["PUN"].shift(96) - df["PUN"].shift(96 + 4)
    df["delta_3h_lag_1d"] = df["PUN"].shift(96) - df["PUN"].shift(96 + 12)
    df["delta_6h_lag_1d"] = df["PUN"].shift(96) - df["PUN"].shift(96 + 24)

    lag = df["PUN"].shift(96)

    df["rolling_mean_6h_lag_1d"] = lag.rolling(24, min_periods=4).mean()
    df["rolling_std_6h_lag_1d"] = lag.rolling(24, min_periods=4).std()
    df["rolling_max_1d_lag_1d"] = lag.rolling(96, min_periods=24).max()

    if "forecast_total_load_MW" in df.columns:
        load = pd.to_numeric(df["forecast_total_load_MW"], errors="coerce")
        roll = load.rolling(96, min_periods=24).mean()

        df["load_ratio_1d"] = load / roll
        df["load_ramp_1h_safe"] = load - load.shift(4)
        df["load_ramp_3h_safe"] = load - load.shift(12)

    if "actual_generation_GWh_hydro" in df.columns:
        hydro = pd.to_numeric(df["actual_generation_GWh_hydro"], errors="coerce")
        roll = hydro.rolling(96, min_periods=24).mean()
        df["hydro_generation_ratio_1d"] = hydro / roll

    return df


# =========================================================
# LOAD DATA
# =========================================================
def load_dataset():
    dbx = get_dbx()
    _, res = dbx.files_download(DROPBOX_DATASET_PATH)

    df = pd.read_parquet(io.BytesIO(res.content))
    df = normalize_datetime_index(df)

    return df


# =========================================================
# PREPARE
# =========================================================
def prepare_xy(df, selected_exog):

    df = add_pun_spike_features(df)

    selected_exog = list(dict.fromkeys(selected_exog + EXTRA_SPIKE_FEATURES))
    selected_exog = [c for c in selected_exog if c in df.columns]

    y = df["PUN"].astype(float)

    exog = (
        df[selected_exog]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
    )

    return y, exog, selected_exog


# =========================================================
# MODEL
# =========================================================
def build_model(params, lags):

    params = params.copy()
    params["objective"] = "quantile"
    params["alpha"] = 0.9

    return ForecasterDirect(
        estimator=LGBMRegressor(**params),
        lags=lags,
        steps=STEPS,
        weight_func=weight_func
    )


# =========================================================
# TRAIN
# =========================================================
def train_model(df):

    dbx = get_dbx()
    _, res = dbx.files_download(DROPBOX_METADATA_PATH)
    meta = json.loads(res.content.decode())

    params = meta["best_params"]
    lags = meta["best_lags"]
    selected_exog = meta["selected_exog"]

    y, exog, selected_exog = prepare_xy(df, selected_exog)

    model = build_model(params, lags)

    model.fit(y=y, exog=exog)

    return model, selected_exog


# =========================================================
# SAVE
# =========================================================
def save_model(model, selected_exog, params, lags, df):

    LOCAL_MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, LOCAL_MODEL_PATH)

    metadata = {
        "best_params": params,
        "best_lags": lags,
        "selected_exog": selected_exog,
        "trained_at": datetime.now(ZoneInfo("Europe/Rome")).isoformat(),
    }

    upload_bytes(
        json.dumps(metadata).encode(),
        DROPBOX_METADATA_PATH
    )

    with open(LOCAL_MODEL_PATH, "rb") as f:
        upload_bytes(f.read(), DROPBOX_MODEL_PATH)


# =========================================================
# RUN
# =========================================================
def run():

    df = load_dataset()

    model, selected_exog = train_model(df)

    print("✅ TRAIN COMPLETATO")

    return model


if __name__ == "__main__":
    run()
