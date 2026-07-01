# ============================================================
# PUN DIRECT 96 FORECAST
# LightGBM p50 + p90 + Optuna Hourly Blend Weights
# ============================================================

import os
import json
import math
import warnings
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    optuna = None
    HAS_OPTUNA = False

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

TARGET_COL = "PUN"
FREQ = "15min"
STEPS = 96

HOLDOUT_ORIGINS = 96

MODEL_P50_NAME = "pun_direct_lgbm_p50.joblib"
MODEL_P90_NAME = "pun_direct_lgbm_p90.joblib"
METADATA_NAME = "pun_direct_metadata.json"
EVAL_NAME = "pun_direct_evaluation.csv"
FORECAST_NAME = "pun_direct_forecast_next_96.csv"
BLEND_WEIGHTS_NAME = "pun_direct_blend_weights.csv"

FUTURE_KNOWN_EXOG = [
    "forecast_total_load_MW",
    "bologna_temperature_2m",
    "bari_wind_speed_80m",
    "cloud_cover_mean",
]

BLEND_CONFIG = {
    "method": "optuna_hourly",
    "default_w90": 0.0,
    "hourly_w90": {},
    "metric": "MAE",
    "n_trials": 80,
    "min_samples_per_hour": 20,
}

LGBM_P50_PARAMS = {
    "objective": "regression",
    "n_estimators": 533,
    "learning_rate": 0.03561019439085495,
    "num_leaves": 22,
    "max_depth": 5,
    "min_child_samples": 47,
    "subsample": 0.6849356442713105,
    "subsample_freq": 1,
    "colsample_bytree": 0.6727299868828402,
    "reg_alpha": 0.9170225492671691,
    "reg_lambda": 6.0848448591907545,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

LGBM_P90_PARAMS = {
    "objective": "quantile",
    "alpha": 0.90,
    "n_estimators": 533,
    "learning_rate": 0.03561019439085495,
    "num_leaves": 22,
    "max_depth": 5,
    "min_child_samples": 47,
    "subsample": 0.6849356442713105,
    "subsample_freq": 1,
    "colsample_bytree": 0.6727299868828402,
    "reg_alpha": 0.9170225492671691,
    "reg_lambda": 6.0848448591907545,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


# ============================================================
# BASIC UTILS
# ============================================================

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Il DataFrame deve avere un DatetimeIndex.")

    df = df.sort_index()

    if df.index.tz is not None:
        df.index = df.index.tz_convert("Europe/Rome").tz_localize(None)

    return df


def infer_and_fix_freq(df: pd.DataFrame, freq: str = FREQ) -> pd.DataFrame:
    df = df.copy().sort_index()

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
    )

    df = df.reindex(full_index)
    return df


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def get_last_valid_target_timestamp(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.Timestamp:
    valid = df[df[target_col].notna()]

    if valid.empty:
        raise ValueError(f"Nessun valore valido trovato nella colonna target '{target_col}'.")

    return valid.index.max()


def make_future_index_after(
    last_ts: pd.Timestamp,
    steps: int = STEPS,
    freq: str = FREQ,
) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_ts + pd.Timedelta(freq),
        periods=steps,
        freq=freq,
    )


def ensure_future_rows(
    df: pd.DataFrame,
    steps: int = STEPS,
    freq: str = FREQ,
) -> pd.DataFrame:
    df = df.copy().sort_index()

    last_valid_target_time = get_last_valid_target_timestamp(df, TARGET_COL)

    required_future_index = make_future_index_after(
        last_valid_target_time,
        steps=steps,
        freq=freq,
    )

    full_index = df.index.union(required_future_index)
    df = df.reindex(full_index).sort_index()

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_time_features_from_index(
    index: pd.DatetimeIndex,
    prefix: str = "",
) -> pd.DataFrame:
    x = pd.DataFrame(index=index)

    x[f"{prefix}hour"] = index.hour
    x[f"{prefix}minute"] = index.minute
    x[f"{prefix}quarter"] = index.minute // 15
    x[f"{prefix}quarter_of_day"] = x[f"{prefix}hour"] * 4 + x[f"{prefix}quarter"]

    x[f"{prefix}day_of_week"] = index.dayofweek
    x[f"{prefix}day_of_year"] = index.dayofyear
    x[f"{prefix}week_of_year"] = index.isocalendar().week.astype(int).to_numpy()
    x[f"{prefix}month"] = index.month
    x[f"{prefix}year"] = index.year

    x[f"{prefix}is_weekend"] = (x[f"{prefix}day_of_week"] >= 5).astype(int)

    x[f"{prefix}is_morning_peak"] = x[f"{prefix}hour"].between(8, 11).astype(int)
    x[f"{prefix}is_evening_peak"] = x[f"{prefix}hour"].between(18, 22).astype(int)
    x[f"{prefix}is_night"] = x[f"{prefix}hour"].between(0, 6).astype(int)
    x[f"{prefix}is_working_hour"] = x[f"{prefix}hour"].between(8, 19).astype(int)

    x[f"{prefix}hour_sin"] = np.sin(2 * np.pi * x[f"{prefix}hour"] / 24)
    x[f"{prefix}hour_cos"] = np.cos(2 * np.pi * x[f"{prefix}hour"] / 24)

    x[f"{prefix}qod_sin"] = np.sin(2 * np.pi * x[f"{prefix}quarter_of_day"] / 96)
    x[f"{prefix}qod_cos"] = np.cos(2 * np.pi * x[f"{prefix}quarter_of_day"] / 96)

    x[f"{prefix}dow_sin"] = np.sin(2 * np.pi * x[f"{prefix}day_of_week"] / 7)
    x[f"{prefix}dow_cos"] = np.cos(2 * np.pi * x[f"{prefix}day_of_week"] / 7)

    x[f"{prefix}month_sin"] = np.sin(2 * np.pi * x[f"{prefix}month"] / 12)
    x[f"{prefix}month_cos"] = np.cos(2 * np.pi * x[f"{prefix}month"] / 12)

    return x


def add_pun_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_datetime_index(df)

    time_df = add_time_features_from_index(df.index, prefix="")
    for c in time_df.columns:
        df[c] = time_df[c]

    lag_map = {
        "pun_lag_15m": 1,
        "pun_lag_30m": 2,
        "pun_lag_1h": 4,
        "pun_lag_2h": 8,
        "pun_lag_4h": 16,
        "pun_lag_8h": 32,
        "pun_lag_12h": 48,
        "pun_lag_1d": 96,
        "pun_lag_2d": 192,
        "pun_lag_3d": 288,
        "pun_lag_7d": 672,
    }

    if TARGET_COL in df.columns:
        for name, lag in lag_map.items():
            df[name] = df[TARGET_COL].shift(lag)

        df["pun_same_q_1d"] = df[TARGET_COL].shift(96)
        df["pun_same_q_2d"] = df[TARGET_COL].shift(192)
        df["pun_same_q_3d"] = df[TARGET_COL].shift(288)
        df["pun_same_q_7d"] = df[TARGET_COL].shift(672)

        same_q_cols = [
            "pun_same_q_1d",
            "pun_same_q_2d",
            "pun_same_q_3d",
            "pun_same_q_7d",
        ]

        df["pun_same_q_mean"] = df[same_q_cols].mean(axis=1)
        df["pun_same_q_std"] = df[same_q_cols].std(axis=1)
        df["pun_same_q_max"] = df[same_q_cols].max(axis=1)
        df["pun_same_q_min"] = df[same_q_cols].min(axis=1)

        s = df[TARGET_COL].shift(1)

        df["pun_roll_mean_1h"] = s.rolling(4, min_periods=2).mean()
        df["pun_roll_std_1h"] = s.rolling(4, min_periods=2).std()

        df["pun_roll_mean_4h"] = s.rolling(16, min_periods=4).mean()
        df["pun_roll_std_4h"] = s.rolling(16, min_periods=4).std()
        df["pun_roll_max_4h"] = s.rolling(16, min_periods=4).max()
        df["pun_roll_min_4h"] = s.rolling(16, min_periods=4).min()

        df["pun_roll_mean_1d"] = s.rolling(96, min_periods=24).mean()
        df["pun_roll_std_1d"] = s.rolling(96, min_periods=24).std()
        df["pun_roll_max_1d"] = s.rolling(96, min_periods=24).max()
        df["pun_roll_min_1d"] = s.rolling(96, min_periods=24).min()

        df["pun_roll_mean_7d"] = s.rolling(672, min_periods=96).mean()
        df["pun_roll_std_7d"] = s.rolling(672, min_periods=96).std()

        df["pun_diff_15m"] = df[TARGET_COL] - df[TARGET_COL].shift(1)
        df["pun_diff_1h"] = df[TARGET_COL] - df[TARGET_COL].shift(4)
        df["pun_diff_4h"] = df[TARGET_COL] - df[TARGET_COL].shift(16)
        df["pun_diff_1d"] = df[TARGET_COL] - df[TARGET_COL].shift(96)
        df["pun_diff_7d"] = df[TARGET_COL] - df[TARGET_COL].shift(672)

        df["pun_ret_15m_calc"] = df[TARGET_COL].pct_change(1)
        df["pun_ret_1h_calc"] = df[TARGET_COL].pct_change(4)
        df["pun_ret_1d_calc"] = df[TARGET_COL].pct_change(96)
        df["pun_ret_7d_calc"] = df[TARGET_COL].pct_change(672)

    if "forecast_total_load_MW" in df.columns:
        df["load_lag_1h"] = df["forecast_total_load_MW"].shift(4)
        df["load_lag_4h"] = df["forecast_total_load_MW"].shift(16)
        df["load_lag_1d"] = df["forecast_total_load_MW"].shift(96)
        df["load_lag_7d"] = df["forecast_total_load_MW"].shift(672)

        df["load_diff_1h"] = df["forecast_total_load_MW"] - df["forecast_total_load_MW"].shift(4)
        df["load_diff_4h"] = df["forecast_total_load_MW"] - df["forecast_total_load_MW"].shift(16)
        df["load_diff_1d"] = df["forecast_total_load_MW"] - df["forecast_total_load_MW"].shift(96)

        df["load_roll_mean_4h"] = (
            df["forecast_total_load_MW"]
            .shift(1)
            .rolling(16, min_periods=4)
            .mean()
        )

        df["load_roll_std_4h"] = (
            df["forecast_total_load_MW"]
            .shift(1)
            .rolling(16, min_periods=4)
            .std()
        )

        df["load_roll_mean_1d"] = (
            df["forecast_total_load_MW"]
            .shift(1)
            .rolling(96, min_periods=24)
            .mean()
        )

        df["load_roll_std_1d"] = (
            df["forecast_total_load_MW"]
            .shift(1)
            .rolling(96, min_periods=24)
            .std()
        )

    if "actual_generation_GWh_hydro" in df.columns:
        df["hydro_lag_1d"] = df["actual_generation_GWh_hydro"].shift(96)
        df["hydro_lag_2d"] = df["actual_generation_GWh_hydro"].shift(192)
        df["hydro_lag_7d"] = df["actual_generation_GWh_hydro"].shift(672)

        df["hydro_roll_mean_1d"] = (
            df["actual_generation_GWh_hydro"]
            .shift(1)
            .rolling(96, min_periods=24)
            .mean()
        )

    zone_cols = [
        "CALA_B16",
        "CNOR_B16",
        "CSUD_B16",
        "NORD_B16",
        "SARD_B16",
        "SICI_B16",
        "SUD_B16",
    ]

    existing_zone_cols = [c for c in zone_cols if c in df.columns]

    if existing_zone_cols:
        df["zone_mean"] = df[existing_zone_cols].mean(axis=1)
        df["zone_std"] = df[existing_zone_cols].std(axis=1)
        df["zone_max"] = df[existing_zone_cols].max(axis=1)
        df["zone_min"] = df[existing_zone_cols].min(axis=1)
        df["zone_spread"] = df["zone_max"] - df["zone_min"]

        for c in existing_zone_cols:
            df[f"{c}_lag_1d"] = df[c].shift(96)
            df[f"{c}_lag_7d"] = df[c].shift(672)

        if "NORD_B16" in df.columns and "SUD_B16" in df.columns:
            df["spread_nord_sud"] = df["NORD_B16"] - df["SUD_B16"]

        if "CNOR_B16" in df.columns and "CSUD_B16" in df.columns:
            df["spread_cnor_csud"] = df["CNOR_B16"] - df["CSUD_B16"]

    if "forecast_total_load_MW" in df.columns:
        df["evening_load"] = df["forecast_total_load_MW"] * df["is_evening_peak"]

    if "load_ramp_1h" in df.columns:
        df["evening_load_ramp"] = df["load_ramp_1h"] * df["is_evening_peak"]

    if "load_forecast_error" in df.columns:
        df["evening_load_forecast_error"] = df["load_forecast_error"] * df["is_evening_peak"]

    if "pun_roll_std_4h" in df.columns:
        df["evening_volatility_4h"] = df["pun_roll_std_4h"] * df["is_evening_peak"]

    if "pun_roll_std_1d" in df.columns:
        df["evening_volatility_1d"] = df["pun_roll_std_1d"] * df["is_evening_peak"]

    if "momentum_4h" in df.columns:
        df["evening_momentum_4h"] = df["momentum_4h"] * df["is_evening_peak"]

    if "momentum_1d" in df.columns:
        df["evening_momentum_1d"] = df["momentum_1d"] * df["is_evening_peak"]

    if "cloud_cover_mean" in df.columns:
        df["evening_cloud_cover"] = df["cloud_cover_mean"] * df["is_evening_peak"]

    if "bari_wind_speed_80m" in df.columns:
        df["evening_wind_bari"] = df["bari_wind_speed_80m"] * df["is_evening_peak"]

    if TARGET_COL in df.columns:
        rolling_q85 = (
            df[TARGET_COL]
            .shift(1)
            .rolling(96, min_periods=24)
            .quantile(0.85)
        )

        rolling_q90 = (
            df[TARGET_COL]
            .shift(1)
            .rolling(96, min_periods=24)
            .quantile(0.90)
        )

        df["recent_spike_q85_flag"] = (
            df[TARGET_COL].shift(1) > rolling_q85
        ).astype(int)

        df["recent_spike_q90_flag"] = (
            df[TARGET_COL].shift(1) > rolling_q90
        ).astype(int)

        df["recent_evening_spike_q85_flag"] = (
            df["recent_spike_q85_flag"] * df["is_evening_peak"]
        )

        df["recent_evening_spike_q90_flag"] = (
            df["recent_spike_q90_flag"] * df["is_evening_peak"]
        )

    df = safe_numeric_df(df)
    return df


# ============================================================
# DIRECT SUPERVISED DATASET
# ============================================================

def get_base_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    feature_cols = []

    for c in df_feat.columns:
        if pd.api.types.is_numeric_dtype(df_feat[c]):
            feature_cols.append(c)

    return feature_cols


def make_direct_X_y_for_horizon(
    df_feat: pd.DataFrame,
    horizon: int,
    base_feature_cols: List[str],
    future_known_exog: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    freq: str = FREQ,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    if future_known_exog is None:
        future_known_exog = []

    df_feat = df_feat.copy().sort_index()

    y = df_feat[target_col].shift(-horizon)

    X = df_feat[base_feature_cols].copy()

    target_times = df_feat.index + pd.Timedelta(freq) * horizon

    future_cal = add_time_features_from_index(
        pd.DatetimeIndex(target_times),
        prefix="future_",
    )
    future_cal.index = df_feat.index

    X = X.join(future_cal)

    existing_future_cols = [c for c in future_known_exog if c in df_feat.columns]

    for c in existing_future_cols:
        X[f"future_{c}"] = df_feat[c].shift(-horizon)

    meta = pd.DataFrame(index=df_feat.index)
    meta["origin_time"] = df_feat.index
    meta["target_time"] = target_times
    meta["horizon"] = horizon

    valid = y.notna()

    X = X.loc[valid]
    y = y.loc[valid]
    meta = meta.loc[valid]

    X = safe_numeric_df(X)

    return X, y, meta


def build_forecast_X_for_single_horizon(
    df_feat: pd.DataFrame,
    horizon: int,
    base_feature_cols: List[str],
    future_known_exog: Optional[List[str]] = None,
    freq: str = FREQ,
) -> pd.DataFrame:

    if future_known_exog is None:
        future_known_exog = []

    df_feat = df_feat.copy().sort_index()

    X = df_feat[base_feature_cols].copy()

    target_times = df_feat.index + pd.Timedelta(freq) * horizon

    future_cal = add_time_features_from_index(
        pd.DatetimeIndex(target_times),
        prefix="future_",
    )
    future_cal.index = df_feat.index

    X = X.join(future_cal)

    existing_future_cols = [c for c in future_known_exog if c in df_feat.columns]

    for c in existing_future_cols:
        X[f"future_{c}"] = df_feat[c].shift(-horizon)

    X = safe_numeric_df(X)

    return X


def filter_feature_columns(
    X_train: pd.DataFrame,
    missing_threshold: float = 0.98,
) -> List[str]:

    cols = []

    for c in X_train.columns:
        s = X_train[c]

        if not pd.api.types.is_numeric_dtype(s):
            continue

        miss_ratio = s.isna().mean()
        if miss_ratio > missing_threshold:
            continue

        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            continue

        cols.append(c)

    return cols


def make_lgbm_pipeline(params: Dict) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(**params)),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def directional_accuracy_from_origin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_origin: np.ndarray,
) -> float:

    true_dir = np.sign(y_true - y_origin)
    pred_dir = np.sign(y_pred - y_origin)

    valid = ~np.isnan(true_dir) & ~np.isnan(pred_dir)

    if valid.sum() == 0:
        return np.nan

    return float((true_dir[valid] == pred_dir[valid]).mean())


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))

    nonzero = np.abs(y_true) > 1e-6

    if nonzero.sum() > 0:
        mape = mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])
    else:
        mape = np.nan

    if len(y_true) >= 2:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    bias = float(np.mean(y_pred - y_true))

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R2": float(r2),
        "Bias": float(bias),
    }


# ============================================================
# OPTUNA BLEND (usato in fase di training)
# ============================================================

def calculate_blended_prediction(
    pred_p50: np.ndarray,
    pred_p90: np.ndarray,
    w90: float,
) -> np.ndarray:

    return (1.0 - w90) * pred_p50 + w90 * pred_p90


def get_w90_for_timestamps(
    index: pd.DatetimeIndex,
    blend_config: Dict = BLEND_CONFIG,
) -> pd.Series:

    w = pd.Series(
        float(blend_config.get("default_w90", 0.0)),
        index=index,
        dtype=float,
    )

    hourly = blend_config.get("hourly_w90", {})

    for hour in range(24):
        val = None

        if hour in hourly:
            val = hourly[hour]
        elif str(hour) in hourly:
            val = hourly[str(hour)]

        if val is not None:
            w.loc[index.hour == hour] = float(val)

    return w


def blend_p50_p90(
    pred_p50: pd.Series,
    pred_p90: pd.Series,
    blend_config: Dict = BLEND_CONFIG,
) -> pd.Series:

    pred_p50 = pred_p50.copy()
    pred_p90 = pred_p90.reindex(pred_p50.index)

    w90 = get_w90_for_timestamps(pred_p50.index, blend_config)

    out = (1.0 - w90) * pred_p50 + w90 * pred_p90
    out.name = "PUN_forecast_blend"

    return out


# ============================================================
# LOAD ARTIFACTS
# ============================================================

def load_direct_artifacts(model_dir: str = ".") -> Dict:
    model_p50_path = os.path.join(model_dir, MODEL_P50_NAME)
    model_p90_path = os.path.join(model_dir, MODEL_P90_NAME)
    metadata_path = os.path.join(model_dir, METADATA_NAME)

    if not os.path.exists(model_p50_path):
        raise FileNotFoundError(f"File non trovato: {model_p50_path}")

    if not os.path.exists(model_p90_path):
        raise FileNotFoundError(f"File non trovato: {model_p90_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"File non trovato: {metadata_path}")

    models_p50 = joblib.load(model_p50_path)
    models_p90 = joblib.load(model_p90_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "models_p50": models_p50,
        "models_p90": models_p90,
        "metadata": metadata,
    }


# ============================================================
# FORECAST NEXT 96
# ============================================================

def forecast_next_96(
    df: pd.DataFrame,
    model_dir: str = ".",
    output_dir: Optional[str] = None,
    save_csv: bool = True,
) -> pd.DataFrame:

    if output_dir is None:
        output_dir = model_dir

    artifacts = load_direct_artifacts(model_dir)

    models_p50 = artifacts["models_p50"]
    models_p90 = artifacts["models_p90"]
    metadata = artifacts["metadata"]

    steps = int(metadata["steps"])
    base_feature_cols = metadata["base_feature_cols"]
    selected_features_by_horizon = metadata["selected_features_by_horizon"]
    future_known_exog = metadata.get("future_known_exog", FUTURE_KNOWN_EXOG)
    blend_config = metadata.get("blend_config", BLEND_CONFIG)

    df = ensure_datetime_index(df)
    df = infer_and_fix_freq(df, FREQ)
    df = safe_numeric_df(df)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Colonna target '{TARGET_COL}' non trovata nel DataFrame.")

    last_origin = get_last_valid_target_timestamp(df, TARGET_COL)

    df = ensure_future_rows(df, steps=steps, freq=FREQ)
    df_feat = add_pun_features(df)

    if last_origin not in df_feat.index:
        raise ValueError("last_origin non trovato in df_feat dopo reindex.")

    rows = []

    for h in range(1, steps + 1):
        h_key = str(h)

        if h_key not in models_p50:
            raise KeyError(f"Modello p50 mancante per horizon {h_key}.")

        if h_key not in models_p90:
            raise KeyError(f"Modello p90 mancante per horizon {h_key}.")

        if h_key not in selected_features_by_horizon:
            raise KeyError(f"Feature selezionate mancanti per horizon {h_key}.")

        X_forecast = build_forecast_X_for_single_horizon(
            df_feat=df_feat,
            horizon=h,
            base_feature_cols=base_feature_cols,
            future_known_exog=future_known_exog,
            freq=FREQ,
        )

        if last_origin not in X_forecast.index:
            raise ValueError(f"Origin {last_origin} non presente in X_forecast per h={h}.")

        selected_cols = selected_features_by_horizon[h_key]

        missing_cols = [c for c in selected_cols if c not in X_forecast.columns]
        for c in missing_cols:
            X_forecast[c] = np.nan

        X_row = X_forecast.loc[[last_origin], selected_cols]

        pred_p50 = float(models_p50[h_key].predict(X_row)[0])
        pred_p90 = float(models_p90[h_key].predict(X_row)[0])

        target_time = last_origin + pd.Timedelta(FREQ) * h

        rows.append(
            {
                "origin_time": last_origin,
                "target_time": target_time,
                "horizon": h,
                "PUN_p50": pred_p50,
                "PUN_p90": pred_p90,
            }
        )

    forecast_df = pd.DataFrame(rows)
    forecast_df["target_time"] = pd.to_datetime(forecast_df["target_time"])
    forecast_df["hour"] = forecast_df["target_time"].dt.hour

    target_index = pd.DatetimeIndex(forecast_df["target_time"])

    w90 = get_w90_for_timestamps(
        index=target_index,
        blend_config=blend_config,
    )

    forecast_df["w90"] = w90.values
    forecast_df["w50"] = 1.0 - forecast_df["w90"]

    forecast_df["PUN_forecast"] = (
        forecast_df["w50"] * forecast_df["PUN_p50"]
        + forecast_df["w90"] * forecast_df["PUN_p90"]
    )

    forecast_df = forecast_df[
        [
            "origin_time",
            "target_time",
            "horizon",
            "hour",
            "PUN_p50",
            "PUN_p90",
            "w50",
            "w90",
            "PUN_forecast",
        ]
    ]

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        forecast_path = os.path.join(output_dir, FORECAST_NAME)
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Forecast salvato in: {forecast_path}")

    return forecast_df


# ============================================================
# NATIVE FEATURE IMPORTANCE (sostituisce SHAP legacy)
# ============================================================

def build_native_importance_df(
    models_p50: Dict,
    metadata: Dict,
) -> pd.DataFrame:
    """
    Importanza nativa LightGBM (gain) per ciascun horizon (1..96),
    ricavata direttamente dai modelli p50 gia' addestrati.
    Sostituisce la vecchia spiegabilita' SHAP, incompatibile con
    l'architettura a 96 modelli indipendenti del nuovo forecaster.
    """
    selected_features_by_horizon = metadata["selected_features_by_horizon"]

    rows = []

    for h_key, pipe in models_p50.items():
        h = int(h_key)
        feats = selected_features_by_horizon.get(h_key, [])

        model = pipe.named_steps["model"]
        importances = model.booster_.feature_importance(importance_type="gain")

        for f, imp in zip(feats, importances):
            rows.append({"horizon": h, "feature": f, "importance": float(imp)})

    df = pd.DataFrame(rows)
    return df


def summarize_native_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    summary = (
        importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return summary
