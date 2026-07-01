# ============================================================
# PUN DIRECT 96 - INFERENCE v2
# Compatibile con il training "quantile grid + festivi IT + CQR"
# (train_direct_quantile_models -> pun_direct_lgbm_quantiles.joblib)
# ============================================================
#
# Espone le 4 funzioni che app.py importa:
#   - load_direct_artifacts(model_dir)   -> {"models_q", "metadata"}
#   - forecast_next_96(df, model_dir)    -> DataFrame forecast day-ahead
#   - build_native_importance_df(models_q, metadata)
#   - summarize_native_importance(importance_df, top_n)
#
# Il forecaster produce, per ogni horizon:
#   - PUN_forecast  = quantile 0.50 (mediana, punto sotto pinball loss)
#   - PUN_q05..PUN_q95 = quantili grezzi dei singoli modelli
#   - PUN_lower / PUN_upper = banda CALIBRATA (CQR) sulla coppia piu' larga
#     (offset conformale additivo per ora del giorno preso dai metadata)
#
# ⚠️ FEATURE ENGINEERING: le funzioni add_pun_features /
# add_time_features_from_index / _italian_holiday_flags DEVONO restare
# IDENTICHE a quelle del training. Sono copiate verbatim qui sotto.
# (Ideale: estrarle in functions/pun_features.py e importarle sia qui
#  che nel training, così la sincronia e' garantita by design.)
# ============================================================

import os
import json
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    holidays = None
    HAS_HOLIDAYS = False


# ============================================================
# CONFIG
# ============================================================

TARGET_COL = "PUN"
FREQ = "15min"
STEPS = 96

MODEL_QUANTILES_NAME = "pun_direct_lgbm_quantiles.joblib"
METADATA_NAME = "pun_direct_metadata.json"
FORECAST_NAME = "pun_direct_forecast_next_96.csv"

FUTURE_KNOWN_EXOG = [
    "forecast_total_load_MW",
    "bologna_temperature_2m",
    "bari_wind_speed_80m",
    "cloud_cover_mean",
]


# ============================================================
# BASIC UTILS  (IDENTICI AL TRAINING)
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
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    return df.reindex(full_index)


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def get_last_valid_target_timestamp(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Timestamp:
    valid = df[df[target_col].notna()]
    if valid.empty:
        raise ValueError(f"Nessun valore valido trovato nella colonna target '{target_col}'.")
    return valid.index.max()


# ============================================================
# FEATURE ENGINEERING  (IDENTICO AL TRAINING - copia verbatim)
# ============================================================

def _italian_holiday_flags(index: pd.DatetimeIndex) -> Dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(index.date)

    if HAS_HOLIDAYS:
        years = range(int(dates.year.min()), int(dates.year.max()) + 2)
        it_holidays = holidays.Italy(years=years)

        is_hol = pd.Series(dates, index=index).isin(it_holidays).to_numpy()
        is_hol_tomorrow = pd.Series(dates + pd.Timedelta(days=1), index=index).isin(it_holidays).to_numpy()
        is_hol_yesterday = pd.Series(dates - pd.Timedelta(days=1), index=index).isin(it_holidays).to_numpy()
    else:
        is_hol = np.zeros(len(index), dtype=bool)
        is_hol_tomorrow = np.zeros(len(index), dtype=bool)
        is_hol_yesterday = np.zeros(len(index), dtype=bool)

    return {
        "is_holiday": is_hol.astype(int),
        "is_day_before_holiday": is_hol_tomorrow.astype(int),   # domani e' festivo
        "is_day_after_holiday": is_hol_yesterday.astype(int),   # ieri era festivo
    }


def add_time_features_from_index(index: pd.DatetimeIndex, prefix: str = "") -> pd.DataFrame:
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

    # --- festivi italiani (feature nota in anticipo) ---
    hol_flags = _italian_holiday_flags(index)
    for name, values in hol_flags.items():
        x[f"{prefix}{name}"] = values

    x[f"{prefix}is_non_working_day"] = (
        (x[f"{prefix}is_weekend"] == 1) | (x[f"{prefix}is_holiday"] == 1)
    ).astype(int)

    return x


def add_pun_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_datetime_index(df)

    time_df = add_time_features_from_index(df.index, prefix="")
    for c in time_df.columns:
        df[c] = time_df[c]

    lag_map = {
        "pun_lag_15m": 1, "pun_lag_30m": 2, "pun_lag_1h": 4, "pun_lag_2h": 8,
        "pun_lag_4h": 16, "pun_lag_8h": 32, "pun_lag_12h": 48, "pun_lag_1d": 96,
        "pun_lag_2d": 192, "pun_lag_3d": 288, "pun_lag_7d": 672,
    }

    if TARGET_COL in df.columns:
        for name, lag in lag_map.items():
            df[name] = df[TARGET_COL].shift(lag)

        df["pun_same_q_1d"] = df[TARGET_COL].shift(96)
        df["pun_same_q_2d"] = df[TARGET_COL].shift(192)
        df["pun_same_q_3d"] = df[TARGET_COL].shift(288)
        df["pun_same_q_7d"] = df[TARGET_COL].shift(672)

        same_q_cols = ["pun_same_q_1d", "pun_same_q_2d", "pun_same_q_3d", "pun_same_q_7d"]
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
        df["load_roll_mean_4h"] = df["forecast_total_load_MW"].shift(1).rolling(16, min_periods=4).mean()
        df["load_roll_std_4h"] = df["forecast_total_load_MW"].shift(1).rolling(16, min_periods=4).std()
        df["load_roll_mean_1d"] = df["forecast_total_load_MW"].shift(1).rolling(96, min_periods=24).mean()
        df["load_roll_std_1d"] = df["forecast_total_load_MW"].shift(1).rolling(96, min_periods=24).std()

    if "actual_generation_GWh_hydro" in df.columns:
        df["hydro_lag_1d"] = df["actual_generation_GWh_hydro"].shift(96)
        df["hydro_lag_2d"] = df["actual_generation_GWh_hydro"].shift(192)
        df["hydro_lag_7d"] = df["actual_generation_GWh_hydro"].shift(672)
        df["hydro_roll_mean_1d"] = df["actual_generation_GWh_hydro"].shift(1).rolling(96, min_periods=24).mean()

    zone_cols = ["CALA_B16", "CNOR_B16", "CSUD_B16", "NORD_B16", "SARD_B16", "SICI_B16", "SUD_B16"]
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
        rolling_q85 = df[TARGET_COL].shift(1).rolling(96, min_periods=24).quantile(0.85)
        rolling_q90 = df[TARGET_COL].shift(1).rolling(96, min_periods=24).quantile(0.90)
        df["recent_spike_q85_flag"] = (df[TARGET_COL].shift(1) > rolling_q85).astype(int)
        df["recent_spike_q90_flag"] = (df[TARGET_COL].shift(1) > rolling_q90).astype(int)
        df["recent_evening_spike_q85_flag"] = df["recent_spike_q85_flag"] * df["is_evening_peak"]
        df["recent_evening_spike_q90_flag"] = df["recent_spike_q90_flag"] * df["is_evening_peak"]

    return safe_numeric_df(df)


def get_base_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    return [c for c in df_feat.columns if pd.api.types.is_numeric_dtype(df_feat[c])]


# ============================================================
# HELPER PER IL FORECAST (righe future + costruzione X)
# ============================================================

def make_future_index_after(last_ts: pd.Timestamp, steps: int = STEPS, freq: str = FREQ) -> pd.DatetimeIndex:
    return pd.date_range(start=last_ts + pd.Timedelta(freq), periods=steps, freq=freq)


def ensure_future_rows(df: pd.DataFrame, steps: int = STEPS, freq: str = FREQ) -> pd.DataFrame:
    df = df.copy().sort_index()
    last_valid_target_time = get_last_valid_target_timestamp(df, TARGET_COL)
    required_future_index = make_future_index_after(last_valid_target_time, steps=steps, freq=freq)
    full_index = df.index.union(required_future_index)
    return df.reindex(full_index).sort_index()


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

    # reindex (invece di selezione diretta): eventuali base cols mancanti -> NaN
    X = df_feat.reindex(columns=base_feature_cols).copy()

    target_times = df_feat.index + pd.Timedelta(freq) * horizon
    future_cal = add_time_features_from_index(pd.DatetimeIndex(target_times), prefix="future_")
    future_cal.index = df_feat.index
    X = X.join(future_cal)

    for c in [c for c in future_known_exog if c in df_feat.columns]:
        X[f"future_{c}"] = df_feat[c].shift(-horizon)

    return safe_numeric_df(X)


# ============================================================
# LOAD ARTIFACTS (v2)
# ============================================================

def load_direct_artifacts(model_dir: str = ".") -> Dict:
    model_path = os.path.join(model_dir, MODEL_QUANTILES_NAME)
    metadata_path = os.path.join(model_dir, METADATA_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File non trovato: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"File non trovato: {metadata_path}")

    models_q = joblib.load(model_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "models_q": models_q,
        "metadata": metadata,
    }


# ============================================================
# FORECAST NEXT 96 (v2: quantile grid + banda CQR)
# ============================================================

def _qcol(q: float) -> str:
    """Nome colonna leggibile per un quantile (0.05 -> PUN_q05, 0.5 -> PUN_q50)."""
    return f"PUN_q{int(round(float(q) * 100)):02d}"


def forecast_next_96(
    df: pd.DataFrame,
    model_dir: str = ".",
    output_dir: Optional[str] = None,
    save_csv: bool = True,
) -> pd.DataFrame:

    if output_dir is None:
        output_dir = model_dir

    artifacts = load_direct_artifacts(model_dir)
    models_q = artifacts["models_q"]
    metadata = artifacts["metadata"]

    steps = int(metadata["steps"])
    base_feature_cols = metadata["base_feature_cols"]
    selected_features_by_horizon = metadata["selected_features_by_horizon"]
    future_known_exog = metadata.get("future_known_exog", FUTURE_KNOWN_EXOG)
    quantile_levels = metadata["quantile_levels"]                 # lista di float
    conformal_pairs = metadata.get("conformal_pairs", [])         # lista di [q_low, q_high]
    conformal_offsets = metadata.get("conformal_offsets", {})     # {"qlow_qhigh": {"0": off, ...}}

    # avviso train/serve skew sui festivi
    if metadata.get("has_holidays_lib") and not HAS_HOLIDAYS:
        print(
            "ATTENZIONE: il modello e' stato allenato con la libreria 'holidays' "
            "ma qui non e' installata: le feature festivi saranno tutte 0 -> possibile degrado."
        )

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

    # coppia conformale per la banda mostrata = la piu' larga disponibile
    primary_pair = None
    if conformal_pairs:
        primary_pair = max(conformal_pairs, key=lambda p: float(p[1]) - float(p[0]))

    rows = []

    for h in range(1, steps + 1):
        h_key = str(h)

        if h_key not in models_q:
            raise KeyError(f"Modelli quantili mancanti per horizon {h_key}.")
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
        # reindex(columns=...) garantisce ordine == selected_cols e riempie i mancanti con NaN
        X_row = X_forecast.loc[[last_origin]].reindex(columns=selected_cols)

        target_time = last_origin + pd.Timedelta(FREQ) * h
        target_hour = int(pd.Timestamp(target_time).hour)

        row = {
            "origin_time": last_origin,
            "target_time": target_time,
            "horizon": h,
            "hour": target_hour,
        }

        # predizione di ogni quantile
        q_preds: Dict[float, float] = {}
        q_dict = models_q[h_key]
        for q in quantile_levels:
            q_key = str(q)
            model = q_dict.get(q_key)
            if model is None:
                raise KeyError(f"Modello quantile {q_key} mancante per horizon {h_key}.")
            pred = float(model.predict(X_row)[0])
            q_preds[float(q)] = pred
            row[_qcol(q)] = pred

        # punto forecast = q50 (fallback: quantile piu' vicino a 0.5)
        if 0.5 in q_preds:
            row["PUN_forecast"] = q_preds[0.5]
        else:
            nearest = min(q_preds, key=lambda qq: abs(qq - 0.5))
            row["PUN_forecast"] = q_preds[nearest]

        # banda calibrata CQR sulla coppia piu' larga
        if primary_pair is not None:
            q_low = float(primary_pair[0])
            q_high = float(primary_pair[1])
            pair_key = f"{q_low}_{q_high}"

            offset = 0.0
            if pair_key in conformal_offsets:
                offset = float(conformal_offsets[pair_key].get(str(target_hour), 0.0))

            low_raw = q_preds.get(q_low)
            high_raw = q_preds.get(q_high)

            if low_raw is not None and high_raw is not None:
                row["PUN_lower"] = low_raw - offset
                row["PUN_upper"] = high_raw + offset
                row["PUN_lower_raw"] = low_raw
                row["PUN_upper_raw"] = high_raw
                row["band_coverage"] = q_high - q_low

        rows.append(row)

    forecast_df = pd.DataFrame(rows)
    forecast_df["target_time"] = pd.to_datetime(forecast_df["target_time"])
    forecast_df = forecast_df.sort_values("target_time").reset_index(drop=True)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        forecast_path = os.path.join(output_dir, FORECAST_NAME)
        forecast_df.to_csv(forecast_path, index=False)
        print(f"Forecast salvato in: {forecast_path}")

    return forecast_df


# ============================================================
# IMPORTANZA FEATURE NATIVA (gain LightGBM, dai modelli q50)
# ============================================================

def build_native_importance_df(
    models_q: Dict,
    metadata: Dict,
    quantile: str = "0.5",
) -> pd.DataFrame:
    """
    Importanza nativa LightGBM (gain) per ciascun horizon, presa dai
    modelli del quantile mediano (q50 di default). Se il quantile
    richiesto non esiste, usa quello piu' vicino a 0.5.
    """
    selected_features_by_horizon = metadata["selected_features_by_horizon"]

    rows = []

    for h_key, q_dict in models_q.items():
        pipe = q_dict.get(quantile)
        if pipe is None:
            avail = sorted(q_dict.keys(), key=lambda k: abs(float(k) - 0.5))
            if not avail:
                continue
            pipe = q_dict[avail[0]]

        feats = selected_features_by_horizon.get(h_key, [])
        model = pipe.named_steps["model"]
        importances = model.booster_.feature_importance(importance_type="gain")

        for f, imp in zip(feats, importances):
            rows.append({"horizon": int(h_key), "feature": f, "importance": float(imp)})

    return pd.DataFrame(rows)


def summarize_native_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    if importance_df is None or importance_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])

    summary = (
        importance_df.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    return summary
