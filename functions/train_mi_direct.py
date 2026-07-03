# ============================================================
# MI DIRECT 96 - TRAINING v3 (generalizzato per zona)
# LightGBM quantile grid + festivi + rolling backtest + CQR
# ------------------------------------------------------------
# Novità rispetto alla v2:
#   1) CQR ASIMMETRICO: offset separati per lato (basso/alto), calibrati
#      su due score distinti -> la banda si allarga di piu' dove serve
#      (spike verso l'alto). Metadata conformal_offsets ora per-lato.
#   2) PUNTO CONFIGURABILE: point_cost_ratio (= costo sotto-stima / costo
#      sovra-stima) -> quantile ottimo tau = r/(r+1), oppure point_quantile
#      fisso. Default = q50. Il quantile scelto viene allenato e salvato.
#   3) FEATURE DI REGIME: z-score prezzo, rapporto volatilita' breve/lunga,
#      flag regime alto/basso (7g), conteggio spike e quarti dall'ultimo
#      spike -> segnali di regime che gli alberi possono splittare.
#   4) IPERPARAMETRI PER ORIZZONTE + OVERRIDE PER ZONA: params LightGBM che
#      variano a fasce di orizzonte (breve/medio/lungo) + lgbm_params_overrides.
#
# ⚠️ Se costruisci il modulo di inference MI, deve leggere:
#    - metadata["point_quantile"]  (quale quantile e' il punto)
#    - metadata["conformal_offsets"][pair][hour] = {"low":.., "high":..}
#      banda: low_cal = pred_qlow - off_low ; high_cal = pred_qhigh + off_high
#    e usare add_price_features IDENTICA (stessi nomi feature).
# ============================================================

import os
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    holidays = None
    HAS_HOLIDAYS = False

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

FREQ = "15min"
STEPS = 96
STEPS_PER_DAY = 96

CALIB_DAYS = 28
BACKTEST_DAYS = 28

QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
CONFORMAL_PAIRS = [(0.05, 0.95), (0.10, 0.90)]

MODEL_QUANTILES_NAME = "mi_direct_lgbm_quantiles.joblib"
METADATA_NAME = "mi_direct_metadata.json"
PINBALL_BY_HOUR_NAME = "mi_direct_eval_pinball_by_hour.csv"
COVERAGE_BY_HOUR_NAME = "mi_direct_eval_coverage_by_hour.csv"

FUTURE_KNOWN_EXOG = [
    # solare day-ahead per zona (ENTSOE A69/B16) - noto in anticipo, driver del prezzo diurno
    "CALA_B16", "CNOR_B16", "CSUD_B16", "NORD_B16", "SARD_B16", "SICI_B16", "SUD_B16",
    # carico: in training = market_load_MW a t+h; al serving iniettiamo il forecast Terna
    "market_load_MW",
    # meteo previsto (Open-Meteo forecast)
    "temperature_mean", "cloud_cover_mean", "wind_speed_mean",
    "bologna_temperature_2m", "bari_wind_speed_80m",
]


def lgbm_quantile_params(alpha: float, horizon: Optional[int] = None,
                         overrides: Optional[Dict] = None) -> Dict:
    """
    Parametri LightGBM per un dato quantile (alpha).

    Modifica #4 — differenziazione per orizzonte:
      - breve (h <= 8, cioe' <= 2h): dinamica veloce -> piu' capacita'
      - medio (8 < h <= 48, <= 12h): base
      - lungo (h > 48, > 12h): piu' liscio/regolarizzato
    `overrides` (dict) permette di forzare qualsiasi parametro per zona.
    """
    p = {
        "objective": "quantile",
        "alpha": float(alpha),
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

    if horizon is not None:
        if horizon <= 8:            # <= 2h
            p.update({"num_leaves": 31, "max_depth": 6, "n_estimators": 600})
        elif horizon <= 48:         # <= 12h
            pass                    # base
        else:                       # > 12h
            p.update({"num_leaves": 16, "max_depth": 5, "n_estimators": 450,
                      "min_child_samples": 60, "reg_lambda": 8.0})

    if overrides:
        p.update(overrides)

    return p


def _resolve_point_quantile(
    quantile_levels: List[float],
    point_quantile: Optional[float],
    point_cost_ratio: Optional[float],
) -> Tuple[List[float], float]:
    """
    Modifica #2 — sceglie il quantile "punto".
      - point_cost_ratio = c_under / c_over  ->  tau = r / (r + 1)
        (se sotto-stimare costa il doppio di sovra-stimare, r=2 -> tau≈0.667,
         il punto sale = previsione piu' alta)
      - altrimenti point_quantile (se dato), altrimenti 0.5.
    Il quantile scelto viene aggiunto alla griglia se assente (cosi' e' allenato).
    Ritorna (quantile_levels_effettivi, point_quantile).
    """
    levels = sorted(set(float(q) for q in quantile_levels))

    if point_cost_ratio is not None:
        r = float(point_cost_ratio)
        tau = r / (r + 1.0)
    elif point_quantile is not None:
        tau = float(point_quantile)
    else:
        tau = 0.5

    tau = float(min(max(tau, 0.01), 0.99))

    if not any(abs(tau - q) < 1e-9 for q in levels):
        levels = sorted(set(levels + [tau]))

    return levels, tau


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
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    return df.reindex(full_index)


def safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def get_last_valid_target_timestamp(df: pd.DataFrame, target_col: str) -> pd.Timestamp:
    valid = df[df[target_col].notna()]
    if valid.empty:
        raise ValueError(f"Nessun valore valido nella colonna target '{target_col}'.")
    return valid.index.max()


# ============================================================
# FEATURE ENGINEERING (target parametrico, prefisso "y_")
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
        "is_day_before_holiday": is_hol_tomorrow.astype(int),
        "is_day_after_holiday": is_hol_yesterday.astype(int),
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

    hol_flags = _italian_holiday_flags(index)
    for name, values in hol_flags.items():
        x[f"{prefix}{name}"] = values

    x[f"{prefix}is_non_working_day"] = (
        (x[f"{prefix}is_weekend"] == 1) | (x[f"{prefix}is_holiday"] == 1)
    ).astype(int)

    return x


def add_price_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Feature engineering per una zona. target_col = colonna prezzo della zona.
    Feature derivate dal prezzo con prefisso "y_". Blocchi esogeni opzionali
    (usati solo se le colonne esistono). Include feature di REGIME (mod. #3).
    """
    df = df.copy()
    df = ensure_datetime_index(df)

    if target_col not in df.columns:
        raise ValueError(f"Colonna target '{target_col}' non presente nel df.")

    time_df = add_time_features_from_index(df.index, prefix="")
    for c in time_df.columns:
        df[c] = time_df[c]

    lag_map = {
        "y_lag_15m": 1, "y_lag_30m": 2, "y_lag_1h": 4, "y_lag_2h": 8,
        "y_lag_4h": 16, "y_lag_8h": 32, "y_lag_12h": 48, "y_lag_1d": 96,
        "y_lag_2d": 192, "y_lag_3d": 288, "y_lag_7d": 672,
    }
    for name, lag in lag_map.items():
        df[name] = df[target_col].shift(lag)

    df["y_same_q_1d"] = df[target_col].shift(96)
    df["y_same_q_2d"] = df[target_col].shift(192)
    df["y_same_q_3d"] = df[target_col].shift(288)
    df["y_same_q_7d"] = df[target_col].shift(672)

    same_q_cols = ["y_same_q_1d", "y_same_q_2d", "y_same_q_3d", "y_same_q_7d"]
    df["y_same_q_mean"] = df[same_q_cols].mean(axis=1)
    df["y_same_q_std"] = df[same_q_cols].std(axis=1)
    df["y_same_q_max"] = df[same_q_cols].max(axis=1)
    df["y_same_q_min"] = df[same_q_cols].min(axis=1)

    s = df[target_col].shift(1)   # prezzo fino a t-15m (no leakage sul target)
    df["y_roll_mean_1h"] = s.rolling(4, min_periods=2).mean()
    df["y_roll_std_1h"] = s.rolling(4, min_periods=2).std()
    df["y_roll_mean_4h"] = s.rolling(16, min_periods=4).mean()
    df["y_roll_std_4h"] = s.rolling(16, min_periods=4).std()
    df["y_roll_max_4h"] = s.rolling(16, min_periods=4).max()
    df["y_roll_min_4h"] = s.rolling(16, min_periods=4).min()
    df["y_roll_mean_1d"] = s.rolling(96, min_periods=24).mean()
    df["y_roll_std_1d"] = s.rolling(96, min_periods=24).std()
    df["y_roll_max_1d"] = s.rolling(96, min_periods=24).max()
    df["y_roll_min_1d"] = s.rolling(96, min_periods=24).min()
    df["y_roll_mean_7d"] = s.rolling(672, min_periods=96).mean()
    df["y_roll_std_7d"] = s.rolling(672, min_periods=96).std()

    df["y_diff_15m"] = df[target_col] - df[target_col].shift(1)
    df["y_diff_1h"] = df[target_col] - df[target_col].shift(4)
    df["y_diff_4h"] = df[target_col] - df[target_col].shift(16)
    df["y_diff_1d"] = df[target_col] - df[target_col].shift(96)
    df["y_diff_7d"] = df[target_col] - df[target_col].shift(672)

    df["y_ret_15m"] = df[target_col].pct_change(1)
    df["y_ret_1h"] = df[target_col].pct_change(4)
    df["y_ret_1d"] = df[target_col].pct_change(96)
    df["y_ret_7d"] = df[target_col].pct_change(672)

    # ---------- FEATURE DI REGIME (mod. #3) ----------
    eps = 1e-6
    roll_mean_1d = s.rolling(96, min_periods=24).mean()
    roll_std_1d = s.rolling(96, min_periods=24).std()
    roll_std_7d = s.rolling(672, min_periods=96).std()

    # z-score del prezzo recente rispetto alla media giornaliera
    df["y_zscore_1d"] = (s - roll_mean_1d) / (roll_std_1d + eps)
    # rapporto volatilita' breve vs lunga (regime di volatilita')
    df["y_vol_ratio_1d_7d"] = roll_std_1d / (roll_std_7d + eps)
    # scostamento relativo dalla media giornaliera
    df["y_rel_dev_1d"] = (s - roll_mean_1d) / (roll_mean_1d.abs() + eps)

    # regime di livello su finestra 7 giorni
    roll_q90_7d = s.rolling(672, min_periods=96).quantile(0.90)
    roll_q10_7d = s.rolling(672, min_periods=96).quantile(0.10)
    df["y_regime_high_7d"] = (s > roll_q90_7d).astype(float)
    df["y_regime_low_7d"] = (s < roll_q10_7d).astype(float)

    # spike (prezzo sopra q90 giornaliero): intensita' e recency
    roll_q90_1d = s.rolling(96, min_periods=24).quantile(0.90)
    is_spike = (s > roll_q90_1d).fillna(False).astype(int)
    df["y_spike_count_1d"] = is_spike.rolling(96, min_periods=24).sum()

    # quarti dall'ultimo spike (vettoriale, robusto)
    pos = np.arange(len(is_spike))
    last_spike = pd.Series(
        np.where(is_spike.values == 1, pos, np.nan), index=is_spike.index
    ).ffill()
    df["y_qtrs_since_spike"] = pos - last_spike.values
    # -------------------------------------------------

    # ---- esogene opzionali (solo se presenti) ----
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

    # ---- interazioni con la fascia serale ----
    if "forecast_total_load_MW" in df.columns:
        df["evening_load"] = df["forecast_total_load_MW"] * df["is_evening_peak"]
    if "cloud_cover_mean" in df.columns:
        df["evening_cloud_cover"] = df["cloud_cover_mean"] * df["is_evening_peak"]
    if "bari_wind_speed_80m" in df.columns:
        df["evening_wind_bari"] = df["bari_wind_speed_80m"] * df["is_evening_peak"]
    df["evening_volatility_4h"] = df["y_roll_std_4h"] * df["is_evening_peak"]
    df["evening_volatility_1d"] = df["y_roll_std_1d"] * df["is_evening_peak"]
    df["evening_regime_high"] = df["y_regime_high_7d"] * df["is_evening_peak"]

    # ---- flag di spike recenti sul prezzo (compatibili con la v2) ----
    df["recent_spike_q85_flag"] = (df[target_col].shift(1) > s.rolling(96, min_periods=24).quantile(0.85)).astype(int)
    df["recent_spike_q90_flag"] = is_spike
    df["recent_evening_spike_q85_flag"] = df["recent_spike_q85_flag"] * df["is_evening_peak"]
    df["recent_evening_spike_q90_flag"] = df["recent_spike_q90_flag"] * df["is_evening_peak"]

    return safe_numeric_df(df)


# ============================================================
# DIRECT SUPERVISED DATASET
# ============================================================

def get_base_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    return [c for c in df_feat.columns if pd.api.types.is_numeric_dtype(df_feat[c])]


def make_direct_X_y_for_horizon(
    df_feat: pd.DataFrame, horizon: int, base_feature_cols: List[str],
    target_col: str, future_known_exog: Optional[List[str]] = None, freq: str = FREQ,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:

    if future_known_exog is None:
        future_known_exog = []

    df_feat = df_feat.copy().sort_index()
    y = df_feat[target_col].shift(-horizon)
    X = df_feat[base_feature_cols].copy()

    target_times = df_feat.index + pd.Timedelta(freq) * horizon
    future_cal = add_time_features_from_index(pd.DatetimeIndex(target_times), prefix="future_")
    future_cal.index = df_feat.index
    X = X.join(future_cal)

    for c in [c for c in future_known_exog if c in df_feat.columns]:
        X[f"future_{c}"] = df_feat[c].shift(-horizon)

    meta = pd.DataFrame(index=df_feat.index)
    meta["origin_time"] = df_feat.index
    meta["target_time"] = target_times
    meta["horizon"] = horizon

    valid = y.notna()
    X, y, meta = X.loc[valid], y.loc[valid], meta.loc[valid]
    return safe_numeric_df(X), y, meta


def filter_feature_columns(X_train: pd.DataFrame, missing_threshold: float = 0.98) -> List[str]:
    cols = []
    for c in X_train.columns:
        s = X_train[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if s.isna().mean() > missing_threshold:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return cols


def make_lgbm_pipeline(params: Dict) -> Pipeline:
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("model", LGBMRegressor(**params))])


# ============================================================
# METRICHE PROBABILISTICHE
# ============================================================

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.maximum(quantile * e, (quantile - 1.0) * e)))


def compute_conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """Quantile (1 - alpha) degli score di nonconformita' (CQR)."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = len(scores)
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


# ============================================================
# TRAIN DIRECT 96 (una zona)
# ============================================================

def train_direct_quantile_models(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str = ".",
    steps: int = STEPS,
    quantile_levels: List[float] = QUANTILE_LEVELS,
    conformal_pairs: List[Tuple[float, float]] = CONFORMAL_PAIRS,
    calib_days: int = CALIB_DAYS,
    backtest_days: int = BACKTEST_DAYS,
    future_known_exog: Optional[List[str]] = None,
    retrain_on_full_data: bool = True,
    # --- nuovi parametri v3 ---
    asymmetric_cqr: bool = True,
    point_quantile: Optional[float] = None,
    point_cost_ratio: Optional[float] = None,
    lgbm_params_overrides: Optional[Dict] = None,
) -> Dict:

    if future_known_exog is None:
        future_known_exog = FUTURE_KNOWN_EXOG

    os.makedirs(output_dir, exist_ok=True)

    df = ensure_datetime_index(df)
    df = infer_and_fix_freq(df, FREQ)
    df = safe_numeric_df(df)

    if target_col not in df.columns:
        raise ValueError(f"Colonna target '{target_col}' non trovata nel DataFrame.")

    # mod. #2: risolvi il quantile "punto" e assicurati sia allenato
    quantile_levels, point_q = _resolve_point_quantile(quantile_levels, point_quantile, point_cost_ratio)
    # assicurati che i quantili delle coppie conformali siano allenati
    for (ql, qh) in conformal_pairs:
        for q in (float(ql), float(qh)):
            if not any(abs(q - x) < 1e-9 for x in quantile_levels):
                quantile_levels = sorted(set(quantile_levels + [q]))

    df_feat = add_price_features(df, target_col)
    base_feature_cols = get_base_feature_columns(df_feat)

    calib_origins = calib_days * STEPS_PER_DAY
    backtest_origins = backtest_days * STEPS_PER_DAY
    total_holdout = calib_origins + backtest_origins

    models_q: Dict[str, Dict[str, Pipeline]] = {}
    selected_features_by_horizon: Dict[str, List[str]] = {}

    pinball_rows = []
    # accumulo score conformali per lato (mod. #1)
    calib_scores_by_pair = {pair: [] for pair in conformal_pairs}
    backtest_raw_by_pair = {pair: [] for pair in conformal_pairs}

    print("============================================================")
    print(f"TRAIN MI DIRECT 96 v3 - target='{target_col}'")
    print("============================================================")
    print(f"Festivi IT: {HAS_HOLIDAYS} | Righe: {len(df_feat)} | Feature base: {len(base_feature_cols)}")
    print(f"Holdout: {calib_days}gg calib + {backtest_days}gg backtest = {total_holdout} origin")
    print(f"Quantili: {quantile_levels}")
    print(f"CQR asimmetrico: {asymmetric_cqr} | Punto = q{point_q}")
    if lgbm_params_overrides:
        print(f"Override params zona: {lgbm_params_overrides}")
    print("============================================================")

    for h in range(1, steps + 1):
        X, y, meta = make_direct_X_y_for_horizon(
            df_feat=df_feat, horizon=h, base_feature_cols=base_feature_cols,
            target_col=target_col, future_known_exog=future_known_exog, freq=FREQ,
        )

        valid_origin = df_feat.loc[X.index, target_col].notna()
        X, y, meta = X.loc[valid_origin], y.loc[valid_origin], meta.loc[valid_origin]

        n = len(X)
        if n <= total_holdout + 200:
            raise ValueError(
                f"Pochi dati per horizon {h} (target='{target_col}'). "
                f"Righe valide={n}, holdout richiesto={total_holdout}."
            )

        split_train_end = n - total_holdout
        split_calib_end = split_train_end + calib_origins

        X_train, y_train = X.iloc[:split_train_end], y.iloc[:split_train_end]
        X_calib, y_calib, meta_calib = X.iloc[split_train_end:split_calib_end], y.iloc[split_train_end:split_calib_end], meta.iloc[split_train_end:split_calib_end]
        X_bt, y_bt, meta_bt = X.iloc[split_calib_end:], y.iloc[split_calib_end:], meta.iloc[split_calib_end:]

        selected_cols = filter_feature_columns(X_train)
        if len(selected_cols) == 0:
            raise ValueError(f"Nessuna feature valida per horizon {h} (target='{target_col}').")
        selected_features_by_horizon[str(h)] = selected_cols

        preds_calib: Dict[float, np.ndarray] = {}
        preds_bt: Dict[float, np.ndarray] = {}

        for q in quantile_levels:
            # mod. #4: params per orizzonte + override per zona
            pipe = make_lgbm_pipeline(lgbm_quantile_params(q, horizon=h, overrides=lgbm_params_overrides))
            pipe.fit(X_train[selected_cols], y_train)

            pred_c = pipe.predict(X_calib[selected_cols])
            pred_b = pipe.predict(X_bt[selected_cols])
            preds_calib[q] = pred_c
            preds_bt[q] = pred_b

            hours_bt = meta_bt["target_time"].dt.hour.values
            for hr in np.unique(hours_bt):
                mask = hours_bt == hr
                pinball_rows.append({
                    "horizon": h, "hour": int(hr), "quantile": q,
                    "pinball": pinball_loss(y_bt.values[mask], pred_b[mask], q),
                    "n": int(mask.sum()),
                })

            if not retrain_on_full_data:
                models_q.setdefault(str(h), {})[str(q)] = pipe

        # mod. #1: accumulo score per lato (basso/alto)
        calib_hours = meta_calib["target_time"].dt.hour.values
        for pair in conformal_pairs:
            q_low, q_high = pair
            e_low = preds_calib[q_low] - y_calib.values     # >0 se y sotto pred_low
            e_high = y_calib.values - preds_calib[q_high]    # >0 se y sopra pred_high
            calib_scores_by_pair[pair].append(pd.DataFrame({
                "hour": calib_hours, "e_low": e_low, "e_high": e_high,
            }))

            bt_hours = meta_bt["target_time"].dt.hour.values
            backtest_raw_by_pair[pair].append(pd.DataFrame({
                "horizon": h, "hour": bt_hours,
                "y": y_bt.values, "pred_low": preds_bt[q_low], "pred_high": preds_bt[q_high],
            }))

        if retrain_on_full_data:
            for q in quantile_levels:
                pipe_final = make_lgbm_pipeline(lgbm_quantile_params(q, horizon=h, overrides=lgbm_params_overrides))
                pipe_final.fit(X[selected_cols], y)
                models_q.setdefault(str(h), {})[str(q)] = pipe_final

        print(f"H={h:02d}/{steps} | features={len(selected_cols)} | train={split_train_end} | calib={len(X_calib)} | backtest={len(X_bt)}")

    # ---- offset conformali per (pair, hour), PER LATO (mod. #1) ----
    conformal_offsets: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pair in conformal_pairs:
        q_low, q_high = pair
        alpha = 1.0 - (q_high - q_low)
        alpha_side = alpha / 2.0 if asymmetric_cqr else alpha
        calib_df = pd.concat(calib_scores_by_pair[pair], ignore_index=True)
        pair_key = f"{q_low}_{q_high}"
        conformal_offsets[pair_key] = {}
        for hr in range(24):
            sub = calib_df[calib_df["hour"] == hr]
            if asymmetric_cqr:
                off_low = compute_conformal_offset(sub["e_low"].values, alpha_side)
                off_high = compute_conformal_offset(sub["e_high"].values, alpha_side)
            else:
                sym = np.maximum(sub["e_low"].values, sub["e_high"].values)
                off = compute_conformal_offset(sym, alpha_side)
                off_low = off_high = off
            conformal_offsets[pair_key][str(hr)] = {"low": float(off_low), "high": float(off_high)}

    # ---- valutazione backtest (raw vs calibrata asimmetrica) ----
    coverage_rows = []
    for pair in conformal_pairs:
        q_low, q_high = pair
        nominal_coverage = q_high - q_low
        pair_key = f"{q_low}_{q_high}"
        bt_df = pd.concat(backtest_raw_by_pair[pair], ignore_index=True)
        bt_df["off_low"] = bt_df["hour"].map(lambda hr: conformal_offsets[pair_key][str(int(hr))]["low"])
        bt_df["off_high"] = bt_df["hour"].map(lambda hr: conformal_offsets[pair_key][str(int(hr))]["high"])
        bt_df["low_raw"] = bt_df["pred_low"]
        bt_df["high_raw"] = bt_df["pred_high"]
        bt_df["low_cal"] = bt_df["pred_low"] - bt_df["off_low"]
        bt_df["high_cal"] = bt_df["pred_high"] + bt_df["off_high"]
        bt_df["covered_raw"] = (bt_df["y"] >= bt_df["low_raw"]) & (bt_df["y"] <= bt_df["high_raw"])
        bt_df["covered_cal"] = (bt_df["y"] >= bt_df["low_cal"]) & (bt_df["y"] <= bt_df["high_cal"])

        for hr, g in bt_df.groupby("hour"):
            coverage_rows.append({
                "pair": pair_key, "hour": int(hr),
                "nominal_coverage": nominal_coverage,
                "empirical_coverage_raw": float(g["covered_raw"].mean()),
                "empirical_coverage_calibrated": float(g["covered_cal"].mean()),
                "mean_width_raw": float((g["high_raw"] - g["low_raw"]).mean()),
                "mean_width_calibrated": float((g["high_cal"] - g["low_cal"]).mean()),
                "mean_offset_low": float(g["off_low"].mean()),
                "mean_offset_high": float(g["off_high"].mean()),
                "n": int(len(g)),
            })
        print(f"Pair {pair_key} | nominale={nominal_coverage:.2f} | "
              f"RAW={float(bt_df['covered_raw'].mean()):.3f} | CAL={float(bt_df['covered_cal'].mean()):.3f}")

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["pair", "hour"])

    pinball_df = pd.DataFrame(pinball_rows)
    pinball_by_hour_quantile = (
        pinball_df.groupby(["hour", "quantile"])
        .apply(lambda g: float(np.average(g["pinball"], weights=g["n"])))
        .reset_index(name="pinball_weighted")
    )
    overall_pinball = (
        pinball_df.groupby("quantile")
        .apply(lambda g: float(np.average(g["pinball"], weights=g["n"])))
    )

    # ---- salvataggio ----
    model_path = os.path.join(output_dir, MODEL_QUANTILES_NAME)
    metadata_path = os.path.join(output_dir, METADATA_NAME)
    pinball_path = os.path.join(output_dir, PINBALL_BY_HOUR_NAME)
    coverage_path = os.path.join(output_dir, COVERAGE_BY_HOUR_NAME)

    joblib.dump(models_q, model_path)
    pinball_by_hour_quantile.to_csv(pinball_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)

    metadata = {
        "market": "MI",
        "target_col": target_col,
        "freq": FREQ,
        "steps": steps,
        "quantile_levels": quantile_levels,
        "point_quantile": point_q,
        "point_cost_ratio": point_cost_ratio,
        "asymmetric_cqr": asymmetric_cqr,
        "conformal_pairs": [[q_low, q_high] for (q_low, q_high) in conformal_pairs],
        "conformal_offsets": conformal_offsets,   # {pair: {hour: {"low":, "high":}}}
        "calib_days": calib_days,
        "backtest_days": backtest_days,
        "retrain_on_full_data": retrain_on_full_data,
        "hparam_scheme": "horizon_bucketed",
        "lgbm_params_overrides": lgbm_params_overrides,
        "has_holidays_lib": HAS_HOLIDAYS,
        "future_known_exog": future_known_exog,
        "base_feature_cols": base_feature_cols,
        "selected_features_by_horizon": selected_features_by_horizon,
        "model_quantiles_file": MODEL_QUANTILES_NAME,
        "pinball_by_hour_file": PINBALL_BY_HOUR_NAME,
        "coverage_by_hour_file": COVERAGE_BY_HOUR_NAME,
        "overall_pinball_by_quantile": {str(q): float(v) for q, v in overall_pinball.items()},
        "trained_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"SALVATO in {output_dir} (target='{target_col}', punto=q{point_q})")

    return {
        "models_q": models_q,
        "metadata": metadata,
        "pinball_by_hour_quantile": pinball_by_hour_quantile,
        "coverage_df": coverage_df,
    }
