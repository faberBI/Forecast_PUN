# backtest_delta_sign_intensity_improved.py
# Obiettivo: stimare SEGNO e INTENSITA del delta = MI_Nord - PUN
# No strategia trading, no PnL, no LONG/SHORT operativo.
# Output principali:
# - probabilita P(delta>0)
# - sign_pred
# - delta_reg_pred_raw
# - delta_pred_final = sign_pred * abs(delta_reg_pred_raw)
# - metriche segno + metriche intensita

import os
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.dummy import DummyClassifier, DummyRegressor
import joblib

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ============================================================
# METRICHE
# ============================================================

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    den = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / den))


def wmape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_pred - y_true)) / (np.sum(np.abs(y_true)) + eps))


def mape_safe(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def sign_array(x, deadband=0.0):
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x), dtype=int)
    out[x > deadband] = 1
    out[x < -deadband] = -1
    return out


def sign_metrics(y_true_delta, y_pred_delta, deadband=0.0, prefix=""):
    y_true_delta = np.asarray(y_true_delta, dtype=float)
    y_pred_delta = np.asarray(y_pred_delta, dtype=float)

    s_true = sign_array(y_true_delta, deadband=deadband)
    s_pred = sign_array(y_pred_delta, deadband=deadband)

    mask = s_true != 0
    if mask.sum() == 0:
        return {
            f"{prefix}sign_match_count": 0,
            f"{prefix}sign_eval_count": 0,
            f"{prefix}sign_accuracy": np.nan,
        }

    correct = s_true[mask] == s_pred[mask]
    return {
        f"{prefix}sign_match_count": int(correct.sum()),
        f"{prefix}sign_eval_count": int(mask.sum()),
        f"{prefix}sign_accuracy": float(correct.mean()),
    }


def regression_metrics(y_true, y_pred, prefix=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}r2": float(r2_score(y_true, y_pred)) if len(y_true) > 2 else np.nan,
        f"{prefix}mape": mape_safe(y_true, y_pred),
        f"{prefix}smape": smape(y_true, y_pred),
        f"{prefix}wmape": wmape(y_true, y_pred),
    }


def safe_auc(y_true_binary, y_score):
    try:
        y_true_binary = np.asarray(y_true_binary).astype(int)
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true_binary)) < 2:
            return np.nan
        return float(roc_auc_score(y_true_binary, y_score))
    except Exception:
        return np.nan


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return float(obj)
    return obj


# ============================================================
# DATETIME E MERGE
# ============================================================

def infer_datetime(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df["__dt__"] = pd.to_datetime(df.index)
        return df

    candidates = [
        "datetime", "date_time", "timestamp", "time", "DateTime", "Timestamp",
        "DATA_ORA", "DataOra", "data_ora", "date", "Date", "DATA"
    ]

    for c in candidates:
        if c in df.columns:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.80:
                df["__dt__"] = parsed
                return df

    needed = {"year", "day_of_year", "quarter_of_day"}
    if needed.issubset(set(df.columns)):
        y = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        doy = pd.to_numeric(df["day_of_year"], errors="coerce").astype("Int64")
        qod_raw = pd.to_numeric(df["quarter_of_day"], errors="coerce")

        q_min = np.nanmin(qod_raw.values)
        qod = qod_raw - 1 if q_min >= 1 else qod_raw

        base = pd.to_datetime(y.astype(str) + "-01-01", errors="coerce")
        df["__dt__"] = base + pd.to_timedelta(doy - 1, unit="D") + pd.to_timedelta(qod * 15, unit="m")
        return df

    needed2 = {"year", "month", "day", "hour", "minute"}
    if needed2.issubset(set(df.columns)):
        df["__dt__"] = pd.to_datetime(
            dict(
                year=df["year"],
                month=df["month"],
                day=df["day"],
                hour=df["hour"],
                minute=df["minute"],
            ),
            errors="coerce",
        )
        return df

    raise ValueError(
        f"Non riesco a ricostruire la datetime per {name}. "
        f"Serve DatetimeIndex, colonna datetime/timestamp, oppure year + day_of_year + quarter_of_day."
    )


def load_and_merge(pun_path: str, mi_path: str, pun_target: str, mi_target: str) -> pd.DataFrame:
    pun = pd.read_parquet(pun_path)
    mi = pd.read_parquet(mi_path)

    pun = infer_datetime(pun, "PUN")
    mi = infer_datetime(mi, "MI_Nord")

    if pun_target not in pun.columns:
        raise ValueError(f"Colonna target PUN non trovata: {pun_target}")
    if mi_target not in mi.columns:
        raise ValueError(f"Colonna target MI non trovata: {mi_target}")

    pun = pun.sort_values("__dt__").drop_duplicates("__dt__")
    mi = mi.sort_values("__dt__").drop_duplicates("__dt__")

    pun = pun.rename(columns={pun_target: "__PUN__"})
    mi = mi.rename(columns={mi_target: "__MI_NORD__"})

    merged = pd.merge(mi, pun, on="__dt__", how="inner", suffixes=("_mi", "_pun"))
    merged = merged.sort_values("__dt__").reset_index(drop=True)

    merged["__PUN__"] = pd.to_numeric(merged["__PUN__"], errors="coerce")
    merged["__MI_NORD__"] = pd.to_numeric(merged["__MI_NORD__"], errors="coerce")
    merged = merged.dropna(subset=["__PUN__", "__MI_NORD__"]).copy()

    merged["delta_real_t"] = merged["__MI_NORD__"] - merged["__PUN__"]
    return merged


# ============================================================
# FEATURE ENGINEERING AVANZATO
# ============================================================

LEAKAGE_COLUMNS = {
    "__PUN__", "__MI_NORD__", "PUN", "Nord",
    "delta_real_t", "delta_target", "sign_target", "abs_delta_target", "__dt__"
}


def signed_streak_from_shifted_sign(x: pd.Series) -> pd.Series:
    arr = np.sign(x.fillna(0).values)
    out = np.zeros(len(arr), dtype=float)
    prev = 0
    count = 0
    for i, v in enumerate(arr):
        if v == 0:
            count = 0
            out[i] = 0
            prev = 0
            continue
        if v == prev:
            count += 1
        else:
            count = 1
        out[i] = count * v
        prev = v
    return pd.Series(out, index=x.index)


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["delta_real_t"]
    shifted = d.shift(1)

    for lag in [1, 2, 3, 4, 8, 12, 16, 24, 48, 96, 192, 672]:
        df[f"delta_lag_{lag}"] = d.shift(lag)
        df[f"delta_abs_lag_{lag}"] = d.shift(lag).abs()
        df[f"delta_sign_lag_{lag}"] = np.sign(d.shift(lag))

    for win in [4, 8, 16, 24, 48, 96, 192, 672]:
        df[f"delta_roll_mean_{win}"] = shifted.rolling(win).mean()
        df[f"delta_roll_std_{win}"] = shifted.rolling(win).std()
        df[f"delta_roll_min_{win}"] = shifted.rolling(win).min()
        df[f"delta_roll_max_{win}"] = shifted.rolling(win).max()
        df[f"delta_roll_median_{win}"] = shifted.rolling(win).median()
        df[f"delta_pos_ratio_{win}"] = (shifted > 0).rolling(win).mean()
        df[f"delta_neg_ratio_{win}"] = (shifted < 0).rolling(win).mean()

        rmean = shifted.rolling(win).mean()
        rstd = shifted.rolling(win).std()
        df[f"delta_zscore_{win}"] = (shifted - rmean) / (rstd + 1e-9)

    for gap in [2, 4, 8, 16, 24, 48, 96]:
        df[f"delta_momentum_{gap}"] = d.shift(1) - d.shift(1 + gap)

    df["delta_streak"] = signed_streak_from_shifted_sign(d.shift(1))
    df["delta_streak_abs"] = df["delta_streak"].abs()
    df["delta_streak_pos"] = (df["delta_streak"] > 0).astype(int)
    df["delta_streak_neg"] = (df["delta_streak"] < 0).astype(int)

    mi = df["__MI_NORD__"]
    pun = df["__PUN__"]
    for lag in [1, 2, 4, 8, 16, 24, 96]:
        mi_lag = mi.shift(lag)
        pun_lag = pun.shift(lag)
        df[f"mi_lag_{lag}"] = mi_lag
        df[f"pun_lag_{lag}"] = pun_lag
        df[f"mi_pun_ratio_lag_{lag}"] = mi_lag / (pun_lag.abs() + 1e-9)
        df[f"mi_minus_pun_lag_{lag}"] = mi_lag - pun_lag

    candidate_pairs = [
        ("pun_ret_1h", "mi_ret_1h"),
        ("pun_ret_1d", "mi_ret_1d"),
        ("pun_ret_7d", "mi_ret_7d"),
    ]
    for a, b in candidate_pairs:
        cols_a = [c for c in df.columns if c == a or c.endswith("_" + a)]
        cols_b = [c for c in df.columns if c == b or c.endswith("_" + b)]
        if cols_a and cols_b:
            ca = cols_a[0]
            cb = cols_b[0]
            df[f"spread_ret_diff_{a}_{b}"] = pd.to_numeric(df[cb], errors="coerce") - pd.to_numeric(df[ca], errors="coerce")
            df[f"spread_ret_sum_abs_{a}_{b}"] = pd.to_numeric(df[cb], errors="coerce").abs() + pd.to_numeric(df[ca], errors="coerce").abs()

    return df


def create_targets(df: pd.DataFrame, horizon_steps: int, sign_deadband: float) -> pd.DataFrame:
    df = df.copy()
    df["delta_target"] = df["delta_real_t"].shift(-horizon_steps)
    df["abs_delta_target"] = df["delta_target"].abs()

    sign_target = np.full(len(df), np.nan)
    sign_target[df["delta_target"] > sign_deadband] = 1
    sign_target[df["delta_target"] < -sign_deadband] = 0
    df["sign_target"] = sign_target
    return df


def select_features(df: pd.DataFrame, allow_current_prices: bool = False, max_missing_ratio: float = 0.40) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    forbidden = set(LEAKAGE_COLUMNS)

    if not allow_current_prices:
        forbidden.update({"__PUN__", "__MI_NORD__", "PUN", "Nord", "PUN_pun", "Nord_mi", "Nord_pun", "PUN_mi"})

    features = []
    for c in numeric_cols:
        base = c.replace("_mi", "").replace("_pun", "")
        if c in forbidden or base in forbidden:
            continue
        if c.startswith("delta_target") or c.startswith("sign_target") or c.startswith("abs_delta_target"):
            continue
        miss = df[c].isna().mean()
        if miss <= max_missing_ratio:
            features.append(c)

    features = sorted(list(set(features)))
    if not features:
        raise ValueError("Nessuna feature selezionata. Controlla colonne numeriche e missing ratio.")
    return features


# ============================================================
# MODELLI
# ============================================================

def make_classifier(random_state=42):
    if HAS_LGBM:
        return LGBMClassifier(
            objective="binary",
            n_estimators=1200,
            learning_rate=0.020,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.20,
            reg_lambda=1.30,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", HistGradientBoostingClassifier(max_iter=800, learning_rate=0.030, max_leaf_nodes=31, l2_regularization=0.20, random_state=random_state)),
    ])


def make_lgb_regressor(random_state=42):
    if HAS_LGBM:
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=1200,
            learning_rate=0.020,
            num_leaves=64,
            max_depth=-1,
            min_child_samples=60,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.15,
            reg_lambda=1.30,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
        ("model", HistGradientBoostingRegressor(max_iter=800, learning_rate=0.030, max_leaf_nodes=31, l2_regularization=0.20, loss="absolute_error", random_state=random_state)),
    ])


def make_rf_regressor(random_state=42, n_estimators=250):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(n_estimators=n_estimators, max_depth=18, min_samples_leaf=5, max_features="sqrt", random_state=random_state, n_jobs=-1)),
    ])


def make_et_regressor(random_state=42, n_estimators=250):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=n_estimators, max_depth=18, min_samples_leaf=4, max_features="sqrt", random_state=random_state, n_jobs=-1)),
    ])


def compute_weight(delta, sign_deadband=0.0):
    a = np.abs(np.asarray(delta, dtype=float))
    valid = a[np.isfinite(a) & (a > sign_deadband)]
    scale = np.nanmedian(valid) if len(valid) else 1.0
    scale = max(scale, 1e-6)
    w = 1.0 + np.minimum(a / scale, 5.0)
    w[~np.isfinite(w)] = 1.0
    return w


def fit_optional_weight(model, X, y, w=None):
    if w is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=w)
        return model
    except Exception:
        pass
    try:
        model.fit(X, y, model__sample_weight=w)
        return model
    except Exception:
        pass
    model.fit(X, y)
    return model


def predict_proba_up(model, X):
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        return np.asarray(pred, dtype=float)

    p = model.predict_proba(X)
    classes = getattr(model, "classes_", np.array([0, 1]))

    if p.shape[1] == 1:
        only_class = int(classes[0])
        return np.ones(len(X)) if only_class == 1 else np.zeros(len(X))

    idx_up = list(classes).index(1) if 1 in classes else 1
    return p[:, idx_up]


@dataclass
class FittedModels:
    clf: object
    reg_lgb: object
    reg_rf: object
    reg_et: object


def fit_models(train_df, features, sign_deadband, random_state, use_tree_ensemble=True, n_estimators_tree=250):
    X = train_df[features]
    y_sign = train_df["sign_target"].astype(int)
    y_delta = train_df["delta_target"].astype(float)
    w = compute_weight(y_delta.values, sign_deadband=sign_deadband)

    if len(np.unique(y_sign)) < 2:
        clf = DummyClassifier(strategy="constant", constant=int(y_sign.iloc[0]))
    else:
        clf = make_classifier(random_state=random_state)

    if y_delta.nunique(dropna=True) <= 1:
        reg_lgb = DummyRegressor(strategy="mean")
        reg_rf = DummyRegressor(strategy="mean")
        reg_et = DummyRegressor(strategy="mean")
    else:
        reg_lgb = make_lgb_regressor(random_state=random_state)
        if use_tree_ensemble:
            reg_rf = make_rf_regressor(random_state=random_state + 11, n_estimators=n_estimators_tree)
            reg_et = make_et_regressor(random_state=random_state + 22, n_estimators=n_estimators_tree)
        else:
            reg_rf = DummyRegressor(strategy="mean")
            reg_et = DummyRegressor(strategy="mean")

    clf = fit_optional_weight(clf, X, y_sign, w)
    reg_lgb = fit_optional_weight(reg_lgb, X, y_delta, w)

    if use_tree_ensemble:
        reg_rf = fit_optional_weight(reg_rf, X, y_delta, w)
        reg_et = fit_optional_weight(reg_et, X, y_delta, w)
    else:
        reg_rf.fit(X, y_delta)
        reg_et.fit(X, y_delta)

    return FittedModels(clf=clf, reg_lgb=reg_lgb, reg_rf=reg_rf, reg_et=reg_et)


def predict_delta_ensemble(models: FittedModels, X, use_tree_ensemble=True, w_lgb=0.60, w_rf=0.20, w_et=0.20):
    p_lgb = np.asarray(models.reg_lgb.predict(X), dtype=float)
    if not use_tree_ensemble:
        return p_lgb, p_lgb, p_lgb, p_lgb

    p_rf = np.asarray(models.reg_rf.predict(X), dtype=float)
    p_et = np.asarray(models.reg_et.predict(X), dtype=float)
    p_ens = w_lgb * p_lgb + w_rf * p_rf + w_et * p_et
    return p_ens, p_lgb, p_rf, p_et


@dataclass
class ThresholdConfig:
    threshold_up: float


def optimize_sign_threshold(y_val_sign, p_up_val):
    y_val_sign = np.asarray(y_val_sign, dtype=int)
    p_up_val = np.asarray(p_up_val, dtype=float)

    best_score = -1e18
    best_thr = 0.50

    for thr in np.round(np.arange(0.35, 0.651, 0.01), 2):
        pred = (p_up_val >= thr).astype(int)
        acc = accuracy_score(y_val_sign, pred)
        bacc = balanced_accuracy_score(y_val_sign, pred)
        mcc = matthews_corrcoef(y_val_sign, pred)
        f1 = f1_score(y_val_sign, pred, zero_division=0)
        score = 2.0 * bacc + 1.0 * mcc + 0.5 * f1 + 0.25 * acc
        if score > best_score:
            best_score = score
            best_thr = float(thr)

    return ThresholdConfig(threshold_up=best_thr)


# ============================================================
# WALK-FORWARD
# ============================================================

def split_train_val(train_df, val_days):
    max_dt = train_df["__dt__"].max()
    val_start = max_dt - pd.Timedelta(days=val_days)
    tr = train_df[train_df["__dt__"] < val_start].copy()
    val = train_df[train_df["__dt__"] >= val_start].copy()
    if len(tr) < 500 or len(val) < 100:
        cut = int(len(train_df) * 0.80)
        tr = train_df.iloc[:cut].copy()
        val = train_df.iloc[cut:].copy()
    return tr, val


def evaluate_full(y_delta, delta_raw, delta_final, p_up, sign_pred, sign_deadband):
    y_delta = np.asarray(y_delta, dtype=float)
    p_up = np.asarray(p_up, dtype=float)
    sign_true_bin = (y_delta > sign_deadband).astype(int)
    sign_pred_bin = (np.asarray(sign_pred) > 0).astype(int)

    out = {"n": int(len(y_delta))}
    out.update(regression_metrics(y_delta, delta_raw, prefix="raw_"))
    out.update(regression_metrics(y_delta, delta_final, prefix="final_"))
    out.update(sign_metrics(y_delta, delta_raw, deadband=sign_deadband, prefix="raw_"))
    out.update(sign_metrics(y_delta, delta_final, deadband=sign_deadband, prefix="final_"))

    out.update({
        "classifier_accuracy": float(accuracy_score(sign_true_bin, sign_pred_bin)),
        "classifier_balanced_accuracy": float(balanced_accuracy_score(sign_true_bin, sign_pred_bin)),
        "classifier_precision_up": float(precision_score(sign_true_bin, sign_pred_bin, zero_division=0)),
        "classifier_recall_up": float(recall_score(sign_true_bin, sign_pred_bin, zero_division=0)),
        "classifier_f1_up": float(f1_score(sign_true_bin, sign_pred_bin, zero_division=0)),
        "classifier_mcc": float(matthews_corrcoef(sign_true_bin, sign_pred_bin)),
        "classifier_auc": safe_auc(sign_true_bin, p_up),
        "mean_real_delta": float(np.mean(y_delta)),
        "mean_raw_delta_pred": float(np.mean(delta_raw)),
        "mean_final_delta_pred": float(np.mean(delta_final)),
        "std_real_delta": float(np.std(y_delta)),
        "std_raw_delta_pred": float(np.std(delta_raw)),
        "std_final_delta_pred": float(np.std(delta_final)),
    })
    return out


def walk_forward_backtest(df, features, test_days, refit_days, val_days, sign_deadband, random_state, use_tree_ensemble=True, n_estimators_tree=250):
    max_dt = df["__dt__"].max()
    test_start = max_dt - pd.Timedelta(days=test_days)

    all_preds = []
    fold_rows = []
    current_start = test_start
    fold = 0

    while current_start <= max_dt:
        current_end = min(current_start + pd.Timedelta(days=refit_days), max_dt + pd.Timedelta(minutes=15))

        train_full = df[df["__dt__"] < current_start].copy()
        test_block = df[(df["__dt__"] >= current_start) & (df["__dt__"] < current_end)].copy()

        train_full = train_full.dropna(subset=["delta_target", "sign_target"])
        test_block = test_block.dropna(subset=["delta_target", "sign_target"])

        if len(train_full) < 1000 or len(test_block) == 0:
            current_start = current_end
            continue

        tr, val = split_train_val(train_full, val_days=val_days)
        tr = tr.dropna(subset=["delta_target", "sign_target"])
        val = val.dropna(subset=["delta_target", "sign_target"])

        if len(tr) < 500 or len(val) < 100:
            current_start = current_end
            continue

        models_inner = fit_models(tr, features, sign_deadband, random_state + fold, use_tree_ensemble, n_estimators_tree)
        X_val = val[features]
        p_up_val = predict_proba_up(models_inner.clf, X_val)
        y_val_sign = val["sign_target"].astype(int).values
        cfg = optimize_sign_threshold(y_val_sign, p_up_val)

        models = fit_models(train_full, features, sign_deadband, random_state + 1000 + fold, use_tree_ensemble, n_estimators_tree)

        X_test = test_block[features]
        y_test_delta = test_block["delta_target"].values

        p_up = predict_proba_up(models.clf, X_test)
        sign_pred = np.where(p_up >= cfg.threshold_up, 1, -1)

        delta_raw, delta_lgb, delta_rf, delta_et = predict_delta_ensemble(models, X_test, use_tree_ensemble=use_tree_ensemble)
        delta_final = sign_pred * np.abs(delta_raw)

        out = test_block[["__dt__", "delta_real_t", "delta_target"]].copy()
        out["fold"] = fold
        out["threshold_up"] = cfg.threshold_up
        out["p_up"] = p_up
        out["p_down"] = 1.0 - p_up
        out["sign_real"] = sign_array(out["delta_target"].values, deadband=sign_deadband)
        out["sign_pred_classifier"] = sign_pred
        out["delta_pred_raw"] = delta_raw
        out["delta_pred_final"] = delta_final
        out["delta_pred_lgb"] = delta_lgb
        out["delta_pred_rf"] = delta_rf
        out["delta_pred_et"] = delta_et
        out["sign_pred_raw"] = sign_array(out["delta_pred_raw"].values, deadband=sign_deadband)
        out["sign_pred_final"] = sign_array(out["delta_pred_final"].values, deadband=sign_deadband)
        out["correct_sign_raw"] = (out["sign_real"] == out["sign_pred_raw"]).astype(float)
        out["correct_sign_final"] = (out["sign_real"] == out["sign_pred_final"]).astype(float)
        out.loc[out["sign_real"] == 0, ["correct_sign_raw", "correct_sign_final"]] = np.nan

        all_preds.append(out)

        m_fold = evaluate_full(y_test_delta, delta_raw, delta_final, p_up, sign_pred, sign_deadband)
        row = {
            "fold": fold,
            "test_start": str(current_start),
            "test_end": str(current_end),
            "threshold_up": cfg.threshold_up,
        }
        row.update({f"fold_{k}": v for k, v in m_fold.items()})
        fold_rows.append(row)

        print(
            f"[FOLD {fold:03d}] {current_start} -> {current_end} | "
            f"thr={cfg.threshold_up:.2f} | "
            f"sign_raw={m_fold['raw_sign_accuracy']:.3f} | "
            f"sign_final={m_fold['final_sign_accuracy']:.3f} | "
            f"MCC={m_fold['classifier_mcc']:.3f} | "
            f"R2_raw={m_fold['raw_r2']:.3f} | R2_final={m_fold['final_r2']:.3f}"
        )

        current_start = current_end
        fold += 1

    if len(all_preds) == 0:
        raise RuntimeError("Nessun fold prodotto. Controlla date, test-days, refit-days, val-days e datetime.")

    preds = pd.concat(all_preds, axis=0).sort_values("__dt__").reset_index(drop=True)
    metrics = evaluate_full(preds["delta_target"].values, preds["delta_pred_raw"].values, preds["delta_pred_final"].values, preds["p_up"].values, preds["sign_pred_classifier"].values, sign_deadband)

    return preds, metrics, pd.DataFrame(fold_rows)


# ============================================================
# REPORT E GRAFICI
# ============================================================

def add_time_buckets(preds):
    preds = preds.copy()
    dt = pd.to_datetime(preds["__dt__"])
    preds["hour"] = dt.dt.hour
    preds["quarter"] = dt.dt.minute // 15
    preds["quarter_of_day"] = preds["hour"] * 4 + preds["quarter"] + 1
    bins = [0, 24, 48, 72, 96]
    labels = ["Q01_Q24", "Q25_Q48", "Q49_Q72", "Q73_Q96"]
    preds["bucket_qday"] = pd.cut(preds["quarter_of_day"], bins=bins, labels=labels, include_lowest=True)
    return preds


def bucket_report(preds, sign_deadband):
    rows = []
    for b, g in preds.groupby("bucket_qday", observed=True):
        if len(g) == 0:
            continue
        m = evaluate_full(g["delta_target"].values, g["delta_pred_raw"].values, g["delta_pred_final"].values, g["p_up"].values, g["sign_pred_classifier"].values, sign_deadband)
        m["bucket"] = str(b)
        rows.append(m)
    return pd.DataFrame(rows)


def save_plots(preds, outdir):
    os.makedirs(outdir, exist_ok=True)
    p = preds.copy()

    plt.figure(figsize=(16, 6))
    plt.plot(p["__dt__"], p["delta_target"], label="Delta reale", linewidth=1.0)
    plt.plot(p["__dt__"], p["delta_pred_raw"], label="Delta predetto raw ensemble", linewidth=1.0)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Delta reale vs delta predetto raw ensemble")
    plt.xlabel("Data")
    plt.ylabel("MI Nord - PUN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delta_real_vs_pred_raw.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(16, 6))
    plt.plot(p["__dt__"], p["delta_target"], label="Delta reale", linewidth=1.0)
    plt.plot(p["__dt__"], p["delta_pred_final"], label="Delta predetto finale sign+intensita", linewidth=1.0)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Delta reale vs delta predetto finale")
    plt.xlabel("Data")
    plt.ylabel("MI Nord - PUN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delta_real_vs_pred_final.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(16, 5))
    plt.plot(p["__dt__"], p["p_up"], label="P(delta > 0)", linewidth=1.0)
    plt.axhline(0.5, color="black", linewidth=0.8)
    plt.title("Probabilita stimata di delta positivo")
    plt.xlabel("Data")
    plt.ylabel("P(delta > 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "proba_up.png"), dpi=160)
    plt.close()

    tmp = p.copy()
    tmp["hit_raw_roll_96"] = tmp["correct_sign_raw"].rolling(96).mean()
    tmp["hit_final_roll_96"] = tmp["correct_sign_final"].rolling(96).mean()
    plt.figure(figsize=(16, 5))
    plt.plot(tmp["__dt__"], tmp["hit_raw_roll_96"], label="Raw sign accuracy rolling 96", linewidth=1.0)
    plt.plot(tmp["__dt__"], tmp["hit_final_roll_96"], label="Final sign accuracy rolling 96", linewidth=1.0)
    plt.axhline(0.50, color="black", linewidth=0.8)
    plt.axhline(0.60, color="orange", linewidth=0.8)
    plt.axhline(0.70, color="green", linewidth=0.8)
    plt.title("Directional accuracy rolling")
    plt.xlabel("Data")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "directional_accuracy_rolling.png"), dpi=160)
    plt.close()

    for col, fname, title in [
        ("delta_pred_raw", "scatter_delta_raw.png", "Scatter delta reale vs predetto raw"),
        ("delta_pred_final", "scatter_delta_final.png", "Scatter delta reale vs predetto finale"),
    ]:
        plt.figure(figsize=(7, 7))
        plt.scatter(p["delta_target"], p[col], s=8, alpha=0.35)
        values = np.r_[p["delta_target"].values, p[col].values]
        values = values[np.isfinite(values)]
        lim = 1.0 if len(values) == 0 else float(np.nanmax(np.abs(values)))
        lim = lim if np.isfinite(lim) and lim > 0 else 1.0
        plt.plot([-lim, lim], [-lim, lim], color="black", linewidth=1.0)
        plt.axhline(0, color="gray", linewidth=0.8)
        plt.axvline(0, color="gray", linewidth=0.8)
        plt.title(title)
        plt.xlabel("Delta reale")
        plt.ylabel("Delta predetto")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=160)
        plt.close()


def save_feature_importance(models, features, outdir):
    rows = []
    if HAS_LGBM and hasattr(models.clf, "feature_importances_"):
        rows.append(pd.DataFrame({"model": "classifier_sign", "feature": features, "importance": models.clf.feature_importances_}))
    if HAS_LGBM and hasattr(models.reg_lgb, "feature_importances_"):
        rows.append(pd.DataFrame({"model": "regressor_lgb_delta", "feature": features, "importance": models.reg_lgb.feature_importances_}))
    if not rows:
        return
    imp = pd.concat(rows, axis=0)
    imp.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)
    top = imp.groupby("feature")["importance"].mean().sort_values(ascending=False).head(50).sort_values()
    plt.figure(figsize=(11, 12))
    plt.barh(top.index, top.values)
    plt.title("Top feature importance media")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importance_top50.png"), dpi=160)
    plt.close()


def fit_and_save_final_models(df, features, outdir, sign_deadband, random_state, use_tree_ensemble, n_estimators_tree):
    train = df.dropna(subset=["delta_target", "sign_target"]).copy()
    models = fit_models(train, features, sign_deadband, random_state, use_tree_ensemble, n_estimators_tree)
    joblib.dump({"models": models, "features": features, "sign_deadband": sign_deadband, "use_tree_ensemble": use_tree_ensemble}, os.path.join(outdir, "final_models.joblib"))
    save_feature_importance(models, features, outdir)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pun-data", required=True)
    parser.add_argument("--mi-data", required=True)
    parser.add_argument("--outdir", default="out_delta_sign_intensity_improved")
    parser.add_argument("--pun-target", default="PUN")
    parser.add_argument("--mi-target", default="Nord")
    parser.add_argument("--horizon-steps", type=int, default=1)
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--refit-days", type=int, default=7)
    parser.add_argument("--val-days", type=int, default=30)
    parser.add_argument("--sign-deadband", type=float, default=0.0)
    parser.add_argument("--max-missing-ratio", type=float, default=0.40)
    parser.add_argument("--allow-current-prices", action="store_true")
    parser.add_argument("--no-tree-ensemble", action="store_true")
    parser.add_argument("--n-estimators-tree", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    use_tree_ensemble = not args.no_tree_ensemble

    print("============================================================")
    print("BACKTEST DELTA SIGN + INTENSITY IMPROVED")
    print("Target: delta = MI_Nord - PUN")
    print("Modello segno: LightGBMClassifier")
    print("Modello intensita: ensemble LGBM + RF + ExtraTrees")
    print("No strategia trading, solo segno e intensita")
    print("============================================================")

    print("[INFO] Caricamento parquet...")
    df = load_and_merge(args.pun_data, args.mi_data, args.pun_target, args.mi_target)
    print(f"[INFO] Righe dopo merge: {len(df):,}")
    print(f"[INFO] Range date: {df['__dt__'].min()} -> {df['__dt__'].max()}")

    print("[INFO] Feature engineering avanzato...")
    df = add_delta_features(df)

    print(f"[INFO] Creo target forward horizon_steps={args.horizon_steps}")
    df = create_targets(df, args.horizon_steps, args.sign_deadband)

    features = select_features(df, allow_current_prices=args.allow_current_prices, max_missing_ratio=args.max_missing_ratio)
    print(f"[INFO] Feature selezionate: {len(features)}")

    with open(os.path.join(args.outdir, "features.json"), "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    df_model = df.dropna(subset=["delta_target", "sign_target"]).copy()
    for c in features:
        df_model[c] = pd.to_numeric(df_model[c], errors="coerce")
    df_model[features] = df_model[features].replace([np.inf, -np.inf], np.nan)

    print("[INFO] Distribuzione sign_target:")
    print("0 = delta futuro negativo, 1 = delta futuro positivo")
    print(df_model["sign_target"].value_counts(dropna=False).sort_index().to_string())

    print("[INFO] Avvio walk-forward...")
    preds, metrics, folds = walk_forward_backtest(
        df=df_model,
        features=features,
        test_days=args.test_days,
        refit_days=args.refit_days,
        val_days=args.val_days,
        sign_deadband=args.sign_deadband,
        random_state=args.random_state,
        use_tree_ensemble=use_tree_ensemble,
        n_estimators_tree=args.n_estimators_tree,
    )

    preds = add_time_buckets(preds)
    buckets = bucket_report(preds, args.sign_deadband)

    preds_path = os.path.join(args.outdir, "predictions_delta_sign_intensity.csv")
    metrics_path = os.path.join(args.outdir, "metrics_global.json")
    folds_path = os.path.join(args.outdir, "fold_thresholds.csv")
    buckets_path = os.path.join(args.outdir, "metrics_by_quarter_bucket.csv")

    preds.to_csv(preds_path, index=False)
    folds.to_csv(folds_path, index=False)
    buckets.to_csv(buckets_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)

    print("[INFO] Salvo grafici...")
    save_plots(preds, args.outdir)

    print("[INFO] Fit finale e salvataggio modelli...")
    fit_and_save_final_models(df_model, features, args.outdir, args.sign_deadband, args.random_state, use_tree_ensemble, args.n_estimators_tree)

    print("")
    print("==================== METRICHE GLOBALI ====================")
    keys = [
        "raw_sign_match_count", "raw_sign_eval_count", "raw_sign_accuracy",
        "final_sign_match_count", "final_sign_eval_count", "final_sign_accuracy",
        "classifier_accuracy", "classifier_balanced_accuracy", "classifier_mcc", "classifier_auc",
        "raw_mae", "raw_rmse", "raw_r2", "raw_mape", "raw_smape", "raw_wmape",
        "final_mae", "final_rmse", "final_r2", "final_mape", "final_smape", "final_wmape",
        "mean_real_delta", "mean_raw_delta_pred", "mean_final_delta_pred",
        "std_real_delta", "std_raw_delta_pred", "std_final_delta_pred",
    ]
    for k in keys:
        if k in metrics:
            v = metrics[k]
            if isinstance(v, float):
                if np.isnan(v):
                    print(f"{k:35s}: nan")
                else:
                    print(f"{k:35s}: {v:.6f}")
            else:
                print(f"{k:35s}: {v}")

    print("")
    print("==================== LETTURA RAPIDA ====================")
    print(f"Raw ensemble: segno corretto {metrics['raw_sign_match_count']} su {metrics['raw_sign_eval_count']} ({metrics['raw_sign_accuracy']:.2%}).")
    print(f"Final sign+intensita: segno corretto {metrics['final_sign_match_count']} su {metrics['final_sign_eval_count']} ({metrics['final_sign_accuracy']:.2%}).")
    print(f"Classifier: balanced accuracy {metrics['classifier_balanced_accuracy']:.2%}, MCC {metrics['classifier_mcc']:.3f}, AUC {metrics['classifier_auc']:.3f}.")

    print("")
    print("==================== OUTPUT GENERATI ====================")
    print(f"Predizioni complete       : {preds_path}")
    print(f"Metriche globali          : {metrics_path}")
    print(f"Soglie per fold           : {folds_path}")
    print(f"Metriche per bucket       : {buckets_path}")
    print(f"Feature list              : {os.path.join(args.outdir, 'features.json')}")
    print(f"Modelli finali            : {os.path.join(args.outdir, 'final_models.joblib')}")
    print(f"Grafico delta raw         : {os.path.join(args.outdir, 'delta_real_vs_pred_raw.png')}")
    print(f"Grafico delta finale      : {os.path.join(args.outdir, 'delta_real_vs_pred_final.png')}")
    print(f"Grafico probabilita UP    : {os.path.join(args.outdir, 'proba_up.png')}")
    print(f"Rolling sign accuracy     : {os.path.join(args.outdir, 'directional_accuracy_rolling.png')}")
    print("==========================================================")


if __name__ == "__main__":
    main()
