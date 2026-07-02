# ============================================================
# MI DIRECT 96 - INFERENCE (serving)
# ============================================================
# Analogo di pun_direct_forecast.py, ma per i modelli MI v3 (per zona).
# Carica gli artefatti salvati da train_mi_direct e produce il forecast
# next-96 con punto (quantile configurabile) e banda CQR ASIMMETRICA.
#
# Riusa add_price_features / add_time_features da train_mi_direct
# => STESSE feature del training, nessun train/serve skew.
#
# Output di forecast_next_96 (colonne GENERICHE, zona-agnostiche):
#   Datetime, origin_time, horizon, hour,
#   q<liv> per ogni quantile (monotoni, crossing corretto),
#   pred   = quantile "punto" (da metadata point_quantile),
#   lower/upper = banda calibrata principale (coppia piu' ampia),
#   band_coverage
# ============================================================

import os
import json
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from train_mi_direct import (
    FREQ, STEPS,
    MODEL_QUANTILES_NAME, METADATA_NAME, HAS_HOLIDAYS,
    ensure_datetime_index, infer_and_fix_freq, safe_numeric_df,
    add_price_features, add_time_features_from_index,
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================
def load_direct_artifacts(model_dir: str) -> Dict:
    """Carica {models_q, metadata} per UNA zona dalla cartella models_mi/<zona>/."""
    model_path = os.path.join(model_dir, MODEL_QUANTILES_NAME)
    meta_path = os.path.join(model_dir, METADATA_NAME)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata non trovato: {meta_path}")

    models_q = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return {"models_q": models_q, "metadata": metadata}


# ============================================================
# COSTRUZIONE RIGA DI PREVISIONE (una origine, un orizzonte)
# ============================================================
def _build_pred_row(df_feat: pd.DataFrame, origin_ts: pd.Timestamp, horizon: int,
                    base_cols: List[str], future_known_exog: List[str]):
    row = df_feat.loc[[origin_ts], base_cols].copy()
    tgt = origin_ts + pd.Timedelta(FREQ) * horizon

    fut = add_time_features_from_index(pd.DatetimeIndex([tgt]), prefix="future_")
    fut.index = row.index
    row = row.join(fut)

    for c in [c for c in future_known_exog if c in df_feat.columns]:
        row[f"future_{c}"] = df_feat[c].get(tgt, np.nan)

    return row, tgt


def _widest_pair(conformal_pairs: List) -> tuple:
    pairs = [tuple(p) for p in conformal_pairs]
    return max(pairs, key=lambda p: p[1] - p[0])


# ============================================================
# FORECAST NEXT 96
# ============================================================
def forecast_next_96(df: pd.DataFrame, model_dir: str,
                     output_dir: Optional[str] = None, save_csv: bool = False) -> pd.DataFrame:
    """
    Previsione dei prossimi 96 quarti d'ora per la zona del modello in model_dir.
    `df` = storico della zona (deve contenere la colonna target del modello).
    """
    art = load_direct_artifacts(model_dir)
    models_q = art["models_q"]
    meta = art["metadata"]

    target_col = meta["target_col"]
    quantile_levels = [float(q) for q in meta["quantile_levels"]]
    point_q = float(meta["point_quantile"])
    conformal_pairs = [tuple(p) for p in meta["conformal_pairs"]]
    offsets = meta["conformal_offsets"]
    base_cols = meta["base_feature_cols"]
    selected = meta["selected_features_by_horizon"]
    future_known_exog = meta.get("future_known_exog", [])

    if target_col not in df.columns:
        raise ValueError(
            f"La colonna target del modello ('{target_col}') non è nel DataFrame passato. "
            f"Colonne disponibili: {list(df.columns)}"
        )

    df = ensure_datetime_index(df)
    df = infer_and_fix_freq(df, FREQ)
    df = safe_numeric_df(df)

    df_feat = add_price_features(df, target_col)

    valid = df_feat[df_feat[target_col].notna()]
    if valid.empty:
        raise ValueError(f"Nessun valore valido nel target '{target_col}'.")
    origin_ts = valid.index.max()

    q_str = [str(q) for q in quantile_levels]
    q_cols = [f"q{q}" for q in quantile_levels]

    rows = []
    for h in range(1, STEPS + 1):
        row, tgt = _build_pred_row(df_feat, origin_ts, h, base_cols, future_known_exog)
        cols = selected[str(h)]
        X_h = row.reindex(columns=cols)

        rec = {"origin_time": origin_ts, "Datetime": tgt, "horizon": h, "hour": int(tgt.hour)}
        for q, qs, qc in zip(quantile_levels, q_str, q_cols):
            rec[qc] = float(models_q[str(h)][qs].predict(X_h)[0])
        rows.append(rec)

    out = pd.DataFrame(rows)

    # --- fix quantile crossing: ordino i quantili previsti per riga ---
    order = np.argsort(quantile_levels)
    ordered_cols = [q_cols[i] for i in order]
    vals_sorted = np.sort(out[ordered_cols].values, axis=1)
    out[ordered_cols] = vals_sorted

    # --- punto = quantile configurato ---
    point_col = f"q{point_q}"
    if point_col not in out.columns:
        # fallback al piu' vicino
        nearest = min(quantile_levels, key=lambda q: abs(q - point_q))
        point_col = f"q{nearest}"
    out["pred"] = out[point_col]

    # --- banda calibrata principale (coppia piu' ampia) ---
    q_low, q_high = _widest_pair(conformal_pairs)
    key = f"{q_low}_{q_high}"
    off = offsets[key]
    low_col, high_col = f"q{q_low}", f"q{q_high}"
    out["lower"] = out[low_col] - out["hour"].map(lambda hr: off[str(int(hr))]["low"])
    out["upper"] = out[high_col] + out["hour"].map(lambda hr: off[str(int(hr))]["high"])
    out["band_coverage"] = float(q_high - q_low)

    # coerenza finale: lower <= pred <= upper
    out["lower"] = np.minimum(out["lower"], out["pred"])
    out["upper"] = np.maximum(out["upper"], out["pred"])

    out = out[["Datetime", "origin_time", "horizon", "hour"] + q_cols +
              ["pred", "lower", "upper", "band_coverage"]]

    if save_csv and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out.to_csv(os.path.join(output_dir, f"mi_forecast_{target_col}.csv".replace(" ", "_")), index=False)

    return out


# ============================================================
# IMPORTANZA FEATURE (gain nativo LightGBM, modelli del punto)
# ============================================================
def build_native_importance_df(models_q: Dict, metadata: Dict) -> pd.DataFrame:
    point_q = str(metadata.get("point_quantile", 0.5))
    selected = metadata.get("selected_features_by_horizon", {})

    rows = []
    for h, qd in models_q.items():
        pipe = qd.get(point_q) or qd.get("0.5") or next(iter(qd.values()))
        model = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else pipe
        imp = getattr(model, "feature_importances_", None)
        feats = selected.get(str(h))
        if imp is None or feats is None or len(imp) != len(feats):
            continue
        for f, v in zip(feats, imp):
            rows.append({"horizon": int(h), "feature": f, "importance": float(v)})

    return pd.DataFrame(rows)


def summarize_native_importance(importance_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if importance_df is None or importance_df.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    s = (importance_df.groupby("feature")["importance"].mean()
         .sort_values(ascending=False).head(top_n))
    return s.reset_index()
