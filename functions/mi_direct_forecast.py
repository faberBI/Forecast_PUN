# ============================================================
# MI DIRECT 96 - INFERENCE (serving)
# ============================================================
# Carica gli artefatti di UNA zona (models_mi/<zona>/) e produce il forecast
# next-96 con punto (quantile configurabile) e banda CQR asimmetrica.
# Riusa add_price_features / add_time_features da train_mi_direct
# => STESSE feature del training (nessun train/serve skew sulle feature base).
#
# Novità di questa versione:
#   - lower_floor: la banda inferiore non scende sotto un minimo (default 0),
#     SENZA censurare il punto (se il punto è negativo, lower segue il punto).
#   - esogene "note nel futuro" (FUTURE_KNOWN_EXOG): a serving il valore futuro
#     non è nello storico -> invece di NaN si usa la PERSISTENZA (ultimo valore
#     noto all'origine). Meglio della mediana dell'imputer.
#   - diagnose_forecast_inputs(): strumento per capire se una previsione "sballa"
#     (feature NaN all'origine, colonne mancanti, ecc.).
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
                    base_cols: List[str], future_known_exog: List[str], last_known: dict):
    row = df_feat.reindex(columns=base_cols).loc[[origin_ts]].copy()
    tgt = origin_ts + pd.Timedelta(FREQ) * horizon

    fut = add_time_features_from_index(pd.DatetimeIndex([tgt]), prefix="future_")
    fut.index = row.index
    row = row.join(fut)

    for c in [c for c in future_known_exog if c in df_feat.columns]:
        val = df_feat[c].get(tgt, np.nan)
        if pd.isna(val):
            val = last_known.get(c, np.nan)   # persistenza: ultimo valore noto
        row[f"future_{c}"] = val

    return row, tgt


def _widest_pair(conformal_pairs: List) -> tuple:
    pairs = [tuple(p) for p in conformal_pairs]
    return max(pairs, key=lambda p: p[1] - p[0])


# ============================================================
# FORECAST NEXT 96
# ============================================================
def forecast_next_96(df: pd.DataFrame, model_dir: str,
                     output_dir: Optional[str] = None, save_csv: bool = False,
                     lower_floor: Optional[float] = 0.0) -> pd.DataFrame:
    """
    Previsione dei prossimi 96 quarti d'ora per la zona del modello in model_dir.
    `df` = storico della zona (deve contenere la colonna target del modello).
    lower_floor: minimo per la banda inferiore (default 0.0; None per disattivare).
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

    # persistenza per le esogene note nel futuro (ultimo valore all'origine)
    last_known = {c: df_feat[c].loc[origin_ts] for c in future_known_exog if c in df_feat.columns}

    q_str = [str(q) for q in quantile_levels]
    q_cols = [f"q{q}" for q in quantile_levels]

    rows = []
    for h in range(1, STEPS + 1):
        row, tgt = _build_pred_row(df_feat, origin_ts, h, base_cols, future_known_exog, last_known)
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

    # --- vincolo di dominio: la banda inferiore non scende sotto lower_floor ---
    if lower_floor is not None:
        out["lower"] = out["lower"].clip(lower=lower_floor)

    # coerenza finale: lower <= pred <= upper (se il punto è < floor, lower segue il punto)
    out["lower"] = np.minimum(out["lower"], out["pred"])
    out["upper"] = np.maximum(out["upper"], out["pred"])

    out = out[["Datetime", "origin_time", "horizon", "hour"] + q_cols +
              ["pred", "lower", "upper", "band_coverage"]]

    if save_csv and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out.to_csv(os.path.join(output_dir, f"mi_forecast_{target_col}.csv".replace(" ", "_")), index=False)

    return out


# ============================================================
# DIAGNOSI: perché una previsione "sballa"?
# ============================================================
def diagnose_forecast_inputs(df: pd.DataFrame, model_dir: str) -> dict:
    """Controlla lo stato delle feature ALL'ORIGINE (l'input che genera tutta la
    previsione). Se molte sono NaN o mancanti, il modello imputa -> forecast sballato.
    Ritorna un dizionario con: origine, n. feature NaN, quali, colonne mancanti dal df."""
    art = load_direct_artifacts(model_dir)
    meta = art["metadata"]
    target_col = meta["target_col"]
    base_cols = meta["base_feature_cols"]
    future_known_exog = meta.get("future_known_exog", [])

    df = ensure_datetime_index(df)
    df = infer_and_fix_freq(df, FREQ)
    df = safe_numeric_df(df)
    df_feat = add_price_features(df, target_col)

    valid = df_feat[df_feat[target_col].notna()]
    if valid.empty:
        return {"error": f"nessun valore valido nel target '{target_col}'"}
    origin_ts = valid.index.max()

    missing = [c for c in base_cols if c not in df_feat.columns]
    orow = df_feat.reindex(columns=base_cols).loc[origin_ts]
    nan_feats = orow[orow.isna()].index.tolist()

    return {
        "origin_ts": str(origin_ts),
        "target_at_origin": float(df_feat[target_col].loc[origin_ts]),
        "n_base_features": len(base_cols),
        "n_missing_from_df": len(missing),
        "missing_from_df": missing[:50],
        "n_nan_at_origin": len(nan_feats),
        "nan_features_at_origin": nan_feats[:50],
        "future_known_exog": future_known_exog,
        "hint": (
            "Se n_nan_at_origin o n_missing_from_df sono alti (es. decine di "
            "esogene), è quello il motivo: l'origine ha feature mancanti/NaN e il "
            "modello imputa. Controlla che l'aggiornamento (mi_update) riempia le "
            "esogene (LOCATIONS/ZONES/credenziali) e che lo schema del dataset "
            "coincida con base_feature_cols."
        ),
    }


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
