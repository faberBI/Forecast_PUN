import os
import json
from pathlib import Path

import dropbox
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# =========================================================
# COSTANTI
# =========================================================
RES_LAGS = [1, 2, 3, 6, 12, 24, 48, 96]

# Regimi critici (stessi del training / gradio)
MIDDAY_QOD = set(range(48, 69))   # 12:00–17:00 inclusi
EVENING_QOD = set(range(76, 87))  # 19:00–21:30 inclusi

# Se vuoi tenerle uguali al training production:
PEAK_WEIGHT_BASE = 2.0
OFFPEAK_WEIGHT_BASE = 1.0
MIDDAY_WEIGHT = 4.0
EVENING_WEIGHT = 5.0


# =========================================================
# WEIGHT FUNCTION
# =========================================================
def peak_weight_func(index):
    """
    Necessaria se i modelli joblib/pickle salvati cercano __main__.peak_weight_func
    o comunque per mantenere coerenza col training.
    """
    index = pd.DatetimeIndex(index)
    qod = index.hour * 4 + (index.minute // 15)

    weights = np.full(len(index), OFFPEAK_WEIGHT_BASE, dtype=float)

    weights[np.isin(qod, list(MIDDAY_QOD))] = MIDDAY_WEIGHT
    weights[np.isin(qod, list(EVENING_QOD))] = EVENING_WEIGHT

    daytime_mask = (index.hour >= 8) & (index.hour < 21)
    weights[(weights == OFFPEAK_WEIGHT_BASE) & daytime_mask] = PEAK_WEIGHT_BASE

    return weights


# =========================================================
# DROPBOX: DOWNLOAD ARTIFACTS
# =========================================================
def download_models_from_dropbox(dropbox_token: str, base_path: str, local_dir):
    """
    Scarica tutti gli artifacts del modello production da Dropbox.
    """
    dbx = dropbox.Dropbox(dropbox_token)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "model_prod.pkl",
        "local_cfg_prod.pkl",
        "selected_exog.pkl",
        "residual_feature_cols.pkl",
        "metadata.json",
    ]

    for f in files:
        dbx_path = f"{base_path}/{f}"
        local_path = local_dir / f

        try:
            _, res = dbx.files_download(dbx_path)
            with open(local_path, "wb") as out:
                out.write(res.content)
        except Exception as e:
            # metadata.json può mancare e non blocca forecast
            if f == "metadata.json":
                continue
            raise RuntimeError(f"Errore download {f} da Dropbox ({dbx_path}): {e}") from e


def load_model_artifacts_from_dropbox(
    dropbox_token: str,
    base_path: str = "/forecast_pun/models",
    local_dir: str | Path = "models",
) -> dict:
    """
    Scarica e carica tutti gli artifacts modello from Dropbox.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    download_models_from_dropbox(
        dropbox_token=dropbox_token,
        base_path=base_path,
        local_dir=local_dir,
    )

    artifacts = {
        "model_prod": joblib.load(local_dir / "model_prod.pkl"),
        "local_cfg_prod": joblib.load(local_dir / "local_cfg_prod.pkl"),
        "selected_exog": joblib.load(local_dir / "selected_exog.pkl"),
        "residual_feature_cols": joblib.load(local_dir / "residual_feature_cols.pkl"),
    }

    metadata_path = local_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            artifacts["metadata"] = json.load(f)
    else:
        artifacts["metadata"] = {}

    return artifacts


# =========================================================
# EXCEL PUN -> DATETIME (robusto)
# =========================================================
def pun_to_datetime(df_pun_excel: pd.DataFrame) -> pd.DataFrame:
    """
    Converte il file PUN reale in formato standard:
    Datetime, PUN, Ora, Periodo

    Gestisce sia:
    - Data seriale Excel
    - Data già in datetime/string
    """
    df = df_pun_excel.copy()

    if pd.api.types.is_numeric_dtype(df["Data"]):
        df["Data"] = pd.to_datetime(df["Data"], unit="D", origin="1899-12-30")
    else:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    df["Data"] = df["Data"].dt.normalize()

    df["Periodo"] = pd.to_numeric(df["Periodo"], errors="coerce")
    df["PUN"] = pd.to_numeric(df["PUN"], errors="coerce")

    # Quarter-hour robusto basato su Periodo 1..96
    df["Datetime"] = df["Data"] + pd.to_timedelta((df["Periodo"] - 1) * 15, unit="m")

    df = (
        df[["Datetime", "PUN", "Ora", "Periodo"]]
        .dropna(subset=["Datetime", "PUN"])
        .sort_values("Datetime")
        .drop_duplicates(subset=["Datetime"], keep="last")
        .reset_index(drop=True)
    )
    return df


# =========================================================
# PLOT FORECAST
# =========================================================
def plot_forecast_pun(preds: pd.DataFrame):
    """
    Plot robusto:
    - se esiste pred_corrected usa quello come forecast principale
    - altrimenti usa pred / pred_base
    """
    df_plot = preds.copy()

    if "Datetime" in df_plot.columns:
        df_plot["Datetime"] = pd.to_datetime(df_plot["Datetime"])

    if "pred_corrected" in df_plot.columns:
        y_col = "pred_corrected"
    elif "pred" in df_plot.columns:
        y_col = "pred"
    else:
        y_col = "pred_base"

    stats = {
        "min": round(df_plot[y_col].min(), 2),
        "max": round(df_plot[y_col].max(), 2),
        "mean": round(df_plot[y_col].mean(), 2),
    }

    idx_min = df_plot[y_col].idxmin()
    idx_max = df_plot[y_col].idxmax()

    fig = go.Figure()

    if "pred_base" in df_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=df_plot["Datetime"],
                y=df_plot["pred_base"],
                mode="lines",
                name="Forecast base",
                line=dict(width=2, dash="dot"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_plot["Datetime"],
            y=df_plot[y_col],
            mode="lines",
            name="Forecast finale",
            line=dict(width=3),
            hovertemplate="<b>%{x}</b><br>PUN: %{y:.2f} €/MWh<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[df_plot.loc[idx_max, "Datetime"]],
            y=[df_plot.loc[idx_max, y_col]],
            mode="markers+text",
            name="Max",
            text=[f"{df_plot.loc[idx_max, y_col]:.2f}"],
            textposition="top center",
            marker=dict(size=10),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[df_plot.loc[idx_min, "Datetime"]],
            y=[df_plot.loc[idx_min, y_col]],
            mode="markers+text",
            name="Min",
            text=[f"{df_plot.loc[idx_min, y_col]:.2f}"],
            textposition="bottom center",
            marker=dict(size=10),
        )
    )

    fig.update_layout(
        title="Forecast Day-Ahead PUN",
        xaxis_title="Datetime",
        yaxis_title="€/MWh",
        height=500,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig, stats


# =========================================================
# BASE FORECAST (già usavi questa)
# =========================================================
def forecast_day_ahead_96_base(
    df_hist,
    best_forecaster,
    meteo_downloader,
    locations,
    selected_exog,
    steps=96
):
    # ======================================================
    # 1. PREPARAZIONE STORICO
    # ======================================================
    df = df_hist.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("15min").ffill()

    if "PUN" not in df.columns:
        raise ValueError("La colonna 'PUN' non è presente in df_hist")

    y = df["PUN"].astype(float).copy()

    window_size = best_forecaster.window_size
    last_window = y.iloc[-window_size:].copy()

    # ======================================================
    # 2. FUTURE INDEX
    # ======================================================
    last_dt = y.index[-1]

    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(minutes=15),
        periods=steps,
        freq="15min"
    )

    # ======================================================
    # 3. METEO FUTURO
    # ======================================================
    weather = meteo_downloader.download_multi_city(
        locations=locations,
        start_date=future_index[0].strftime("%Y-%m-%d"),
        end_date=future_index[-1].strftime("%Y-%m-%d")
    )

    weather["Datetime"] = pd.to_datetime(weather["Datetime"]).dt.floor("15min")

    cloud_cols = [c for c in weather.columns if "cloud_cover" in c]
    if cloud_cols and "cloud_cover_mean" not in weather.columns:
        weather["cloud_cover_mean"] = weather[cloud_cols].mean(axis=1)

    wind_cols = [c for c in weather.columns if "wind_speed_80m" in c]
    if wind_cols and "wind_speed_mean" not in weather.columns:
        weather["wind_speed_mean"] = weather[wind_cols].mean(axis=1)

    weather = (
        weather
        .set_index("Datetime")
        .sort_index()
        .reindex(future_index)
        .ffill()
        .bfill()
    )

    # ======================================================
    # 4. EXOG FUTURE
    # ======================================================
    exog_future = pd.DataFrame(index=future_index)

    exog_future["minute"] = future_index.minute
    exog_future["quarter_of_day"] = future_index.hour * 4 + (future_index.minute // 15)
    exog_future["hour"] = future_index.hour
    exog_future["day_of_week"] = future_index.dayofweek
    exog_future["month"] = future_index.month
    exog_future["is_weekend"] = (future_index.dayofweek >= 5).astype(int)

    exog_future = pd.concat([exog_future, weather], axis=1)

    terna_cols = [
        "forecast_total_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
        "load_ramp_1h",
        "load_forecast_error",
        "market_load_MW",
        "forecast_market_load_MW",
        "cloud_cover_mean",
    ]

    for col in terna_cols:
        if col in df.columns and df[col].dropna().shape[0] > 0:
            exog_future[col] = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]
        elif col not in exog_future.columns:
            exog_future[col] = 0.0

    for col in selected_exog:
        if col not in exog_future.columns:
            if col in df.columns and df[col].dropna().shape[0] > 0:
                exog_future[col] = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]
            else:
                exog_future[col] = 0.0

    exog_future = exog_future[selected_exog].copy()
    exog_future = exog_future.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in exog_future.columns:
        exog_future[c] = pd.to_numeric(exog_future[c], errors="coerce").fillna(0.0)

    preds = best_forecaster.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )

    preds = pd.Series(preds.values, index=future_index, name="pred")

    out = pd.DataFrame({
        "Datetime": future_index,
        "pred": preds.values
    })

    return out


# =========================================================
# HELPERS RESIDUAL
# =========================================================
def _get_window_size_from_model(model) -> int:
    lags = model.lags
    return int(lags) if np.isscalar(lags) else int(max(lags))


def _make_residual_row_forecast(ts,
                                base_pred: float,
                                exog_row: pd.Series,
                                residual_history: pd.Series,
                                horizon: int,
                                exog_cols: list[str]) -> dict:
    qod = ts.hour * 4 + (ts.minute // 15)

    row = {
        "base_pred": float(base_pred),
        "hour": ts.hour,
        "dayofweek": ts.dayofweek,
        "horizon": int(horizon),
        "quarter_of_day": qod,
        "sin_qod": np.sin(2 * np.pi * qod / 96.0),
        "cos_qod": np.cos(2 * np.pi * qod / 96.0),
        "sin_dow": np.sin(2 * np.pi * ts.dayofweek / 7.0),
        "cos_dow": np.cos(2 * np.pi * ts.dayofweek / 7.0),
        "is_midday_critical": int(qod in MIDDAY_QOD),
        "is_evening_critical": int(qod in EVENING_QOD),
    }

    row["midday_base_interaction"] = row["is_midday_critical"] * row["base_pred"]
    row["evening_base_interaction"] = row["is_evening_critical"] * row["base_pred"]

    for col in exog_cols:
        row[col] = float(exog_row[col]) if col in exog_row.index else 0.0

    for lag in RES_LAGS:
        row[f"res_lag_{lag}"] = (
            float(residual_history.iloc[-lag]) if len(residual_history) >= lag else 0.0
        )

    if len(residual_history) >= 24:
        row["res_roll_mean_24"] = float(residual_history.iloc[-24:].mean())
        std = residual_history.iloc[-24:].std()
        row["res_roll_std_24"] = float(std) if not pd.isna(std) else 0.0
    else:
        row["res_roll_mean_24"] = 0.0
        row["res_roll_std_24"] = 0.0

    return row


def _predict_residual_forecast(ts,
                               base_pred: float,
                               exog_row: pd.Series,
                               residual_history: pd.Series,
                               horizon: int,
                               residual_feature_cols: list[str],
                               local_cfg_prod: dict,
                               exog_cols: list[str]) -> tuple[float, str]:
    qod = ts.hour * 4 + (ts.minute // 15)

    if qod in MIDDAY_QOD:
        regime = "midday"
    elif qod in EVENING_QOD:
        regime = "evening"
    else:
        return 0.0, "off"

    cfg = local_cfg_prod.get(regime, {})
    if (
        not cfg.get("use_correction")
        or cfg.get("model") is None
        or cfg.get("best_alpha", 0) <= 0
    ):
        return 0.0, regime

    row = _make_residual_row_forecast(
        ts=ts,
        base_pred=base_pred,
        exog_row=exog_row,
        residual_history=residual_history,
        horizon=horizon,
        exog_cols=exog_cols,
    )

    X = pd.DataFrame([row], index=[ts])

    for col in residual_feature_cols:
        if col not in X.columns:
            X[col] = 0.0

    X = X[residual_feature_cols].fillna(0.0)

    raw_pred = float(cfg["model"].predict(X)[0])
    shrunk_pred = raw_pred * float(cfg["best_alpha"])

    return shrunk_pred, regime


# =========================================================
# COSTRUZIONE EXOG FUTURE
# =========================================================
def build_future_exog(
    df_hist: pd.DataFrame,
    meteo_downloader,
    locations,
    selected_exog,
    steps: int = 96
):
    df = df_hist.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("15min").ffill()

    y = df["PUN"].astype(float).copy()
    last_dt = y.index[-1]

    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(minutes=15),
        periods=steps,
        freq="15min"
    )

    weather = meteo_downloader.download_multi_city(
        locations=locations,
        start_date=future_index[0].strftime("%Y-%m-%d"),
        end_date=future_index[-1].strftime("%Y-%m-%d")
    )

    weather["Datetime"] = pd.to_datetime(weather["Datetime"]).dt.floor("15min")

    cloud_cols = [c for c in weather.columns if "cloud_cover" in c]
    if cloud_cols and "cloud_cover_mean" not in weather.columns:
        weather["cloud_cover_mean"] = weather[cloud_cols].mean(axis=1)

    wind_cols = [c for c in weather.columns if "wind_speed_80m" in c]
    if wind_cols and "wind_speed_mean" not in weather.columns:
        weather["wind_speed_mean"] = weather[wind_cols].mean(axis=1)

    weather = (
        weather
        .set_index("Datetime")
        .sort_index()
        .reindex(future_index)
        .ffill()
        .bfill()
    )

    exog_future = pd.DataFrame(index=future_index)

    exog_future["minute"] = future_index.minute
    exog_future["quarter_of_day"] = future_index.hour * 4 + (future_index.minute // 15)
    exog_future["hour"] = future_index.hour
    exog_future["day_of_week"] = future_index.dayofweek
    exog_future["month"] = future_index.month
    exog_future["is_weekend"] = (future_index.dayofweek >= 5).astype(int)

    exog_future = pd.concat([exog_future, weather], axis=1)

    terna_cols = [
        "forecast_total_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
        "load_ramp_1h",
        "load_forecast_error",
        "market_load_MW",
        "forecast_market_load_MW",
        "cloud_cover_mean",
    ]

    for col in terna_cols:
        if col in df.columns and df[col].dropna().shape[0] > 0:
            exog_future[col] = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]
        elif col not in exog_future.columns:
            exog_future[col] = 0.0

    for col in selected_exog:
        if col not in exog_future.columns:
            if col in df.columns and df[col].dropna().shape[0] > 0:
                exog_future[col] = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]
            else:
                exog_future[col] = 0.0

    exog_future = exog_future[selected_exog].copy()
    exog_future = exog_future.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in exog_future.columns:
        exog_future[c] = pd.to_numeric(exog_future[c], errors="coerce").fillna(0.0)

    return future_index, exog_future


# =========================================================
# RESIDUAL HISTORY REALE
# =========================================================
def build_recent_residual_history(
    df_hist: pd.DataFrame,
    model_prod,
    selected_exog: list[str],
    warmup_steps: int = 96 * 7,
    block_steps: int = 96
) -> pd.Series:
    """
    Costruisce la residual history reale usando gli ultimi giorni osservati.
    """
    df = df_hist.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("15min").ffill()

    missing = [c for c in selected_exog if c not in df.columns]
    if missing:
        raise ValueError(f"Missing exog columns in df_hist: {missing}")

    y = df["PUN"].astype(float).copy()
    exog = df[selected_exog].copy()

    if len(df) < warmup_steps + 200:
        return pd.Series([0.0] * 96, dtype=float)

    y_history_core = y.iloc[:-warmup_steps]
    y_warmup = y.iloc[-warmup_steps:]
    exog_warmup = exog.iloc[-warmup_steps:]

    window_size = _get_window_size_from_model(model_prod)
    history_y = y_history_core.copy().sort_index().asfreq("15min")

    residual_history = []

    for start in range(0, len(y_warmup), block_steps):
        end = min(start + block_steps, len(y_warmup))
        idx_block = y_warmup.index[start:end]

        exog_block = exog_warmup.loc[idx_block]
        last_window = history_y.iloc[-window_size:].asfreq("15min")

        base_block = model_prod.predict(
            steps=len(idx_block),
            last_window=last_window,
            exog=exog_block
        )
        base_block = pd.Series(base_block, index=idx_block)

        actual_residual_block = y_warmup.loc[idx_block] - base_block
        residual_history.append(actual_residual_block)

        history_y = pd.concat([history_y, y_warmup.loc[idx_block]]).sort_index().asfreq("15min")

    if len(residual_history) == 0:
        return pd.Series([0.0] * 96, dtype=float)

    return pd.concat(residual_history).astype(float)


# =========================================================
# FORECAST PRODUCTION BASE + RESIDUAL
# =========================================================
def forecast_day_ahead_96_full_production(
    df_hist: pd.DataFrame,
    model_prod,
    local_cfg_prod: dict,
    residual_feature_cols: list,
    meteo_downloader,
    locations,
    selected_exog: list,
    steps: int = 96,
    warmup_steps: int = 96 * 7
) -> pd.DataFrame:
    """
    Forecast day-ahead production-grade:
    - base model
    - residual history reale su warmup
    - residual correction ricorsiva step-by-step
    """

    df = df_hist.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().asfreq("15min").ffill()

    missing = [c for c in selected_exog if c not in df.columns]
    if missing:
        raise ValueError(f"Missing exog columns in df_hist: {missing}")

    y_hist = df["PUN"].astype(float).copy()

    # 1) exog future
    future_index, exog_future = build_future_exog(
        df_hist=df,
        meteo_downloader=meteo_downloader,
        locations=locations,
        selected_exog=selected_exog,
        steps=steps
    )

    # 2) base pred
    window_size = _get_window_size_from_model(model_prod)
    last_window = y_hist.iloc[-window_size:].copy().asfreq("15min")

    base_pred = model_prod.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )
    base_pred = pd.Series(base_pred, index=future_index, name="pred_base")

    # 3) residual history reale
    residual_history = build_recent_residual_history(
        df_hist=df,
        model_prod=model_prod,
        selected_exog=selected_exog,
        warmup_steps=warmup_steps,
        block_steps=96
    )

    # 4) residual correction ricorsiva
    res_preds = []
    regimes = []

    for i, ts in enumerate(future_index, start=1):
        exog_row = exog_future.loc[ts]

        r_pred, regime = _predict_residual_forecast(
            ts=ts,
            base_pred=float(base_pred.loc[ts]),
            exog_row=exog_row,
            residual_history=residual_history,
            horizon=i,
            residual_feature_cols=residual_feature_cols,
            local_cfg_prod=local_cfg_prod,
            exog_cols=selected_exog,
        )

        residual_history = pd.concat([
            residual_history,
            pd.Series([r_pred], index=[ts])
        ])

        res_preds.append(r_pred)
        regimes.append(regime)

    residual_pred = pd.Series(res_preds, index=future_index, name="residual_correction")
    final_pred = base_pred + residual_pred

    out = pd.DataFrame({
        "Datetime": future_index,
        "pred_base": base_pred.values,
        "residual_correction": residual_pred.values,
        "pred_corrected": final_pred.values,
        "regime_used": regimes,
    })

    # compatibilità con vecchia UI / salvataggi
    out["pred"] = out["pred_corrected"]

    return out

