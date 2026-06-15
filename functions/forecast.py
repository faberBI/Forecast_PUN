import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dropbox
from pathlib import Path
import joblib

RES_LAGS = [1, 2, 3, 6, 12, 24, 48, 96]

MIDDAY_QOD = set(range(48, 69))   # 12:00–17:00
EVENING_QOD = set(range(76, 87))  # 19:00–21:30


def download_models_from_dropbox(dropbox_token: str, base_path: str, local_dir):
    dbx = dropbox.Dropbox(dropbox_token)

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "model_prod.pkl",
        "local_cfg_prod.pkl",
        "selected_exog.pkl",
        "residual_feature_cols.pkl",
    ]

    for f in files:
        dbx_path = f"{base_path}/{f}"
        local_path = local_dir / f

        metadata, res = dbx.files_download(dbx_path)

        with open(local_path, "wb") as out:
            out.write(res.content)

def load_model_artifacts_from_dropbox(dropbox_token: str,
                                      base_path: str = "/forecast_pun/models",
                                      local_dir: str | Path = "models") -> dict:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    download_models_from_dropbox(
        dropbox_token=dropbox_token,
        base_path=base_path,
        local_dir=local_dir
    )

    return {
        "model_prod": joblib.load(local_dir / "model_prod.pkl"),
        "local_cfg_prod": joblib.load(local_dir / "local_cfg_prod.pkl"),
        "selected_exog": joblib.load(local_dir / "selected_exog.pkl"),
        "residual_feature_cols": joblib.load(local_dir / "residual_feature_cols.pkl"),
    }

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

    # -----------------------------
    # meteo futuro
    # -----------------------------
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

    # -----------------------------
    # exog future
    # -----------------------------
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

    y_hist = df["PUN"].astype(float).copy()

    # --------------------------------------------
    # 1. EXOG FUTURE + FUTURE INDEX
    # --------------------------------------------
    future_index, exog_future = build_future_exog(
        df_hist=df,
        meteo_downloader=meteo_downloader,
        locations=locations,
        selected_exog=selected_exog,
        steps=steps
    )

    # --------------------------------------------
    # 2. BASE PRED
    # --------------------------------------------
    window_size = _get_window_size_from_model(model_prod)
    last_window = y_hist.iloc[-window_size:].copy().asfreq("15min")

    base_pred = model_prod.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )
    base_pred = pd.Series(base_pred, index=future_index, name="pred_base")

    # --------------------------------------------
    # 3. RESIDUAL HISTORY REALE
    # --------------------------------------------
    residual_history = build_recent_residual_history(
        df_hist=df,
        model_prod=model_prod,
        selected_exog=selected_exog,
        warmup_steps=warmup_steps,
        block_steps=96
    )

    # --------------------------------------------
    # 4. RESIDUAL CORRECTION RICORSIVA
    # --------------------------------------------
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

        # aggiorna history con il residuo predetto → ricorsivo
        residual_history = pd.concat([
            residual_history,
            pd.Series([r_pred], index=[ts])
        ])

        res_preds.append(r_pred)
        regimes.append(regime)

    residual_pred = pd.Series(res_preds, index=future_index, name="residual_correction")
    final_pred = base_pred + residual_pred

    # --------------------------------------------
    # 5. OUTPUT
    # --------------------------------------------
    out = pd.DataFrame({
        "Datetime": future_index,
        "pred_base": base_pred.values,
        "residual_correction": residual_pred.values,
        "pred_corrected": final_pred.values,
        "regime_used": regimes,
    })

    return out

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
    df = df.sort_index().ffill()

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

    # cloud cover medio se utile
    cloud_cols = [c for c in weather.columns if "cloud_cover" in c]
    if cloud_cols and "cloud_cover_mean" not in weather.columns:
        weather["cloud_cover_mean"] = weather[cloud_cols].mean(axis=1)

    # wind medio se utile
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

    # -------------------------
    # 4.1 feature deterministiche
    # -------------------------
    exog_future["minute"] = future_index.minute
    exog_future["quarter_of_day"] = future_index.hour * 4 + (future_index.minute // 15)
    exog_future["hour"] = future_index.hour
    exog_future["day_of_week"] = future_index.dayofweek
    exog_future["month"] = future_index.month
    exog_future["is_weekend"] = (future_index.dayofweek >= 5).astype(int)

    # -------------------------
    # 4.2 meteo
    # -------------------------
    exog_future = pd.concat([exog_future, weather], axis=1)

    # -------------------------
    # 4.3 Terna-like features:
    # carry-forward ultimo valore disponibile
    # -------------------------
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

    # ======================================================
    # 5. FEATURE STORICHE "STATICHE" / BACKFILL
    # ======================================================
    # Per tutte le feature richieste dal modello che non abbiamo
    # costruito direttamente, usiamo ultimo valore disponibile se esiste,
    # altrimenti 0.0
    # ======================================================
    for col in selected_exog:
        if col not in exog_future.columns:
            if col in df.columns and df[col].dropna().shape[0] > 0:
                exog_future[col] = pd.to_numeric(df[col], errors="coerce").dropna().iloc[-1]
            else:
                exog_future[col] = 0.0

    # ======================================================
    # 6. ALLINEAMENTO EXOG
    # ======================================================
    exog_future = exog_future[selected_exog].copy()
    exog_future = exog_future.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    for c in exog_future.columns:
        exog_future[c] = pd.to_numeric(exog_future[c], errors="coerce").fillna(0.0)

    # ======================================================
    # 7. FORECAST BASE
    # ======================================================
    preds = best_forecaster.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )

    preds = pd.Series(preds.values, index=future_index, name="pred")

    # ======================================================
    # 8. OUTPUT
    # ======================================================
    out = pd.DataFrame({
        "Datetime": future_index,
        "pred": preds.values
    })

    return out

def pun_to_datetime(df_pun_excel: pd.DataFrame) -> pd.DataFrame:
    df = df_pun_excel.copy()

    # =========================================
    # 1. CONVERSIONE DATA (robusta)
    # =========================================
    if pd.api.types.is_numeric_dtype(df["Data"]):
        # formato Excel seriale
        df["Data"] = pd.to_datetime(df["Data"], unit="D", origin="1899-12-30")
    else:
        # già datetime o stringa data
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    df["Data"] = df["Data"].dt.normalize()

    # =========================================
    # 2. TIPI SICURI
    # =========================================
    df["Periodo"] = pd.to_numeric(df["Periodo"], errors="coerce")
    df["PUN"] = pd.to_numeric(df["PUN"], errors="coerce")

    # =========================================
    # 3. COSTRUZIONE DATETIME (CORRETTA)
    # =========================================
    df["Datetime"] = df["Data"] + pd.to_timedelta((df["Periodo"] - 1) * 15, unit="m")

    # =========================================
    # 4. OUTPUT
    # =========================================
    df = (
        df[["Datetime", "PUN", "Ora", "Periodo"]]
        .dropna(subset=["Datetime", "PUN"])
        .sort_values("Datetime")
        .drop_duplicates(subset=["Datetime"], keep="last")
        .reset_index(drop=True)
    )

    return df


def plot_forecast_pun(preds: pd.DataFrame):

    df_plot = preds.copy()

    if "Datetime" in df_plot.columns:
        df_plot["Datetime"] = pd.to_datetime(df_plot["Datetime"])

    # ============================
    # METRICHE
    # ============================
    stats = {
        "min": round(df_plot["pred"].min(), 2),
        "max": round(df_plot["pred"].max(), 2),
        "mean": round(df_plot["pred"].mean(), 2),
    }

    # ============================
    # INDEX MIN/MAX
    # ============================
    idx_min = df_plot["pred"].idxmin()
    idx_max = df_plot["pred"].idxmax()

    # ============================
    # FIGURA
    # ============================
    fig = go.Figure()

    # linea forecast
    fig.add_trace(
        go.Scatter(
            x=df_plot["Datetime"],
            y=df_plot["pred"],
            mode="lines",
            name="Forecast",
            line=dict(width=3),
            hovertemplate="<b>%{x}</b><br>PUN: %{y:.2f} €/MWh<extra></extra>"
        )
    )

    # massimo
    fig.add_trace(
        go.Scatter(
            x=[df_plot.loc[idx_max, "Datetime"]],
            y=[df_plot.loc[idx_max, "pred"]],
            mode="markers+text",
            name="Max",
            text=[f"{df_plot.loc[idx_max, 'pred']:.2f}"],
            textposition="top center",
            marker=dict(size=10),
        )
    )

    # minimo
    fig.add_trace(
        go.Scatter(
            x=[df_plot.loc[idx_min, "Datetime"]],
            y=[df_plot.loc[idx_min, "pred"]],
            mode="markers+text",
            name="Min",
            text=[f"{df_plot.loc[idx_min, 'pred']:.2f}"],
            textposition="bottom center",
            marker=dict(size=10),
        )
    )

    # highlight fascia peak (8-20)
    for ts in df_plot["Datetime"]:
        if 8 <= ts.hour < 21:
            fig.add_vrect(
                x0=ts,
                x1=ts + pd.Timedelta(minutes=15),
                fillcolor="orange",
                opacity=0.02,
                line_width=0
            )

    fig.update_layout(
        title="Forecast Day-Ahead PUN",
        xaxis_title="Datetime",
        yaxis_title="€/MWh",
        height=500,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig, stats

