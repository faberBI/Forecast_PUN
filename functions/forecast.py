def forecast_day_ahead_96_base(
    df_hist,
    best_forecaster,
    meteo_downloader,
    locations,
    selected_exog,
    steps=96
):
    import pandas as pd
    import numpy as np

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
