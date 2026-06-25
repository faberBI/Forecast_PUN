import os
import io
import json
import joblib
import dropbox
import pandas as pd
import numpy as np


# ==========================================================
# BASIC UTILS
# ==========================================================

def sanitize_filename(text):
    text = str(text).strip()
    text = text.replace("/", "_").replace("\\", "_")
    text = text.replace("(", "").replace(")", "")
    text = text.replace(" ", "_")
    text = text.replace("__", "_")
    return text


def normalize_dropbox_path(path):
    path = str(path).replace("\\", "/")

    if not path.startswith("/"):
        path = "/" + path

    return path


def get_dropbox_client(dropbox_token):
    if dropbox_token is None or str(dropbox_token).strip() == "":
        raise ValueError("DROPBOX_TOKEN mancante o vuoto.")

    return dropbox.Dropbox(dropbox_token)


def normalize_hist_df(df_hist):
    """
    Normalizza storico:
    - Datetime index
    - ordinamento
    - no duplicati
    - freq 15min
    - ffill
    """

    if df_hist is None:
        raise ValueError("df_hist è None.")

    df = df_hist.copy()

    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.set_index("Datetime")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    try:
        df = df.asfreq("15min")
    except Exception:
        pass

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill()

    return df


# ==========================================================
# DROPBOX LOW LEVEL
# ==========================================================

def dbx_download_bytes(dropbox_token, dropbox_path):
    dbx = get_dropbox_client(dropbox_token)
    dropbox_path = normalize_dropbox_path(dropbox_path)

    try:
        _, res = dbx.files_download(dropbox_path)
        return res.content

    except Exception as e:
        raise RuntimeError(
            f"Download Dropbox fallito: {dropbox_path} | "
            f"{type(e).__name__}: {e}"
        )


def dbx_upload_bytes(dropbox_token, dropbox_path, content_bytes):
    dbx = get_dropbox_client(dropbox_token)
    dropbox_path = normalize_dropbox_path(dropbox_path)

    try:
        dbx.files_upload(
            content_bytes,
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )

        print(f"💾 Upload Dropbox OK: {dropbox_path}")

    except Exception as e:
        raise RuntimeError(
            f"Upload Dropbox fallito: {dropbox_path} | "
            f"{type(e).__name__}: {e}"
        )


def dbx_file_exists(dropbox_token, dropbox_path):
    dbx = get_dropbox_client(dropbox_token)
    dropbox_path = normalize_dropbox_path(dropbox_path)

    try:
        dbx.files_get_metadata(dropbox_path)
        return True

    except Exception:
        return False


# ==========================================================
# DROPBOX READERS / WRITERS
# ==========================================================

def read_json_from_dropbox(dropbox_token, dropbox_path):
    content = dbx_download_bytes(
        dropbox_token=dropbox_token,
        dropbox_path=dropbox_path
    )

    return json.loads(content.decode("utf-8"))


def read_parquet_from_dropbox(dropbox_token, dropbox_path):
    content = dbx_download_bytes(
        dropbox_token=dropbox_token,
        dropbox_path=dropbox_path
    )

    return pd.read_parquet(io.BytesIO(content))


def upload_df_parquet_to_dropbox(
    df,
    dropbox_token,
    dropbox_path,
    index=False
):
    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=index
    )

    buffer.seek(0)

    dbx_upload_bytes(
        dropbox_token=dropbox_token,
        dropbox_path=dropbox_path,
        content_bytes=buffer.getvalue()
    )


# ==========================================================
# LOAD MODEL FROM DROPBOX
# ==========================================================

def load_mi_model_payload_from_dropbox(
    dropbox_token,
    model_path_from_json,
    dropbox_models_dir
):
    """
    Carica un modello MI da Dropbox.

    Gestisce:
    - models_retrained\\file.joblib
    - models_retrained/file.joblib
    - file.joblib
    """

    if model_path_from_json is None:
        raise ValueError("model_path_from_json è None.")

    dbx = get_dropbox_client(dropbox_token)

    clean_path = str(model_path_from_json).replace("\\", "/").strip()
    file_name = os.path.basename(clean_path)

    dropbox_models_dir = normalize_dropbox_path(dropbox_models_dir).rstrip("/")

    try:
        folder_res = dbx.files_list_folder(dropbox_models_dir)

        available_models = {
            entry.name: entry.path_display
            for entry in folder_res.entries
            if entry.name.endswith(".joblib")
        }

    except Exception as e:
        raise RuntimeError(
            f"Errore lettura cartella modelli Dropbox: "
            f"{dropbox_models_dir} -> {e}"
        )

    if file_name not in available_models:
        raise FileNotFoundError(
            f"Modello non trovato su Dropbox: {file_name}. "
            f"Cartella: {dropbox_models_dir}. "
            f"Disponibili: {list(available_models.keys())[:20]}"
        )

    model_dropbox_path = available_models[file_name]

    try:
        _, res = dbx.files_download(model_dropbox_path)
        payload = joblib.load(io.BytesIO(res.content))

    except Exception as e:
        raise RuntimeError(
            f"Errore download/load modello: {model_dropbox_path} -> {e}"
        )

    required_keys = ["forecaster", "selected_exog", "target"]

    for k in required_keys:
        if k not in payload:
            raise ValueError(
                f"Chiave '{k}' mancante nel payload modello: {model_dropbox_path}"
            )

    return payload, model_dropbox_path




def prepare_terna(terna, start, end):
    def clean(df):
        return terna.clean_terna_df(df)

    load = clean(terna.get_total_load(start, end))
    market = clean(terna.get_market_load(start, end))

    wind = clean(terna.get_generation(start, end, "Wind")).rename(
        columns={"actual_generation_MW": "wind_generation_MW"}
    )
    solar = clean(terna.get_generation(start, end, "Photovoltaic")).rename(
        columns={"actual_generation_MW": "solar_generation_MW"}
    )
    hydro = clean(terna.get_generation(start, end, "Hydro")).rename(
        columns={"actual_generation_MW": "hydro_generation_MW"}
    )

    df = load.merge(market, on="date", how="outer")
    df = terna.safe_merge(df, wind, "wind")
    df = terna.safe_merge(df, solar, "solar")
    df = terna.safe_merge(df, hydro, "hydro")

    df = terna.clean_terna_features(df)

    numeric_cols = [
        "total_load_MW",
        "market_load_MW",
        "forecast_total_load_MW",
        "forecast_market_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
        "load_ramp_1h",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Datetime"] = pd.to_datetime(df["date"])

    # hourly -> 15min
    df = (
        df.set_index("Datetime")
          .sort_index()
          .resample("15min")
          .ffill()
          .reset_index()
    )

    return df

def shift_terna_only(df: pd.DataFrame, shift_steps: int = 1) -> pd.DataFrame:
    """
    Shifta SOLO le feature Terna per evitare leakage.
    """
    df = df.copy()

    terna_cols = [
        "forecast_total_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
        "load_ramp_1h",
        "load_forecast_error",
    ]

    cols_to_shift = [c for c in terna_cols if c in df.columns]
    df[cols_to_shift] = df[cols_to_shift].shift(shift_steps)

    return df
    
# ==========================================================
# BUILD EXOG FUTURE - STILE PUN
# ==========================================================

def build_mi_exog_future_pun_style(
    df_hist,
    target_col,
    future_index,
    selected_exog,
    meteo_downloader=None,
    locations=None
):
    """
    Costruisce exog_future con logica semplice/stabile stile PUN:
    - feature calendario future
    - meteo futuro se disponibile
    - feature target-derived come fallback costante
    - tutte le selected_exog mancanti vengono prese dallo storico o messe a 0
    """

    df = normalize_hist_df(df_hist)

    exog_future = pd.DataFrame(index=future_index)

    # ======================================================
    # 1. FEATURE TEMPORALI
    # ======================================================
    exog_future["minute"] = future_index.minute
    exog_future["hour"] = future_index.hour
    exog_future["quarter"] = future_index.minute // 15
    exog_future["quarter_of_day"] = (
        future_index.hour * 4 + (future_index.minute // 15)
    )

    exog_future["day_of_week"] = future_index.dayofweek
    exog_future["day_of_year"] = future_index.dayofyear
    exog_future["month"] = future_index.month
    exog_future["year"] = future_index.year
    exog_future["is_weekend"] = (future_index.dayofweek >= 5).astype(int)

    exog_future["hour_sin"] = np.sin(
        2 * np.pi * exog_future["quarter_of_day"] / 96
    )
    exog_future["hour_cos"] = np.cos(
        2 * np.pi * exog_future["quarter_of_day"] / 96
    )

    exog_future["dow_sin"] = np.sin(
        2 * np.pi * exog_future["day_of_week"] / 7
    )
    exog_future["dow_cos"] = np.cos(
        2 * np.pi * exog_future["day_of_week"] / 7
    )

    # ======================================================
    # 2. METEO FUTURO
    # ======================================================
    if meteo_downloader is not None and locations is not None:

        try:
            weather = meteo_downloader.download_multi_city(
                locations=locations,
                start_date=future_index[0].strftime("%Y-%m-%d"),
                end_date=future_index[-1].strftime("%Y-%m-%d")
            )

            if weather is not None and not weather.empty:

                weather = weather.copy()
                weather["Datetime"] = (
                    pd.to_datetime(weather["Datetime"])
                    .dt.floor("15min")
                )

                cloud_cols = [
                    c for c in weather.columns
                    if "cloud_cover" in c
                ]

                if cloud_cols and "cloud_cover_mean" not in weather.columns:
                    weather["cloud_cover_mean"] = weather[cloud_cols].mean(axis=1)

                wind_cols = [
                    c for c in weather.columns
                    if "wind_speed_80m" in c
                ]

                if wind_cols and "wind_speed_mean" not in weather.columns:
                    weather["wind_speed_mean"] = weather[wind_cols].mean(axis=1)

                precip_cols = [
                    c for c in weather.columns
                    if "precipitation" in c
                ]

                if precip_cols and "precipitation_mean" not in weather.columns:
                    weather["precipitation_mean"] = weather[precip_cols].mean(axis=1)

                weather = (
                    weather
                    .set_index("Datetime")
                    .sort_index()
                    .reindex(future_index)
                    .ffill()
                    .bfill()
                )

                exog_future = pd.concat(
                    [exog_future, weather],
                    axis=1
                )

        except Exception as e:
            print(f"⚠️ Meteo non disponibile nel forecast MI: {e}")

    # ======================================================
    # 3. FEATURE DERIVATE DAL TARGET
    # ======================================================
    if target_col in df.columns:

        y = pd.to_numeric(df[target_col], errors="coerce")

        fallback_features = {
            "lag_15m": y.shift(1).dropna(),
            "lag_30m": y.shift(2).dropna(),
            "lag_1h": y.shift(4).dropna(),
            "lag_2h": y.shift(8).dropna(),
            "lag_4h": y.shift(16).dropna(),

            "lag_1d": y.shift(96).dropna(),
            "lag_2d": y.shift(96 * 2).dropna(),
            "lag_7d": y.shift(96 * 7).dropna(),

            "mi_ret_1h": y.pct_change(4).shift(1).dropna(),
            "mi_ret_1d": y.pct_change(96).shift(1).dropna(),
            "mi_ret_7d": y.pct_change(96 * 7).shift(1).dropna(),

            "rolling_mean_4h": y.shift(1).rolling(16).mean().dropna(),
            "rolling_mean_24h": y.shift(1).rolling(96).mean().dropna(),
            "rolling_std_24h": y.shift(1).rolling(96).std().dropna(),
            "rolling_std_7d": y.shift(1).rolling(96 * 7).std().dropna(),
            "rolling_max_24h": y.shift(1).rolling(96).max().dropna(),
            "rolling_min_24h": y.shift(1).rolling(96).min().dropna(),

            "momentum_4h": (y.shift(1) - y.shift(16)).dropna(),
            "momentum_1d": (y.shift(1) - y.shift(96)).dropna(),
        }

        for col, ser in fallback_features.items():
            if col not in exog_future.columns:
                if len(ser) > 0:
                    exog_future[col] = ser.iloc[-1]
                else:
                    exog_future[col] = 0.0

    # ======================================================
    # 4. FILL SELECTED EXOG MANCANTI
    # ======================================================
    for col in selected_exog:

        if col in exog_future.columns:
            continue

        if col in df.columns and df[col].dropna().shape[0] > 0:
            exog_future[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .dropna()
                .iloc[-1]
            )
        else:
            exog_future[col] = 0.0

    # ======================================================
    # 5. CLEAN FINALE
    # ======================================================
    exog_future = exog_future.loc[:, ~exog_future.columns.duplicated()]
    exog_future = exog_future[selected_exog].copy()

    for c in exog_future.columns:
        exog_future[c] = pd.to_numeric(
            exog_future[c],
            errors="coerce"
        )

    exog_future = (
        exog_future
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .bfill()
        .fillna(0.0)
    )

    return exog_future


# ==========================================================
# FORECAST SINGLE MODEL MI
# ==========================================================

def forecast_next_96_single_mi_model_from_dropbox(
    df_hist,
    dropbox_token,
    model_path_from_json,
    dropbox_models_dir,
    nome_df=None,
    target=None,
    meteo=None,
    locations=None,
    terna=None,
    steps=96,
    freq="15min",
    lookback_days=10,
    terna_shift_steps=1
):

    payload, model_dropbox_path = load_mi_model_payload_from_dropbox(
        dropbox_token=dropbox_token,
        model_path_from_json=model_path_from_json,
        dropbox_models_dir=dropbox_models_dir
    )

    forecaster = payload["forecaster"]
    selected_exog = payload["selected_exog"]

    if target is None:
        target = payload["target"]

    if nome_df is None:
        nome_df = payload.get("nome_df", None)

    df = normalize_hist_df(df_hist)

    y = pd.to_numeric(df[target], errors="coerce").dropna()

    last_window = y.iloc[-forecaster.window_size:].copy()
    last_dt = y.index[-1]

    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(freq),
        periods=steps,
        freq=freq
    )

    # =========================
    # BUILD EXOG FUTURE BASE
    # =========================
    exog_future = build_mi_exog_future_pun_style(
        df_hist=df,
        target_col=target,
        future_index=future_index,
        selected_exog=selected_exog,
        meteo_downloader=meteo,
        locations=locations
    )

    # =========================
    # ✅ TERNA
    # =========================

    if terna is not None:
        try:
            terna_df = prepare_terna(terna,(last_dt - pd.Timedelta(days=lookback_days)).strftime("%d/%m/%Y"), future_index[-1].strftime("%d/%m/%Y"))
            terna_df = shift_terna_only(terna_df)
        except Exception as e:
            print(f"⚠️ Terna failed: {e}")
            terna_df = pd.DataFrame(index=df_hist.index)
    else:
        terna_df = pd.DataFrame(index=df_hist.index)


    exog_future = exog_future.merge(
            terna_df,
            on="Datetime",
            how="left"
        )

    # =========================
    # ✅ ENTSOE (FIX CRITICO)
    # =========================
    ZONES = [
        ("NORD", "10Y1001A1001A73I"),
        ("CNOR", "10Y1001A1001A70O"),
        ("CSUD", "10Y1001A1001A71M"),
        ("SUD",  "10Y1001A1001A788"),
        ("SARD", "10Y1001A1001A74G"),
        ("SICI", "10Y1001A1001A75E"),
        ("CALA", "10Y1001C--00096J")
    ]

    entsoe = EntsoeDownloader(
        token=st.secrets["ENTSOE_TOKEN"],
        zones=ZONES,
        start_date=future_index[0].to_pydatetime(),
        end_date=future_index[-1].to_pydatetime()
    )

    entsoe_feat = entsoe.build_features()

    entsoe_feat["Datetime"] = pd.to_datetime(entsoe_feat["Timestamp"])
    entsoe_feat = entsoe_feat.drop(columns=["Timestamp"])

    # 👉 merge ENTSOE
    exog_future = exog_future.merge(
        entsoe_feat,
        on="Datetime",
        how="left"
    )

    exog_future = exog_future.ffill()

    # =========================
    # PREDICT
    # =========================
    preds = forecaster.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )

    preds = pd.Series(preds, index=future_index)

    forecast_df = pd.DataFrame({
        "Datetime": future_index,
        "nome_df": nome_df,
        "target": target,
        "pred": preds.values,
        "created_at": pd.Timestamp.now(),
        "model_dropbox_path": model_dropbox_path
    })

    return forecast_df, exog_future, exog_future.copy()


# ==========================================================
# FORECAST ALL MODELS MI
# ==========================================================

def forecast_next_96_all_mi_models_dropbox(
    dfs,
    dropbox_token,
    dropbox_results_json_path,
    dropbox_models_dir,
    dropbox_forecasts_dir,
    forecast_history_long_path,
    forecast_history_wide_path,
    errors_path,
    meteo=None,
    locations=None,
    terna=None,
    terna_zone="Italy",
    steps=96,
    freq="15min",
    lookback_days=10,
    use_commodities=True,
    terna_shift_steps=1,
    save_single_files=True,
    save_exog_files=True,
    save_feature_frame=False,
    append_history=True,
    skip_errors=True
):
    """
    Forecast next 96 per tutti i modelli MI.
    """

    dropbox_forecasts_dir = normalize_dropbox_path(
        dropbox_forecasts_dir
    ).rstrip("/")

    results = read_json_from_dropbox(
        dropbox_token=dropbox_token,
        dropbox_path=dropbox_results_json_path
    )

    all_forecasts = []
    errors = []

    for nome_df, targets in results.items():

        if nome_df not in dfs:

            msg = f"{nome_df} non presente in dfs"
            print(f"⚠️ {msg}")

            errors.append({
                "nome_df": nome_df,
                "target": None,
                "error": msg
            })

            continue

        df_hist = dfs[nome_df]

        for target, res in targets.items():

            if not isinstance(res, dict):
                continue

            if res.get("status") != "ok":
                print(f"⚠️ {nome_df} / {target} status non ok → skip")
                continue

            model_path_from_json = res.get("model_path")

            if model_path_from_json is None:

                msg = f"model_path mancante per {nome_df} / {target}"
                print(f"⚠️ {msg}")

                errors.append({
                    "nome_df": nome_df,
                    "target": target,
                    "error": msg
                })

                continue

            try:
                print("\n" + "=" * 100)
                print(f"🔮 FORECAST MI NEXT {steps}: {nome_df} / {target}")
                print("=" * 100)

                forecast_df, exog_future, feature_frame = (
                    forecast_next_96_single_mi_model_from_dropbox(
                        df_hist=df_hist,
                        dropbox_token=dropbox_token,
                        model_path_from_json=model_path_from_json,
                        dropbox_models_dir=dropbox_models_dir,
                        nome_df=nome_df,
                        target=target,
                        meteo=meteo,
                        locations=locations,
                        steps=steps,
                        freq=freq,
                        lookback_days=lookback_days,
                        terna=terna,
                        terna_shift_steps=terna_shift_steps
                    )
                )

                all_forecasts.append(forecast_df)

                safe_nome_df = sanitize_filename(nome_df)
                safe_target = sanitize_filename(target)

                if save_single_files:

                    single_path = (
                        f"{dropbox_forecasts_dir}/"
                        f"forecast_next_{steps}__{safe_nome_df}__{safe_target}.parquet"
                    )

                    upload_df_parquet_to_dropbox(
                        df=forecast_df,
                        dropbox_token=dropbox_token,
                        dropbox_path=single_path,
                        index=False
                    )

                if save_exog_files:

                    exog_to_save = exog_future.copy()
                    exog_to_save.insert(0, "Datetime", exog_to_save.index)

                    exog_path = (
                        f"{dropbox_forecasts_dir}/"
                        f"exog_future_next_{steps}__{safe_nome_df}__{safe_target}.parquet"
                    )

                    upload_df_parquet_to_dropbox(
                        df=exog_to_save,
                        dropbox_token=dropbox_token,
                        dropbox_path=exog_path,
                        index=False
                    )

            except Exception as e:
                msg = f"{nome_df} / {target} -> {str(e)}"
                print("💣 ERRORE REALE:", msg)
                raise RuntimeError(msg)

    if len(all_forecasts) == 0:

        df_errors = pd.DataFrame(errors)

        if len(df_errors) > 0:
            upload_df_parquet_to_dropbox(
                df=df_errors,
                dropbox_token=dropbox_token,
                dropbox_path=errors_path,
                index=False
            )

        if not skip_errors:
            raise RuntimeError(
                "Nessun forecast prodotto. Errori:\n"
                + df_errors.to_string(index=False)
            )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            df_errors
        )

    df_new_long = pd.concat(
        all_forecasts,
        ignore_index=True
    )

    df_new_long["Datetime"] = pd.to_datetime(df_new_long["Datetime"])
    df_new_long["created_at"] = pd.to_datetime(df_new_long["created_at"])

    if append_history and dbx_file_exists(
        dropbox_token=dropbox_token,
        dropbox_path=forecast_history_long_path
    ):

        df_old_long = read_parquet_from_dropbox(
            dropbox_token=dropbox_token,
            dropbox_path=forecast_history_long_path
        )

        df_all_long = pd.concat(
            [df_old_long, df_new_long],
            ignore_index=True
        )

    else:
        df_all_long = df_new_long.copy()

    df_all_long["Datetime"] = pd.to_datetime(df_all_long["Datetime"])

    if "created_at" in df_all_long.columns:
        df_all_long["created_at"] = pd.to_datetime(df_all_long["created_at"])
    else:
        df_all_long["created_at"] = pd.Timestamp.now()

    df_all_long = (
        df_all_long
        .sort_values(["Datetime", "nome_df", "target", "created_at"])
        .drop_duplicates(
            subset=["Datetime", "nome_df", "target"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    upload_df_parquet_to_dropbox(
        df=df_all_long,
        dropbox_token=dropbox_token,
        dropbox_path=forecast_history_long_path,
        index=False
    )

    df_all_wide = (
        df_all_long
        .assign(
            model=lambda x: (
                x["nome_df"].astype(str)
                + "__"
                + x["target"].astype(str)
            )
        )
        .pivot_table(
            index="Datetime",
            columns="model",
            values="pred",
            aggfunc="first"
        )
        .reset_index()
    )

    upload_df_parquet_to_dropbox(
        df=df_all_wide,
        dropbox_token=dropbox_token,
        dropbox_path=forecast_history_wide_path,
        index=False
    )

    df_errors = pd.DataFrame(errors)

    if len(df_errors) > 0:

        upload_df_parquet_to_dropbox(
            df=df_errors,
            dropbox_token=dropbox_token,
            dropbox_path=errors_path,
            index=False
        )

    print("\n" + "=" * 100)
    print("✅ FORECAST MI COMPLETATO")
    print(f"📄 History long: {forecast_history_long_path}")
    print(f"📄 History wide: {forecast_history_wide_path}")
    print(
        "📊 Modelli forecastati:",
        df_new_long[["nome_df", "target"]]
        .drop_duplicates()
        .shape[0]
    )
    print("=" * 100)

    return df_all_long, df_all_wide, df_errors
