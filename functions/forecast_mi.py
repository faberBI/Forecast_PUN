import os
import io
import json
import joblib
import dropbox
import pandas as pd
import streamlit as st


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


def read_joblib_from_dropbox(dropbox_token, dropbox_path):
    content = dbx_download_bytes(
        dropbox_token=dropbox_token,
        dropbox_path=dropbox_path
    )

    return joblib.load(io.BytesIO(content))


# ==========================================================
# DROPBOX WRITERS
# ==========================================================

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
# MODEL PATH RESOLVER
# ==========================================================

def resolve_mi_model_dropbox_path(
    model_path_from_json,
    dropbox_models_dir
):
    """
    Converte il model_path salvato nel JSON nel path Dropbox effettivo.

    Gestisce:
    - /forecast_mi/models_retrained/model.joblib
    - models_retrained/model.joblib
    - model.joblib
    """

    if model_path_from_json is None:
        raise ValueError("model_path_from_json è None.")

    model_path_from_json = str(model_path_from_json).replace("\\", "/")

    if model_path_from_json.startswith("/"):
        return model_path_from_json

    file_name = os.path.basename(model_path_from_json)

    dropbox_models_dir = normalize_dropbox_path(
        dropbox_models_dir
    ).rstrip("/")

    return f"{dropbox_models_dir}/{file_name}"

@st.cache_data(show_spinner=False)
def load_mi_model_payload_from_dropbox(
    dropbox_token,
    model_path_from_json,
    dropbox_models_dir
):
    """
    Carica modello MI da Dropbox usando il filename reale trovato nella cartella models_retrained.
    Robusto contro:
    - path Windows nel JSON
    - models_retrained\\file.joblib
    - path parziali
    """

    if model_path_from_json is None:
        raise ValueError("model_path_from_json è None.")

    dbx = dropbox.Dropbox(dropbox_token)

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
            f"Errore lettura cartella modelli Dropbox: {dropbox_models_dir} -> {e}"
        )

    if file_name not in available_models:
        raise FileNotFoundError(
            f"Modello non trovato su Dropbox: {file_name}. "
            f"Cartella: {dropbox_models_dir}. "
            f"Disponibili: {list_available_models_safe(available_models)}"
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


def list_available_models_safe(available_models):
    try:
        return list(available_models.keys())[:20]
    except Exception:
        return []


# ==========================================================
# FORECAST SINGLE MI MODEL
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
    terna_zone="Italy",
    steps=96,
    freq="15min",
    lookback_days=10,
    use_commodities=True,
    terna_shift_steps=1
):
    """
    Forecast next 96 per un singolo modello MI salvato su Dropbox.

    Dipendenze richieste nello stesso file:
    - normalize_hist_df
    - build_forecast_feature_frame_same_features
    """

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

    # ======================================================
    # STORICO TARGET
    # ======================================================

    df = normalize_hist_df(df_hist)

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' non presente in df_hist. "
            f"Prime colonne disponibili: {df.columns.tolist()[:50]}"
        )

    y = pd.to_numeric(
        df[target],
        errors="coerce"
    ).dropna()

    if len(y) == 0:
        raise ValueError(
            f"Target '{target}' vuoto dopo conversione numerica."
        )

    window_size = forecaster.window_size

    if len(y) < window_size:
        raise ValueError(
            f"Serie troppo corta per last_window. "
            f"len(y)={len(y)}, window_size={window_size}"
        )

    last_window = y.iloc[-window_size:].copy()

    # ======================================================
    # FUTURE INDEX
    # ======================================================

    last_dt = y.index[-1]

    future_index = pd.date_range(
        start=last_dt + pd.Timedelta(freq),
        periods=steps,
        freq=freq
    )

    # ======================================================
    # EXOG FUTURE
    # ======================================================

    exog_future, feature_frame = build_forecast_feature_frame_same_features(
        df_hist=df,
        target_col=target,
        future_index=future_index,
        selected_exog=selected_exog,
        meteo=meteo,
        locations=locations,
        terna=terna,
        terna_zone=terna_zone,
        lookback_days=lookback_days,
        use_commodities=use_commodities,
        terna_shift_steps=terna_shift_steps
    )

    exog_future = exog_future[selected_exog].copy()

    # ======================================================
    # PREDICT
    # ======================================================

    preds = forecaster.predict(
        steps=steps,
        last_window=last_window,
        exog=exog_future
    )

    preds = pd.Series(
        np.asarray(preds).reshape(-1),
        index=future_index,
        name="pred"
    )

    created_at = pd.Timestamp.now()

    forecast_df = pd.DataFrame({
        "Datetime": future_index,
        "nome_df": nome_df,
        "target": target,
        "pred": preds.values,
        "created_at": created_at,
        "model_dropbox_path": model_dropbox_path
    })

    return forecast_df, exog_future, feature_frame


# ==========================================================
# FORECAST ALL MI MODELS
# ==========================================================
def normalize_hist_df(df_hist):

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

    Legge da Dropbox:
    - JSON risultati/configurazioni
    - modelli joblib

    Usa:
    - dfs già caricati dai parquet Dropbox

    Salva su Dropbox:
    - forecast singoli
    - exog future
    - forecast history long
    - forecast history wide
    - errors
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

    # ======================================================
    # LOOP MODELLI
    # ======================================================

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
                        terna=terna,
                        terna_zone=terna_zone,
                        steps=steps,
                        freq=freq,
                        lookback_days=lookback_days,
                        use_commodities=use_commodities,
                        terna_shift_steps=terna_shift_steps
                    )
                )

                all_forecasts.append(forecast_df)

                safe_nome_df = sanitize_filename(nome_df)
                safe_target = sanitize_filename(target)

                # ==================================================
                # SAVE SINGLE FORECAST
                # ==================================================

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

                # ==================================================
                # SAVE EXOG FUTURE
                # ==================================================

                if save_exog_files:

                    exog_to_save = exog_future.copy()

                    exog_to_save.insert(
                        0,
                        "Datetime",
                        exog_to_save.index
                    )

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

                # ==================================================
                # SAVE FEATURE FRAME DEBUG
                # ==================================================

                if save_feature_frame:

                    ff = feature_frame.copy()

                    ff.insert(
                        0,
                        "Datetime_index",
                        ff.index
                    )

                    ff_path = (
                        f"{dropbox_forecasts_dir}/"
                        f"feature_frame_debug__{safe_nome_df}__{safe_target}.parquet"
                    )

                    upload_df_parquet_to_dropbox(
                        df=ff,
                        dropbox_token=dropbox_token,
                        dropbox_path=ff_path,
                        index=False
                    )

            except Exception as e:

                msg = str(e)

                print(f"❌ Errore forecast {nome_df} / {target}: {msg}")

                errors.append({
                    "nome_df": nome_df,
                    "target": target,
                    "model_path_from_json": model_path_from_json,
                    "error": msg
                })

                if not skip_errors:
                    raise

                continue

    # ======================================================
    # NESSUN FORECAST
    # ======================================================

    if len(all_forecasts) == 0:

        df_errors = pd.DataFrame(errors)

        if len(df_errors) > 0:

            upload_df_parquet_to_dropbox(
                df=df_errors,
                dropbox_token=dropbox_token,
                dropbox_path=errors_path,
                index=False
            )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
            df_errors
        )

    # ======================================================
    # NEW LONG
    # ======================================================

    df_new_long = pd.concat(
        all_forecasts,
        ignore_index=True
    )

    df_new_long["Datetime"] = pd.to_datetime(
        df_new_long["Datetime"]
    )

    if "created_at" in df_new_long.columns:
        df_new_long["created_at"] = pd.to_datetime(
            df_new_long["created_at"]
        )
    else:
        df_new_long["created_at"] = pd.Timestamp.now()

    # ======================================================
    # APPEND HISTORY LONG
    # ======================================================

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

    df_all_long["Datetime"] = pd.to_datetime(
        df_all_long["Datetime"]
    )

    if "created_at" in df_all_long.columns:
        df_all_long["created_at"] = pd.to_datetime(
            df_all_long["created_at"]
        )
    else:
        df_all_long["created_at"] = pd.Timestamp.now()

    df_all_long = (
        df_all_long
        .sort_values(
            ["Datetime", "nome_df", "target", "created_at"]
        )
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

    # ======================================================
    # WIDE HISTORY
    # ======================================================

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

    # ======================================================
    # ERRORS
    # ======================================================

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


import numpy as np


# ==========================================================
# UTILS
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


# ==========================================================
# DROPBOX LOW LEVEL
