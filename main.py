import os
import traceback
from datetime import date
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
import plotly.graph_objects as go
import plotly.express as px
st.set_page_config(page_title="PUN Dataset Manager", layout="wide")

from functions.create_datasets import (PUNFeatureEngineering, MeteoDownloader, TernaClient, ks_drift, upload_to_dropbox, load_from_dropbox)
from functions.forecast import (pun_to_datetime, plot_forecast_pun)
from functions.create_datasets import EntsoeDownloader

# --- NUOVO MODELLO: PUN Direct 96 (LightGBM p50/p90 + blend Optuna) ---
from functions.pun_direct_forecast import (
    load_direct_artifacts,
    forecast_next_96,
    build_native_importance_df,
    summarize_native_importance,
)

import yaml
import dropbox

CONFIG_PATH = "config/config.yaml"

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

FEATURES_OLD = config["features"]["FEATURES_OLD"]
FEATURES_NEW = config["features"]["FEATURES_NEW"]
SELECTED_EXOG = config["features"]["SELECTED_EXOG"]

# =========================================================
# CONFIG UI
# =========================================================
st.title("⚡ PUN Dataset Manager")
st.caption("Aggiornamento dataset intraday PUN / Meteo / Terna")

# =========================================================
# PATH CONFIG
# =========================================================
HISTORICAL_PATH = "dati_input/final_dataset_historical.parquet"
OUTPUT_PATH = "dati_output/final_dataset_intra_day.parquet"
PUN_INPUT_PATH = "dati_input/Add_on_PUN.xlsx"


# =========================================================
# CONSTANTS
# =========================================================
LOCATIONS = [
    {"name": "milano", "lat": 45.4642, "lon": 9.1900},
    {"name": "torino", "lat": 45.0703, "lon": 7.6869},
    {"name": "roma", "lat": 41.9028, "lon": 12.4964},
    {"name": "bologna", "lat": 44.4949, "lon": 11.3426},
    {"name": "bari", "lat": 41.1171, "lon": 16.8719},
    {"name": "palermo", "lat": 38.1157, "lon": 13.3615},
]

FEATURES_DROP = ["Data", "Ora", "Periodo", "date", "target", "Minute"]


# =========================================================
# LOADERS
# =========================================================
@st.cache_data(show_spinner=False)
def load_historical(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
    df = df.sort_index()
    return df


@st.cache_data(show_spinner=False)
def load_output_if_exists(path: str):
    if os.path.exists(path):
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.set_index("Datetime")
        df = df.sort_index()
        return df
    return None


# =========================================================
# HELPERS
# =========================================================
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


def aggregate_meteo(df: pd.DataFrame) -> pd.DataFrame:
    def mean_cols(substr):
        cols = [c for c in df.columns if substr in c]
        return df[cols].mean(axis=1) if cols else np.nan

    df = df.copy()
    df["temperature_mean"] = mean_cols("temperature_2m")
    df["cloud_cover_mean"] = mean_cols("cloud_cover")
    df["wind_speed_mean"] = mean_cols("wind_speed_80m")

    precip = [c for c in df.columns if "precipitation" in c]
    if precip:
        df["precipitation_mean"] = df[precip].mean(axis=1)

    return df


def prepare_meteo(meteo, start_date_meteo: str, end_date_meteo: str) -> pd.DataFrame:
    df = meteo.download_multi_city(LOCATIONS, start_date_meteo, end_date_meteo)
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("h")
    df = df.groupby("Datetime").mean(numeric_only=True).reset_index()
    df = aggregate_meteo(df)

    # hourly -> 15min
    df = (
        df.set_index("Datetime")
          .sort_index()
          .resample("15min")
          .ffill()
          .reset_index()
    )

    return df


def prepare_pun(pun_fe, pun_path: str) -> pd.DataFrame:
    raw = pd.read_excel(pun_path)
    df = pun_fe.prepare_dataset(raw, merge_commodities=True)

    # Il PUN è già quarter-hour tramite Data/Ora/Periodo
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    return df

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


def merge_all(pun_df: pd.DataFrame, meteo_df: pd.DataFrame, terna_df: pd.DataFrame) -> pd.DataFrame:
    df = pun_df.merge(meteo_df, on="Datetime", how="left")
    df = df.merge(terna_df, on="Datetime", how="left")
    return df

def add_features(df):
    df = df.copy()

    numeric_cols = [
        "total_load_MW",
        "actual_generation_MW",
        "renewable_generation_MW",
        "forecast_total_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if {"total_load_MW", "actual_generation_MW"} <= set(df.columns):
        df["net_load"] = df["total_load_MW"] - df["actual_generation_MW"]

    if {"renewable_generation_MW", "total_load_MW"} <= set(df.columns):
        df["renewable_share"] = df["renewable_generation_MW"] / (df["total_load_MW"] + 1e-6)

    if "total_load_MW" in df.columns:
        df["load_ramp_1h"] = df["total_load_MW"].diff(4)

    if {"forecast_total_load_MW", "total_load_MW"} <= set(df.columns):
        df["load_forecast_error"] = df["forecast_total_load_MW"] - df["total_load_MW"]

    return df

def make_quarter_hour(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Minute"] = np.tile([0, 15, 30, 45], len(df) // 4 + 1)[: len(df)]
    df["Datetime"] = df["Datetime"] + pd.to_timedelta(df["Minute"], unit="m")

    df = df.set_index("Datetime").sort_index()
    df = df[~df.index.duplicated()]
    df = df.asfreq("15min")
    df = df.reset_index()

    return df


def validate_required_columns(df: pd.DataFrame, required_cols: list, df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name}: mancano le colonne richieste: {missing}")


# =========================================================
# CORE PIPELINE (invariata: aggiornamento dataset storico)
# =========================================================
def pipeline_run():
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)

    today = date.today()

    # storico
    df_historical = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]).copy()

    if not isinstance(df_historical.index, pd.DatetimeIndex):
      df_historical["Datetime"] = pd.to_datetime(df_historical["Datetime"])
      df_historical = df_historical.set_index("Datetime")

    df_historical = df_historical.sort_index().asfreq("15min").ffill()

    last_date = df_historical.index.max()
    # =========================================================
    # LOOKBACK per feature tipo lag_7d / pun_ret_7d / momentum_1d
    # =========================================================
    LOOKBACK_DAYS = 8  # 7 giorni + margine
    lookback_start_dt = pd.Timestamp(last_date).floor("D") - pd.Timedelta(days=LOOKBACK_DAYS)
    end_date_dt = pd.Timestamp(today)

    START_DATE_METEO = lookback_start_dt.strftime("%Y-%m-%d")
    END_DATE_METEO = end_date_dt.strftime("%Y-%m-%d")

    START_DATE_TERNA = lookback_start_dt.strftime("%d/%m/%Y")
    END_DATE_TERNA = end_date_dt.strftime("%d/%m/%Y")

    START_DATE_PUN = lookback_start_dt.strftime("%Y-%m-%d")

    log(f"Ultima data storico: {last_date}")
    log(f"Lookback start: {lookback_start_dt}")
    log(f"Download Meteo da {START_DATE_METEO} a {END_DATE_METEO}")
    log(f"Download Terna da {START_DATE_TERNA} a {END_DATE_TERNA}")
    log(f"Start PUN: {START_DATE_PUN}")

    # =========================================================
    # PIPELINE NUOVI DATI (CON LOOKBACK)
    # =========================================================
    pun_fe = PUNFeatureEngineering(start=START_DATE_PUN, pun_col="PUN")
    meteo = MeteoDownloader()
    #
    log("Preparazione PUN...")
    pun_hist = df_historical.reset_index()[["Datetime", "PUN"]].copy()

    # 2️⃣ nuovi dati PUN (da Excel → già feature engineering completo)
    pun_df_new = prepare_pun(pun_fe, PUN_INPUT_PATH)

    # 3️⃣ prendi solo colonne raw PUN
    pun_df_new = pun_df_new[["Datetime", "PUN"]].copy()

    # 4️⃣ concat tutto
    pun_full = pd.concat([pun_hist, pun_df_new], ignore_index=True)

    pun_full = (
      pun_full
      .drop_duplicates(subset=["Datetime"], keep="last")
      .sort_values("Datetime")
      )

    # 5️⃣ ora calcoli i lag DIRETTAMENTE qui (molto più semplice)
    pun_full = pun_full.sort_values("Datetime")

    pun_full["lag_2d"] = pun_full["PUN"].shift(96*2)
    pun_full["lag_7d"] = pun_full["PUN"].shift(96*7)

    pun_full["pun_ret_1d"] = pun_full["PUN"].pct_change(96).shift(1)
    pun_full["pun_ret_7d"] = pun_full["PUN"].pct_change(96*7).shift(1)

    pun_full["momentum_1d"] = pun_full["PUN"].shift(1) - pun_full["PUN"].shift(96)
    # minute
    pun_full["minute"] = pd.to_datetime(pun_full["Datetime"]).dt.minute
    # pun_ret_1h (4 quarter)
    pun_full["pun_ret_1h"] = (pun_full["PUN"].pct_change(4).shift(1))
    # momentum_4h (16 quarter)
    pun_full["momentum_4h"] = (pun_full["PUN"].shift(1) - pun_full["PUN"].shift(16))


    # 6️⃣ tieni solo la parte nuova
    pun_full["Datetime"] = pd.to_datetime(pun_full["Datetime"])

    pun_df = pun_full[pun_full["Datetime"] >= lookback_start_dt].copy()
    #

    log("Preparazione Meteo...")
    meteo_df = prepare_meteo(meteo, START_DATE_METEO, END_DATE_METEO)

    terna_client_id = st.secrets.get("TERNA_CLIENT_ID", os.getenv("TERNA_CLIENT_ID", ""))
    terna_client_secret = st.secrets.get("TERNA_CLIENT_SECRET", os.getenv("TERNA_CLIENT_SECRET", ""))

    if not terna_client_id or not terna_client_secret:
        raise ValueError(
            "Credenziali Terna mancanti. Inserisci TERNA_CLIENT_ID e TERNA_CLIENT_SECRET in st.secrets oppure env vars."
        )

    terna = TernaClient(
        client_id=terna_client_id,
        client_secret=terna_client_secret,
    )

    log("Preparazione Terna...")
    terna_df = prepare_terna(terna, START_DATE_TERNA, END_DATE_TERNA)
    terna_df = shift_terna_only(terna_df, shift_steps=1)

    # =========================
    # ✅ ENTSOE
    # =========================
    log("Preparazione ENTSOE...")

    ZONES = [
        ("NORD", "10Y1001A1001A73I"),
        ("CNOR", "10Y1001A1001A70O"),
        ("CSUD", "10Y1001A1001A71M"),
        ("SUD",  "10Y1001A1001A788"),
        ("SARD", "10Y1001A1001A74G"),
        ("SICI", "10Y1001A1001A75E"),
        ("CALA", "10Y1001C--00096J")]

    entsoe = EntsoeDownloader(
        token=st.secrets["ENTSOE_TOKEN"],
        zones=ZONES,
        start_date=lookback_start_dt.to_pydatetime(),
        end_date=end_date_dt.to_pydatetime()
        )

    entsoe_feat = entsoe.build_features()

    log(f"ENTSOE max: {entsoe_feat.index.max()}")


    log("Merge dataset...")

    df_new_raw = merge_all(pun_df, meteo_df, terna_df)

    df_new_raw["Datetime"] = pd.to_datetime(df_new_raw["Datetime"])
    df_new_raw = df_new_raw.set_index("Datetime")

    # ✅ join ENTSOE (SAFE)
    df_new_raw = df_new_raw.join(entsoe_feat, how="left")
    df_new_raw = df_new_raw.reset_index()


    ent_cols = [c for c in entsoe_feat.columns]
    df_new_raw[ent_cols] = df_new_raw[ent_cols].fillna(0.0)

    log("Feature engineering...")
    df_new = add_features(df_new_raw)

    # pulizia colonne inutili
    df_new.drop(columns=FEATURES_DROP, errors="ignore", inplace=True)
    df_new = df_new.reset_index(drop=True)

    # =========================================================
    # TIENI SOLO LE RIGHE DAVVERO NUOVE
    # =========================================================
    df_new["Datetime"] = pd.to_datetime(df_new["Datetime"])
    df_new = df_new[df_new["Datetime"] > last_date].copy()

    missing = [c for c in FEATURES_NEW if c not in df_new.columns]
    extra = [c for c in df_new.columns if c not in FEATURES_NEW]

    if missing:
        st.error(f"❌ Feature mancanti: {missing}")
        raise ValueError("Schema mismatch -> stop pipeline")

    if extra:
        st.warning(f"⚠️ Feature extra ignorate: {extra}")

    validate_required_columns(
        df_historical.reset_index(),
        ["Datetime"] + FEATURES_OLD,
        "df_historical.reset_index()"
    )
    validate_required_columns(df_new, FEATURES_NEW, "df new pipeline")

    # storico già pronto
    df_old = df_historical[FEATURES_OLD].copy().reset_index()

    # nuove righe già pronte
    df_new = df_new[FEATURES_NEW].copy()

    # concat finale
    df_final = pd.concat([df_old, df_new], axis=0, ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["Datetime"], keep="last")
    df_final = df_final.sort_values("Datetime").reset_index(drop=True)
    df_final.set_index("Datetime", inplace=True)

    log(f"Shape finale: {df_final.shape}")
    log(f"Ultima data finale: {df_final.index.max()}")

    df_final.to_parquet(OUTPUT_PATH)
    log(f"Salvato file: {OUTPUT_PATH}")

    upload_to_dropbox(
        OUTPUT_PATH,
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    )
    log("Salvato dataset su Dropbox ✅")

    return df_final, log_lines

# =========================================================
# UI - STATUS
# =========================================================
col1, col2 = st.columns(2)
df_historical = None

try:
    df_historical = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]).copy()

    if not isinstance(df_historical.index, pd.DatetimeIndex):
        df_historical["Datetime"] = pd.to_datetime(df_historical["Datetime"])
        df_historical = df_historical.set_index("Datetime")

    df_historical = df_historical.sort_index()

    last_hist_date = df_historical.index.max()

    with col1:
        st.metric("📅 Ultima data DB storico", str(last_hist_date))

except Exception as e:
    with col1:
        st.warning("⚠️ Dropbox non disponibile")
        st.metric("📅 Ultima data DB storico", "non disponibile")


st.divider()


# =========================================================
# UI - CONTROL PANEL
# =========================================================
left, right = st.columns([1, 2])

with left:
    run_update = st.button("🔄 Aggiorna dataset", use_container_width=True)

with right:
    st.info(
        "Il bottone esegue la pipeline completa: "
        "PUN + Meteo + Terna → merge → feature engineering → allineamento schema → salvataggio parquet."
    )


# =========================================================
# EXECUTION
# =========================================================
WINDOW = 96 * 7   # 7 giorni


drift_cols = SELECTED_EXOG

if run_update:
    try:
        with st.spinner("🚀 Aggiornamento dataset in corso..."):
            df_updated, logs = pipeline_run()

            # ✅ DRIFT CHECK
            drift_df = ks_drift(
                df_historical.tail(WINDOW),
                df_updated.tail(WINDOW),
                drift_cols
            )

            st.subheader("📊 Covariate Drift (KS Test)")

            if not drift_df.empty:
                st.dataframe(drift_df)


                n_drift = drift_df["drift_flag"].sum()

                if n_drift >= 5:
                    st.error("🚨 Drift forte → retrain consigliato")
                elif n_drift > 0:
                    st.warning("⚠️ Drift moderato → monitorare")
                else:
                    st.success("✅ Nessun drift significativo")

            else:
                st.warning("⚠️ Drift non calcolabile (dati insufficienti)")

        st.success("✅ Dataset aggiornato correttamente")

        st.subheader("📝 Log esecuzione")
        st.code("\n".join(logs), language="text")

        st.subheader("📊 Preview dataset finale")
        st.dataframe(df_updated.tail(100), use_container_width=True)

        st.subheader("📈 Info dataset finale")
        c1, c2, c3 = st.columns(3)
        c1.metric("Righe", f"{len(df_updated):,}".replace(",", "."))
        c2.metric("Colonne", df_updated.shape[1])
        c3.metric("Ultima data", str(df_updated.index.max()))

        st.download_button(
            label="⬇️ Scarica snapshot CSV (ultime 500 righe)",
            data=df_updated.tail(500).to_csv(index=True).encode("utf-8"),
            file_name="final_dataset_intra_day_last500.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # invalida cache letture
        st.cache_data.clear()

    except Exception as e:
        st.error(f"❌ Errore durante l'aggiornamento: {e}")
        st.code(traceback.format_exc(), language="python")


st.divider()


# =========================================================
# VIEW DATA
# =========================================================
st.subheader("📚 Preview DB storico")
if df_historical is not None:
    st.dataframe(df_historical.tail(50), use_container_width=True)


st.subheader("📦 Preview DB aggiornato")

try:
    df_output = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]).copy()

    if not isinstance(df_output.index, pd.DatetimeIndex):
        df_output["Datetime"] = pd.to_datetime(df_output["Datetime"])
        df_output = df_output.set_index("Datetime")

    df_output = df_output.sort_index()

    st.dataframe(df_output.tail(50), use_container_width=True)

except:
    st.warning("⚠️ Nessun dato disponibile su Dropbox")

# =========================================================
# LOAD MODEL FROM DROPBOX — PUN DIRECT 96 (NUOVO MODELLO)
# =========================================================
# Il vecchio modello (skforecast ForecasterDirect + pipeline.py + SHAP)
# e' stato sostituito dal nuovo forecaster "PUN Direct 96":
# 96 coppie di modelli LightGBM (p50/p90), uno per ciascun horizon
# di 15 minuti, con blend p50/p90 ottimizzato via Optuna per fascia oraria.
#
# I 3 artefatti attesi su Dropbox sono quelli prodotti da
# functions/pun_direct_forecast.py::train_direct_models(...):
#   - pun_direct_lgbm_p50.joblib
#   - pun_direct_lgbm_p90.joblib
#   - pun_direct_metadata.json

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROPBOX_MODEL_DIR = "/forecast_pun/models_direct"

MODEL_P50_PATH = MODEL_DIR / "pun_direct_lgbm_p50.joblib"
MODEL_P90_PATH = MODEL_DIR / "pun_direct_lgbm_p90.joblib"
METADATA_PATH = MODEL_DIR / "pun_direct_metadata.json"

DROPBOX_MODEL_P50_PATH = f"{DROPBOX_MODEL_DIR}/pun_direct_lgbm_p50.joblib"
DROPBOX_MODEL_P90_PATH = f"{DROPBOX_MODEL_DIR}/pun_direct_lgbm_p90.joblib"
DROPBOX_METADATA_PATH = f"{DROPBOX_MODEL_DIR}/pun_direct_metadata.json"


def make_dbx_client():
    """
    Usa Dropbox token da Streamlit secrets oppure env.
    """
    token = st.secrets.get("DROPBOX_TOKEN", os.getenv("DROPBOX_TOKEN", ""))

    if not token:
        raise RuntimeError(
            "DROPBOX_TOKEN mancante. Inseriscilo in st.secrets oppure nelle env vars."
        )

    return dropbox.Dropbox(oauth2_access_token=token)


def download_file_from_dropbox(dbx, dropbox_path: str, local_path: Path):
    """
    Scarica un file da Dropbox e lo salva localmente.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        metadata, res = dbx.files_download(dropbox_path)
    except Exception as e:
        raise RuntimeError(f"Download Dropbox fallito: {dropbox_path} -> {e}") from e

    with open(local_path, "wb") as f:
        f.write(res.content)

    return metadata


def local_file_ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


@st.cache_resource(show_spinner=False)
def load_model_artifacts_from_dropbox(force_download: bool = False):
    """
    Carica il modello PUN Direct 96 (p50/p90 + metadata) da Dropbox.

    Ritorna:
    {
        "models_p50": dict[str horizon -> Pipeline],
        "models_p90": dict[str horizon -> Pipeline],
        "metadata": dict,
    }
    """
    dbx = make_dbx_client()

    must_download = (
        force_download
        or not local_file_ok(MODEL_P50_PATH)
        or not local_file_ok(MODEL_P90_PATH)
        or not local_file_ok(METADATA_PATH)
    )

    if must_download:
        today_minus_1 = pd.Timestamp.today() - pd.Timedelta(days=1)
        st.info(
            f"📥 Download modello PUN Direct 96 da Dropbox aggiornato al "
            f"{today_minus_1.strftime('%d-%m-%Y')}..."
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_MODEL_P50_PATH,
            local_path=MODEL_P50_PATH,
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_MODEL_P90_PATH,
            local_path=MODEL_P90_PATH,
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_METADATA_PATH,
            local_path=METADATA_PATH,
        )

    try:
        return load_direct_artifacts(model_dir=str(MODEL_DIR))
    except Exception as e:
        st.error(f"❌ Errore caricamento modello da file locale: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="python")
        raise


force_model_download = st.sidebar.button("🔁 Forza download modello da Dropbox")

artifacts = load_model_artifacts_from_dropbox(
    force_download=force_model_download
)

models_p50 = artifacts["models_p50"]
models_p90 = artifacts["models_p90"]
model_metadata = artifacts["metadata"]

st.success("✅ Modello PUN Direct 96 caricato da Dropbox")

n_base_features = len(model_metadata.get("base_feature_cols", []))
overall_metrics = model_metadata.get("overall_metrics", {})

st.caption(
    f"📦 Modello: PUN Direct 96 (LightGBM p50/p90 + blend Optuna orario) | "
    f"steps: {model_metadata.get('steps', 96)} | "
    f"feature base: {n_base_features}"
)

if overall_metrics:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (holdout)", f"{overall_metrics.get('MAE', float('nan')):.3f}")
    c2.metric("RMSE (holdout)", f"{overall_metrics.get('RMSE', float('nan')):.3f}")
    c3.metric("Bias (holdout)", f"{overall_metrics.get('Bias', float('nan')):.3f}")
    c4.metric("R2 (holdout)", f"{overall_metrics.get('R2', float('nan')):.3f}")

#
try:
    df_hist = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]).copy()

    if not isinstance(df_hist.index, pd.DatetimeIndex):
        df_hist["Datetime"] = pd.to_datetime(df_hist["Datetime"])
        df_hist = df_hist.set_index("Datetime")

    df_hist = df_hist.sort_index().asfreq("15min").ffill()

except:
    st.warning("⚠️ Dataset non disponibile su Dropbox")
    st.stop()

FORECAST_PATH = "dati_output/forecast_history.parquet"


# =========================================================
# SESSION STATE FORECAST
# =========================================================
if "pun_preds" not in st.session_state:
    st.session_state["pun_preds"] = None

if "pun_importance_df" not in st.session_state:
    st.session_state["pun_importance_df"] = None

if "pun_forecast_done" not in st.session_state:
    st.session_state["pun_forecast_done"] = False

# =========================================================
# FORECAST BUTTON
# =========================================================
run_forecast = st.button(
    "📈 Esegui Forecast Day Ahead",
    use_container_width=True
)

if run_forecast:

    try:

        with st.spinner("📈 Calcolo forecast next 96 (PUN Direct 96)..."):

            forecast_df = forecast_next_96(
                df=df_hist,
                model_dir=str(MODEL_DIR),
                output_dir=str(MODEL_DIR),
                save_csv=False,
            )

        # Adatto le colonne al formato atteso dal resto dell'app
        # (Datetime / pred, come produceva il vecchio forecaster)
        preds = forecast_df.rename(
            columns={
                "target_time": "Datetime",
                "PUN_forecast": "pred",
            }
        ).copy()

        st.session_state["pun_preds"] = preds.copy()
        st.session_state["pun_forecast_done"] = True

        st.success("✅ Forecast completato")

        # =====================================================
        # IMPORTANZA FEATURE (sostituisce lo SHAP legacy)
        # =====================================================
        with st.spinner("⏳ Calcolo importanza feature (gain LightGBM)..."):
            importance_df = build_native_importance_df(
                models_p50=models_p50,
                metadata=model_metadata,
            )

        st.session_state["pun_importance_df"] = importance_df

        # =====================================================
        # SAVE FORECAST
        # =====================================================
        preds_to_save = preds.copy()
        preds_to_save["created_at"] = pd.Timestamp.now()

        if os.path.exists(FORECAST_PATH):

            df_old = pd.read_parquet(FORECAST_PATH)

            df_all = pd.concat(
                [df_old, preds_to_save],
                ignore_index=True
            )

        else:

            df_all = preds_to_save.copy()

            st.info(
                "📦 Primo forecast salvato "
                "(inizializzazione storico)"
            )

        df_all = (
            df_all
            .drop_duplicates(
                subset=["Datetime"],
                keep="last"
            )
            .sort_values("Datetime")
        )

        df_all.to_parquet(FORECAST_PATH)

        upload_to_dropbox(
            FORECAST_PATH,
            "/forecast_pun/forecast_history.parquet",
            st.secrets["DROPBOX_TOKEN"]
        )

        st.success("✅ Salvato su Dropbox")
        st.success(
            f"✅ Salvati {len(preds_to_save)} nuovi forecast"
        )

        st.write(
            f"📊 Totale forecast storico: {len(df_all)}"
        )

    except Exception as e:

        st.error(
            f"❌ Errore forecast: "
            f"{type(e).__name__}: {e}"
        )

        st.code(
            traceback.format_exc(),
            language="python"
        )

# =========================================================
# RENDER PERSISTENTE FORECAST + IMPORTANZA FEATURE
# =========================================================
if (
    st.session_state.get("pun_forecast_done")
    and st.session_state.get("pun_preds") is not None
):

    preds = st.session_state["pun_preds"].copy()
    importance_df = st.session_state["pun_importance_df"]

    st.divider()

    st.subheader("📊 Preview Forecast")

    st.dataframe(
        preds.tail(50),
        use_container_width=True
    )

    # =====================================================
    # FORECAST PANEL
    # =====================================================
    fig, stats = plot_forecast_pun(preds)

    st.subheader("📈 Forecast intraday PUN")

    st.dataframe(
        preds,
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Min", stats["min"])
    c2.metric("Max", stats["max"])
    c3.metric("Mean", stats["mean"])

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Scarica forecast",
        data=preds.to_csv(index=False),
        file_name="forecast_day_ahead.csv",
        mime="text/csv",
    )

    # =====================================================
    # BANDA DI INCERTEZZA p50 / p90
    # =====================================================
    st.divider()
    st.subheader("📐 Banda p50 / p90 e peso blend orario")

    fig_band = go.Figure()
    fig_band.add_trace(go.Scatter(
        x=preds["Datetime"], y=preds["PUN_p90"],
        mode="lines", name="p90", line=dict(color="rgba(255,0,0,0.3)")
    ))
    fig_band.add_trace(go.Scatter(
        x=preds["Datetime"], y=preds["PUN_p50"],
        mode="lines", name="p50", line=dict(color="rgba(0,0,255,0.4)")
    ))
    fig_band.add_trace(go.Scatter(
        x=preds["Datetime"], y=preds["pred"],
        mode="lines", name="Forecast (blend)", line=dict(color="black", width=2)
    ))
    st.plotly_chart(fig_band, use_container_width=True)

    fig_w = px.bar(preds, x="Datetime", y="w90", title="Peso w90 applicato per fascia oraria")
    st.plotly_chart(fig_w, use_container_width=True)

    # =====================================================
    # IMPORTANZA FEATURE (nativa LightGBM, sostituisce SHAP)
    # =====================================================
    st.divider()

    st.header("🔎 Importanza feature (gain LightGBM) — next 96")

    st.caption(
        "Nota: il nuovo modello PUN Direct 96 usa 96 modelli LightGBM "
        "indipendenti (uno per horizon). Al posto dello SHAP per-step "
        "del vecchio forecaster, qui viene mostrata l'importanza nativa "
        "(gain) media sui vari horizon: e' meno granulare ma non richiede "
        "ricalcolo pesante ad ogni forecast."
    )

    if importance_df is None or importance_df.empty:
        st.warning("⚠️ Nessuna importanza feature disponibile")
    else:
        top_n_explain = st.slider(
            "Numero feature da mostrare",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            key="top_n_importance_native",
        )

        summary = summarize_native_importance(importance_df, top_n=top_n_explain)

        fig_imp = px.bar(
            summary.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Top feature — importanza media (gain) su tutti gli horizon",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.dataframe(summary, use_container_width=True)

        with st.expander("📄 Tabella completa importanza per horizon"):
            st.dataframe(importance_df, use_container_width=True)

# =========================================================
# 📉 MODEL MONITORING (Concept Drift)
# =========================================================
st.divider()
st.header("📉 Model Monitoring (Concept Drift)")

FORECAST_PATH = "dati_output/forecast_history.parquet"
ERROR_PATH = "dati_output/error_history.parquet"

from io import BytesIO

# =========================================================
# 1. UPLOAD FILE PUN REALE
# =========================================================
today = date.today()
uploaded_file = st.file_uploader(f"📥 Carica file PUN rileavati a {today}", type=["xlsx"])
# =========================================================
# 2. PROCESSAMENTO
# =========================================================
if uploaded_file is not None:

    try:
        df_pun_excel = pd.read_excel(uploaded_file)

        df_real = pun_to_datetime(df_pun_excel)

        st.success("✅ File PUN caricato e trasformato")

        st.dataframe(df_real.head())

        # =====================================================
        # 3. CARICA FORECAST STORICO
        # =====================================================
        if os.path.exists(FORECAST_PATH):
          df_forecast = pd.read_parquet(FORECAST_PATH)
        else:
          st.warning("⚠️ Nessun forecast disponibile")
          st.stop()

        df_forecast["Datetime"] = pd.to_datetime(df_forecast["Datetime"])
        st.info(f"Forecast caricati: {len(df_forecast)} righe")
        # limite sui reali disponibili
        max_real_dt = df_real.index.max()
        # uso SOLO forecast "maturi"
        df_forecast_eval = df_forecast[df_forecast["Datetime"] <= max_real_dt].copy()

        # =================================================
        # 4. MERGE
        # =================================================
        df_eval = df_forecast_eval.merge(
                df_real,
                on="Datetime",
                how="inner"
            )

        if df_eval.empty:
          st.warning("⚠️ Nessun matching tra forecast e dati reali")

        else:
            # ==============================================
            # 5. ERRORI
            # ==============================================
            run_ts = pd.Timestamp.now()
            df_eval["error"] = df_eval["PUN"] - df_eval["pred"]
            df_eval["abs_error"] = df_eval["error"].abs()
            df_eval["created_at"] = run_ts

            # =====================================================
            # ✅ READ DA DROPBOX (SOURCE OF TRUTH)
            # =====================================================
            #try:
              #df_old = load_from_dropbox("/forecast_pun/error_history.parquet",st.secrets["DROPBOX_TOKEN"])
              #df_all = pd.concat([df_old, df_eval], ignore_index=True)

            #except Exception:
            # ✅ primo run (file non esiste ancora)
            df_all = df_eval.copy()
            # =====================================================
            # ✅ CLEAN + APPEND SICURO
            # =====================================================
            df_all = df_all.sort_values(["Datetime", "created_at"])

            # ✅ evita duplicati (stesso timestamp)
            df_all = df_all.drop_duplicates(subset=["Datetime"], keep="last")

            # =====================================================
            # ✅ SAVE LOCALE + DROPBOX
            # =====================================================
            df_all.to_parquet(ERROR_PATH)
            upload_to_dropbox(ERROR_PATH, "/forecast_pun/error_history.parquet", st.secrets["DROPBOX_TOKEN"])

            st.success("✅ Error history aggiornato")

            # ==============================================
            # 7. METRICHE
            # ==============================================
            df_all["mae_rolling"] = df_all["abs_error"].rolling(96).mean()
            df_all["rmse_rolling"] = (df_all["error"]**2).rolling(96).mean()**0.5
            # ==============================================
            # 8. UI
            # ==============================================
            st.subheader("📊 Metriche")
            c1, c2 = st.columns(2)
            c1.metric("MAE recente", round(df_all["mae_rolling"].iloc[-1], 2))
            c2.metric("RMSE recente", round(df_all["rmse_rolling"].iloc[-1], 2))
            st.subheader("📈 Trend errori")
            st.line_chart(
                df_all[["mae_rolling", "rmse_rolling"]].dropna()
            )
            st.subheader("📊 Errori ultimi punti")
            st.dataframe(df_all[["Datetime", "pred", "PUN", "error"]].tail(50))
            # ==============================================
            # 9. ALERT
            # ==============================================
            df_all['error_abs_perc'] = np.abs(df_all['pred']-df_all['PUN'])/df_all['PUN']
            df_all["hour"] = df_all["Datetime"].dt.hour
            fig = px.box(df_all, x="hour",y="error",    title="Errore per ora del giorno")
            st.plotly_chart(fig)
            fig = go.Figure()
            fig.add_trace(
              go.Scatter(
              x=df_all["Datetime"],
              y=df_all["PUN"],
              mode="lines",
              name="PUN reale",
              line=dict(color="blue")
              ))

            fig.add_trace(
              go.Scatter(
              x=df_all["Datetime"],
              y=df_all["pred"],
              mode="lines",
              name="Forecast",
              line=dict(color="red")))

            st.plotly_chart(fig, use_container_width=True)
            #
            mape_mean = df_all['error_abs_perc'].mean() * 100  # in %
            st.write(f"MAPE medio: {mape_mean:.2f}%")
            if mape_mean > 20:
              st.error("🚨 Concept drift forte → retraining urgente")
            elif mape_mean > 15:
              st.warning("⚠️ Performance degradata → monitoring stretto (retraining consigliato ma non urgente)")
            elif mape_mean > 10:
              st.info("🟡 Buone performance (range 10–15%) → modello stabile ma migliorabile")
            else:
              st.success("✅ Modello calibrato perfettamente (<10%)")
            # ==============================================
            # 10. DOWNLOAD
            # ==============================================
            st.download_button(
                label="⬇️ Scarica error history",
                data=df_all.to_csv(index=False),
                file_name="error_history.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Errore processamento PUN: {e}")
        st.code(traceback.format_exc(), language="python")
