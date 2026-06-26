import os
import traceback
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
import plotly.graph_objects as go
st.set_page_config(page_title="PUN Dataset Manager", layout="wide")
                   
from functions.create_datasets import (PUNFeatureEngineering, MeteoDownloader, TernaClient, ks_drift, upload_to_dropbox, load_from_dropbox)
from functions.forecast import (forecast_day_ahead_96_base, pun_to_datetime, plot_forecast_pun)
from functions.forecast_mi import forecast_next_96_all_mi_models_dropbox
from functions.create_datasets import EntsoeDownloader
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
import dropbox

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
# CORE PIPELINE
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
    # da cancellare
    
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
# LOAD MODEL FROM DROPBOX
# =========================================================

from pathlib import Path
import json
import joblib
import traceback
import dropbox

try:
    import pipeline  
except Exception as e:
    st.error(
        "❌ Impossibile importare pipeline.py. "
        "Devi copiare lo stesso pipeline.py usato su Hugging Face nella repo Streamlit."
    )
    st.code(traceback.format_exc(), language="python")
    st.stop()


MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DROPBOX_MODEL_DIR = "/forecast_pun/models"

MODEL_PATH = MODEL_DIR / "model_prod.pkl"
SELECTED_EXOG_PATH = MODEL_DIR / "selected_exog.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

DROPBOX_MODEL_PATH = f"{DROPBOX_MODEL_DIR}/model_prod.pkl"
DROPBOX_SELECTED_EXOG_PATH = f"{DROPBOX_MODEL_DIR}/selected_exog.pkl"
DROPBOX_METADATA_PATH = f"{DROPBOX_MODEL_DIR}/metadata.json"


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
    Carica modello production da Dropbox.

    File attesi:
    - model_prod.pkl
    - selected_exog.pkl
    - metadata.json

    Ritorna:
    {
        "model": ForecasterDirect,
        "selected_exog": list[str],
        "metadata": dict
    }
    """
    dbx = make_dbx_client()

    must_download = (
        force_download
        or not local_file_ok(MODEL_PATH)
        or not local_file_ok(SELECTED_EXOG_PATH)
        or not local_file_ok(METADATA_PATH)
    )

    if must_download:
        today_minus_1 = pd.Timestamp.today() - pd.Timedelta(days=1)
        st.info(
            f"📥 Download modello production da Dropbox aggiornato al "
            f"{today_minus_1.strftime('%d-%m-%Y')}..."
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_MODEL_PATH,
            local_path=MODEL_PATH,
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_SELECTED_EXOG_PATH,
            local_path=SELECTED_EXOG_PATH,
        )

        download_file_from_dropbox(
            dbx=dbx,
            dropbox_path=DROPBOX_METADATA_PATH,
            local_path=METADATA_PATH,
        )

    try:
        model = joblib.load(MODEL_PATH)
        selected_exog = joblib.load(SELECTED_EXOG_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "model": model,
            "selected_exog": selected_exog,
            "metadata": metadata,
        }

    except Exception as e:
        st.error(f"❌ Errore caricamento modello da file locale: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="python")
        raise


def compute_sample_weights(index: pd.DatetimeIndex) -> np.ndarray:
    """Peso per ogni timestamp in base alla fascia oraria (off-peak/midday/evening)."""
    qod = quarter_of_day(index)
    weights = np.full(len(index), OFFPEAK_WEIGHT_BASE, dtype=float)
    weights[np.isin(qod, list(MIDDAY_QOD))] = MIDDAY_WEIGHT
    weights[np.isin(qod, list(EVENING_QOD))] = EVENING_WEIGHT
    return weights


def weight_func(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Firma richiesta da skforecast: ForecasterDirect chiama questa funzione
    con l'indice temporale dei dati di training e usa il risultato come
    sample_weight per ciascuno dei modelli step-wise.
    """
    return compute_sample_weights(index)

force_model_download = st.sidebar.button("🔁 Forza download modello da Dropbox")

artifacts = load_model_artifacts_from_dropbox(
    force_download=force_model_download
)

model_base = artifacts["model"]
selected_exog = artifacts["selected_exog"]
model_metadata = artifacts["metadata"]

st.success("✅ Modello production caricato da Dropbox")

trained_at = model_metadata.get("trained_at", "n/d")
mode = model_metadata.get("mode", "n/d")
n_exog = model_metadata.get("n_selected_exog", len(selected_exog))

st.caption(f"📦 Model: model_prod.pkl | trained_at: {trained_at} | mode: {mode}")
st.caption(f"🧬 Exog caricate: {n_exog}")
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
    
run_forecast = st.button("📈 Esegui Forecast Day Ahead", use_container_width=True)

FORECAST_PATH = "dati_output/forecast_history.parquet"

if run_forecast:

    preds = forecast_day_ahead_96_base(
        df_hist=df_hist,
        best_forecaster=model_base,
        meteo_downloader=MeteoDownloader(),
        locations=LOCATIONS,
        selected_exog=selected_exog,
        steps=96
    )

    st.success("✅ Forecast completato")

    # ==========================================
    # ✅ FIX TIMEZONE SOLO PER DISPLAY
    # ==========================================
    preds_view = preds.copy()
    st.subheader("📊 Preview Forecast")
    st.dataframe(preds_view.tail(50), use_container_width=True)

    # ==========================================
    # SAVE FORECAST (parquet)
    # ==========================================
    
    preds_to_save = preds.copy()
    preds_to_save["created_at"] = pd.Timestamp.now()

    if os.path.exists(FORECAST_PATH):
        df_old = pd.read_parquet(FORECAST_PATH)
        df_all = pd.concat([df_old, preds_to_save], ignore_index=True)
    else:
        df_all = preds_to_save.copy()
        st.info("📦 Primo forecast salvato (inizializzazione storico)")

    df_all = df_all.drop_duplicates(subset=["Datetime"], keep="last")
    df_all = df_all.sort_values("Datetime")

    df_all.to_parquet(FORECAST_PATH)

    upload_to_dropbox(
      FORECAST_PATH,
      "/forecast_pun/forecast_history.parquet",
      st.secrets["DROPBOX_TOKEN"]
      )

    st.success("✅ Salvato su Dropbox")

    # log UI
    st.success(f"✅ Salvati {len(preds_to_save)} nuovi forecast")
    st.write(f"📊 Totale forecast storico: {len(df_all)}")
    # UI
    fig, stats = plot_forecast_pun(preds)
    st.subheader("📈 Forecast intraday PUN")
    st.dataframe(preds)
  
    c1, c2, c3 = st.columns(3)
    c1.metric("Min", stats["min"])
    c2.metric("Max", stats["max"])
    c3.metric("Mean", stats["mean"])
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="⬇️ Scarica forecast",
        data=preds.to_csv(index=False),
        file_name="forecast_day_ahead.csv",
        mime="text/csv"
    )

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
        st.write(df_eval)
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
            # ==============================================#
            # 6. SAVE ERROR HISTORY
            # ==============================================#
            # =====================================================
            # ✅ READ DA DROPBOX (SOURCE OF TRUTH)
            # =====================================================
            try:
              df_old = load_from_dropbox("/forecast_pun/error_history.parquet",st.secrets["DROPBOX_TOKEN"])
              df_all = pd.concat([df_old, df_eval], ignore_index=True)

            except Exception:
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
            import plotly.express as px
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



# =========================================================
# CONFIG
# =========================================================
import io
def dbx_path(*parts):
    return "/" + "/".join([str(p).strip("/") for p in parts])

def load_config_mi():
    with open("config/config_mi.yaml", "r") as f:
        return yaml.safe_load(f)

MI = load_config_mi()["mi"]
MI_DROPBOX_ROOT = MI["dropbox"]["root"]

MI_DATASETS_DIR = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["datasets_dir"])
MI_MODELS_DIR = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["models_dir"])
MI_RESULTS_JSON_PATH = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["results_json"])

MI_FORECASTS_DIR = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["forecasts_dir"])
MI_FORECAST_HISTORY_LONG = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["forecast_history_long"])
MI_FORECAST_HISTORY_WIDE = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["forecast_history_wide"])
MI_FORECAST_ERRORS_PATH = dbx_path(MI_DROPBOX_ROOT, MI["dropbox"]["forecast_errors"])

MI_STEPS = MI["forecast"]["steps"]
MI_FREQ = MI["forecast"]["freq"]
MI_LOOKBACK_DAYS = MI["forecast"]["lookback_days"]

colonne_analisi_mi = MI["mercati"]


# =========================================================
# HELPERS
# =========================================================

def make_market_key(nome: str):
    return (
        str(nome)
        .lower()
        .replace("(", "")
        .replace(")", "")
        .replace("  ", " ")
        .replace(" ", "_")
        .replace("__", "_")
    )


# =========================================================
# LOAD DATASETS
# =========================================================

@st.cache_data(show_spinner=False)
def load_mi_datasets_from_dropbox_cached(dropbox_token: str):

    dfs_mi = {}

    for nome in colonne_analisi_mi:

        file_name = f"MI_{nome}.parquet"
        dropbox_path = f"{MI_DATASETS_DIR}/{file_name}"

        try:
            df = load_from_dropbox(dropbox_path, dropbox_token).copy()

            # Datetime handling
            if "Datetime" in df.columns:
                df["Datetime"] = pd.to_datetime(df["Datetime"])
                df = df.set_index("Datetime")

            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]

            # Frequenza
            try:
                df = df.asfreq("15min")
            except:
                pass

            # Clean dati
            df = df.replace([np.inf, -np.inf], np.nan).ffill()

            key = make_market_key(nome)
            dfs_mi[key] = df

        except Exception as e:
            print(f"⚠️ Errore loading {dropbox_path}: {e}")

    return dfs_mi


# =========================================================
# LOAD JSON MODELS
# =========================================================

def load_mi_json_from_dropbox(dropbox_token: str):

    dbx = dropbox.Dropbox(dropbox_token)

    try:
        _, res = dbx.files_download(MI_RESULTS_JSON_PATH)
        return json.loads(res.content.decode("utf-8"))

    except Exception as e:
        raise RuntimeError(f"Errore lettura JSON MI: {e}")


# =========================================================
# CHECK JSON VS DATASETS
# =========================================================

def check_mi_json_vs_dfs_keys(dropbox_token, dfs_mi):

    results = load_mi_json_from_dropbox(dropbox_token)

    json_keys = set(results.keys())
    dfs_keys = set(dfs_mi.keys())

    return {
        "common": sorted(json_keys & dfs_keys),
        "json_not_in_dfs": sorted(json_keys - dfs_keys),
        "dfs_not_in_json": sorted(dfs_keys - json_keys)
    }


# =========================================================
# PLOT FORECAST
# =========================================================

def plot_mi(df_long, nome_df, target):

    tmp = df_long[
        (df_long["nome_df"] == nome_df) &
        (df_long["target"] == target)
    ].copy()

    if tmp.empty:
        return None

    tmp["Datetime"] = pd.to_datetime(tmp["Datetime"])
    tmp = tmp.sort_values("Datetime")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tmp["Datetime"],
        y=tmp["pred"],
        mode="lines+markers",
        name=f"{nome_df} / {target}"
    ))

    fig.update_layout(
        title=f"{nome_df} - {target}",
        template="plotly_white"
    )

    return fig


# =========================================================
# DEBUG DROPBOX
# =========================================================

def debug_mi_dropbox(dropbox_token: str):

    st.subheader("🧪 Debug MI Dropbox")

    dbx = dropbox.Dropbox(dropbox_token)

    report = {
        "datasets_ok": [],
        "datasets_missing": [],
        "models_ok": [],
        "models_missing": [],
        "payload_errors": [],
    }

    # DATASETS
    for nome in colonne_analisi_mi:
        file_name = f"MI_{nome}.parquet"
        path = f"{MI_DATASETS_DIR}/{file_name}"

        try:
            dbx.files_get_metadata(path)
            report["datasets_ok"].append(file_name)
        except:
            report["datasets_missing"].append(file_name)

    # JSON
    try:
        results = load_mi_json_from_dropbox(dropbox_token)
        st.success("✅ JSON OK")
    except Exception as e:
        st.error(f"❌ JSON ERROR: {e}")
        return

    # MODELS
    for nome_df, targets in results.items():
        for target, res in targets.items():

            if res.get("status") != "ok":
                continue

            model_path = res.get("model_path")

            if not model_path:
                report["models_missing"].append(f"{nome_df}/{target}")
                continue


            model_path = str(model_path).replace("\\", "/")
            file_name = os.path.basename(model_path)
            model_path = f"{MI_MODELS_DIR}/{file_name}"

            #
            try:
                res = dbx.files_list_folder(MI_MODELS_DIR)

                for entry in res.entries:
                    if entry.name == file_name:
                        real_path = entry.path_display

                        _, file_res = dbx.files_download(real_path)
                        report["models_ok"].append(real_path)

                        payload = joblib.load(io.BytesIO(file_res.content))

                        for key in ["forecaster", "selected_exog", "target"]:
                            if key not in payload:
                                report["payload_errors"].append(
                                f"{real_path} missing '{key}'")

                        break
                    else:
                        report["models_missing"].append(file_name)

            except Exception as e:
                report["models_missing"].append(f"{file_name} -> {e}")

    # OUTPUT
    st.write("✅ Dataset OK:", len(report["datasets_ok"]))
    st.write("❌ Dataset missing:", report["datasets_missing"])
    st.write("✅ Models OK:", len(report["models_ok"]))
    st.write("❌ Models missing:", report["models_missing"])

    if report["payload_errors"]:
        st.warning("⚠️ Payload errors:")
        st.write(report["payload_errors"])

    if not report["datasets_missing"] and not report["models_missing"]:
        st.success("✅ Sistema MI pronto")
    else:
        st.error("❌ Problemi trovati")

    return report

def pipeline_run_mi():

    log_lines = []

    def log(msg):
        log_lines.append(msg)

    DROPBOX_TOKEN = st.secrets["DROPBOX_TOKEN"]

    # =========================
    # LOAD MI HISTORICAL
    # =========================
    dfs_mi = load_mi_datasets_from_dropbox_cached(DROPBOX_TOKEN)

    if not dfs_mi:
        raise ValueError("❌ Nessun dataset MI")

    # =========================
    # LOAD EXCEL UNICO
    # =========================
    df_excel = pd.read_excel("dati_input/MI.xlsx")

    df_excel["Data"] = pd.to_datetime(df_excel["Data"], dayfirst=True)
    df_excel["Datetime"] = (
        df_excel["Data"]
        + pd.to_timedelta((df_excel["Periodo"] - 1) * 15, unit="m")
    )
    df_excel = df_excel.sort_values("Datetime")

    today = pd.Timestamp.today()

    updated_dfs = {}

  
    MARKET_TO_EXCEL = {
      "italia_senza_vincoli": "Italia (senza vincoli)",
      "calabria": "Calabria",
      "centro_nord": "Centro Nord",
      "centro_sud": "Centro Sud",
      "nord": "Nord",
      "sardegna": "Sardegna",
      "sicilia": "Sicilia",
      "sud": "Sud",
      "italia_coupling": "Italia  Coupling"  # ⚠️ doppio spazio!
    }

    # =========================
    # LOOP MERCATI
    # =========================
    for nome, df_historical in dfs_mi.items():

        log(f"--- {nome} ---")

        df_historical = df_historical.copy()
        df_historical.index = pd.to_datetime(df_historical.index)

        # ✅ timezone safe
        if df_historical.index.tz is not None:
            df_historical.index = (
                df_historical.index
                .tz_convert("Europe/Rome")
                .tz_localize(None)
            )

        df_historical = df_historical.sort_index().asfreq("15min").ffill()

        last_date = df_historical.index.max()

        # =========================
        # LOOKBACK
        # =========================
        
        lookback_start_dt = last_date.floor("D") - pd.Timedelta(days=MI_LOOKBACK_DAYS)
        col_name = MARKET_TO_EXCEL.get(nome)

        if col_name is None or col_name not in df_excel.columns:
          log(f"❌ Colonna NON trovata per {nome} → {col_name}")
          continue

        log(f"✅ {nome} → match Excel: {col_name}")

        # =========================
        # TARGET (Y)
        # =========================
        df_new_y = df_excel[["Datetime", col_name]].copy()

        df_new_y = df_new_y.rename(columns={col_name: "target"})

        df_new_y["target"] = (
            df_new_y["target"]
            .astype(str)
            .str.replace(",", ".")
            .astype(float)
        )

        # =========================
        # CONCAT STORICO + NUOVI
        # =========================
        df_hist = df_historical.reset_index()["Datetime"]

        df_full = pd.concat([df_hist, df_new_y])
        df_full = df_full.drop_duplicates("Datetime", keep="last")
        df_full = df_full.sort_values("Datetime")

        # =========================
        # FEATURE ENGINEERING
        # =========================
        df_full["lag_2d"] = df_full["target"].shift(96 * 2)
        df_full["lag_7d"] = df_full["target"].shift(96 * 7)

        df_full["ret_1d"] = df_full["target"].pct_change(96).shift(1)
        df_full["ret_7d"] = df_full["target"].pct_change(96 * 7).shift(1)

        df_full["momentum_1d"] = df_full["target"].shift(1) - df_full["target"].shift(96)
        df_full["momentum_4h"] = df_full["target"].shift(1) - df_full["target"].shift(16)

        df_full["ret_1h"] = df_full["target"].pct_change(4).shift(1)
        df_full["minute"] = pd.to_datetime(df_full["Datetime"]).dt.minute

        # =========================
        # BASE LOOKBACK
        # =========================
        df_base = df_full[df_full["Datetime"] >= lookback_start_dt]

        # =========================
        # METEO
        # =========================
        meteo_df = prepare_meteo(
            MeteoDownloader(),
            lookback_start_dt.strftime("%Y-%m-%d"),
            today.strftime("%Y-%m-%d")
        )

        # =========================
        # TERNA
        # =========================
        terna_df = prepare_terna(
            TernaClient(
                st.secrets["TERNA_CLIENT_ID"],
                st.secrets["TERNA_CLIENT_SECRET"]
            ),
            lookback_start_dt.strftime("%d/%m/%Y"),
            today.strftime("%d/%m/%Y")
        )

        terna_df = shift_terna_only(terna_df)

        # =========================
        # ENTSOE ✅ CORRETTO
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
            start_date=lookback_start_dt.to_pydatetime(),
            end_date=today.to_pydatetime()
        )

        entsoe_feat = entsoe.build_features()

        # ✅ timezone safe
        if entsoe_feat.index.tz is not None:
            entsoe_feat.index = (
                entsoe_feat.index
                .tz_convert("Europe/Rome")
                .tz_localize(None)
            )

        # =========================
        # MERGE ✅ INDEX BASED
        # =========================
        df_new = df_base.merge(meteo_df, on="Datetime", how="left")
        df_new = df_new.merge(terna_df, on="Datetime", how="left")

        df_new["Datetime"] = pd.to_datetime(df_new["Datetime"])
        df_new = df_new.set_index("Datetime")

        # 🔥 join ENTSOE
        df_new = df_new.join(entsoe_feat, how="left")

        # =========================
        # FEATURES FINALI
        # =========================
        df_new = df_new.reset_index()
        df_new = add_features(df_new)

        # =========================
        # SOLO NUOVE
        # =========================
        df_new = df_new[df_new["Datetime"] > last_date] 
        if df_new.empty:
          log(f"{nome} → NESSUN UPDATE ❌ (df_new vuoto)")
          continue

        # =========================
        # FINAL
        # =========================
        df_final = pd.concat([
            df_historical.reset_index(),
            df_new
        ])

        df_final = df_final.drop_duplicates("Datetime").sort_values("Datetime")
        df_final = df_final.set_index("Datetime")

        # =========================
        # SAVE
        # =========================
        filename = f"MI_{nome}.parquet"
        local_path = f"/tmp/{filename}"

        df_final.to_parquet(local_path)

        upload_to_dropbox(
            local_path,
            f"{MI_DATASETS_DIR}/{filename}",
            DROPBOX_TOKEN
        )

        log(f"✅ {nome} aggiornato: {df_final.shape}")
        updated_dfs[nome] = df_final

    return updated_dfs, log_lines

# ============================⚡ MI PIPELINE ============================ #

st.divider()
st.header("⚡ MI Forecast + Monitoring")

DROPBOX_TOKEN = st.secrets.get("DROPBOX_TOKEN", "")

if not DROPBOX_TOKEN:
    st.error("❌ DROPBOX_TOKEN mancante")
    st.stop()


# =========================================================
# LOAD DATASETS
# =========================================================
try:
    if "dfs_mi" not in st.session_state:
        dfs_mi = load_mi_datasets_from_dropbox_cached(DROPBOX_TOKEN)
        st.session_state["dfs_mi"] = dfs_mi
    else:
        dfs_mi = st.session_state["dfs_mi"]

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset MI", len(dfs_mi))

    if dfs_mi:
        last_dt = max(v.index.max() for v in dfs_mi.values())
        col2.metric("Ultima data globale", str(last_dt))
        col3.metric("Mercati", len(dfs_mi))
    else:
        col2.warning("Nessun dataset")

    # =========================================================
    # ✅ STATO PER MERCATO (QUESTO TI SERVE 🔥)
    # =========================================================
    st.subheader("📊 Stato aggiornamento per mercato")

    status = []

    for nome, df in dfs_mi.items():
        status.append({
            "market": nome,
            "last_date": df.index.max(),
            "rows": len(df)
        })

    df_status = pd.DataFrame(status).sort_values("last_date", ascending=False)

    st.dataframe(df_status, use_container_width=True)

except Exception:
    st.error("Errore loading MI")
    st.code(traceback.format_exc())
    st.stop()


# =========================================================
# PREVIEW
# =========================================================
with st.expander("📚 Preview dataset"):
    mkt = st.selectbox("Mercato", list(dfs_mi.keys()), key="preview_mi")

    df_sel = dfs_mi[mkt]
    st.dataframe(df_sel.tail(50), use_container_width=True)


# =========================================================
# ✅ INIT STATE
# =========================================================
if "forecast_done" not in st.session_state:
    st.session_state["forecast_done"] = False


# =========================================================
# ✅ BUTTONS
# =========================================================
col1, col2 = st.columns(2)

run_update = col1.button("🧱 Update DB + KS Drift", use_container_width=True)
run_forecast = col2.button("📈 Forecast + Monitoring", use_container_width=True)


WINDOW = 96 * 7


# =========================================================
# ✅ TRIGGER FORECAST
# =========================================================
if run_forecast:
    st.session_state["forecast_done"] = True


# =========================================================
# ✅ UPDATE DB + KS + PREVIEW
# =========================================================
if run_update:

    try:
        st.subheader("🧱 Update dataset")

        dfs_old = dfs_mi.copy()

        with st.spinner("Aggiornamento DB MI..."):
            dfs_new, logs = pipeline_run_mi()

        st.cache_data.clear()

        st.session_state["dfs_mi"] = dfs_new

        st.success("✅ DB aggiornato")
        st.code("\n".join(logs))

        # ✅ SUMMARY
        st.subheader("📊 Update summary")

        for nome in dfs_new:
            old_len = len(dfs_old.get(nome, []))
            new_len = len(dfs_new[nome])
            st.write(f"{nome} → +{new_len - old_len} righe")

        # ✅ KS DRIFT
        st.subheader("📊 KS Drift")

        drift_results = []

        for nome in dfs_new:

            if nome not in dfs_old:
                continue

            df_old = dfs_old[nome]
            df_new = dfs_new[nome]

            if len(df_old) < WINDOW or len(df_new) < WINDOW:
                continue

            cols = [c for c in df_new.columns if c != "target"]

            drift_df = ks_drift(
                df_old.tail(WINDOW),
                df_new.tail(WINDOW),
                cols
            )

            if not drift_df.empty:
                drift_df["market"] = nome
                drift_results.append(drift_df)

        if drift_results:
            drift_all = pd.concat(drift_results)
            st.dataframe(drift_all)

            n_drift = drift_all["drift_flag"].sum()

            if n_drift >= 10:
                st.error("🚨 Drift forte → retrain!")
            elif n_drift > 0:
                st.warning("⚠️ Drift moderato")
            else:
                st.success("✅ Nessun drift")

        else:
            st.warning("⚠️ Drift non calcolabile")

        # ✅ PREVIEW DATASET
        st.subheader("📦 Preview dataset aggiornato")

        dfs_mi = st.session_state["dfs_mi"]

        mkt_preview = st.selectbox(
            "Mercato",
            list(dfs_mi.keys()),
            key="preview_updated_mi"
        )

        df_preview = dfs_mi[mkt_preview]

        st.write("📅 Ultima data:", df_preview.index.max())
        st.write("📦 Shape:", df_preview.shape)

        st.dataframe(df_preview.tail(50), use_container_width=True)

    except Exception:
        st.error("❌ Errore update + KS")
        st.code(traceback.format_exc())


# =========================================================
# ✅ FORECAST + MONITORING
# =========================================================
if st.session_state["forecast_done"]:
  try:
        st.subheader("📈 Forecast")

        # ✅ CALCOLA UNA VOLTA SOLA
        if "forecast_long" not in st.session_state:

            dfs_mi = st.session_state["dfs_mi"]

            try:
                terna = TernaClient(
                    st.secrets["TERNA_CLIENT_ID"],
                    st.secrets["TERNA_CLIENT_SECRET"]
                )
            except Exception as e:
                print(f"⚠️ Terna disabilitato: {e}")
                terna = None

            meteo = MeteoDownloader()

            df_long, _, _ = forecast_next_96_all_mi_models_dropbox(
                dfs=dfs_mi,
                dropbox_token=DROPBOX_TOKEN,
                dropbox_results_json_path=MI_RESULTS_JSON_PATH,
                dropbox_models_dir=MI_MODELS_DIR,
                dropbox_forecasts_dir=MI_FORECASTS_DIR,
                forecast_history_long_path=MI_FORECAST_HISTORY_LONG,
                forecast_history_wide_path=MI_FORECAST_HISTORY_WIDE,
                errors_path=MI_FORECAST_ERRORS_PATH,
                meteo=meteo,
                locations=LOCATIONS,
                terna=terna,
                steps=MI_STEPS,
                freq=MI_FREQ,
                lookback_days=MI_LOOKBACK_DAYS
            )

            st.session_state["forecast_long"] = df_long

        df_long = st.session_state["forecast_long"]

        if df_long.empty:
            st.warning("⚠️ Forecast vuoto")
            st.stop()

        st.success("✅ Forecast disponibile")

        # =================================================
        # ✅ MONITORING
        # =================================================
        st.subheader("📉 Monitoring")

        uploaded_file = st.file_uploader(
            "Carica Excel MI reali",
            type=["xlsx"],
            key="monitoring_file"
        )

        if uploaded_file is None:
            st.info("⬆️ Carica Excel per vedere il monitoring")
            st.stop()

        df_real = pd.read_excel(uploaded_file)

        df_real["Data"] = pd.to_datetime(df_real["Data"], dayfirst=True)
        df_real["Datetime"] = (
            df_real["Data"]
            + pd.to_timedelta((df_real["Periodo"] - 1) * 15, unit="m")
        )

        df_forecast = df_long.copy()
        df_forecast["Datetime"] = pd.to_datetime(df_forecast["Datetime"])

        all_eval = []

        for col in df_real.columns:
            if col in ["Data", "Ora", "Periodo", "Datetime", "Italia"]:
                continue

            nome_df = make_market_key(col)
            df_pred = df_forecast[df_forecast["nome_df"] == nome_df]
            df_real_sel = df_real[['Datetime', col]]
            df_pred_sel = df_pred[['Datetime','pred']]
            if df_pred_sel.empty:
                continue
            df_tmp = df_pred_sel.merge(
                df_real_sel,
                on="Datetime",
                how="inner"
            )
            if df_tmp.empty:
                continue
            
            df_tmp["real"] = pd.to_numeric(df_tmp[col], errors="coerce")
            df_tmp["pred"] = pd.to_numeric(df_tmp["pred"], errors="coerce")
            df_tmp["nome_df"] = nome_df
            df_tmp["abs_error"] = (df_tmp["real"] - df_tmp["pred"]).abs()
            df_tmp["error_abs_perc"] = df_tmp["abs_error"] / (df_tmp["real"] + 1e-6)

            all_eval.append(df_tmp)

        if not all_eval:
            st.warning("⚠️ Nessun match forecast vs real")
            st.stop()

        df_eval = pd.concat(all_eval)
        # =========================================================
        # ✅ SELECT MODEL
        # =========================================================
        st.subheader("🎯 Selezione modello")

        models = sorted(df_eval["nome_df"].unique())

        selected_model = st.selectbox(
            "Seleziona mercato",
            models
        )

        df_model = df_eval[df_eval["nome_df"] == selected_model]

        if df_model.empty:
            st.warning("⚠️ Nessun dato per il modello selezionato")
            st.stop()

        # =========================================================
        # ✅ METRICHE (SOLO MODELLO)
        # =========================================================
        df_model = df_model.sort_values("Datetime").copy()

        df_model["mae"] = df_model["abs_error"].rolling(96).mean()
        df_model["rmse"] = (df_model["abs_error"] ** 2).rolling(96).mean() ** 0.5

        c1, c2 = st.columns(2)
        c1.metric("MAE", round(df_model["mae"].iloc[-1], 2))
        c2.metric("RMSE", round(df_model["rmse"].iloc[-1], 2))

        st.line_chart(df_model.set_index("Datetime")[["mae", "rmse"]].dropna())

        # =========================================================
        # ✅ REAL vs PRED
        # =========================================================
        st.subheader("📊 Reale vs Predetto")

        plot_df = df_model.set_index("Datetime")[["real", "pred"]].tail(300)
        st.line_chart(plot_df)

        # =========================================================
        # ✅ ERRORE PER ORA
        # =========================================================
        st.subheader("⏱️ Errore per ora")

        err_hour = df_model.groupby(df_model["Datetime"].dt.hour)["abs_error"].mean()
        st.line_chart(err_hour)

        # =========================================================
        # ✅ DRIFT
        # =========================================================
        mape = df_model["error_abs_perc"].mean() * 100

        if mape > 25:
            st.error(f"🚨 Drift forte ({round(mape,2)}%)")
        elif mape > 18:
            st.warning(f"⚠️ Drift moderato ({round(mape,2)}%)")
        else:
            st.success(f"✅ Modello stabile ({round(mape,2)}%)")
  except Exception:
        st.error("❌ Errore forecast")
        import traceback
        st.code(traceback.format_exc())
