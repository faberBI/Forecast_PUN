import os
import traceback
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
import plotly.graph_objects as go
st.set_page_config(page_title="PUN Dataset Manager", layout="wide")
                   
from functions.create_datasets import (PUNFeatureEngineering, MeteoDownloader, TernaClient, ks_drift, df_to_supabase_records, load_dataset_history_from_supabase)
from functions.forecast import (forecast_day_ahead_96_base, pun_to_datetime, plot_forecast_pun)

import yaml

CONFIG_PATH = "config/config.yaml"

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

config = load_config()

FEATURES_OLD = config["features"]["FEATURES_OLD"]
FEATURES_NEW = config["features"]["FEATURES_NEW"]
SELECTED_EXOG = config["features"]["SELECTED_EXOG"]


from supabase import create_client

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

supabase = create_client(url, key)


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
    return aggregate_meteo(df)


def prepare_pun(pun_fe, pun_path: str) -> pd.DataFrame:
    raw = pd.read_excel(pun_path)
    df = pun_fe.prepare_dataset(raw, merge_commodities=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.floor("h")
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

    # LOGICA ORIGINALE: niente get_forecast_load separato
    df = load.merge(market, on="date", how="outer")
    df = terna.safe_merge(df, wind, "wind")
    df = terna.safe_merge(df, solar, "solar")
    df = terna.safe_merge(df, hydro, "hydro")

    df = terna.clean_terna_features(df)

    # coercizione robusta ai numerici
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
    return df


def merge_all(pun_df: pd.DataFrame, meteo_df: pd.DataFrame, terna_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge_asof(
        pun_df.sort_values("Datetime"),
        meteo_df.sort_values("Datetime"),
        on="Datetime",
        direction="backward",
        tolerance=pd.Timedelta("90min"),
    )

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
    df_historical = load_historical(HISTORICAL_PATH).copy()
    last_date = df_historical.index.max()

    # =========================================================
    # DATE SETUP
    # =========================================================
    end_date_dt = pd.Timestamp(today)
    start_date_dt = last_date

    START_DATE_METEO = start_date_dt.strftime("%Y-%m-%d")
    END_DATE_METEO = end_date_dt.strftime("%Y-%m-%d")

    START_DATE_TERNA = start_date_dt.strftime("%d/%m/%Y")
    END_DATE_TERNA = end_date_dt.strftime("%d/%m/%Y")

    START_DATE_PUN = start_date_dt.strftime("%Y-%m-%d")

    log(f"Ultima data storico: {last_date}")
    log(f"Download Meteo da {START_DATE_METEO} a {END_DATE_METEO}")
    log(f"Download Terna da {START_DATE_TERNA} a {END_DATE_TERNA}")
    log(f"Start PUN: {START_DATE_PUN}")

    # =========================================================
    # PIPELINE
    # =========================================================
    pun_fe = PUNFeatureEngineering(start=START_DATE_PUN, pun_col="PUN")
    meteo = MeteoDownloader()

    log("Preparazione PUN...")
    pun_df = prepare_pun(pun_fe, PUN_INPUT_PATH)

    log("Preparazione Meteo...")
    meteo_df = prepare_meteo(meteo, START_DATE_METEO, END_DATE_METEO)

    # meglio usare secrets o env in produzione
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

    log("Merge dataset...")
    df = merge_all(pun_df, meteo_df, terna_df)

    log("Feature engineering...")
    df = add_features(df)

    log("Quarter-hour expansion...")
    df = make_quarter_hour(df)

    df.drop(columns=FEATURES_DROP, errors="ignore", inplace=True)
    df = df.reset_index(drop=True)

    missing = [c for c in FEATURES_NEW if c not in df.columns]
    extra = [c for c in df.columns if c not in FEATURES_NEW]

    if missing:
        st.error(f"❌ Feature mancanti: {missing}")
        raise ValueError("Schema mismatch -> stop pipeline")

    if extra:
        st.warning(f"⚠️ Feature extra ignorate: {extra}")

    # =========================================================
    # FEATURE ALIGNMENT - FEDELE AL CODICE ORIGINALE
    # =========================================================
    validate_required_columns(df_historical.reset_index(), ["Datetime"] + FEATURES_OLD, "df_historical.reset_index()")
    validate_required_columns(df, FEATURES_NEW, "df new pipeline")

    df_old = df_historical[FEATURES_OLD].copy()
    df_new = df[FEATURES_NEW].copy()

    # elimina da df_old tutto ciò che è già presente in df_new per Datetime
    df_old = df_old[~df_old.index.isin(df_new["Datetime"])]  
    df_old.reset_index(inplace=True)

    df_final = pd.concat([df_old, df_new],axis = 0, ignore_index=True)
    df_final = df_final.sort_values("Datetime").reset_index(drop=True)
    df_final.set_index("Datetime", inplace=True)

    log(f"Shape finale: {df_final.shape}")
    log(f"Ultima data finale: {df_final.index.max()}")

    # =========================================================
    # =========================================================
    df_final.to_parquet(OUTPUT_PATH)
    log(f"Salvato file: {OUTPUT_PATH}")
    # ✅ SAVE SU SUPABASE (VERSIONE CORRETTA)

    df_to_db = df_final.reset_index().copy()
    st.dataframe(df_final.tail(50))
    st.dataframe(df_to_db.tail(50))
    records = df_to_supabase_records(df_to_db)
    supabase.table("dataset_history").upsert(records).execute()
    log("✅ Dataset salvato su Supabase")
    return df_final, log_lines



# =========================================================
# UI - STATUS
# =========================================================
col1, col2 = st.columns(2)

try:
    df_historical = load_historical(HISTORICAL_PATH)
    last_hist_date = df_historical.index.max()
    with col1:
        st.metric("📅 Ultima data DB storico", str(last_hist_date))
except Exception as e:
    with col1:
        st.error(f"Errore lettura storico: {e}")
    df_historical = None

try:
    df_output = load_output_if_exists(OUTPUT_PATH)
    if df_output is not None:
        with col2:
            st.metric("🆕 Ultima data DB aggiornato", str(df_output.index.max()))
    else:
        with col2:
            st.metric("🆕 Ultima data DB aggiornato", "file non presente")
except Exception as e:
    with col2:
        st.error(f"Errore lettura output: {e}")


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


st.subheader("📦 Preview DB aggiornato (Supabase)")

response = supabase.table("dataset_history") \
    .select("*") \
    .order("Datetime", desc=True) \
    .limit(50) \
    .execute()

df_output = pd.DataFrame(response.data)

if not df_output.empty:
    # conversione datetime per sicurezza
    df_output["Datetime"] = pd.to_datetime(df_output["Datetime"])

    st.dataframe(df_output, use_container_width=True)
else:
    st.warning("⚠️ Nessun dato presente sul db SQL")


from pathlib import Path
import gdown

# ===== costanti (stesse del training!)
OFFPEAK_WEIGHT_BASE = 1.0
PEAK_WEIGHT_BASE = 1.5
MIDDAY_WEIGHT = 2.5
EVENING_WEIGHT = 3.0

MIDDAY_QOD = range(48, 68)
EVENING_QOD = range(76, 87)


def peak_weight_func(index):
    index = pd.DatetimeIndex(index)
    qod = index.hour * 4 + (index.minute // 15)

    weights = np.full(len(index), OFFPEAK_WEIGHT_BASE, dtype=float)

    weights[np.isin(qod, list(MIDDAY_QOD))] = MIDDAY_WEIGHT
    weights[np.isin(qod, list(EVENING_QOD))] = EVENING_WEIGHT

    daytime_mask = (index.hour >= 8) & (index.hour < 21)
    weights[(weights == OFFPEAK_WEIGHT_BASE) & daytime_mask] = PEAK_WEIGHT_BASE

    return weights



MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "pun_full_pipeline.pkl"
FILE_ID = st.secrets.get("MODEL_PRODUCTION", os.getenv("MODEL_PRODUCTION", "")) 

import traceback
import joblib
import gdown
import pandas as pd
import streamlit as st

@st.cache_resource
def load_model_bundle():

    if not FILE_ID:
        st.error("❌ MODEL_PRODUCTION non configurato nei secrets")
        raise ValueError("Missing FILE_ID")

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        today_minus_1 = pd.Timestamp.today() - pd.Timedelta(days=1)
        st.info(
            f"📥 Download modello aggiornato al {today_minus_1.strftime('%d-%m-%Y')}..."
        )

        gdown.download(
            f"https://drive.google.com/uc?export=download&id={FILE_ID}",
            str(MODEL_PATH),
            quiet=False,
        )

    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle

    except Exception as e:
        st.error(f"❌ Errore caricamento modello: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="python")
        raise


# ✅ USO
bundle = load_model_bundle()
model_base = bundle["base_model"]

st.write(f'Modello aggiornato caricato ✅')

selected_exog = model_base.exog_names_in_

st.write(f'Varibaili esogene aggiornate✅')




df_hist = load_dataset_history_from_supabase()

if df_hist.empty:
    st.error("❌ Dataset vuoto in Supabase → prima esegui aggiornamento")
    st.stop()

df_hist["Datetime"] = pd.to_datetime(df_hist["Datetime"])
df_hist = df_hist.set_index("Datetime").sort_index()
df_hist = df_hist.asfreq("15min").ffill()

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
    # SAVE FORECAST (gestione file non esistente)
    # ==========================================
    preds_to_save = preds.copy()
    preds_to_save["created_at"] = pd.Timestamp.now()

    if os.path.exists(FORECAST_PATH):
        df_old = pd.read_parquet(FORECAST_PATH)
        df_all = pd.concat([df_old, preds_to_save])
        df_all = df_all.drop_duplicates(subset=["Datetime"])
    else:
        df_all = preds_to_save
        st.info("📦 Primo forecast salvato (inizializzazione storico)")

    df_all = df_all.sort_values("Datetime")

    df_all.to_parquet(FORECAST_PATH)
    preds_to_db = preds_to_save.copy()
    records = df_to_supabase_records(preds_to_db)
    supabase.table("forecast_history").upsert(records).execute()



    st.write(f"✅ Forecast salvati: {len(preds_to_save)}")

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
        response = supabase.table("forecast_history").select("*").execute()
        df_forecast = pd.DataFrame(response.data)
        if df_forecast.empty:
          st.warning("⚠️ Nessun forecast presente in Supabase")
          st.stop()
        
        df_forecast["Datetime"] = pd.to_datetime(df_forecast["Datetime"])
        st.info(f"Forecast caricati: {len(df_forecast)} righe")
        # =================================================
        # 4. MERGE
        # =================================================
        df_eval = df_forecast.merge(
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
            df_eval["error"] = df_eval["PUN"] - df_eval["pred"]
            df_eval["abs_error"] = df_eval["error"].abs()
            # ==============================================
            # 6. SAVE ERROR HISTORY
            # ==============================================
            if os.path.exists(ERROR_PATH):
                df_old = pd.read_parquet(ERROR_PATH)
                df_all = pd.concat([df_old, df_eval])
                df_all = df_all.drop_duplicates(subset=["Datetime"])
            else:
                df_all = df_eval
            df_all = df_all.sort_values("Datetime")
            df_all.to_parquet(ERROR_PATH)
            if "Datetime" not in df_all.columns:
              df_err_db = df_all.reset_index()
            else:
              df_err_db = df_all.copy()
            records = df_to_supabase_records(df_err_db)
            supabase.table("error_history").upsert(records).execute()

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
            if df_all["mae_rolling"].iloc[-1] > 10:
                st.error("🚨 Concept drift rilevato → retraining consigliato")
            else:
                st.success("✅ Modello stabile")
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
