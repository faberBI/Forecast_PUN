import traceback
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

from functions.create_datasets import (
    PUNFeatureEngineering, MeteoDownloader, TernaClient,
    ks_drift, upload_to_dropbox, load_from_dropbox, EntsoeDownloader
)

st.set_page_config(page_title="PUN Dataset Manager", layout="wide")

st.title("⚡ PUN Dataset Manager")

OUTPUT_PATH = "dati_output/final_dataset_intra_day.parquet"
PUN_INPUT_PATH = "dati_input/Add_on_PUN.xlsx"

LOCATION_SAMPLE = {"name": "roma", "lat": 41.9, "lon": 12.5}


# =========================================================
# PIPELINE
# =========================================================
def pipeline_run():

    log_lines = []

    def log(msg):
        log_lines.append(msg)

    log("🚀 START PIPELINE")

    # =========================
    # LOAD DB
    # =========================
    df_historical = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    ).copy()

    df_historical["Datetime"] = pd.to_datetime(df_historical.index)
    df_historical = df_historical.sort_index().asfreq("15min").ffill()

    last_date = df_historical.index.max()

    log(f"Last storico: {last_date}")

    # =========================
    # PUN
    # =========================
    pun_fe = PUNFeatureEngineering(start=str(last_date.date()), pun_col="PUN")

    pun_new = pd.read_excel(PUN_INPUT_PATH)
    pun_new = pun_fe.prepare_dataset(pun_new, merge_commodities=True)

    pun_new["Datetime"] = pd.to_datetime(pun_new["Datetime"])

    log(f"PUN max: {pun_new['Datetime'].max()}")

    pun_full = pd.concat([
        df_historical.reset_index()[["Datetime", "PUN"]],
        pun_new[["Datetime", "PUN"]]
    ])

    pun_full = (
        pun_full
        .drop_duplicates("Datetime", keep="last")
        .sort_values("Datetime")
    )

    # =========================
    # METEO
    # =========================
    meteo = MeteoDownloader()

    meteo_df = pd.DataFrame({
        "Datetime": pun_full["Datetime"]
    })

    # =========================
    # TERNA
    # =========================
    terna = TernaClient(
        client_id=st.secrets["TERNA_CLIENT_ID"],
        client_secret=st.secrets["TERNA_CLIENT_SECRET"]
    )

    terna_df = pd.DataFrame({
        "Datetime": pun_full["Datetime"]
    })

    # =========================
    # ENTSOE
    # =========================
    log("ENTSOE...")

    entsoe = EntsoeDownloader(
        token=st.secrets["ENTSOE_TOKEN"],
        zones=[("NORD", "10Y1001A1001A73I")],
        start_date=pun_full["Datetime"].min().to_pydatetime(),
        end_date=date.today()
    )

    entsoe_feat = entsoe.build_features()

    log(f"ENTSOE max: {entsoe_feat.index.max()}")

    # =========================
    # MERGE
    # =========================
    df_new = pun_full.copy()

    df_new = df_new.set_index("Datetime")
    df_new = df_new.join(entsoe_feat, how="left")
    df_new = df_new.reset_index()

    # =========================
    # SOLO NUOVE
    # =========================
    df_new["Datetime"] = pd.to_datetime(df_new["Datetime"])

    df_new = df_new[df_new["Datetime"] > last_date]

    log(f"Nuove righe: {len(df_new)}")

    # =========================
    # FINAL
    # =========================
    df_final = pd.concat([
        df_historical.reset_index(),
        df_new
    ])

    df_final = df_final.drop_duplicates("Datetime")
    df_final = df_final.sort_values("Datetime")
    df_final = df_final.set_index("Datetime")

    log(f"Final max: {df_final.index.max()}")

    # =========================
    # SAVE
    # =========================
    df_final.to_parquet(OUTPUT_PATH)

    upload_to_dropbox(
        OUTPUT_PATH,
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    )

    log("✅ SALVATO")

    return df_final, log_lines


# =========================================================
# UI
# =========================================================
df_historical = load_from_dropbox(
    "/forecast_pun/dataset_history.parquet",
    st.secrets["DROPBOX_TOKEN"]
)

st.write("Last storico:", df_historical.index.max())

run = st.button("🔄 Aggiorna DB")

WINDOW = 96 * 7

if run:

    st.write("CLICK ✅")

    try:
        df_updated, logs = pipeline_run()

        st.write("LOG:", logs)

        st.write("UPDATED MAX:", df_updated.index.max())

        drift = ks_drift(
            df_historical.tail(WINDOW),
            df_updated.tail(WINDOW),
            [c for c in df_updated.columns if c != "PUN"]
        )

        st.dataframe(drift)

        st.success("✅ DONE")

    except Exception as e:
        st.error(str(e))
        st.code(traceback.format_exc())

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

            if os.path.exists(ERROR_PATH):
              df_old = pd.read_parquet(ERROR_PATH)
              df_all = pd.concat([df_old, df_eval], ignore_index=True)
            else:
              df_all = df_eval.copy()
            
            df_all = df_all.sort_values(["Datetime", "created_at"])
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
        LOOKBACK_DAYS = 8
        lookback_start_dt = last_date.floor("D") - pd.Timedelta(days=LOOKBACK_DAYS)

        # =========================
        # MATCH COLONNA EXCEL
        # =========================
        col_name = None

        for c in df_excel.columns:
            if nome.replace("_", " ").lower() in str(c).lower():
                col_name = c
                break

        if col_name is None:
            log(f"⚠️ colonna non trovata per {nome}")
            continue

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
        df_hist = df_historical.reset_index()[["Datetime", "target"]]

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

# ============================⚡ MI PIPELINE: UPDATE + FORECAST + MONITORING ⚡=========================== #

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
    dfs_mi = load_mi_datasets_from_dropbox_cached(DROPBOX_TOKEN)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset MI", len(dfs_mi))

    if dfs_mi:
        last_dt = max(v.index.max() for v in dfs_mi.values())
        col2.metric("Ultima data", str(last_dt))
        col3.metric("Mercati", len(colonne_analisi_mi))
    else:
        col2.warning("Nessun dataset")

except Exception:
    dfs_mi = {}
    st.error("Errore loading MI")
    st.code(traceback.format_exc())


# =========================================================
# PREVIEW
# =========================================================
with st.expander("📚 Preview dataset"):
    if dfs_mi:
        mkt = st.selectbox("Mercato", list(dfs_mi.keys()))
        st.dataframe(dfs_mi[mkt].tail(50), use_container_width=True)
    else:
        st.warning("Niente dataset")


# =========================================================
# BUTTONS
# =========================================================

col1, col2 = st.columns(2)

with col1:
    run_update = st.button("🧱 Update DB + KS Drift", use_container_width=True)

with col2:
    run_forecast = st.button("📈 Forecast + Monitoring", use_container_width=True)

WINDOW = 96 * 7


# =========================================================
# ✅ BOTTONE 1: UPDATE DB + KS DRIFT
# =========================================================
if run_update:

    if not dfs_mi:
        st.error("❌ Nessun dataset MI")
        st.stop()

    try:
        st.subheader("🧱 Update dataset")

        dfs_old = dfs_mi.copy()

        with st.spinner("Aggiornamento DB MI..."):
            dfs_new, logs = pipeline_run_mi()

        st.success("✅ DB aggiornato")
        st.code("\n".join(logs))

        # ===== KS DRIFT =====
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

            if drift_df.empty:
                continue

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

        # aggiorna runtime
        dfs_mi = dfs_new

    except Exception:
        st.error("❌ Errore update + KS")
        st.code(traceback.format_exc())


# =========================================================
# ✅ BOTTONE 2: FORECAST + MONITORING (per ciascun mercato MI)
# =========================================================
if run_forecast:

    if not dfs_mi:
        st.error("❌ Nessun dataset MI")
        st.stop()

    try:
        st.subheader("📈 Forecast")

        terna = TernaClient(
            client_id=st.secrets["TERNA_CLIENT_ID"],
            client_secret=st.secrets["TERNA_CLIENT_SECRET"]
        )

        meteo = MeteoDownloader()

        df_long, df_wide, df_errors = forecast_next_96_all_mi_models_dropbox(
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

        if df_errors is not None and not df_errors.empty:
            st.error("❌ Errori modelli")
            st.dataframe(df_errors)

        if df_long is None or df_long.empty:
            st.error("❌ Forecast vuoto")
            st.stop()

        st.success("✅ Forecast completato")
        st.dataframe(df_long.tail(100))

        # =================================================
        # MONITORING
        # =================================================
        st.subheader("📉 Monitoring")

        uploaded_file = st.file_uploader("Carica Excel MI reali", type=["xlsx"])

        if uploaded_file is None:
            st.info("⬆️ Carica il file Excel con i prezzi reali per calcolare l'errore")
            st.stop()

        df_real = pd.read_excel(uploaded_file)

        df_real["Data"] = pd.to_datetime(df_real["Data"], dayfirst=True)
        df_real["Datetime"] = (
            df_real["Data"]
            + pd.to_timedelta((df_real["Periodo"] - 1) * 15, unit="m")
        )

        df_forecast = load_from_dropbox(
            MI_FORECAST_HISTORY_LONG,
            DROPBOX_TOKEN
        ).copy()

        df_forecast["Datetime"] = pd.to_datetime(df_forecast["Datetime"])

        if df_forecast["Datetime"].dt.tz is not None:
            df_forecast["Datetime"] = (
                df_forecast["Datetime"]
                .dt.tz_convert("Europe/Rome")
                .dt.tz_localize(None)
            )

        # ===== Matching forecast vs real, per ciascun mercato =====
        all_eval = []

        for col in df_real.columns:

            if col in ["Data", "Ora", "Periodo", "Datetime", "Italia"]:
                continue

            nome_df = make_market_key(col)

            df_pred = df_forecast[df_forecast["nome_df"] == nome_df]

            if df_pred.empty:
                continue

            df_tmp = df_pred.merge(
                df_real[["Datetime", col]],
                on="Datetime",
                how="inner"
            )

            if df_tmp.empty:
                continue

            df_tmp = df_tmp.rename(columns={col: "real"})

            df_tmp["abs_error"] = (df_tmp["real"] - df_tmp["pred"]).abs()
            df_tmp["error_abs_perc"] = df_tmp["abs_error"] / (df_tmp["real"] + 1e-6)

            all_eval.append(df_tmp)

        if not all_eval:
            st.warning("⚠️ Nessun match forecast vs real")
            st.stop()

        df_eval = pd.concat(all_eval)

        df_eval["mae"] = df_eval["abs_error"].rolling(96).mean()
        df_eval["rmse"] = (df_eval["abs_error"] ** 2).rolling(96).mean() ** 0.5

        c1, c2 = st.columns(2)
        c1.metric("MAE", round(df_eval["mae"].iloc[-1], 2))
        c2.metric("RMSE", round(df_eval["rmse"].iloc[-1], 2))

        st.line_chart(df_eval[["mae", "rmse"]].dropna())

        mape = df_eval["error_abs_perc"].mean() * 100

        if mape > 25:
            st.error("🚨 Drift forte")
        elif mape > 18:
            st.warning("⚠️ Drift moderato")
        else:
            st.success("✅ Modello stabile")

        st.success("✅ Pipeline completata")

    except Exception:
        st.error("❌ Errore forecast/monitoring")
        st.code(traceback.format_exc())
