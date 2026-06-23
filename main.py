import os
import traceback
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp
import plotly.graph_objects as go
import yaml
import dropbox

from functions.create_datasets import (PUNFeatureEngineering, MeteoDownloader, TernaClient, ks_drift, upload_to_dropbox, load_from_dropbox, EntsoeDownloader)
from functions.forecast import (forecast_day_ahead_96_base, pun_to_datetime, plot_forecast_pun)


from functions.forecast_mi import (forecast_next_96_all_mi_models_dropbox)

st.set_page_config(page_title="PUN Dataset Manager", layout="wide")

CONFIG_PATH = "config/config.yaml"

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def load_config_mi():
    with open("config/config_mi.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()
MI = load_config_mi()["mi"]

FEATURES_OLD = config["features"]["FEATURES_OLD"]
FEATURES_NEW = config["features"]["FEATURES_NEW"]
SELECTED_EXOG = config["features"]["SELECTED_EXOG"]

# =========================================================
# CONFIG UI
# =========================================================
st.title("⚡ PUN Dataset Manager")
st.caption("Aggiornamento dataset intraday PUN / Meteo / Terna/ Entsoe")
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


def debug_mi_dropbox(dropbox_token: str):

    st.subheader("🧪 Debug MI Dropbox")
    # 5. CHECK MODELS FROM JSON
    # =====================================================
    for nome_df, targets in results.items():

        if not isinstance(targets, dict):
            continue

        for target, res in targets.items():

            if not isinstance(res, dict):
                continue

            if res.get("status") != "ok":
                continue

            raw_model_path = res.get("model_path")

            if not raw_model_path:
                report["models_missing"].append(f"{nome_df}/{target} -> model_path mancante")
                continue

            # pulizia path Windows/Linux
            clean_path = str(raw_model_path).replace("\\", "/").strip()
            file_name = os.path.basename(clean_path)

            # evita controlli duplicati
            if file_name in checked_models:
                continue

            checked_models.add(file_name)

            if file_name not in available_models:
                report["models_missing"].append(file_name)
                continue

            real_dropbox_path = available_models[file_name]

            try:
                _, file_res = dbx.files_download(real_dropbox_path)

                payload = joblib.load(io.BytesIO(file_res.content))

                for key in ["forecaster", "selected_exog", "target"]:
                    if key not in payload:
                        report["payload_errors"].append(
                            f"{file_name} missing '{key}'"
                        )

                report["models_ok"].append(file_name)

            except Exception as e:
                report["models_missing"].append(f"{file_name} -> {e}")

    # =====================================================
    # OUTPUT
    # =====================================================
    st.write("✅ Dataset OK:", len(report["datasets_ok"]))
    st.write("❌ Dataset missing:", report["datasets_missing"])

    st.write("✅ Models OK:", len(report["models_ok"]))
    st.write("❌ Models missing:", report["models_missing"])

    if report["payload_errors"]:
        st.warning("⚠️ Payload errors:")
        st.write(report["payload_errors"])

    st.write("🔎 JSON vs datasets:")
    st.json(report["json_vs_datasets"])

    if (
        not report["datasets_missing"]
        and not report["models_missing"]
        and not report["payload_errors"]
    ):
        st.success("✅ Sistema MI pronto")
    else:
        st.error("❌ Problemi trovati")

    return report


    dbx = dropbox.Dropbox(dropbox_token)

    report = {
        "datasets_ok": [],
        "datasets_missing": [],
        "models_ok": [],
        "models_missing": [],
        "payload_errors": [],
        "json_vs_datasets": {}
    }

    # =====================================================
    # 1. CHECK DATASETS
    # =====================================================
    for nome in colonne_analisi_mi:
        file_name = f"MI_{nome}.parquet"
        path = f"{MI_DATASETS_DIR}/{file_name}"

        try:
            dbx.files_get_metadata(path)
            report["datasets_ok"].append(file_name)
        except Exception as e:
            report["datasets_missing"].append(f"{file_name} -> {e}")

    # =====================================================
    # 2. LOAD JSON
    # =====================================================
    try:
        results = load_mi_json_from_dropbox(dropbox_token)
        st.success("✅ JSON OK")
    except Exception as e:
        st.error(f"❌ JSON ERROR: {e}")
        return report

    # =====================================================
    # 3. CHECK JSON VS DATASETS
    # =====================================================
    json_keys = set(results.keys())
    dfs_keys = set([make_market_key(x) for x in colonne_analisi_mi])

    report["json_vs_datasets"] = {
        "common": sorted(json_keys & dfs_keys),
        "json_not_in_dfs": sorted(json_keys - dfs_keys),
        "dfs_not_in_json": sorted(dfs_keys - json_keys),
    }

    # =====================================================
    # 4. LIST MODELS FOLDER ONCE
    # =====================================================
    try:
        folder_res = dbx.files_list_folder(MI_MODELS_DIR)

        available_models = {
            entry.name: entry.path_display
            for entry in folder_res.entries
            if entry.name.endswith(".joblib")
        }

    except Exception as e:
        st.error(f"❌ Errore lettura cartella modelli: {MI_MODELS_DIR} -> {e}")
        return report

    # evita doppioni se stesso file appare più volte nel JSON
    checked_models = set()



    # =========================================================
    # 3. CHECK MATCH KEYS
    # =========================================================
    dfs_keys = [make_market_key(x) for x in colonne_analisi_mi]
    json_keys = list(results.keys())

    report["json_vs_datasets"] = {
        "common": list(set(json_keys) & set(dfs_keys)),
        "json_not_in_dfs": list(set(json_keys) - set(dfs_keys)),
        "dfs_not_in_json": list(set(dfs_keys) - set(json_keys))
    }

    # =========================================================
    # 4. CHECK MODELS
    # =========================================================
    st.write("🤖 Checking models...")

    for nome_df, targets in results.items():

        for target, res in targets.items():

            if not isinstance(res, dict):
                continue

            if res.get("status") != "ok":
                continue

            model_path = res.get("model_path")

            if not model_path:
                report["models_missing"].append(f"{nome_df}/{target} (no path)")
                continue

            # resolve path
            if not model_path.startswith("/"):
                model_path = f"{MI_MODELS_DIR}/{os.path.basename(model_path)}"

            try:
                _, res_file = dbx.files_download(model_path)
                report["models_ok"].append(model_path)

                # CHECK PAYLOAD
                try:
                    payload = joblib.load(io.BytesIO(res_file.content))

                    for key in ["forecaster", "selected_exog", "target"]:
                        if key not in payload:
                            report["payload_errors"].append(
                                f"{model_path} missing '{key}'"
                            )

                except Exception as e:
                    report["payload_errors"].append(
                        f"{model_path} corrupted: {e}"
                    )
            #
            except Exception as e:
                report["models_missing"].append(f"{model_path} -> {str(e)}")


    # =========================================================
    # OUTPUT
    # =========================================================
    st.subheader("📊 Report")

    st.write("✅ Dataset OK:", len(report["datasets_ok"]))
    st.write("❌ Dataset mancanti:", report["datasets_missing"])

    st.write("✅ Modelli OK:", len(report["models_ok"]))
    st.write("❌ Modelli mancanti:", report["models_missing"])

    if report["payload_errors"]:
        st.warning("⚠️ Problemi payload modelli:")
        st.write(report["payload_errors"])

    st.write("🔎 JSON vs datasets:")
    st.json(report["json_vs_datasets"])

    if (
        not report["datasets_missing"]
        and not report["models_missing"]
        and not report["payload_errors"]
        and not report["json_vs_datasets"]["json_not_in_dfs"]
        and not report["json_vs_datasets"]["dfs_not_in_json"]
    ):
        st.success("✅ TUTTO OK - pronto per forecast MI")
    else:
        st.error("❌ Problemi trovati")

    return report

# =========================================================
# CORE PIPELINE
# =========================================================
def pipeline_run():

    df_historical = df_historical.sort_index().asfreq("15min").ffill()

    last_date = df_historical.index.max()
    today = pd.Timestamp.today()

    # =========================
    # LOOKBACK
    # =========================
    LOOKBACK_DAYS = 8
    lookback_start_dt = pd.Timestamp(last_date).floor("D") - pd.Timedelta(days=LOOKBACK_DAYS)
    end_date_dt = pd.Timestamp(today)

    log(f"Ultima data storico: {last_date}")

    # =========================
    # PUN
    # =========================
    pun_fe = PUNFeatureEngineering(start=lookback_start_dt.strftime("%Y-%m-%d"), pun_col="PUN")

    pun_hist = df_historical.reset_index()[["Datetime", "PUN"]]
    pun_df_new = prepare_pun(pun_fe, PUN_INPUT_PATH)[["Datetime", "PUN"]]

    pun_full = pd.concat([pun_hist, pun_df_new])
    pun_full = pun_full.drop_duplicates("Datetime", keep="last").sort_values("Datetime")

    # lag + returns
    pun_full["lag_2d"] = pun_full["PUN"].shift(96*2)
    pun_full["lag_7d"] = pun_full["PUN"].shift(96*7)

    pun_full["pun_ret_1d"] = pun_full["PUN"].pct_change(96).shift(1)
    pun_full["pun_ret_7d"] = pun_full["PUN"].pct_change(96*7).shift(1)

    pun_full["momentum_1d"] = pun_full["PUN"].shift(1) - pun_full["PUN"].shift(96)
    pun_full["momentum_4h"] = pun_full["PUN"].shift(1) - pun_full["PUN"].shift(16)

    pun_full["pun_ret_1h"] = pun_full["PUN"].pct_change(4).shift(1)
    pun_full["minute"] = pd.to_datetime(pun_full["Datetime"]).dt.minute

    pun_df = pun_full[pun_full["Datetime"] >= lookback_start_dt].copy()

    # =========================
    # METEO
    # =========================
    log("Meteo...")
    meteo = MeteoDownloader()

    meteo_df = prepare_meteo(
        meteo,
        lookback_start_dt.strftime("%Y-%m-%d"),
        end_date_dt.strftime("%Y-%m-%d")
    )

    # =========================
    # TERNA
    # =========================
    log("Terna...")

    terna = TernaClient(
        client_id=st.secrets["TERNA_CLIENT_ID"],
        client_secret=st.secrets["TERNA_CLIENT_SECRET"]
    )

    terna_df = prepare_terna(
        terna,
        lookback_start_dt.strftime("%d/%m/%Y"),
        end_date_dt.strftime("%d/%m/%Y")
    )

    terna_df = shift_terna_only(terna_df)

    # =========================
    # 🔥 ENTSOE
    # =========================
    log("ENTSOE...")

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
        end_date=end_date_dt.to_pydatetime()
    )

    
    entsoe_feat = entsoe.build_features()

    entsoe_feat["Datetime"] = pd.to_datetime(entsoe_feat["Timestamp"])
    entsoe_feat = entsoe_feat.drop(columns=["Timestamp"])


    entsoe_df["Timestamp"] = pd.to_datetime(entsoe_df["Timestamp"])
    entsoe_df["feature"] = entsoe_df["Zone"] + "_" + entsoe_df["ProductionType"]

    entsoe_feat = (
        entsoe_df
        .pivot_table(index="Timestamp", columns="feature", values="MW", aggfunc="mean")
        .asfreq("15min")
        .ffill()
        .reset_index()
    )

    ent_cols = [c for c in entsoe_feat.columns if c != "Timestamp"]

    # =========================
    # MERGE
    # =========================
    log("Merge...")

    df_new = merge_all(pun_df, meteo_df, terna_df)


    df_new = df_new.merge(
        entsoe_feat,
        on="Datetime",
        how="left"
        )

    # ✅ ENTSOE CLEAN
    df_new[ent_cols] = df_new[ent_cols].fillna(0.0).astype(float)

    # ✅ AGGIUNTA DINAMICA FEATURE (CRITICO)
    global FEATURES_NEW, SELECTED_EXOG
    FEATURES_NEW = list(set(FEATURES_NEW + ent_cols))
    SELECTED_EXOG = list(set(SELECTED_EXOG + ent_cols))

    # =========================
    # FEATURES
    # =========================
    df_new = add_features(df_new)

    df_new.drop(columns=FEATURES_DROP, errors="ignore", inplace=True)

    df_new["Datetime"] = pd.to_datetime(df_new["Datetime"]).dt.tz_localize(None)

    df_new = df_new[df_new["Datetime"] > last_date]

    # =========================
    # FINAL DATASET
    # =========================
    df_old = df_historical[FEATURES_OLD].reset_index()
    df_new = df_new[FEATURES_NEW]

    df_final = pd.concat([df_old, df_new])
    df_final = df_final.drop_duplicates("Datetime").sort_values("Datetime")
    df_final.set_index("Datetime", inplace=True)

    # =========================
    # CHECK
    # =========================
    assert df_final.index.is_monotonic_increasing
    assert df_final.index.is_unique

    # =========================
    # SAVE
    # =========================
    df_final.to_parquet(OUTPUT_PATH)

    upload_to_dropbox(
        OUTPUT_PATH,
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    )

    log("✅ Dataset aggiornato con ENTSOE")

    return df_final, log_lines

    def log(msg):
        log_lines.append(msg)

    # =========================
    # LOAD STORICO
    # =========================
    df_historical = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    ).copy()

    if not isinstance(df_historical.index, pd.DatetimeIndex):
        df_historical["Datetime"] = pd.to_datetime(df_historical["Datetime"])


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


if run_update:
    try:
        with st.spinner("🚀 Aggiornamento dataset in corso..."):
            df_updated, logs = pipeline_run()
            drift_cols = [c for c in df_updated.columns if c != "PUN"]
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

    df_view = df_historical.copy()

    df_view.index = (
        df_view.index
        .tz_convert("Europe/Rome")
        .tz_localize(None)
    )

    st.dataframe(df_view.tail(50), use_container_width=True)




st.subheader("📦 Preview DB aggiornato")

try:
    df_output = load_from_dropbox(
        "/forecast_pun/dataset_history.parquet",
        st.secrets["DROPBOX_TOKEN"]
    ).copy()

    if not isinstance(df_output.index, pd.DatetimeIndex):
        df_output["Datetime"] = pd.to_datetime(df_output["Datetime"])
        df_output = df_output.set_index("Datetime")

    df_output = df_output.sort_index()

    # ✅ FIX TIMEZONE (SOLO VISUALIZZAZIONE)
    df_view1 = df_output.copy()
    df_view1.index = (
        df_view1.index
        .tz_convert("Europe/Rome")
        .tz_localize(None)
    )

    st.dataframe(df_view1.tail(50), use_container_width=True)

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

# FONDAMENTALE:
# pipeline.py deve essere presente nella root della webapp Streamlit.
# Serve perché model_prod.pkl contiene riferimenti a funzioni definite in pipeline.py,
# ad esempio production_weight_func.
try:
    import pipeline  # noqa: F401
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
#


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

    preds_view.index = (
        preds_view.index
        .tz_convert("Europe/Rome")
        .tz_localize(None)
    )

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
        # ✅ FIX TIMEZONE PER MERGE# ✅ FIX TIMEZONE PERdf_forecast["Datetime"] = (
        
        df_forecast["Datetime"] = (df_forecast["Datetime"].dt.tz_convert("Europe/Rome").dt.tz_localize(None))

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

st.write('MI_DATASET_DIR', MI_DATASETS_DIR)
st.write("MODELS DIR:", MI_MODELS_DIR)

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

# =========================================================
# ⚡ MI PIPELINE: UPDATE + FORECAST + MONITORING
# =========================================================

st.divider()
st.header("⚡ MI Forecast + Monitoring")

DROPBOX_TOKEN = st.secrets.get("DROPBOX_TOKEN", "")

if not DROPBOX_TOKEN:
    st.error("❌ DROPBOX_TOKEN mancante")
    st.stop()


# =========================================================
# DEBUG
# =========================================================
if st.sidebar.button("🧪 Debug MI Dropbox"):
    debug_mi_dropbox(DROPBOX_TOKEN)


# =========================================================
# LOAD DATASETS
# =========================================================
col1, col2, col3 = st.columns(3)

try:
    dfs_mi = load_mi_datasets_from_dropbox_cached(DROPBOX_TOKEN)

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

        df_view2 = dfs_mi[mkt].copy()
        df_view2.index = (
            df_view2.index
            .tz_convert("Europe/Rome")
            .tz_localize(None)
        )

        st.dataframe(df_view2.tail(50))

    else:
        st.warning("Niente dataset")


# =========================================================
# CONTROL
# =========================================================
run_pipeline = st.button("🚀 Run MI Pipeline (Update + Forecast + Monitoring)")

WINDOW = 96 * 7   # 7 giorni


# =========================================================
# PIPELINE RUN
# =========================================================
if run_pipeline:

    if not dfs_mi:
        st.error("❌ Nessun dataset MI")
        st.stop()

    try:

        # =================================================
        # 1️⃣ UPDATE DATASET
        # =================================================
        st.subheader("🧱 Step 1: Update dataset")

        with st.spinner("Aggiornamento dataset MI..."):
            dfs_updated, logs = pipeline_run_mi()

        st.success("✅ Dataset MI aggiornati")
        st.code("\n".join(logs), language="text")

        # tieni old per KS
        dfs_old = dfs_mi.copy()
        dfs_mi = dfs_updated


        # =================================================
        # ✅ KS DRIFT (TUTTI I MERCATI)
        # =================================================
        st.subheader("📊 Covariate Drift (KS Test)")

        drift_results = []

        for nome in dfs_mi:

            if nome not in dfs_old:
                continue

            df_old = dfs_old[nome]
            df_new = dfs_mi[nome]

            if len(df_old) < WINDOW or len(df_new) < WINDOW:
                continue

            df_old_win = df_old.tail(WINDOW)
            df_new_win = df_new.tail(WINDOW)

            cols = [c for c in df_new.columns if c != "target"]

            drift_df = ks_drift(df_old_win, df_new_win, cols)

            if drift_df.empty:
                continue

            drift_df["market"] = nome
            drift_results.append(drift_df)

        if drift_results:

            drift_all = pd.concat(drift_results)

            st.dataframe(drift_all)

            n_drift = drift_all["drift_flag"].sum()

            if n_drift >= 10:
                st.error("🚨 Drift forte (MI)")
            elif n_drift > 0:
                st.warning("⚠️ Drift moderato")
            else:
                st.success("✅ Nessun drift")

        else:
            st.warning("⚠️ Drift non calcolabile")


        # =================================================
        # 2️⃣ FORECAST
        # =================================================
        st.subheader("📈 Step 2: Forecast")

        terna = TernaClient(
            client_id=st.secrets["TERNA_CLIENT_ID"],
            client_secret=st.secrets["TERNA_CLIENT_SECRET"]
        )

        meteo = MeteoDownloader()

        with st.spinner("Running forecast MI..."):

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
            st.error("❌ ERRORI MODELLI")
            st.dataframe(df_errors)

        if df_long is None or df_long.empty:
            st.error("❌ Forecast vuoto")
            st.stop()

        st.success("✅ Forecast completato")
        st.dataframe(df_long.tail(100))


        # =================================================
        # 3️⃣ UPLOAD REALI
        # =================================================
        st.subheader("📥 Step 3: Upload reali")

        uploaded_file = st.file_uploader(
            "Carica Excel MI reali",
            type=["xlsx"]
        )

        if uploaded_file is None:
            st.info("⬆️ Carica file reale MI")
            st.stop()

        df_real = pd.read_excel(uploaded_file)

        df_real["Data"] = pd.to_datetime(df_real["Data"], dayfirst=True)

        df_real["Datetime"] = (
            df_real["Data"]
            + pd.to_timedelta((df_real["Periodo"] - 1) * 15, unit="m")
        )

        df_real = df_real.sort_values("Datetime")


        # =================================================
        # 4️⃣ MONITORING
        # =================================================
        st.subheader("📉 Step 4: Monitoring")

        df_forecast = load_from_dropbox(
            MI_FORECAST_HISTORY_LONG,
            DROPBOX_TOKEN
        ).copy()

        df_forecast["Datetime"] = pd.to_datetime(df_forecast["Datetime"])
        df_forecast["Datetime"] = (df_forecast["Datetime"].dt.tz_convert("Europe/Rome").dt.tz_localize(None))
        all_eval = []

        for col in df_real.columns:

            if col in ["Data", "Ora", "Periodo", "Datetime"]:
                continue

            nome_df = make_market_key(col)

            df_pred = df_forecast[
                df_forecast["nome_df"] == nome_df
            ]

            if df_pred.empty:
                continue

            df_tmp = df_pred.merge(
                df_real[["Datetime", col]],
                on="Datetime"
            )

            if df_tmp.empty:
                continue

            df_tmp = df_tmp.rename(columns={col: "real"})

            df_tmp["error"] = df_tmp["real"] - df_tmp["pred"]
            df_tmp["abs_error"] = df_tmp["error"].abs()
            df_tmp["error_abs_perc"] = df_tmp["abs_error"] / (df_tmp["real"] + 1e-6)

            all_eval.append(df_tmp)

        if not all_eval:
            st.warning("⚠️ Nessun match forecast vs real")
            st.stop()

        df_eval = pd.concat(all_eval)

        df_eval["mae"] = df_eval["abs_error"].rolling(96).mean()
        df_eval["rmse"] = (df_eval["error"]**2).rolling(96).mean()**0.5

        st.subheader("📊 Metriche")

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

        st.cache_data.clear()

    except Exception as e:
        st.error("❌ ERRORE PIPELINE MI")
        st.code(traceback.format_exc())
