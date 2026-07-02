# ============================================================
# mi_section.py — Sezione MI da richiamare dentro app.py (PUN)
# ============================================================
# NON è una pagina a sé: espone render_mi() che disegna l'intera sezione MI.
# PUN e MI restano SEMPRE entrambi visibili (nessun selettore): in app.py
# aggiungi l'import in cima e la chiamata come ULTIMA riga del file:
#
#     from mi_section import render_mi          # in cima, con gli altri import
#     ...   (tutto il codice PUN, invariato)   ...
#     st.divider()
#     render_mi()                               # in fondo: la sezione MI segue il PUN
#
# Richiede alla radice del progetto: mi_direct_forecast.py, train_mi_direct.py,
# functions/ (gli stessi helper del PUN) e il secret st.secrets["DROPBOX_TOKEN"].
#
# ⚠️ ASSUNZIONI (in cima, facili da cambiare):
#   - Dropbox: /forecast_mi/<zona>/... e /forecast_mi/models_mi/<zona>/
#   - Excel: colonna Datetime + colonna prezzo della zona (nome = ZONE_TARGET),
#     tollera Excel 'largo' (tutte le zone) o con un'unica colonna numerica.
# ============================================================

import os
import traceback
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import dropbox

# helper Dropbox condivisi con il PUN
from functions.create_datasets import load_from_dropbox, upload_to_dropbox

# motore di inference MI + nomi file artefatti
from mi_direct_forecast import (
    load_direct_artifacts,
    forecast_next_96,
    build_native_importance_df,
    summarize_native_importance,
)
from train_mi_direct import MODEL_QUANTILES_NAME, METADATA_NAME


# ============================================================
# CONFIG / ASSUNZIONI
# ============================================================
DROPBOX_ROOT = "/forecast_mi"

ZONE_TARGET = {
    "italia_senza_vincoli": "Italia (senza vincoli)",
    "calabria":             "Calabria",
    "centro_nord":          "Centro Nord",
    "centro_sud":           "Centro Sud",
    "nord":                 "Nord",
    "sardegna":             "Sardegna",
    "sicilia":              "Sicilia",
    "sud":                  "Sud",
    "italia_coupling":      "Italia Coupling",
}


def zone_paths(zone_key: str) -> dict:
    base = f"{DROPBOX_ROOT}/{zone_key}"
    return {
        "dataset": f"{base}/dataset.parquet",
        "model_dir": f"{DROPBOX_ROOT}/models_mi/{zone_key}",
        "forecast_history": f"{base}/forecast_history.parquet",
        "error_history": f"{base}/error_history.parquet",
    }


# ============================================================
# DROPBOX HELPERS
# ============================================================
def make_dbx_client() -> dropbox.Dropbox:
    token = st.secrets.get("DROPBOX_TOKEN", os.getenv("DROPBOX_TOKEN", ""))
    if not token:
        raise RuntimeError("DROPBOX_TOKEN mancante (st.secrets o env).")
    return dropbox.Dropbox(oauth2_access_token=token)


def _dropbox_path_exists(path: str) -> bool:
    """True se esiste, False se NON esiste; su altri errori RILANCIA (non azzerare storici)."""
    try:
        make_dbx_client().files_get_metadata(path)
        return True
    except dropbox.exceptions.ApiError as e:
        if (isinstance(e.error, dropbox.files.GetMetadataError)
                and e.error.is_path() and e.error.get_path().is_not_found()):
            return False
        raise


def _download_file(dbx, dropbox_path: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _, res = dbx.files_download(dropbox_path)
    with open(local_path, "wb") as f:
        f.write(res.content)


@st.cache_resource(show_spinner=False)
def load_zone_model(zone_key: str):
    """Scarica gli artefatti della zona da Dropbox e li carica. Cache per zona."""
    paths = zone_paths(zone_key)
    local_dir = Path("models_mi_local") / zone_key
    dbx = make_dbx_client()
    for name in (MODEL_QUANTILES_NAME, METADATA_NAME):
        _download_file(dbx, f"{paths['model_dir']}/{name}", local_dir / name)
    return load_direct_artifacts(str(local_dir))


# ============================================================
# EXCEL / DATASET HELPERS
# ============================================================
def _find_datetime_col(df: pd.DataFrame):
    for cand in ("Datetime", "datetime", "Data", "data", "DateTime", "timestamp", "Timestamp"):
        if cand in df.columns:
            return cand
    for c in df.columns:
        try:
            pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None


def parse_zone_excel(uploaded_file, target_col: str) -> pd.DataFrame:
    """Excel -> df indicizzato Datetime con la sola colonna prezzo della zona (rinominata target_col)."""
    raw = pd.read_excel(uploaded_file)

    dt_col = _find_datetime_col(raw)
    if dt_col is None:
        raise ValueError(f"Nessuna colonna data/ora riconosciuta. Colonne: {list(raw.columns)}")

    raw[dt_col] = pd.to_datetime(raw[dt_col], errors="coerce")
    raw = raw.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    raw.index.name = "Datetime"

    if target_col in raw.columns:
        price_col = target_col
    else:
        norm = {str(c).strip().lower(): c for c in raw.columns}
        price_col = norm.get(str(target_col).strip().lower())

    if price_col is None:
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c])]
        if len(num_cols) == 1:
            price_col = num_cols[0]
        else:
            raise ValueError(
                f"Colonna prezzo per '{target_col}' non trovata. Colonne disponibili: {list(raw.columns)}"
            )

    out = raw[[price_col]].copy()
    out.columns = [target_col]
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out[~out.index.duplicated(keep="last")]
    return out


def _load_dataset(zone_key: str):
    paths = zone_paths(zone_key)
    if not _dropbox_path_exists(paths["dataset"]):
        return None
    df = load_from_dropbox(paths["dataset"], st.secrets["DROPBOX_TOKEN"]).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")
    return df.sort_index()


# ============================================================
# RENDER — l'intera sezione MI
# ============================================================
def render_mi():
    st.sidebar.header("⚡ Zona MI")
    zone_key = st.sidebar.selectbox(
        "Seleziona la zona",
        options=list(ZONE_TARGET.keys()),
        format_func=lambda k: ZONE_TARGET[k],
    )
    target_col = ZONE_TARGET[zone_key]
    paths = zone_paths(zone_key)

    st.title(f"📈 MI — {target_col}")
    st.caption(f"Modello direct 96 v3 (quantile grid + CQR asimmetrico) — zona **{target_col}**")

    # ------------------------------------------------------------
    # 1) DATI INPUT — aggiorna / sovrascrivi
    # ------------------------------------------------------------
    st.divider()
    st.header("1 · Dati input")

    col_a, col_b = st.columns(2)
    df_dataset = None
    try:
        df_dataset = _load_dataset(zone_key)
        with col_a:
            if df_dataset is not None and not df_dataset.empty:
                st.metric("📅 Ultima data dataset", str(df_dataset.index.max()))
                st.metric("Righe", f"{len(df_dataset):,}".replace(",", "."))
            else:
                st.info("Nessun dataset ancora su Dropbox per questa zona.")
    except Exception as e:
        with col_a:
            st.warning(f"⚠️ Impossibile leggere il dataset: {e}")

    with col_b:
        st.info(
            "Carica un Excel/parquet con i nuovi prezzi della zona: verranno uniti "
            "allo storico su Dropbox (dedup per timestamp) e il dataset verrà sovrascritto."
        )

    up_data = st.file_uploader(
        f"📥 Nuovi dati per «{target_col}» (Excel o parquet)",
        type=["xlsx", "parquet"], key="mi_data_upload",
    )

    if up_data is not None and st.button("🔄 Aggiorna e sovrascrivi dataset", use_container_width=True):
        try:
            if up_data.name.lower().endswith(".parquet"):
                new_df = pd.read_parquet(up_data)
                if not isinstance(new_df.index, pd.DatetimeIndex):
                    dt_col = _find_datetime_col(new_df)
                    new_df[dt_col] = pd.to_datetime(new_df[dt_col], errors="coerce")
                    new_df = new_df.dropna(subset=[dt_col]).set_index(dt_col)
                    new_df.index.name = "Datetime"
                if target_col not in new_df.columns:
                    raise ValueError(f"Il parquet non contiene la colonna '{target_col}'.")
                new_df = new_df[[target_col]].copy()
                new_df[target_col] = pd.to_numeric(new_df[target_col], errors="coerce")
                new_df = new_df[~new_df.index.duplicated(keep="last")]
            else:
                new_df = parse_zone_excel(up_data, target_col)

            st.success(f"✅ Letti {len(new_df)} nuovi record")
            st.dataframe(new_df.tail(10), use_container_width=True)

            if df_dataset is not None and not df_dataset.empty:
                merged = pd.concat([df_dataset, new_df], axis=0)
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            else:
                merged = new_df.sort_index()

            merged = merged.asfreq("15min")

            local = Path("mi_tmp"); local.mkdir(exist_ok=True)
            local_path = local / f"dataset_{zone_key}.parquet"
            merged.to_parquet(local_path)
            upload_to_dropbox(str(local_path), paths["dataset"], st.secrets["DROPBOX_TOKEN"])

            st.success(f"✅ Dataset «{target_col}» aggiornato e sovrascritto su Dropbox "
                       f"({len(merged)} righe, fino a {merged.index.max()})")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"❌ Errore aggiornamento dati: {e}")
            st.code(traceback.format_exc(), language="python")

    # ------------------------------------------------------------
    # 2) MODELLO — carica + forecast
    # ------------------------------------------------------------
    st.divider()
    st.header("2 · Modello e forecast")

    if st.sidebar.button("🔁 Ricarica modello MI da Dropbox"):
        load_zone_model.clear()

    if "mi_fc" not in st.session_state:
        st.session_state["mi_fc"] = {}

    try:
        artifacts = load_zone_model(zone_key)
        model_meta = artifacts["metadata"]
        models_q = artifacts["models_q"]
        st.success(f"✅ Modello «{target_col}» caricato da Dropbox")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Steps", model_meta.get("steps", 96))
        c2.metric("Punto", f"q{round(float(model_meta.get('point_quantile', 0.5)), 3)}")
        c3.metric("CQR asimm.", "sì" if model_meta.get("asymmetric_cqr") else "no")
        c4.metric("Quantili", len(model_meta.get("quantile_levels", [])))

        op = model_meta.get("overall_pinball_by_quantile", {})
        if op:
            st.caption("Pinball backtest interno: " +
                       " · ".join(f"q{k}={float(v):.3f}" for k, v in list(op.items())[:7]))
    except Exception as e:
        st.error(f"❌ Modello non disponibile per «{target_col}»: {e}")
        st.info("Alleni la zona e carichi gli artefatti su Dropbox in "
                f"`{paths['model_dir']}/` (i file {MODEL_QUANTILES_NAME} e {METADATA_NAME}).")
        return

    run_fc = st.button("📈 Esegui forecast next 96", use_container_width=True)

    if run_fc:
        try:
            if df_dataset is None or df_dataset.empty:
                st.warning("⚠️ Nessun dataset per questa zona: carica prima i dati input (sezione 1).")
                return
            with st.spinner("Calcolo forecast..."):
                fc = forecast_next_96(df_dataset, model_dir=str(Path("models_mi_local") / zone_key))
            st.session_state["mi_fc"][zone_key] = fc

            fc_save = fc.copy()
            fc_save["created_at"] = pd.Timestamp.now()
            if _dropbox_path_exists(paths["forecast_history"]):
                try:
                    old = load_from_dropbox(paths["forecast_history"], st.secrets["DROPBOX_TOKEN"])
                    allfc = pd.concat([old, fc_save], ignore_index=True)
                except Exception:
                    st.error("Impossibile leggere il forecast history: non sovrascrivo.")
                    return
            else:
                allfc = fc_save.copy()
            allfc = allfc.drop_duplicates(subset=["Datetime"], keep="last").sort_values("Datetime")

            local = Path("mi_tmp"); local.mkdir(exist_ok=True)
            p = local / f"forecast_{zone_key}.parquet"
            allfc.to_parquet(p)
            upload_to_dropbox(str(p), paths["forecast_history"], st.secrets["DROPBOX_TOKEN"])
            st.success(f"✅ Forecast salvato ({len(fc)} punti; storico {len(allfc)})")
        except Exception as e:
            st.error(f"❌ Errore forecast: {e}")
            st.code(traceback.format_exc(), language="python")

    fc = st.session_state["mi_fc"].get(zone_key)
    if fc is not None:
        st.subheader("Forecast intraday")
        c1, c2, c3 = st.columns(3)
        c1.metric("Min", round(float(fc["pred"].min()), 2))
        c2.metric("Max", round(float(fc["pred"].max()), 2))
        c3.metric("Media", round(float(fc["pred"].mean()), 2))

        fig = go.Figure()
        if {"upper", "lower"} <= set(fc.columns):
            cov = float(fc["band_coverage"].iloc[0]) if "band_coverage" in fc.columns else None
            band_name = "Banda calibrata" + (f" ~{cov:.0%}" if cov is not None else "")
            fig.add_trace(go.Scatter(x=fc["Datetime"], y=fc["upper"], mode="lines",
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=fc["Datetime"], y=fc["lower"], mode="lines",
                                     fill="tonexty", fillcolor="rgba(255,0,0,0.12)",
                                     line=dict(width=0), name=band_name))
        fig.add_trace(go.Scatter(x=fc["Datetime"], y=fc["pred"], mode="lines",
                                 line=dict(color="black", width=2), name="Forecast (punto)"))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fc, use_container_width=True)
        st.download_button("⬇️ Scarica forecast", fc.to_csv(index=False),
                           file_name=f"forecast_mi_{zone_key}.csv", mime="text/csv")

        if {"upper", "lower"} <= set(fc.columns):
            bw = fc.assign(width=fc["upper"] - fc["lower"])
            st.plotly_chart(px.bar(bw, x="Datetime", y="width",
                                   title="Ampiezza banda calibrata (€/MWh)"),
                            use_container_width=True)

    # ------------------------------------------------------------
    # 3) EXPLAINABILITY
    # ------------------------------------------------------------
    st.divider()
    st.header("3 · Explainability (gain LightGBM)")
    st.caption("Importanza nativa (gain) dei modelli del quantile-punto, media sugli orizzonti.")

    if st.button("🔎 Calcola importanza feature", use_container_width=True):
        try:
            with st.spinner("Calcolo importanza..."):
                imp_df = build_native_importance_df(models_q, model_meta)
            st.session_state.setdefault("mi_imp", {})[zone_key] = imp_df
        except Exception as e:
            st.error(f"❌ Errore importanza: {e}")

    imp_df = st.session_state.get("mi_imp", {}).get(zone_key)
    if imp_df is not None:
        if imp_df.empty:
            st.warning("⚠️ Nessuna importanza disponibile.")
        else:
            top_n = st.slider("Feature da mostrare", 5, 50, 25, 5, key="mi_topn")
            summ = summarize_native_importance(imp_df, top_n=top_n)
            st.plotly_chart(
                px.bar(summ.sort_values("importance"), x="importance", y="feature",
                       orientation="h", title="Top feature — gain medio sugli orizzonti"),
                use_container_width=True,
            )
            st.dataframe(summ, use_container_width=True)
            with st.expander("Tabella completa per orizzonte"):
                st.dataframe(imp_df, use_container_width=True)

    # ------------------------------------------------------------
    # 4) MONITORING (concept drift) — upload Excel prezzi reali
    # ------------------------------------------------------------
    st.divider()
    st.header("4 · Monitoring")

    today = date.today()
    up_real = st.file_uploader(
        f"📥 Prezzi reali «{target_col}» rilevati a {today} (Excel)",
        type=["xlsx"], key="mi_real_upload",
    )

    if up_real is not None:
        try:
            df_real = parse_zone_excel(up_real, target_col).rename(columns={target_col: "y"})
            st.success("✅ File prezzi reali caricato")
            st.dataframe(df_real.head(), use_container_width=True)

            if not _dropbox_path_exists(paths["forecast_history"]):
                st.warning("⚠️ Nessun forecast history su Dropbox per questa zona.")
                return
            df_fc = load_from_dropbox(paths["forecast_history"], st.secrets["DROPBOX_TOKEN"]).copy()
            df_fc["Datetime"] = pd.to_datetime(df_fc["Datetime"])
            st.info(f"Forecast in archivio: {len(df_fc)} righe")

            max_real = df_real.index.max()
            df_fc_eval = df_fc[df_fc["Datetime"] <= max_real].copy()
            df_eval = df_fc_eval.merge(df_real.reset_index(), on="Datetime", how="inner")

            if df_eval.empty:
                st.warning("⚠️ Nessun match tra forecast e prezzi reali.")
            else:
                run_ts = pd.Timestamp.now()
                df_eval["error"] = df_eval["y"] - df_eval["pred"]
                df_eval["abs_error"] = df_eval["error"].abs()
                df_eval["created_at"] = run_ts

                if _dropbox_path_exists(paths["error_history"]):
                    try:
                        old = load_from_dropbox(paths["error_history"], st.secrets["DROPBOX_TOKEN"])
                    except Exception:
                        st.error("Impossibile leggere lo storico errori: interrompo per non sovrascriverlo.")
                        return
                    df_all = pd.concat([old, df_eval], ignore_index=True)
                else:
                    df_all = df_eval.copy()

                df_all = df_all.sort_values(["Datetime", "created_at"])
                df_all = df_all.drop_duplicates(subset=["Datetime"], keep="last")

                local = Path("mi_tmp"); local.mkdir(exist_ok=True)
                p = local / f"error_{zone_key}.parquet"
                df_all.to_parquet(p)
                upload_to_dropbox(str(p), paths["error_history"], st.secrets["DROPBOX_TOKEN"])
                st.success("✅ Error history aggiornato")

                df_all["mae_rolling"] = df_all["abs_error"].rolling(96).mean()
                df_all["rmse_rolling"] = (df_all["error"] ** 2).rolling(96).mean() ** 0.5
                c1, c2 = st.columns(2)
                c1.metric("MAE recente", round(float(df_all["mae_rolling"].iloc[-1]), 2))
                c2.metric("RMSE recente", round(float(df_all["rmse_rolling"].iloc[-1]), 2))

                st.subheader("Trend errori")
                st.line_chart(df_all[["mae_rolling", "rmse_rolling"]].dropna())

                st.subheader("Errore per ora del giorno")
                df_all["hour"] = df_all["Datetime"].dt.hour
                st.plotly_chart(px.box(df_all, x="hour", y="error"), use_container_width=True)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_all["Datetime"], y=df_all["y"], mode="lines",
                                         name="Reale", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=df_all["Datetime"], y=df_all["pred"], mode="lines",
                                         name="Forecast", line=dict(color="red")))
                st.plotly_chart(fig, use_container_width=True)

                df_all["ape"] = np.abs(df_all["pred"] - df_all["y"]) / df_all["y"].abs().clip(lower=1.0)
                mape = float(df_all["ape"].mean() * 100)
                st.write(f"MAPE medio: {mape:.2f}%")
                if mape > 20:
                    st.error("🚨 Concept drift forte → retraining urgente")
                elif mape > 15:
                    st.warning("⚠️ Performance degradata → monitoring stretto")
                elif mape > 10:
                    st.info("🟡 Buone performance (10–15%)")
                else:
                    st.success("✅ Ben calibrato (<10%)")

                st.download_button("⬇️ Scarica error history", df_all.to_csv(index=False),
                                   file_name=f"error_history_mi_{zone_key}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ Errore monitoring: {e}")
            st.code(traceback.format_exc(), language="python")
