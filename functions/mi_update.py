# ============================================================
# mi_update.py — aggiornamento dataset MI (append righe complete su Drive)
# ============================================================
# Replica del pipeline_run del PUN, ma per le 9 zone MI:
#   1) scarica gli ESOGENI una volta sola (uguali per tutte le zone):
#      commodities (yfinance) + meteo + Terna market_load + ENTSOE _B16
#   2) per ogni zona: prende la y dall'Excel GME, calcola le feature
#      autoregressive con le formule confermate, RIALLINEA alle colonne del
#      dataset esistente su Drive, fa append, deduplica e risalva
#   3) test KS (drift) vecchio vs nuovo
#
# Mantiene i modelli attuali (nessun riallenamento). Le righe nuove hanno le
# STESSE colonne dello storico.
#
# ⚠️ CAVEAT (concordato):
#   - mi_ret_1h/1d/7d: formula non ricostruibile -> NaN nelle righe nuove
#     (l'imputer del modello la riempie; effetto piccolo e circoscritto).
#   - ret_1h/1d/7d: erano NaN nello storico (colonne morte) -> restano NaN.
#   - high_vol_regime: soglia = quantile 0.80 GLOBALE della vol24h (come lo
#     storico); il quantile si sposta un filo man mano che il dataset cresce.
#
# Riusa i client di functions/create_datasets.py e gdrive_io.py.
# Import "pigri" delle dipendenze pesanti, così il file si importa ovunque.
# ============================================================

import os
import numpy as np
import pandas as pd

import gdrive_io as gdrive

FREQ = "15min"
STEPS_PER_DAY = 96
LOOKBACK_DAYS = 8   # margine per lag_7d / rolling_7d

GDRIVE_ROOT = "PUN Forecast/forecast_pun/forecast_mi"

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

# Excel di input MI: export MI (infragiornaliero) — SEPARATO da Add_on_PUN.xlsx
# (il PUN è day-ahead; qui la y sono i prezzi MI). Stessa struttura GME
# (Data/Ora/Periodo + colonne zona). Committalo nel repo in dati_input/.
MI_INPUT_PATH = "dati_input/Add_on_MI.xlsx"

# città meteo — IDENTICHE al PUN (lista di dict, come vuole download_multi_city)
LOCATIONS = [
    {"name": "milano",  "lat": 45.4642, "lon": 9.1900},
    {"name": "torino",  "lat": 45.0703, "lon": 7.6869},
    {"name": "roma",    "lat": 41.9028, "lon": 12.4964},
    {"name": "bologna", "lat": 44.4949, "lon": 11.3426},
    {"name": "bari",    "lat": 41.1171, "lon": 16.8719},
    {"name": "palermo", "lat": 38.1157, "lon": 13.3615},
]
ZONES_ENTSOE = [
    ("NORD", "10Y1001A1001A73I"),
    ("CNOR", "10Y1001A1001A70O"),
    ("CSUD", "10Y1001A1001A71M"),
    ("SUD",  "10Y1001A1001A788"),
    ("SARD", "10Y1001A1001A74G"),
    ("SICI", "10Y1001A1001A75E"),
    ("CALA", "10Y1001C--00096J"),
]

# --- KS drift: calcolato su TUTTE le esogene, mostra le prime N per entità di drift ---
KS_TOP_N = 15
_METEO_CITIES = ("milano", "torino", "roma", "bologna", "bari", "palermo")
_METEO_MEANS = ("temperature_mean", "cloud_cover_mean", "wind_speed_mean", "precipitation_mean")
_COMMODITY_SPREADS = ("gas_eua_ratio", "spark_spread_proxy", "ttf_vol_z", "gas_regime")


def exogenous_cols(df) -> list:
    """Colonne ESOGENE (commodities+derivate, meteo, carico, _B16) su cui ha
    senso misurare il covariate drift. Esclude target, calendario e autoregressive."""
    cols = []
    for c in df.columns:
        if ("_price" in c) or ("_B16" in c) or (c == "market_load_MW") \
           or (c in _COMMODITY_SPREADS) or (c in _METEO_MEANS) \
           or any(c.startswith(city + "_") for city in _METEO_CITIES):
            cols.append(c)
    return cols


def zone_paths(zone_key: str) -> dict:
    base = f"{GDRIVE_ROOT}/{zone_key}"
    return {"dataset": f"{base}/dataset.parquet"}


# ============================================================
# EXCEL GME -> y della zona (stessa logica di mi_section.parse_zone_excel)
# ============================================================
def parse_zone_excel(uploaded_file, target_col: str) -> pd.DataFrame:
    raw = pd.read_excel(uploaded_file)
    raw.columns = [" ".join(str(c).split()) for c in raw.columns]   # collassa spazi doppi/interni

    if "Data" in raw.columns and "Periodo" in raw.columns:
        data = pd.to_datetime(raw["Data"], dayfirst=True, errors="coerce")
        periodo = pd.to_numeric(raw["Periodo"], errors="coerce")
        raw = raw.assign(Datetime=data + pd.to_timedelta((periodo - 1) * 15, unit="m"))
        dt_col = "Datetime"
    else:
        dt_col = "Datetime" if "Datetime" in raw.columns else raw.columns[0]
        raw[dt_col] = pd.to_datetime(raw[dt_col], dayfirst=True, errors="coerce")

    raw = raw.dropna(subset=[dt_col]).sort_values(dt_col).set_index(dt_col)
    raw.index.name = "Datetime"

    tgt = " ".join(str(target_col).split())
    if tgt in raw.columns:
        price_col = tgt
    else:
        norm = {c.lower(): c for c in raw.columns}
        price_col = norm.get(tgt.lower())
    if price_col is None:
        raise ValueError(f"Colonna '{target_col}' non trovata. Colonne: {list(raw.columns)}")

    def _num(s):
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")
        s = s.astype(str).str.strip()
        has_comma = s.str.contains(",", na=False)
        s = s.where(~has_comma, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
        return pd.to_numeric(s, errors="coerce")

    out = raw[[price_col]].copy()
    out.columns = [target_col]
    out[target_col] = _num(out[target_col])
    out = out[out[target_col].notna()]
    return out[~out.index.duplicated(keep="last")]


# ============================================================
# ESOGENE (una volta sola, condivise tra le zone)
# ============================================================
def download_exogenous(start_dt: pd.Timestamp, end_dt: pd.Timestamp,
                       secrets: dict) -> pd.DataFrame:
    """
    Scarica e assembla gli esogeni a 15 min sulla finestra [start_dt, end_dt].
    Riusa i client del PUN (functions/create_datasets.py). Ritorna un df
    indicizzato Datetime con: commodities+derivate, meteo (per città + medie),
    market_load_MW, prezzi zonali _B16 (+ lag).
    """
    from functions.create_datasets import (
        PUNFeatureEngineering, MeteoDownloader, TernaClient, EntsoeDownloader
    )

    s_meteo = start_dt.strftime("%Y-%m-%d"); e_meteo = end_dt.strftime("%Y-%m-%d")
    s_terna = start_dt.strftime("%d/%m/%Y"); e_terna = end_dt.strftime("%d/%m/%Y")

    # griglia 15 min di riferimento
    grid = pd.date_range(start_dt.floor("D"), end_dt.ceil("D"), freq=FREQ)
    base = pd.DataFrame(index=grid); base.index.name = "Datetime"

    # ---- commodities (giornaliere -> 15min via ffill) + derivate ----
    fe = PUNFeatureEngineering(start=s_meteo)
    comm = fe.build_commodity_dataset()                    # colonne: Data + *_price
    comm["Data"] = pd.to_datetime(comm["Data"]).dt.normalize()
    g = base.reset_index()
    g["Data"] = g["Datetime"].dt.normalize()
    comm15 = g.merge(comm, on="Data", how="left").drop(columns=["Data"])
    comm15 = comm15.set_index("Datetime").sort_index().ffill()
    comm15 = fe.add_commodity_features(comm15)             # lag/ret/vol/rolling + spreads
    exog = comm15

    # ---- meteo (per città + medie) ----
    meteo = MeteoDownloader()
    mdf = meteo.download_multi_city(LOCATIONS, s_meteo, e_meteo)
    mdf["Datetime"] = pd.to_datetime(mdf["Datetime"]).dt.floor("h")
    mdf = mdf.groupby("Datetime").mean(numeric_only=True)
    # medie aggregate (come aggregate_meteo del PUN)
    def _mean(sub):
        cols = [c for c in mdf.columns if sub in c]
        return mdf[cols].mean(axis=1) if cols else np.nan
    mdf["temperature_mean"] = _mean("temperature_2m")
    mdf["cloud_cover_mean"] = _mean("cloud_cover")
    mdf["wind_speed_mean"] = _mean("wind_speed_80m")
    prec = [c for c in mdf.columns if "precipitation" in c]
    if prec:
        mdf["precipitation_mean"] = mdf[prec].mean(axis=1)
    mdf = mdf.sort_index().resample(FREQ).ffill()
    exog = exog.join(mdf, how="left")

    # ---- Terna: market_load_MW ----
    tid = secrets.get("TERNA_CLIENT_ID", os.getenv("TERNA_CLIENT_ID", ""))
    tsec = secrets.get("TERNA_CLIENT_SECRET", os.getenv("TERNA_CLIENT_SECRET", ""))
    if tid and tsec:
        terna = TernaClient(client_id=tid, client_secret=tsec)
        mk = terna.clean_terna_df(terna.get_market_load(s_terna, e_terna))
        mk["Datetime"] = pd.to_datetime(mk["date"])
        mk = (mk.set_index("Datetime").sort_index()
                .resample(FREQ).ffill())
        keep = [c for c in mk.columns if c in ("market_load_MW",)]
        exog = exog.join(mk[keep], how="left")

    # ---- ENTSOE: prezzi zonali _B16 (+ lag) ----
    ent_token = secrets.get("ENTSOE_TOKEN", os.getenv("ENTSOE_TOKEN", ""))
    if ent_token:
        entsoe = EntsoeDownloader(
            token=ent_token, zones=ZONES_ENTSOE,
            start_date=start_dt.to_pydatetime(), end_date=end_dt.to_pydatetime(),
        )
        ent = entsoe.build_features()
        exog = exog.join(ent, how="left")

    return exog.sort_index()


# ============================================================
# FEATURE AUTOREGRESSIVE "LEGACY" (schema dei modelli attuali)
# ============================================================
def _it_holiday_flags(index):
    try:
        import holidays
        it = holidays.IT()
        dates = pd.DatetimeIndex(index.date)
        is_hol = pd.Series(dates, index=index).isin(it).to_numpy()
        is_tom = pd.Series(dates + pd.Timedelta(days=1), index=index).isin(it).to_numpy()
        is_yst = pd.Series(dates - pd.Timedelta(days=1), index=index).isin(it).to_numpy()
        wd = index.dayofweek
        bridge = ((wd == 0) & is_tom) | ((wd == 4) & is_yst)
    except ImportError:
        n = len(index); is_hol = np.zeros(n, bool); bridge = np.zeros(n, bool)
    return {"is_holiday": is_hol.astype(int), "is_bridge_day": np.asarray(bridge).astype(int)}


def compute_legacy_autoregressive(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Calcola calendario (DERIVA DALLA DATA) + autoregressive (DERIVA DAL TARGET)
    con le formule confermate. mi_ret_* e ret_* -> NaN (caveat). Mantiene le esogene."""
    df = df.copy()
    y = df[target_col]
    s = y.shift(1)
    idx = df.index

    # ---- CALENDARIO (tutte le feature "DERIVA DALLA DATA") ----
    df["hour"] = idx.hour
    df["minute"] = idx.minute
    df["Minute"] = idx.minute
    df["quarter"] = idx.minute // 15
    df["quarter_of_day"] = df["hour"] * 4 + df["quarter"]
    df["day_of_week"] = idx.dayofweek
    df["day_of_year"] = idx.dayofyear
    df["week_of_year"] = idx.isocalendar().week.astype(int).to_numpy()
    df["month"] = idx.month
    df["year"] = idx.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    hol = _it_holiday_flags(idx)
    df["is_holiday"] = hol["is_holiday"]
    df["is_bridge_day"] = hol["is_bridge_day"]

    # ---- AUTOREGRESSIVE (DERIVA DAL TARGET) ----
    df["lag_1d"] = y.shift(96)
    df["lag_2d"] = y.shift(192)
    df["lag_7d"] = y.shift(672)
    df["rolling_mean_4h"] = s.rolling(16, min_periods=4).mean()
    df["rolling_mean_24h"] = s.rolling(96, min_periods=24).mean()
    df["rolling_std_24h"] = s.rolling(96, min_periods=24).std()
    df["rolling_std_7d"] = s.rolling(672, min_periods=96).std()
    df["rolling_max_24h"] = s.rolling(96, min_periods=24).max()
    df["rolling_min_24h"] = s.rolling(96, min_periods=24).min()
    df["momentum_4h"] = y.shift(1) - y.shift(16)
    df["momentum_1d"] = y.shift(1) - y.shift(96)

    thr = df["rolling_std_24h"].quantile(0.80)
    df["high_vol_regime"] = (df["rolling_std_24h"] > thr).astype(int)

    # non ricostruibili / morte -> NaN (l'imputer del modello le riempie)
    for c in ["mi_ret_1h", "mi_ret_1d", "mi_ret_7d", "ret_1h", "ret_1d", "ret_7d"]:
        df[c] = np.nan

    return df


# ============================================================
# UPDATE DI UNA ZONA
# ============================================================
def update_zone(zone_key: str, y_new: pd.DataFrame, exog: pd.DataFrame,
                secrets: dict) -> tuple:
    """Aggiorna il dataset.parquet della zona su Drive. Ritorna (df_final, ks_df)."""
    from functions.create_datasets import ks_drift

    svc = gdrive.get_service_from_secrets(secrets)
    path = zone_paths(zone_key)["dataset"]
    target_col = ZONE_TARGET[zone_key]

    # storico
    df_hist = gdrive.read_parquet(svc, path)
    if not isinstance(df_hist.index, pd.DatetimeIndex):
        df_hist["Datetime"] = pd.to_datetime(df_hist["Datetime"])
        df_hist = df_hist.set_index("Datetime")
    df_hist = df_hist.sort_index()
    schema = list(df_hist.columns)                       # colonne da riprodurre
    last_date = df_hist.index.max()

    # serie prezzo completa (storico + nuova) per calcolare lag/rolling a cavallo
    y_full = pd.concat([df_hist[[target_col]], y_new]).sort_index()
    y_full = y_full[~y_full.index.duplicated(keep="last")]
    y_full = y_full.asfreq(FREQ)

    # righe nuove = oltre l'ultima data storica
    new_idx = y_full.index[y_full.index > last_date]
    if len(new_idx) == 0:
        return df_hist, pd.DataFrame()

    # costruisci il blocco nuovo: y + esogene + autoregressive
    block = y_full.copy()
    block = block.join(exog, how="left")                 # esogene (già a 15min)
    block = compute_legacy_autoregressive(block, target_col)

    df_new = block.loc[new_idx].copy()
    # riallinea ESATTAMENTE allo schema dello storico (mancanti -> NaN, extra -> via)
    df_new = df_new.reindex(columns=schema)

    # append + dedup
    df_final = pd.concat([df_hist, df_new], axis=0)
    df_final = df_final[~df_final.index.duplicated(keep="last")].sort_index()

    # salva su Drive
    local = "mi_tmp"; os.makedirs(local, exist_ok=True)
    lp = os.path.join(local, f"dataset_{zone_key}.parquet")
    df_final.reset_index().rename(columns={"index": "Datetime"}).set_index("Datetime").to_parquet(lp)
    gdrive.upload_file(svc, lp, path, overwrite=True)

    # KS drift: su TUTTE le esogene, poi tieni le prime N per entità di drift.
    # (ks_drift ordina già per ks_stat decrescente)
    ks_cols = [c for c in exogenous_cols(df_hist) if c in df_final.columns]
    try:
        ks_full = ks_drift(df_hist, df_final, ks_cols) if ks_cols else pd.DataFrame()
        ks_df = ks_full.head(KS_TOP_N) if not ks_full.empty else ks_full
    except Exception:
        ks_df = pd.DataFrame()

    return df_final, ks_df


# ============================================================
# UPDATE DI TUTTE LE ZONE (esogene scaricate UNA volta)
# ============================================================
def update_all_zones(excel_file, secrets: dict, only_zones=None, log=print) -> dict:
    """
    excel_file: Excel GME (Data/Ora/Periodo + colonne zona).
    secrets: dict-like con gcp_service_account, TERNA_*, ENTSOE_TOKEN.
    Ritorna {zone_key: {"rows_added": n, "last": ts, "ks": df}}.
    """
    today = pd.Timestamp.today().normalize()

    # una zona qualsiasi per leggere l'ultima data e fissare la finestra di lookback
    ref_zone = (only_zones or list(ZONE_TARGET.keys()))[0]
    svc = gdrive.get_service_from_secrets(secrets)
    df_ref = gdrive.read_parquet(svc, zone_paths(ref_zone)["dataset"])
    if not isinstance(df_ref.index, pd.DatetimeIndex):
        df_ref["Datetime"] = pd.to_datetime(df_ref["Datetime"]); df_ref = df_ref.set_index("Datetime")
    last_date = df_ref.index.max()
    start_dt = pd.Timestamp(last_date).floor("D") - pd.Timedelta(days=LOOKBACK_DAYS)
    end_dt = today

    log(f"Finestra esogeni: {start_dt.date()} → {end_dt.date()}")
    log("Scarico esogeni (una volta per tutte le zone)...")
    exog = download_exogenous(start_dt, end_dt, secrets)
    log(f"Esogeni pronti: {exog.shape[0]} righe, {exog.shape[1]} colonne")

    results = {}
    zones = list(ZONE_TARGET.keys()) if only_zones is None else only_zones
    for zk in zones:
        target_col = ZONE_TARGET[zk]
        try:
            y_new = parse_zone_excel(excel_file, target_col)
            df_final, ks_df = update_zone(zk, y_new, exog, secrets)
            added = int((df_final.index > last_date).sum())
            results[zk] = {"rows_added": added, "last": df_final.index.max(), "ks": ks_df}
            log(f"✅ {zk}: +{added} righe (fino a {df_final.index.max()})")
        except Exception as e:
            results[zk] = {"error": f"{type(e).__name__}: {e}"}
            log(f"❌ {zk}: {e}")

    return results
