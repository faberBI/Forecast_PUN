import os
import io
import json
import joblib
import dropbox
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# ==========================================================
# DEFAULT CONFIG
# ==========================================================

FORECAST_STEPS = 96
FORECAST_FREQ = "15min"
LOOKBACK_DAYS = 10

# ==========================================================
# FORECAST ALL MI MODELS
# ==========================================================
def normalize_hist_df(df_hist):
    """
([np.inf, -np.inf], np.nan)    Normalizza storico modello:
    df = df.ffill()

    return df
    - Datetime come index
    - frequenza 15min
    - ordinamento
    - ffill minimo
    """

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

def add_temporal_features_like_training(df):
    """
    Aggiunge le feature temporali esattamente come nella tua classe:
    hour, minute, quarter, quarter_of_day, day_of_week,
    day_of_year, week_of_year, month, year, is_weekend,
    hour_sin/cos, dow_sin/cos, doy_sin/cos, month_sin/cos,
    is_holiday, is_bridge_day.
    """

    df = df.copy()

    if "Datetime" not in df.columns:
        df["Datetime"] = pd.to_datetime(df.index)

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df["Data"] = df["Datetime"].dt.normalize()

    df["hour"] = df["Datetime"].dt.hour
    df["minute"] = df["Datetime"].dt.minute
    df["quarter"] = df["minute"] // 15

    df["quarter_of_day"] = (
        df["hour"] * 4 + df["quarter"]
    )

    df["day_of_week"] = df["Datetime"].dt.dayofweek
    df["day_of_year"] = df["Datetime"].dt.dayofyear

    df["week_of_year"] = (
        df["Datetime"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    df["month"] = df["Datetime"].dt.month
    df["year"] = df["Datetime"].dt.year

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["quarter_of_day"] / 96
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["quarter_of_day"] / 96
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["doy_sin"] = np.sin(
        2 * np.pi * df["day_of_year"] / 365
    )

    df["doy_cos"] = np.cos(
        2 * np.pi * df["day_of_year"] / 365
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    it_holidays = holidays.IT()

    df["is_holiday"] = (
        df["Data"]
        .dt.date
        .isin(it_holidays)
        .astype(int)
    )

    df["is_bridge_day"] = (
        df["Data"]
        .dt.date
        .apply(is_bridge_day_italy)
        .astype(int)
    )

    return df

def prepare_meteo_range(
    meteo,
    locations,
    start_date,
    end_date,
    full_index
):
    """
    Scarica meteo con MeteoDownloader e lo porta a 15min.

    Stessa logica:
    - download_multi_city
    - floor("h")
    - groupby mean
    - aggregate_meteo
    - reindex 15min ffill/bfill
    """

    df = meteo.download_multi_city(
        locations,
        start_date,
        end_date
    )

    df = df.copy()

    df["Datetime"] = (
        pd.to_datetime(df["Datetime"])
        .dt.floor("h")
    )

    df = (
        df
        .groupby("Datetime")
        .mean(numeric_only=True)
        .reset_index()
    )

    df = aggregate_meteo(df)

    df["Datetime"] = pd.to_datetime(df["Datetime"])

    df = (
        df
        .sort_values("Datetime")
        .set_index("Datetime")
    )

    df = df[~df.index.duplicated(keep="last")]

    df = (
        df
        .reindex(full_index)
        .ffill()
        .bfill()
    )

    return df

def prepare_terna_actual_range(
    terna,
    start_date_terna,
    end_date_terna,
    zone="Italy",
    shift_steps=1
):
    """
    Scarica Terna ACTUAL fino a oggi.

    Logica:
    - total_load
    - market_load
    - forecast_load
    - generation Wind/Solar/Hydro actual
    - conversione MW -> GWh
    - feature:
        load_ramp_1h
        load_forecast_error
        renewable_share
        net_load_proxy
    - shift anti leakage
    """

    def clean(df):
        if df is None or df.empty:
            return pd.DataFrame()

        return terna.clean_terna_df(df)

    dfs = []

    # ======================================================
    # TOTAL LOAD
    # ======================================================

    try:
        load = clean(
            terna.get_total_load(
                start_date_terna,
                end_date_terna,
                zone=zone
            )
        )

        if not load.empty:
            dfs.append(load)

    except Exception as e:
        print(f"⚠️ Terna total_load non disponibile: {e}")

    # ======================================================
    # MARKET LOAD
    # ======================================================

    try:
        market = clean(
            terna.get_market_load(
                start_date_terna,
                end_date_terna,
                zone=zone
            )
        )

        if not market.empty:
            dfs.append(market)

    except Exception as e:
        print(f"⚠️ Terna market_load non disponibile: {e}")

    # ======================================================
    # FORECAST LOAD
    # ======================================================

    try:
        forecast_load = clean(
            terna.get_forecast_load(
                start_date_terna,
                end_date_terna,
                zone=zone
            )
        )

        if not forecast_load.empty:
            dfs.append(forecast_load)

    except Exception as e:
        print(f"⚠️ Terna forecast_load non disponibile: {e}")

    # ======================================================
    # GENERATION WIND / SOLAR / HYDRO
    # ======================================================

    try:
        wind = clean(
            terna.get_generation(
                start_date_terna,
                end_date_terna,
                "Wind"
            )
        ).rename(
            columns={"actual_generation_MW": "wind_generation_MW"}
        )

        if not wind.empty:
            dfs.append(wind)

    except Exception as e:
        print(f"⚠️ Terna Wind generation non disponibile: {e}")

    try:
        solar = clean(
            terna.get_generation(
                start_date_terna,
                end_date_terna,
                "Photovoltaic"
            )
        ).rename(
            columns={"actual_generation_MW": "solar_generation_MW"}
        )

        if not solar.empty:
            dfs.append(solar)

    except Exception as e:
        print(f"⚠️ Terna Solar generation non disponibile: {e}")

    try:
        hydro = clean(
            terna.get_generation(
                start_date_terna,
                end_date_terna,
                "Hydro"
            )
        ).rename(
            columns={"actual_generation_MW": "hydro_generation_MW"}
        )

        if not hydro.empty:
            dfs.append(hydro)

    except Exception as e:
        print(f"⚠️ Terna Hydro generation non disponibile: {e}")

    if len(dfs) == 0:
        return pd.DataFrame()

    # ======================================================
    # MERGE OUTER SU date
    # ======================================================

    base = dfs[0]

    for tmp in dfs[1:]:
        base = base.merge(
            tmp,
            on="date",
            how="outer"
        )

    base = base.loc[:, ~base.columns.duplicated()]
    base["date"] = pd.to_datetime(base["date"]).dt.floor("15min")
    base = base.sort_values("date")

    # ======================================================
    # FORCE NUMERIC
    # ======================================================

    for c in base.columns:
        if c != "date":
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # ======================================================
    # GENERATION GWh
    # ======================================================

    if "wind_generation_MW" in base.columns:
        base["actual_generation_GWh"] = mw_to_gwh_15min(
            base["wind_generation_MW"]
        )

    if "solar_generation_MW" in base.columns:
        base["actual_generation_GWh_solar"] = mw_to_gwh_15min(
            base["solar_generation_MW"]
        )

    if "hydro_generation_MW" in base.columns:
        base["actual_generation_GWh_hydro"] = mw_to_gwh_15min(
            base["hydro_generation_MW"]
        )

    # ======================================================
    # FEATURES TERNA
    # ======================================================

    if "total_load_MW" in base.columns:
        base["load_ramp_1h"] = (
            base["total_load_MW"]
            .diff(4)
        )

    if (
        "forecast_total_load_MW" in base.columns
        and "total_load_MW" in base.columns
    ):
        base["load_forecast_error"] = (
            base["forecast_total_load_MW"]
            - base["total_load_MW"]
        )

    if (
        "actual_generation_GWh_solar" in base.columns
        and "actual_generation_GWh_hydro" in base.columns
        and "total_load_MW" in base.columns
    ):
        base["renewable_share"] = (
            base["actual_generation_GWh_solar"]
            + base["actual_generation_GWh_hydro"]
        ) / (base["total_load_MW"] + 1e-9)

    if (
        "total_load_MW" in base.columns
        and "actual_generation_GWh" in base.columns
    ):
        base["net_load_proxy"] = (
            base["total_load_MW"]
            - base["actual_generation_GWh"]
        )

    base["Datetime"] = pd.to_datetime(base["date"])

    keep_cols = [
        "Datetime",
        "date",
        "total_load_MW",
        "market_load_MW",
        "forecast_total_load_MW",
        "forecast_market_load_MW",
        "actual_generation_GWh",
        "actual_generation_GWh_solar",
        "actual_generation_GWh_hydro",
        "load_ramp_1h",
        "load_forecast_error",
        "renewable_share",
        "net_load_proxy"
    ]

    base = base[
        [c for c in keep_cols if c in base.columns]
    ].copy()

    base = (
        base
        .drop_duplicates(subset=["Datetime"])
        .sort_values("Datetime")
        .set_index("Datetime")
    )

    base = base.replace([np.inf, -np.inf], np.nan)

    # shift anti leakage sulle feature Terna
    base = shift_terna_only(
        base,
        shift_steps=shift_steps
    )

    return base

def extend_terna_to_full_index(terna_df, full_index):
    """
    Terna actual disponibile fino a oggi.
    Per domani: ffill ultimo valore noto.
    """

    if terna_df is None or terna_df.empty:
        return pd.DataFrame(index=full_index)

    df = terna_df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df = (
        df
        .reindex(full_index)
        .ffill()
        .bfill()
    )

    return df

def prepare_commodity_features_range(
    full_index,
    start_date=None
):
    """
    Scarica commodity con PUNFeatureEngineering.build_commodity_dataset
    e applica add_commodity_features.

    Poi porta tutto a 15min e ffill.
    """

    try:
        if start_date is None:
            start_date = full_index.min().strftime("%Y-%m-%d")

        fe = PUNFeatureEngineering(
            start=start_date,
            pun_col="dummy"
        )

        commodities = fe.build_commodity_dataset()

        if commodities is None or commodities.empty:
            return pd.DataFrame(index=full_index)

        commodities = commodities.copy()

        commodities["Data"] = pd.to_datetime(
            commodities["Data"]
        ).dt.normalize()

        base = pd.DataFrame(index=full_index)
        base["Datetime"] = base.index
        base["Data"] = base.index.normalize()

        base = (
            base
            .reset_index(drop=True)
            .merge(
                commodities,
                on="Data",
                how="left"
            )
            .sort_values("Datetime")
            .ffill()
        )

        base = fe.add_commodity_features(base)

        base = (
            base
            .set_index("Datetime")
            .drop(columns=["Data"], errors="ignore")
            .reindex(full_index)
            .ffill()
            .bfill()
        )

        return base

    except Exception as e:
        print(f"⚠️ Commodity features non disponibili: {e}")
        return pd.DataFrame(index=full_index)
        
def add_target_features_like_training(df, target_col):
    """
    Replica la parte target-based della tua PUNFeatureEngineering:

    - lag_15m, lag_30m, lag_1h, lag_2h, lag_4h
    - lag_1d, lag_2d, lag_7d
    - pun_ret_1h, pun_ret_1d, pun_ret_7d
    - rolling_mean_4h, rolling_mean_24h
    - rolling_std_24h, rolling_std_7d
    - rolling_max_24h, rolling_min_24h
    - high_vol_regime
    - momentum_4h, momentum_1d

    Nota:
    sul futuro target_col è NaN. Quindi:
    - lag_1d/2d/7d sono noti se cadono nello storico.
    - lag brevi oltre i primi step diventano NaN e poi vengono ffillati.
    - i lags veri del ForecasterDirect stanno comunque dentro last_window.
    """

    df = df.copy()

    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' non presente nel dataframe.")

    y = pd.to_numeric(
        df[target_col],
        errors="coerce"
    )

    # ======================================================
    # LAGS
    # ======================================================

    lag_map = {
        "lag_15m": 1,
        "lag_30m": 2,
        "lag_1h": 4,
        "lag_2h": 8,
        "lag_4h": 16,
        "lag_1d": 96,
        "lag_2d": 96 * 2,
        "lag_7d": 96 * 7,
    }

    for col_name, lag in lag_map.items():
        df[col_name] = y.shift(lag)

    # ======================================================
    # RETURNS - NOMI ESATTI DELLA TUA VERSIONE
    # ======================================================

    df["pun_ret_1h"] = (
        y
        .pct_change(4)
        .shift(1)
    )

    df["pun_ret_1d"] = (
        y
        .pct_change(96)
        .shift(1)
    )

    df["pun_ret_7d"] = (
        y
        .pct_change(96 * 7)
        .shift(1)
    )

    # ======================================================
    # ROLLING FEATURES
    # ======================================================

    df["rolling_mean_4h"] = (
        y
        .shift(1)
        .rolling(16)
        .mean()
    )

    df["rolling_mean_24h"] = (
        y
        .shift(1)
        .rolling(96)
        .mean()
    )

    df["rolling_std_24h"] = (
        y
        .shift(1)
        .rolling(96)
        .std()
    )

    df["rolling_std_7d"] = (
        y
        .shift(1)
        .rolling(96 * 7)
        .std()
    )

    df["rolling_max_24h"] = (
        y
        .shift(1)
        .rolling(96)
        .max()
    )

    df["rolling_min_24h"] = (
        y
        .shift(1)
        .rolling(96)
        .min()
    )

    # ======================================================
    # REGIME
    # ======================================================

    vol_24h = (
        y
        .shift(1)
        .rolling(96)
        .std()
    )

    threshold = vol_24h.quantile(0.8)

    df["high_vol_regime"] = (
        vol_24h > threshold
    ).astype(int)

    # ======================================================
    # MOMENTUM
    # ======================================================

    df["momentum_4h"] = (
        y.shift(1)
        - y.shift(16)
    )

    df["momentum_1d"] = (
        y.shift(1)
        - y.shift(96)
    )

    return df
    
def build_forecast_feature_frame_same_features(
    df_hist,
    target_col,
    future_index,
    selected_exog,
    meteo=None,
    locations=None,
    terna=None,
    terna_zone="Italy",
    lookback_days=10,
    use_commodities=True,
    terna_shift_steps=1
):

    """
    Costruisce frame storico + futuro usando le stesse feature
    della tua PUNFeatureEngineering.

    Output:
    - exog_future = solo future_index e selected_exog
    - feature_frame = debug completo storico+futuro
    """

    hist = normalize_hist_df(df_hist)

    if target_col not in hist.columns:
        raise ValueError(
            f"Target '{target_col}' non presente nello storico."
        )

    hist_last_dt = hist.index.max()

    full_start = max(
        hist.index.min(),
        hist_last_dt - pd.Timedelta(days=lookback_days)
    )

    full_end = future_index[-1]

    full_index = pd.date_range(
        start=full_start,
        end=full_end,
        freq="15min"
    )

    # ======================================================
    # BASE STORICO + FUTURO
    # ======================================================

    base = pd.DataFrame(index=full_index)
    base["Datetime"] = base.index

    hist_reindexed = (
        hist
        .reindex(full_index)
        .ffill()
    )

    # porto tutte le colonne storiche disponibili
    for col in hist_reindexed.columns:
        base[col] = hist_reindexed[col]

    # IMPORTANTISSIMO:
    # target futuro non deve essere ffillato.
    # Dopo ultimo dato storico deve restare NaN.
    base.loc[
        base.index > hist_last_dt,
        target_col
    ] = np.nan

    # ======================================================
    # TEMPORAL FEATURES
    # ======================================================

    base = add_temporal_features_like_training(base)

    base = base.set_index("Datetime", drop=False)
    base = base.loc[:, ~base.columns.duplicated()]

    # ======================================================
    # METEO
    # ======================================================

    if meteo is not None and locations is not None:

        try:
            start_meteo = full_start.strftime("%Y-%m-%d")
            end_meteo = full_end.strftime("%Y-%m-%d")

            meteo_df = prepare_meteo_range(
                meteo=meteo,
                locations=locations,
                start_date=start_meteo,
                end_date=end_meteo,
                full_index=full_index
            )

            base = pd.concat(
                [base, meteo_df],
                axis=1
            )

            base = base.loc[:, ~base.columns.duplicated()]

        except Exception as e:
            print(f"⚠️ Meteo non integrato: {e}")

    # ======================================================
    # TERNA ACTUAL FINO A OGGI
    # ======================================================

    if terna is not None:

        try:
            start_terna = full_start.strftime("%d/%m/%Y")
            end_terna = pd.Timestamp.today().strftime("%d/%m/%Y")

            terna_hist = prepare_terna_actual_range(
                terna=terna,
                start_date_terna=start_terna,
                end_date_terna=end_terna,
                zone=terna_zone,
                shift_steps=terna_shift_steps
            )

            terna_full = extend_terna_to_full_index(
                terna_df=terna_hist,
                full_index=full_index
            )

            base = pd.concat(
                [base, terna_full],
                axis=1
            )

            base = base.loc[:, ~base.columns.duplicated()]

        except Exception as e:
            print(f"⚠️ Terna non integrata: {e}")

    # ======================================================
    # COMMODITIES
    # ======================================================

    if use_commodities:

        try:
            commodities = prepare_commodity_features_range(
                full_index=full_index,
                start_date=full_start.strftime("%Y-%m-%d")
            )

            base = pd.concat(
                [base, commodities],
                axis=1
            )

            base = base.loc[:, ~base.columns.duplicated()]

        except Exception as e:
            print(f"⚠️ Commodities non integrate: {e}")

    # ======================================================
    # TARGET-BASED FEATURES
    # ======================================================

    base = add_target_features_like_training(
        df=base,
        target_col=target_col
    )

    base = base.loc[:, ~base.columns.duplicated()]
    base = base.replace([np.inf, -np.inf], np.nan)

    # ======================================================
    # EXOG FUTURE
    # ======================================================

    exog_future = base.loc[
        future_index,
        :
    ].copy()

    # ======================================================
    # GARANTISCO selected_exog
    # ======================================================

    for col in selected_exog:

        if col in exog_future.columns:
            continue

        # fallback da storico originale
        if col in hist.columns and hist[col].dropna().shape[0] > 0:
            exog_future[col] = (
                pd.to_numeric(hist[col], errors="coerce")
                .dropna()
                .iloc[-1]
            )

        # fallback da base completo
        elif col in base.columns and base[col].dropna().shape[0] > 0:
            exog_future[col] = (
                pd.to_numeric(base[col], errors="coerce")
                .dropna()
                .iloc[-1]
            )

        else:
            exog_future[col] = 0.0

    exog_future = exog_future[selected_exog].copy()

    # numeric cleanup
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

    return exog_future, base


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
