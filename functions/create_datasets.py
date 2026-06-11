import pandas as pd
import numpy as np
import holidays
import yfinance as yf
import dropbox
import streamlit as st

class PUNFeatureEngineering:

    def __init__(
        self,
        start="2018-01-01",
        pun_col="PUN"
    ):

        self.start = start
        self.pun_col = pun_col

        self.it_holidays = holidays.IT()

    # =====================================================
    # DOWNLOAD COMMODITIES
    # =====================================================

    def download_close(
        self,
        ticker,
        column_name
    ):

        df = yf.download(
            ticker,
            start=self.start,
            auto_adjust=True,
            progress=False
        )

        df = df[["Close"]].copy()

        df.columns = [column_name]

        df.index = pd.to_datetime(df.index)

        return df

    def build_commodity_dataset(self):

        tickers = {

            "TTF=F": "ttf_price",
            "NG=F": "ng_price",
            "BZ=F": "brent_price",
            "CL=F": "wti_price",
            "MTF=F": "coal_price",
            "CO2.L": "eua_price"
        }

        dfs = []

        for ticker, col in tickers.items():

            try:

                tmp = self.download_close(
                    ticker,
                    col
                )

                dfs.append(tmp)

            except Exception as e:

                print(f"Errore download {ticker}: {e}")

        df = pd.concat(
            dfs,
            axis=1
        )

        df = (
            df
            .sort_index()
            .ffill()
        )

        df = df.reset_index()


        # ✅ FIX UNIVERSALE
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "Data"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Data"})
        else:
            df = df.rename(columns={df.columns[0]: "Data"})

        df["Data"] = pd.to_datetime(df["Data"]).dt.normalize()


        return df

    # =====================================================
    # BRIDGE DAYS
    # =====================================================

    def is_bridge_day(self, date):

        ts = pd.Timestamp(date)

        weekday = ts.weekday()

        prev_day = (
            ts - pd.Timedelta(days=1)
        ).date()

        next_day = (
            ts + pd.Timedelta(days=1)
        ).date()

        holiday_dates = set(
            self.it_holidays.keys()
        )

        # lunedì prima di festivo
        if weekday == 0 and next_day in holiday_dates:
            return 1

        # venerdì dopo festivo
        if weekday == 4 and prev_day in holiday_dates:
            return 1

        return 0

    # =====================================================
    # ADD COMMODITY FEATURES
    # =====================================================

    def add_commodity_features(
        self,
        df
    ):

        commodity_cols = [

            "ttf_price",
            "ng_price",
            "brent_price",
            "coal_price",
            "eua_price"
        ]

        for col in commodity_cols:

            if col not in df.columns:
                continue

            # lag daily
            df[f"{col}_lag_1d"] = (
                df[col]
                .shift(96)
            )

            df[f"{col}_lag_7d"] = (
                df[col]
                .shift(96 * 7)
            )

            # returns
            df[f"{col}_ret_1d"] = (
                df[col]
                .pct_change(96)
                .shift(1)
            )

            df[f"{col}_ret_7d"] = (
                df[col]
                .pct_change(96 * 7)
                .shift(1)
            )

            # volatility
            df[f"{col}_vol_7d"] = (
                df[col]
                .shift(1)
                .rolling(96 * 7)
                .std()
            )

            # rolling mean
            df[f"{col}_rolling_mean_7d"] = (
                df[col]
                .shift(1)
                .rolling(96 * 7)
                .mean()
            )

        # =================================================
        # SPREADS / REGIMES
        # =================================================

        if (
            "ttf_price" in df.columns
            and "eua_price" in df.columns
        ):

            df["gas_eua_ratio"] = (
                df["ttf_price"].shift(1)
                / df["eua_price"].shift(1)
            )

            df["spark_spread_proxy"] = (
                df["ttf_price"].shift(1)
                - df["eua_price"].shift(1) * 0.2
            )

            ttf_vol_short = (
                df["ttf_price"]
                .shift(1)
                .rolling(96 * 7)
                .std()
            )

            ttf_vol_long = (
                df["ttf_price"]
                .shift(1)
                .rolling(96 * 30)
                .std()
            )

            df["ttf_vol_z"] = (
                ttf_vol_short / ttf_vol_long
            )

            df["gas_regime"] = (
                df["ttf_vol_z"] > 1.2
            ).astype(int)

        return df

    # =====================================================
    # MAIN FEATURE ENGINEERING
    # =====================================================

    def prepare_dataset(
        self,
        df,
        merge_commodities=True
    ):

        df = df.copy()

        # =================================================
        # PARSE DATE
        # =================================================

        df["Data"] = pd.to_datetime(
            df["Data"],
            dayfirst=True
        )

        # =================================================
        # CLEAN NUMERIC
        # =================================================

        numeric_cols = [self.pun_col]

        for col in numeric_cols:

            if col in df.columns:

                if df[col].dtype == object:

                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", ".")
                        .astype(float)
                    )

        # =================================================
        # DATETIME
        # =================================================

        df["minutes_from_midnight"] = (
            (df["Periodo"] - 1) * 15
        )

        df["Datetime"] = (
            df["Data"] +
            pd.to_timedelta(
                df["minutes_from_midnight"],
                unit="m"
            )
        )

        df = (
            df
            .sort_values("Datetime")
            .reset_index(drop=True)
        )

        # =================================================
        # TEMPORAL FEATURES
        # =================================================

        df["hour"] = (
            df["Datetime"].dt.hour
        )

        df["minute"] = (
            df["Datetime"].dt.minute
        )

        df["quarter"] = (
            df["minute"] // 15
        )

        df["quarter_of_day"] = (
            df["hour"] * 4 + df["quarter"]
        )

        df["day_of_week"] = (
            df["Datetime"].dt.dayofweek
        )

        df["day_of_year"] = (
            df["Datetime"].dt.dayofyear
        )

        df["week_of_year"] = (
            df["Datetime"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        df["month"] = (
            df["Datetime"].dt.month
        )

        df["year"] = (
            df["Datetime"].dt.year
        )

        df["is_weekend"] = (
            df["day_of_week"] >= 5
        ).astype(int)

        # =================================================
        # CYCLICAL FEATURES
        # =================================================

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

        # =================================================
        # HOLIDAYS
        # =================================================

        df["is_holiday"] = (
            df["Data"]
            .dt.date
            .isin(self.it_holidays)
            .astype(int)
        )

        df["is_bridge_day"] = (
            df["Data"]
            .dt.date
            .apply(self.is_bridge_day)
            .astype(int)
        )

        # =================================================
        # LAGS
        # =================================================

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

            df[col_name] = (
                df[self.pun_col]
                .shift(lag)
            )

        # =================================================
        # RETURNS
        # =================================================

        df["pun_ret_1h"] = (
            df[self.pun_col]
            .pct_change(4)
            .shift(1)
        )

        df["pun_ret_1d"] = (
            df[self.pun_col]
            .pct_change(96)
            .shift(1)
        )

        df["pun_ret_7d"] = (
            df[self.pun_col]
            .pct_change(96 * 7)
            .shift(1)
        )

        # =================================================
        # ROLLING FEATURES
        # =================================================

        df["rolling_mean_4h"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(16)
            .mean()
        )

        df["rolling_mean_24h"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(96)
            .mean()
        )

        df["rolling_std_24h"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(96)
            .std()
        )

        df["rolling_std_7d"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(96 * 7)
            .std()
        )

        df["rolling_max_24h"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(96)
            .max()
        )

        df["rolling_min_24h"] = (
            df[self.pun_col]
            .shift(1)
            .rolling(96)
            .min()
        )

        # =================================================
        # REGIME
        # =================================================

        vol_24h = (
            df[self.pun_col]
            .shift(1)
            .rolling(96)
            .std()
        )

        threshold = vol_24h.quantile(0.8)

        df["high_vol_regime"] = (
            vol_24h > threshold
        ).astype(int)

        # =================================================
        # MOMENTUM
        # =================================================

        df["momentum_4h"] = (
            df[self.pun_col].shift(1)
            - df[self.pun_col].shift(16)
        )

        df["momentum_1d"] = (
            df[self.pun_col].shift(1)
            - df[self.pun_col].shift(96)
        )

        # =================================================
        # MERGE COMMODITIES
        # =================================================

        if merge_commodities:

            commodities = (
                self.build_commodity_dataset()
            )

            df = df.merge(
                commodities,
                on="Data",
                how="left"
            )

            df = (
                df
                .sort_values("Datetime")
                .ffill()
            )

            df = self.add_commodity_features(df)

        # =================================================
        # TARGET
        # =================================================

        df["target"] = (
            df[self.pun_col]
            .shift(-96)
        )

        # =================================================
        # CLEANUP
        # =================================================

        df.drop(
            columns=["minutes_from_midnight"],
            inplace=True,
            errors="ignore"
        )

        return df
    

import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests


class MeteoDownloader:

    def __init__(self):

        # =====================================================
        # CLIENT SETUP (cache + retry)
        # =====================================================

        cache_session = requests_cache.CachedSession(
            ".cache",
            expire_after=3600
        )

        retry_session = retry(
            cache_session,
            retries=5,
            backoff_factor=0.2
        )

        self.client = openmeteo_requests.Client(
            session=retry_session
        )

        self.url = "https://api.open-meteo.com/v1/forecast"

    # =====================================================
    # MAIN DOWNLOAD FUNCTION
    # =====================================================

    def download_weather(
        self,
        latitude,
        longitude,
        start_date,
        end_date,
        timezone="Europe/Rome"
    ):

        hourly_vars = [

            "temperature_2m",
            "cloud_cover",

            "wind_speed_10m",
            "wind_speed_80m",
            "wind_speed_120m",
            "wind_speed_180m",

            "apparent_temperature",

            "precipitation",
            "precipitation_probability",
            "rain",

            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",

            "soil_temperature_0cm",
            "soil_temperature_6cm",
            "soil_temperature_18cm"
        ]

        params = {

            "latitude": latitude,
            "longitude": longitude,

            "hourly": hourly_vars,

            "timezone": timezone,

            "start_date": start_date,
            "end_date": end_date
        }

        responses = self.client.weather_api(
            self.url,
            params=params
        )

        response = responses[0]
        hourly = response.Hourly()

        # =====================================================
        # TIME INDEX
        # =====================================================

        time_index = pd.date_range(

            start=pd.to_datetime(
                hourly.Time(),
                unit="s",
                utc=True
            ),

            end=pd.to_datetime(
                hourly.TimeEnd(),
                unit="s",
                utc=True
            ),

            freq=pd.Timedelta(
                seconds=hourly.Interval()
            ),

            inclusive="left"
        ).tz_convert(
            response.Timezone().decode()
        )

        # =====================================================
        # BUILD DATAFRAME
        # =====================================================

        data = {
            "Datetime": time_index
        }

        for i, var in enumerate(hourly_vars):

            data[var] = (
                hourly
                .Variables(i)
                .ValuesAsNumpy()
            )

        df = pd.DataFrame(data)

        # =====================================================
        # CLEAN TIMEZONE
        # =====================================================

        df["Datetime"] = (
            df["Datetime"]
            .dt.tz_localize(None)
        )

        return df

    # =====================================================
    # MULTI-CITY AGGREGATION (VERY USEFUL FOR PUN)
    # =====================================================

    def download_multi_city(
        self,
        locations,
        start_date,
        end_date
    ):

        dfs = []

        for loc in locations:

            df = self.download_weather(
                latitude=loc["lat"],
                longitude=loc["lon"],
                start_date=start_date,
                end_date=end_date
            )

            df = df.add_prefix(loc["name"] + "_")

            df = df.rename(
                columns={
                    f"{loc['name']}_Datetime": "Datetime"
                }
            )

            dfs.append(df)

        merged = dfs[0]

        for df in dfs[1:]:

            merged = merged.merge(
                df,
                on="Datetime",
                how="outer"
            )

        return merged


import requests
import pandas as pd
import time
from datetime import datetime, timedelta


class TernaClient:

    def __init__(self, client_id, client_secret):

        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://api.terna.it"
        self.token = None

        self.get_token()

    # =================================================
    # TOKEN
    # =================================================

    def get_token(self):

        url = f"{self.base_url}/public-api/access-token"

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials"
        }

        r = requests.post(url, data=payload)

        if r.status_code != 200:
            raise ValueError(r.text)

        self.token = r.json()["access_token"]
        print("TOKEN OK")

    # =================================================
    # HEADERS
    # =================================================

    def headers(self):

        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json"
        }

    # =================================================
    # SPLIT DATE (API LIMIT SAFE)
    # =================================================

    def split_dates(self, start, end, max_days=60):

        start = datetime.strptime(start, "%d/%m/%Y")
        end = datetime.strptime(end, "%d/%m/%Y")

        chunks = []

        while start <= end:

            chunk_end = min(start + timedelta(days=max_days - 1), end)

            chunks.append((
                start.strftime("%d/%m/%Y"),
                chunk_end.strftime("%d/%m/%Y")
            ))

            start = chunk_end + timedelta(days=1)

        return chunks

    # =================================================
    # SAFE REQUEST
    # =================================================

    def mw_to_gwh(self, df, col):

        # 15 minuti = 0.25 ore
        df[col.replace("_MW", "_GWh")] = df[col] * 0.25 / 1000

        return df

    def safe_get(self, url, params, max_retries=5):

        for i in range(max_retries):

            r = requests.get(
                url,
                headers=self.headers(),
                params=params
            )

            if r.status_code == 200:
                return r.json()

            if r.status_code == 401:
                self.get_token()
                continue

            if r.status_code >= 500:
                time.sleep(2 * (i + 1))
                continue

            print(r.text)
            time.sleep(2)

        raise ValueError("API failed")

    # =================================================
    # NORMALIZER (FOR 15-MIN DATA)
    # =================================================

    def normalize_15min(self, js, key, value_col):

        if not js or key not in js:
            return pd.DataFrame()

        df = pd.DataFrame(js[key])

        if df.empty:
            return df

        # ⚡ FORCE 15-MIN DATETIME
        df["date"] = pd.to_datetime(df["date"])

        # align to 15-min grid
        df["date"] = df["date"].dt.floor("15min")

        if value_col in df.columns:
            df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

        return df

    # =================================================
    # TOTAL LOAD (15 MIN ALIGNED)
    # =================================================

    def get_total_load(self, start, end, zone="Italy"):

        url = f"{self.base_url}/load/v2.0/total-load"
        dfs = []

        for s, e in self.split_dates(start, end):

            js = self.safe_get(url, {
                "dateFrom": s,
                "dateTo": e,
                "biddingZone": zone
            })

            df = self.normalize_15min(js, "total_load", "total_load_MW")

            if not df.empty:
                dfs.append(df)

            time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        return (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )

    def get_forecast_load(self, start, end, zone="Italy"):

        url = f"{self.base_url}/load/v2.0/forecast-load"
        dfs = []

        for s, e in self.split_dates(start, end):

            js = self.safe_get(url, {
                "dateFrom": s,
                "dateTo": e,
                "biddingZone": zone
            })

            df = self.normalize_15min(
                js,
                "forecast_load",
                "forecast_total_load_MW"
            )

            if not df.empty:
                dfs.append(df)

            time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        return (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )


    # =================================================
    # MARKET LOAD (15 MIN)
    # =================================================

    def get_market_load(self, start, end, zone="Italy"):

        url = f"{self.base_url}/load/v2.0/market-load"
        dfs = []

        for s, e in self.split_dates(start, end):

            js = self.safe_get(url, {
                "dateFrom": s,
                "dateTo": e,
                "biddingZone": zone
            })

            df = self.normalize_15min(js, "market_load", "market_load_MW")

            if not df.empty:
                dfs.append(df)

            time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        return (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )

    # =================================================
    # GENERATION (FORCED 15 MIN)
    # =================================================

    def get_generation(self, start, end, gtype="Wind"):
    
            url = f"{self.base_url}/generation/v2.0/actual-generation"
            dfs = []
    
            for s, e in self.split_dates(start, end):
            
                js = self.safe_get(url, {
                    "dateFrom": s,
                    "dateTo": e,
                    "type": gtype
                })
    
                df = self.normalize_15min(
                    js,
                    "actual_generation",
                    "actual_generation_MW"
                )
    
                if not df.empty:
                    dfs.append(df)
    
                time.sleep(1)
    
            if not dfs:
                return pd.DataFrame()
    
            return (
                pd.concat(dfs)
                .drop_duplicates()
                .sort_values("date")
                .reset_index(drop=True)
            )


    # =================================================
    # ENERGY BALANCE (15 MIN)
    # =================================================

    def get_energy_balance(self, start, end, gtype="Wind"):

        url = f"{self.base_url}/generation/v2.0/energy-balance"
        dfs = []

        for s, e in self.split_dates(start, end):

            js = self.safe_get(url, {
                "dateFrom": s,
                "dateTo": e,
                "type": gtype
            })

            df = self.normalize_15min(
                js,
                "energy_balance",
                "energy_balance_MW"
            )

            if not df.empty:
                dfs.append(df)

            time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        return (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )

    # =================================================
    # RENEWABLES (15 MIN SAFE VERSION)
    # =================================================

    def get_renewable_generation(self, start, end, sources=None):

        url = f"{self.base_url}/generation/v2.0/renewable-generation"
        dfs = []

        if sources is None:
            sources = [
                "Wind",
                "Hydro",
                "Photovoltaic",
                "Geothermal"
            ]

        for s, e in self.split_dates(start, end):

            for source in sources:

                js = self.safe_get(url, {
                    "dateFrom": s,
                    "dateTo": e,
                    "type": source
                })

                df = self.normalize_15min(
                    js,
                    "renewable_generation",
                    "renewable_generation_MW"
                )

                if not df.empty:
                    df["source"] = source
                    dfs.append(df)

                time.sleep(1)

        if not dfs:
            return pd.DataFrame()

        return (
            pd.concat(dfs)
            .drop_duplicates()
            .sort_values("date")
            .reset_index(drop=True)
        )


    def clean_terna_df(self, df, prefix=None) :

      df = df.copy( )

      # standardizza nome tempo
      if "date" not in df.columns:
          raise ValueError("Missing 'date' column"  )

      df["date"] = pd.to_datetime(df["date"]  )

      # elimina colonne duplicate fastidiose
      drop_cols = [c for c in df.columns if c in ["date_tz", "date_offset"]]
      df = df.drop(columns=drop_cols, errors="ignore" )

      # opzionale: prefix per evitare collisioni future
      if prefix:
          df = df.add_prefix(prefix + "_")
          df = df.rename(columns={f"{prefix}_date": "date"} )

      return df

    def clean_terna_features(self, df):

        df = df.copy()

        # 1. remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # 2. keep only useful columns
        cols_keep = [

            "date",

            "total_load_MW",
            "market_load_MW",
            "forecast_total_load_MW",
            "forecast_market_load_MW",

            "actual_generation_GWh",
            "actual_generation_GWh_solar",
            "actual_generation_GWh_hydro",

            "load_ramp_1h"
        ]

        df = df[[c for c in cols_keep if c in df.columns]]

        # 3. sort time
        df = df.sort_values("date")

        return df

    def safe_merge(self, base_df, new_df, name) :

        new_df = self.clean_terna_df(new_df)

        # evita doppioni
        new_df = new_df.loc[:, ~new_df.columns.duplicated() ]

        return base_df.merge(
            new_df,
            on="date",
            how="left",
            suffixes=("", f"_{name}")
        )
    
    def add_terna_features(self, df):

        df = df.copy()

        # =========================================================
        # FORCE NUMERIC (IMPORTANTISSIMO)
        # =========================================================

        numeric_cols = [c for c in df.columns if c != "date"]

        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # =========================================================
        # renewable share
        # =========================================================

        if all(c in df.columns for c in [
            "actual_generation_GWh_solar",
            "actual_generation_GWh_hydro",
            "total_load_MW"
        ]):

            df["renewable_share"] = (
                df["actual_generation_GWh_solar"]
                + df["actual_generation_GWh_hydro"]
            ) / df["total_load_MW"]

        # =========================================================
        # forecast error load
        # =========================================================

        if all(c in df.columns for c in [
            "forecast_total_load_MW",
            "total_load_MW"
        ]):

            df["load_forecast_error"] = (
                df["forecast_total_load_MW"]
                - df["total_load_MW"]
            )

        # =========================================================
        # net load proxy
        # =========================================================

        if all(c in df.columns for c in [
            "total_load_MW",
            "actual_generation_GWh"
        ]):

            df["net_load_proxy"] = (
                df["total_load_MW"]
                - df["actual_generation_GWh"]
            )

        df["Datetime"] = pd.to_datetime(df["date"])

        return df


from scipy.stats import ks_2samp

def ks_drift(df_old, df_new, cols, alpha=0.05):

    drift = {}

    for c in cols:
        if c in df_old.columns and c in df_new.columns:

            x_old = df_old[c].dropna()
            x_new = df_new[c].dropna()

            if len(x_old) > 50 and len(x_new) > 50:
                stat, p = ks_2samp(x_old, x_new)

                drift[c] = {
                    "ks_stat": stat,
                    "p_value": p,
                    "drift_flag": p < alpha
                }

    df = pd.DataFrame(drift).T

    if not df.empty:
        df = df.sort_values("ks_stat", ascending=False)

    return df


def df_to_supabase_records(df: pd.DataFrame):

    df = df.copy()

    # 1️⃣ datetime -> string
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 2️⃣ inf -> NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 3️⃣ NaN -> None (CRITICO)
    df = df.astype(object)
    df = df.where(pd.notnull(df), None)
    return df.to_dict(orient="records")



def load_dataset_history_from_supabase(supabase):
    all_rows = []
    page_size = 1000
    start = 0

    while True:
        resp = (
            supabase.table("dataset_history")
            .select("*")
            .order("Datetime", desc=False)
            .range(start, start + page_size - 1)
            .execute()
        )

        page = resp.data or []
        if not page:
            break

        all_rows.extend(page)

        if len(page) < page_size:
            break

        start += page_size

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").sort_index()
    df = df.asfreq("15min").ffill()

    return df


def upload_to_dropbox(local_path, dropbox_path, token):
    dbx = dropbox.Dropbox(token)

    with open(local_path, "rb") as f:
        dbx.files_upload(
            f.read(),
            dropbox_path,
            mode=dropbox.files.WriteMode.overwrite
        )


def load_from_dropbox(dropbox_path, token):
    dbx = dropbox.Dropbox(token)

    metadata, res = dbx.files_download(dropbox_path)

    file_bytes = io.BytesIO(res.content)
    df = pd.read_parquet(file_bytes)

    return df


