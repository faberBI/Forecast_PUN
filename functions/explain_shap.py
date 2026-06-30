import numpy as np
import pandas as pd
import shap


def build_forecast_explainability_shap_df(
    forecaster,
    last_window: pd.Series,
    exog: pd.DataFrame,
    steps: int = 96,
) -> pd.DataFrame:
    """
    Calcola i valori SHAP firmati per ciascuno dei `steps` quarti d'ora
    previsti, usando un TreeExplainer per ogni estimatore step-wise del
    ForecasterDirect (skforecast 0.22.0).

    shap_value > 0  -> la feature spinge la previsione verso l'ALTO
    shap_value < 0  -> la feature spinge la previsione verso il BASSO

    Parametri
    ---------
    forecaster : ForecasterDirect
        Modello già fittato (lo stesso usato per forecaster.predict()).
    last_window : pandas Series
        Stessa serie passata a forecaster.predict(last_window=...) /
        forecast_day_ahead_96_base per costruire i lag.
    exog : pandas DataFrame
        Stesse esogene future passate a forecaster.predict(exog=...).
        Deve coprire tutti gli `steps` richiesti.
    steps : int
        Numero di step da spiegare (default 96 = giornata intera a 15min).

    Ritorna
    -------
    DataFrame long: Datetime, step, hour, feature, feature_value, shap_value
    """
    # X_predict: DataFrame con UNA RIGA PER STEP, colonne = lag + exog,
    # esattamente nell'ordine/scaling con cui ogni estimatore è stato allenato.
    # Metodo pubblico ufficiale di skforecast 0.22.0 -> nessuna ricostruzione
    # manuale dei lag, niente rischio di disallineamento con il training.
    X_predict = forecaster.create_predict_X(
        steps=steps,
        last_window=last_window,
        exog=exog,
    )

    estimators = forecaster.estimators_  # dict {step: estimator}, step parte da 1

    records = []

    for i, (dt, row) in enumerate(X_predict.iterrows()):
        step = i + 1
        est = estimators.get(step)

        if est is None:
            continue

        X_row = row.to_frame().T.astype(float)

        explainer = shap.TreeExplainer(est)
        shap_vals = explainer.shap_values(X_row)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.asarray(shap_vals).reshape(-1)

        for feat, val, shap_v in zip(X_row.columns, X_row.iloc[0].values, shap_vals):
            records.append({
                "Datetime": dt,
                "step": step,
                "hour": pd.Timestamp(dt).hour,
                "feature": feat,
                "feature_value": val,
                "shap_value": float(shap_v),
            })

    return pd.DataFrame(records)


def summarize_signed_importance(explain_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """
    Aggrega su tutti gli step: media firmata (direzione media dell'impatto)
    e media assoluta (per il ranking di importanza).
    """
    agg = (
        explain_df.groupby("feature")
        .agg(
            shap_mean=("shap_value", "mean"),
            shap_abs_mean=("shap_value", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values("shap_abs_mean", ascending=False)
        .head(top_n)
    )
    return agg


def plot_signed_importance_bar(summary: pd.DataFrame, title: str):
    import plotly.express as px

    summary = summary.sort_values("shap_mean")
    fig = px.bar(
        summary,
        x="shap_mean",
        y="feature",
        orientation="h",
        color="shap_mean",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=title,
        labels={"shap_mean": "Impatto medio (segno) su PUN", "feature": "Feature"},
    )
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_signed_heatmap(explain_df: pd.DataFrame, top_features: list):
    import plotly.express as px

    heat_df = (
        explain_df[explain_df["feature"].isin(top_features)]
        .groupby(["hour", "feature"], as_index=False)["shap_value"]
        .mean()
    )

    fig = px.density_heatmap(
        heat_df,
        x="hour",
        y="feature",
        z="shap_value",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Impatto medio firmato per ora sul forecast next 96",
        labels={"hour": "Ora forecast", "feature": "Feature", "shap_value": "Impatto medio"},
    )
    fig.update_layout(height=550, xaxis=dict(dtick=1), margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_signed_waterfall_for_step(explain_df: pd.DataFrame, datetime_value, top_n: int = 20):
    """
    Contributo firmato di ciascuna feature per UN singolo quarto d'ora
    (per la vista "Per 15 minuti forecast" già presente nell'app).
    """
    import plotly.express as px

    slot_df = (
        explain_df[explain_df["Datetime"] == datetime_value]
        .copy()
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(top_n)
        .sort_values("shap_value")
    )

    fig = px.bar(
        slot_df,
        x="shap_value",
        y="feature",
        orientation="h",
        color="shap_value",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=f"Contributi firmati alla previsione — {datetime_value}",
        labels={"shap_value": "Impatto (SHAP)", "feature": "Feature"},
        hover_data={"feature_value": True},
    )
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=20))
    return fig
