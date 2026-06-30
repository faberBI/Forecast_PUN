# =========================================================
# 🔎 HELPERS EXPLAINABILITY FORECAST NEXT 96
# =========================================================

import plotly.express as px


def _safe_list(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray, pd.Index)):
        return list(x)
    return None


def _get_regressor_for_step(forecaster, step: int):
    """
    Recupera il regressore dello step da ForecasterDirect.
    step va da 1 a 96.
    """
    regressors = getattr(forecaster, "regressors_", None)

    if isinstance(regressors, dict):
        if step in regressors:
            return regressors[step]
        if str(step) in regressors:
            return regressors[str(step)]

    if isinstance(regressors, (list, tuple)):
        idx = step - 1
        if 0 <= idx < len(regressors):
            return regressors[idx]

    return None


def _get_feature_names_from_estimator(estimator, forecaster=None, n_features=None, selected_exog=None):
    """
    Recupera i nomi feature in modo robusto.
    """
    candidates = []

    candidates.append(_safe_list(getattr(estimator, "feature_names_in_", None)))
    candidates.append(_safe_list(getattr(estimator, "feature_name_", None)))

    try:
        booster = getattr(estimator, "booster_", None)
        if booster is not None:
            candidates.append(_safe_list(booster.feature_name()))
    except Exception:
        pass

    if forecaster is not None:
        for attr in [
            "X_train_features_names_out_",
            "X_train_features_names_in_",
            "feature_names_in_",
            "exog_names_in_",
        ]:
            candidates.append(_safe_list(getattr(forecaster, attr, None)))

        lags_names = (
            _safe_list(getattr(forecaster, "lags_names", None))
            or _safe_list(getattr(forecaster, "lags_names_", None))
        )

        if lags_names is not None and selected_exog is not None:
            candidates.append(list(lags_names) + list(selected_exog))

    if selected_exog is not None:
        candidates.append(list(selected_exog))

    for names in candidates:
        if names is not None and n_features is not None and len(names) == n_features:
            return names

    if n_features is not None:
        return [f"feature_{i}" for i in range(n_features)]

    return None


def _extract_importance_from_forecaster_method(forecaster, step: int):
    """
    Usa get_feature_importances(step=...) se disponibile.
    """
    if not hasattr(forecaster, "get_feature_importances"):
        return pd.DataFrame(columns=["feature", "importance"])

    try:
        imp = forecaster.get_feature_importances(step=step)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    if imp is None or len(imp) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    imp = imp.copy()

    col_map = {}
    for c in imp.columns:
        cl = str(c).lower()

        if cl in ["feature", "features", "variable", "variable_name"]:
            col_map[c] = "feature"

        elif cl in [
            "importance",
            "feature_importance",
            "importance_gain",
            "gain",
            "split",
        ]:
            col_map[c] = "importance"

    imp = imp.rename(columns=col_map)

    if "feature" not in imp.columns or "importance" not in imp.columns:
        return pd.DataFrame(columns=["feature", "importance"])

    imp = imp[["feature", "importance"]].copy()
    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce")
    imp = imp.dropna(subset=["importance"])
    imp["importance"] = imp["importance"].abs()

    return imp


def _extract_importance_from_estimator(estimator, forecaster=None, selected_exog=None):
    """
    Fallback diretto su estimator LightGBM / sklearn.
    """
    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance"])

    values = None

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)

    elif hasattr(estimator, "coef_"):
        values = np.asarray(estimator.coef_, dtype=float).ravel()
        values = np.abs(values)

    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if values is None or len(values) == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    names = _get_feature_names_from_estimator(
        estimator=estimator,
        forecaster=forecaster,
        n_features=len(values),
        selected_exog=selected_exog,
    )

    imp = pd.DataFrame({
        "feature": names,
        "importance": values,
    })

    imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce")
    imp = imp.dropna(subset=["importance"])
    imp = imp[imp["importance"] >= 0]

    return imp


def build_forecast_explainability_df(
    forecaster,
    selected_exog,
    preds: pd.DataFrame,
    steps: int = 96,
):
    """
    Costruisce spiegabilità allineata ai 96 timestamp previsti.

    Output:
    Datetime | step | hour | minute | slot_15m | feature | importance | importance_norm
    """

    if preds is None or preds.empty:
        return pd.DataFrame()

    preds_tmp = preds.copy()

    if "Datetime" not in preds_tmp.columns:
        if isinstance(preds_tmp.index, pd.DatetimeIndex):
            preds_tmp = preds_tmp.reset_index().rename(columns={"index": "Datetime"})
        else:
            raise ValueError("preds deve avere colonna Datetime o DatetimeIndex.")

    preds_tmp["Datetime"] = pd.to_datetime(preds_tmp["Datetime"])
    preds_tmp = preds_tmp.sort_values("Datetime").head(steps).reset_index(drop=True)

    rows = []

    for i, row in preds_tmp.iterrows():
        step = i + 1
        dt = row["Datetime"]

        imp = _extract_importance_from_forecaster_method(forecaster, step)

        if imp.empty:
            estimator = _get_regressor_for_step(forecaster, step)
            imp = _extract_importance_from_estimator(
                estimator=estimator,
                forecaster=forecaster,
                selected_exog=selected_exog,
            )

        if imp.empty:
            continue

        imp = imp.copy()
        imp["Datetime"] = dt
        imp["step"] = step
        imp["hour"] = dt.hour
        imp["minute"] = dt.minute
        imp["slot_15m"] = dt.strftime("%H:%M")

        rows.append(imp)

    if not rows:
        return pd.DataFrame(
            columns=[
                "Datetime",
                "step",
                "hour",
                "minute",
                "slot_15m",
                "feature",
                "importance",
                "importance_norm",
            ]
        )

    df_imp = pd.concat(rows, ignore_index=True)

    df_imp["importance"] = pd.to_numeric(df_imp["importance"], errors="coerce")
    df_imp = df_imp.dropna(subset=["importance"])
    df_imp = df_imp[df_imp["importance"] >= 0]

    df_imp = (
        df_imp
        .groupby(
            ["Datetime", "step", "hour", "minute", "slot_15m", "feature"],
            as_index=False
        )["importance"]
        .mean()
    )

    step_sum = df_imp.groupby("step")["importance"].transform("sum")

    df_imp["importance_norm"] = np.where(
        step_sum > 0,
        df_imp["importance"] / step_sum,
        0.0,
    )

    return df_imp


def summarize_importance(df_imp: pd.DataFrame, top_n: int = 25):
    """
    Aggrega importanza normalizzata.
    """
    if df_imp is None or df_imp.empty:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_pct"])

    out = (
        df_imp
        .groupby("feature", as_index=False)["importance_norm"]
        .mean()
        .rename(columns={"importance_norm": "importance_mean"})
        .sort_values("importance_mean", ascending=False)
    )

    total = out["importance_mean"].sum()

    if total > 0:
        out["importance_pct"] = out["importance_mean"] / total * 100
    else:
        out["importance_pct"] = 0.0

    out = out.head(top_n).copy()
    out["importance_pct"] = out["importance_pct"].round(2)

    return out


def plot_importance_bar(df_summary: pd.DataFrame, title: str):
    """
    Barplot top feature.
    """
    if df_summary is None or df_summary.empty:
        return None

    plot_df = df_summary.sort_values("importance_pct", ascending=True)

    fig = px.bar(
        plot_df,
        x="importance_pct",
        y="feature",
        orientation="h",
        title=title,
        labels={
            "importance_pct": "Importanza media normalizzata (%)",
            "feature": "Feature",
        },
        text="importance_pct",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig.update_layout(
        height=max(450, 24 * len(plot_df)),
        margin=dict(l=20, r=40, t=60, b=20),
        yaxis=dict(categoryorder="total ascending"),
    )

    return fig
