"""U3 Saber Hacer - Supervised analysis on the COVID Tracking Project case study.

Business question (same case as U2, grain: state x day):
    With the data a state reports today, can we anticipate the hospital load
    of the next 7 days?

Two supervised tasks share the same input variables:
    - Regression: percent change of hospitalized patients (7-day average) in 7 days.
    - Classification: alert when that change is greater than +5 %.

Evaluation: models are compared and tuned with temporal cross-validation on
Apr-Oct 2020 and tested on Nov 2020-Feb 2021 with monthly walk-forward
retraining (each month is predicted with data available 7 days before it).

Run from the repository root:
    python src/u3_supervised_analysis.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "all-states-history.csv"
CLEAN_DIR = ROOT / "data" / "clean"
FIG_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs" / "tables"

SEED: int = 42
HORIZON_DAYS: int = 7
ALERT_THRESHOLD_PCT: float = 5.0
MIN_HOSPITALIZED: int = 20
TERRITORIES: set[str] = {"AS", "GU", "MP", "PR", "VI"}
START_DATE = pd.Timestamp("2020-04-15")
TRAIN_END = pd.Timestamp("2020-10-24")  # targets of this day look at 2020-10-31
TEST_START = pd.Timestamp("2020-11-01")
TEST_END = pd.Timestamp("2021-02-28")  # last day whose target (t + 7) exists

FEATURES: list[str] = [
    "hosp_growth_3d",
    "hosp_growth_7d",
    "hosp_growth_prev_7d",
    "cases_growth_3d",
    "cases_growth_7d",
    "cases_growth_prev_7d",
    "positivity_7d",
    "cases_per_hosp",
    "deaths_per_100_hosp",
]
FEATURE_LABELS: dict[str, str] = {
    "hosp_growth_3d": "Cambio % hospitalizados, últimos 3 días",
    "hosp_growth_7d": "Cambio % hospitalizados, últimos 7 días",
    "hosp_growth_prev_7d": "Cambio % hospitalizados, semana previa",
    "cases_growth_3d": "Cambio % casos, últimos 3 días",
    "cases_growth_7d": "Cambio % casos, últimos 7 días",
    "cases_growth_prev_7d": "Cambio % casos, semana previa",
    "positivity_7d": "Positividad % (7 días)",
    "cases_per_hosp": "Casos nuevos por hospitalizado",
    "deaths_per_100_hosp": "Muertes por cada 100 hospitalizados",
}

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 8,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


# --------------------------------------------------------------------------- data

def load_and_clean(raw_file: Path) -> pd.DataFrame:
    """Return one clean row per state and day (50 states + DC)."""
    columns = ["date", "state", "positiveIncrease", "deathIncrease",
               "totalTestResultsIncrease", "hospitalizedCurrently"]
    raw = pd.read_csv(raw_file, usecols=columns, parse_dates=["date"])
    clean = raw[~raw["state"].isin(TERRITORIES)].copy()
    clean = clean.drop_duplicates(["state", "date"]).sort_values(["state", "date"])
    clean = clean.rename(columns={
        "positiveIncrease": "new_cases",
        "deathIncrease": "new_deaths",
        "totalTestResultsIncrease": "new_tests",
        "hospitalizedCurrently": "hospitalized",
    })
    # Negative daily increases are retroactive corrections, not real events.
    for col in ["new_cases", "new_deaths", "new_tests"]:
        clean[col] = clean[col].clip(lower=0)

    by_state = clean.groupby("state")
    for col in ["new_cases", "new_deaths", "new_tests"]:
        clean[f"{col}_7d"] = by_state[col].transform(lambda s: s.rolling(7, min_periods=7).mean())
    # The 7-day average removes day-of-week reporting noise from the hospital census.
    clean["hospitalized_7d"] = by_state["hospitalized"].transform(lambda s: s.rolling(7, min_periods=5).mean())
    return clean.reset_index(drop=True)


def pct_change(current: pd.Series, previous: pd.Series) -> pd.Series:
    return (current / previous.where(previous > 0) - 1) * 100


def build_model_dataset(clean: pd.DataFrame) -> pd.DataFrame:
    """Create input variables known at day t and the targets at t + 7."""
    data = clean.copy()
    by_state = data.groupby("state")
    hosp = data["hospitalized_7d"]
    hosp_lag7 = by_state["hospitalized_7d"].shift(7)
    hosp_lag14 = by_state["hospitalized_7d"].shift(14)
    hosp_lead = by_state["hospitalized_7d"].shift(-HORIZON_DAYS)
    daily_hosp_lag3 = by_state["hospitalized"].shift(3)
    cases_lag3 = by_state["new_cases_7d"].shift(3)
    cases_lag7 = by_state["new_cases_7d"].shift(7)
    cases_lag14 = by_state["new_cases_7d"].shift(14)

    data["hosp_growth_3d"] = pct_change(data["hospitalized"], daily_hosp_lag3)
    data["hosp_growth_7d"] = pct_change(hosp, hosp_lag7)
    data["hosp_growth_prev_7d"] = pct_change(hosp_lag7, hosp_lag14)
    data["cases_growth_3d"] = pct_change(data["new_cases_7d"], cases_lag3)
    data["cases_growth_7d"] = pct_change(data["new_cases_7d"], cases_lag7)
    data["cases_growth_prev_7d"] = pct_change(cases_lag7, cases_lag14)
    positivity = data["new_cases_7d"] / data["new_tests_7d"].where(data["new_tests_7d"] > 0) * 100
    data["positivity_7d"] = positivity.clip(upper=100)
    data["cases_per_hosp"] = data["new_cases_7d"] / hosp
    data["deaths_per_100_hosp"] = data["new_deaths_7d"] / hosp * 100
    data["hospitalized_7d_in_7d"] = hosp_lead
    data["target_change_pct"] = pct_change(hosp_lead, hosp)
    data["target_alert"] = (data["target_change_pct"] > ALERT_THRESHOLD_PCT).astype(int)

    valid = (
        data["date"].between(START_DATE, TEST_END)
        & (hosp >= MIN_HOSPITALIZED) & (hosp_lag7 >= MIN_HOSPITALIZED)
        & hosp_lead.notna() & (cases_lag7 > 0)
    )
    data = data.loc[valid].replace([np.inf, -np.inf], np.nan)
    # Remove the rows between both periods so no training target looks into the test period.
    data = data[(data["date"] <= TRAIN_END) | (data["date"] >= TEST_START)].copy()
    data["split"] = np.where(data["date"] <= TRAIN_END, "train", "test")
    keep = ["date", "state", "hospitalized", "hospitalized_7d", *FEATURES, "hospitalized_7d_in_7d",
            "target_change_pct", "target_alert", "split"]
    return data[keep].reset_index(drop=True)


def temporal_folds(dates: pd.Series, n_folds: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds on dates, leaving a 7-day gap before each validation block."""
    unique_dates = np.sort(dates.unique())
    blocks = np.array_split(unique_dates, n_folds + 1)
    folds = []
    for block in blocks[1:]:
        val_start = block[0]
        train_idx = np.where(dates < val_start - np.timedelta64(HORIZON_DAYS, "D"))[0]
        val_idx = np.where(dates.isin(block))[0]
        folds.append((train_idx, val_idx))
    return folds


# --------------------------------------------------------------------------- models

def linear_pipeline(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


def tree_pipeline(model) -> Pipeline:
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def regression_models() -> dict[str, Pipeline]:
    return {
        "Regresión lineal (Ridge)": linear_pipeline(Ridge(alpha=1.0)),
        "Random Forest": tree_pipeline(RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED)),
        "Gradient Boosting": tree_pipeline(HistGradientBoostingRegressor(random_state=SEED)),
    }


def classification_models() -> dict[str, Pipeline]:
    return {
        "Regresión logística": linear_pipeline(LogisticRegression(
            class_weight="balanced", max_iter=1000)),
        "Random Forest": tree_pipeline(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1,
            random_state=SEED)),
        "Gradient Boosting": tree_pipeline(HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED)),
    }


def regression_scores(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def classification_scores(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_score),
    }


@dataclass
class TaskResult:
    comparison: pd.DataFrame
    best_name: str
    tuned_model: Pipeline  # fitted on the training period only
    tuning: dict
    test_pred: np.ndarray  # walk-forward predictions of the tuned model
    test_score: np.ndarray | None = None  # walk-forward probabilities (classification)


def cross_validate_regression(model, X, y, folds) -> dict[str, float]:
    scores = [regression_scores(y.iloc[va], clone(model).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va]))
              for tr, va in folds]
    return pd.DataFrame(scores).mean().to_dict()


def cross_validate_classification(model, X, y, folds) -> dict[str, float]:
    scores = []
    for tr, va in folds:
        fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
        scores.append(classification_scores(y.iloc[va], fitted.predict(X.iloc[va]),
                                            fitted.predict_proba(X.iloc[va])[:, 1]))
    return pd.DataFrame(scores).mean().to_dict()


def walk_forward(model, data: pd.DataFrame, test: pd.DataFrame, target: str,
                 proba: bool = False) -> np.ndarray:
    """Retrain at the start of each test month with data whose target is already known."""
    predictions = np.empty(len(test))
    months = test["date"].dt.to_period("M")
    for month in months.unique():
        cutoff = month.start_time - pd.Timedelta(days=HORIZON_DAYS)
        history = data[data["date"] < cutoff]
        fitted = clone(model).fit(history[FEATURES], history[target])
        rows = np.where(months == month)[0]
        X_month = test.iloc[rows][FEATURES]
        predictions[rows] = fitted.predict_proba(X_month)[:, 1] if proba else fitted.predict(X_month)
    return predictions


def baseline_regression(X: pd.DataFrame) -> np.ndarray:
    """Naive trend: next week changes as much as the last week did."""
    return X["hosp_growth_7d"].to_numpy()


def baseline_classification(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    growth = X["hosp_growth_7d"].to_numpy()
    return (growth > ALERT_THRESHOLD_PCT).astype(int), growth


REG_GRIDS = {
    "Random Forest": {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [5, 20, 50],
                      "model__max_features": [0.5, 1.0]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, 6, None],
                          "model__min_samples_leaf": [20, 100], "model__l2_regularization": [0.0, 1.0]},
    "Regresión lineal (Ridge)": {"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
}
CLF_GRIDS = {
    "Random Forest": {"model__max_depth": [8, 16, None], "model__min_samples_leaf": [5, 20, 50],
                      "model__max_features": [0.5, 1.0]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.1], "model__max_depth": [3, 6, None],
                          "model__min_samples_leaf": [20, 100], "model__l2_regularization": [0.0, 1.0]},
    "Regresión logística": {"model__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
}


def run_regression(data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, folds) -> TaskResult:
    """Compare defaults, tune every model, keep the best tuned one by CV and test it."""
    target = "target_change_pct"
    X_tr, y_tr, y_te = train[FEATURES], train[target], test[target]
    cv_base = pd.DataFrame([regression_scores(y_tr.iloc[va], baseline_regression(X_tr.iloc[va]))
                            for _, va in folds]).mean().to_dict()
    rows = [{"model": "Línea base (tendencia)", "cv_MAE_default": cv_base["MAE"], "cv_MAE": cv_base["MAE"],
             "best_params": "-", "n_candidates": 0,
             **{f"test_{k}": v for k, v in regression_scores(y_te, baseline_regression(test[FEATURES])).items()}}]
    tuned, predictions = {}, {}
    for name, model in regression_models().items():
        default_cv = cross_validate_regression(model, X_tr, y_tr, folds)
        search = GridSearchCV(model, REG_GRIDS[name], cv=folds, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(X_tr, y_tr)
        tuned[name] = search.best_estimator_
        predictions[name] = walk_forward(search.best_estimator_, data, test, target)
        rows.append({"model": name, "cv_MAE_default": default_cv["MAE"], "cv_MAE": -search.best_score_,
                     "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                     "n_candidates": len(search.cv_results_["params"]),
                     **{f"test_{k}": v for k, v in regression_scores(y_te, predictions[name]).items()}})
        print(f"[regression] {name}: CV MAE {default_cv['MAE']:.3f} -> {-search.best_score_:.3f}", flush=True)
    comparison = pd.DataFrame(rows)
    best_name = comparison.iloc[1:].sort_values("cv_MAE").iloc[0]["model"]
    return TaskResult(comparison, best_name, tuned[best_name], {"selected": best_name}, predictions[best_name])


def run_classification(data: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, folds) -> TaskResult:
    """Same procedure as regression, optimizing F1 of the alert class."""
    target = "target_alert"
    X_tr, y_tr, y_te = train[FEATURES], train[target], test[target]
    cv_base = pd.DataFrame([classification_scores(y_tr.iloc[va], *baseline_classification(X_tr.iloc[va]))
                            for _, va in folds]).mean().to_dict()
    rows = [{"model": "Línea base (tendencia)", "cv_F1_default": cv_base["F1"], "cv_F1": cv_base["F1"],
             "best_params": "-", "n_candidates": 0,
             **{f"test_{k}": v for k, v in classification_scores(y_te, *baseline_classification(test[FEATURES])).items()}}]
    tuned, outputs = {}, {}
    for name, model in classification_models().items():
        default_cv = cross_validate_classification(model, X_tr, y_tr, folds)
        search = GridSearchCV(model, CLF_GRIDS[name], cv=folds, scoring="f1", n_jobs=-1)
        search.fit(X_tr, y_tr)
        proba = walk_forward(search.best_estimator_, data, test, target, proba=True)
        pred = (proba >= 0.5).astype(int)
        tuned[name], outputs[name] = search.best_estimator_, (pred, proba)
        rows.append({"model": name, "cv_F1_default": default_cv["F1"], "cv_F1": search.best_score_,
                     "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                     "n_candidates": len(search.cv_results_["params"]),
                     **{f"test_{k}": v for k, v in classification_scores(y_te, pred, proba).items()}})
        print(f"[classification] {name}: CV F1 {default_cv['F1']:.3f} -> {search.best_score_:.3f}", flush=True)
    comparison = pd.DataFrame(rows)
    best_name = comparison.iloc[1:].sort_values("cv_F1", ascending=False).iloc[0]["model"]
    pred, proba = outputs[best_name]
    return TaskResult(comparison, best_name, tuned[best_name], {"selected": best_name}, pred, proba)


# --------------------------------------------------------------------------- figures

def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=200, facecolor="white")
    plt.close(fig)


def figure_workflow() -> None:
    steps = [
        ("1. Datos", "CSV original\ndel COVID\nTracking\nProject"),
        ("2. Limpieza", "50 estados\ny DC,\ncorrecciones,\npromedios\nde 7 días"),
        ("3. Variables", "9 indicadores\nconocidos\nen el día t"),
        ("4. División", "Entrena:\nabr-oct 2020\nPrueba:\nnov 2020-\nfeb 2021"),
        ("5. Entrenar", "Validación\ncruzada\ntemporal\n(5 bloques)"),
        ("6. Optimizar", "GridSearchCV\ny selección\ndel mejor\nmodelo"),
        ("7. Evaluar", "Prueba con\nreentrena-\nmiento\nmensual"),
    ]
    width, step = 1.32, 1.5
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_xlim(0, len(steps) * step)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, text) in enumerate(steps):
        x = 0.06 + i * step
        color = BLUE if i < 4 else (ORANGE if i < 6 else AQUA)
        ax.add_patch(FancyBboxPatch((x, 0.06), width, 0.88, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + width / 2, 0.8, title, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
        ax.text(x + width / 2, 0.4, text, ha="center", va="center", fontsize=5.9, color=INK_2, linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + step + 0.01, 0.5), xytext=(x + width + 0.01, 0.5),
                        arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.2})
    save(fig, "fig1_flujo_modelo.png")


def figure_comparison(reg: pd.DataFrame, clf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, table, metric, title, fmt in [
        (axes[0], reg, "MAE", "Regresión: MAE (menor es mejor)", "{:.1f}"),
        (axes[1], clf, "F1", "Clasificación: F1 (mayor es mejor)", "{:.2f}"),
    ]:
        names = table["model"].str.replace(" (Ridge)", "", regex=False).str.replace(" (tendencia)", "", regex=False)
        y = np.arange(len(names))
        h = 0.36
        ax.barh(y - h / 2 - 0.01, table[f"cv_{metric}"], height=h, color=BLUE, label="Validación cruzada (abr-oct 2020)")
        ax.barh(y + h / 2 + 0.01, table[f"test_{metric}"], height=h, color=ORANGE, label="Prueba (nov 2020-feb 2021)")
        for yi, v in zip(y + h / 2 + 0.01, table[f"test_{metric}"]):
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=7, color=INK)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=8.5, loc="left", color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, table[[f"cv_{metric}", f"test_{metric}"]].to_numpy().max() * 1.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig2_comparacion_modelos.png")


def figure_national_forecast(test: pd.DataFrame, pred_pct: np.ndarray) -> dict:
    frame = test[["date", "state", "hospitalized_7d", "hospitalized_7d_in_7d"]].copy()
    frame = frame.rename(columns={"hospitalized_7d_in_7d": "hospitalized_in_7d"})
    frame["predicted_in_7d"] = frame["hospitalized_7d"] * (1 + pred_pct / 100)
    frame["naive_in_7d"] = frame["hospitalized_7d"] * (1 + test["hosp_growth_7d"].to_numpy() / 100)
    national = frame.groupby("date")[["hospitalized_in_7d", "predicted_in_7d", "naive_in_7d"]].sum()
    national.index = national.index + pd.Timedelta(days=HORIZON_DAYS)  # plot at the forecast day
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    ax.plot(national.index, national["hospitalized_in_7d"] / 1000, color=INK_2, lw=2, label="Real")
    ax.plot(national.index, national["predicted_in_7d"] / 1000, color=BLUE, lw=2, label="Modelo optimizado")
    ax.plot(national.index, national["naive_in_7d"] / 1000, color=ORANGE, lw=1.5, ls="--", label="Línea base")
    ax.set_ylabel("Hospitalizados, promedio 7 días (miles)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=7.5)
    fig.autofmt_xdate()
    save(fig, "fig3_pronostico_nacional.png")
    patients_mae_model = float((frame["predicted_in_7d"] - frame["hospitalized_in_7d"]).abs().mean())
    patients_mae_naive = float((frame["naive_in_7d"] - frame["hospitalized_in_7d"]).abs().mean())
    peak_day = national["hospitalized_in_7d"].idxmax()
    return {"patients_MAE_model": patients_mae_model, "patients_MAE_baseline": patients_mae_naive,
            "national_peak_date": str(peak_day.date()),
            "national_peak_real": float(national.loc[peak_day, "hospitalized_in_7d"]),
            "national_peak_pred": float(national.loc[peak_day, "predicted_in_7d"])}


def figure_classifier(y_te: pd.Series, pred: np.ndarray, proba: np.ndarray, base_score: np.ndarray) -> list[list[int]]:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Sin alerta", "Alerta"],
                                            cmap="Blues", colorbar=False, ax=axes[0])
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Valor real")
    axes[0].set_title("Matriz de confusión (prueba)", fontsize=8.5, loc="left", color=INK)
    RocCurveDisplay.from_predictions(y_te, proba, ax=axes[1], name="Modelo optimizado",
                                     curve_kwargs={"color": BLUE, "lw": 2})
    RocCurveDisplay.from_predictions(y_te, base_score, ax=axes[1], name="Línea base",
                                     curve_kwargs={"color": ORANGE, "lw": 1.5, "ls": "--"})
    axes[1].plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":")
    axes[1].set_xlabel("Tasa de falsos positivos")
    axes[1].set_ylabel("Verdaderos positivos (recall)")
    axes[1].set_title("Curva ROC (prueba)", fontsize=8.5, loc="left", color=INK)
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower right")
    axes[1].grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    save(fig, "fig4_clasificador_prueba.png")
    return confusion_matrix(y_te, pred).tolist()


def figure_importance(reg_model, clf_model, X_te, y_reg, y_clf) -> dict:
    reg_imp = permutation_importance(reg_model, X_te, y_reg, scoring="neg_mean_absolute_error",
                                     n_repeats=10, random_state=SEED, n_jobs=-1)
    clf_imp = permutation_importance(clf_model, X_te, y_clf, scoring="f1",
                                     n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.DataFrame({"regression": reg_imp.importances_mean, "classification": clf_imp.importances_mean},
                       index=FEATURES)
    imp["regression_share"] = imp["regression"].clip(lower=0) / imp["regression"].clip(lower=0).sum() * 100
    imp["classification_share"] = imp["classification"].clip(lower=0) / imp["classification"].clip(lower=0).sum() * 100
    imp = imp.sort_values("regression_share")
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = np.arange(len(imp))
    h = 0.36
    ax.barh(y + h / 2 + 0.01, imp["regression_share"], height=h, color=BLUE, label="Regresión")
    ax.barh(y - h / 2 - 0.01, imp["classification_share"], height=h, color=ORANGE, label="Clasificación")
    ax.set_yticks(y, [FEATURE_LABELS[f] for f in imp.index])
    ax.set_xlabel("Importancia relativa por permutación (%)")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    save(fig, "fig5_importancia_variables.png")
    return imp[["regression_share", "classification_share"]].round(1).to_dict()


# --------------------------------------------------------------------------- main

def main() -> None:
    for folder in (CLEAN_DIR, FIG_DIR, TABLE_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    clean = load_and_clean(RAW_FILE)
    clean.to_csv(CLEAN_DIR / "covid_states_clean.csv", index=False)
    dataset = build_model_dataset(clean)
    dataset.to_csv(CLEAN_DIR / "covid_model_dataset.csv", index=False)

    dataset = dataset.sort_values(["date", "state"]).reset_index(drop=True)
    train = dataset[dataset["split"] == "train"].reset_index(drop=True)
    test = dataset[dataset["split"] == "test"].reset_index(drop=True)
    folds = temporal_folds(train["date"])

    reg = run_regression(dataset, train, test, folds)
    clf = run_classification(dataset, train, test, folds)
    reg.comparison.to_csv(TABLE_DIR / "regression_comparison.csv", index=False)
    clf.comparison.to_csv(TABLE_DIR / "classification_comparison.csv", index=False)

    figure_workflow()
    figure_comparison(reg.comparison, clf.comparison)
    forecast = figure_national_forecast(test, reg.test_pred)
    cm = figure_classifier(test["target_alert"], clf.test_pred, clf.test_score,
                           baseline_classification(test[FEATURES])[1])
    importance = figure_importance(reg.tuned_model, clf.tuned_model, test[FEATURES],
                                   test["target_change_pct"], test["target_alert"])

    results = {
        "dataset": {
            "raw_rows": int(pd.read_csv(RAW_FILE, usecols=["state"]).shape[0]),
            "clean_rows": int(len(clean)),
            "model_rows": int(len(dataset)),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "states": int(dataset["state"].nunique()),
            "train_period": [str(train["date"].min().date()), str(train["date"].max().date())],
            "test_period": [str(test["date"].min().date()), str(test["date"].max().date())],
            "alert_share_train": float(train["target_alert"].mean()),
            "alert_share_test": float(test["target_alert"].mean()),
            "target_mean_train": float(train["target_change_pct"].mean()),
            "target_std_train": float(train["target_change_pct"].std()),
            "missing_by_feature": dataset[FEATURES].isna().sum().astype(int).to_dict(),
        },
        "regression": {"comparison": reg.comparison.to_dict("records"), "best": reg.best_name,
                       "selection": reg.tuning, "forecast": forecast},
        "classification": {"comparison": clf.comparison.to_dict("records"), "best": clf.best_name,
                           "selection": clf.tuning, "confusion_matrix": cm},
        "permutation_importance": importance,
    }
    (ROOT / "outputs" / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str),
                                                   encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
