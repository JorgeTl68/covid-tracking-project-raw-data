"""U4 Saber Hacer - Unsupervised analysis on the COVID Tracking Project case study.

Question (same case as U2 and U3, the U3 clean dataset is the input):
    Without any label, which states lived a similar pandemic? Each state is
    described by the shape of its weekly curve of new cases (Mar 2020-Feb 2021),
    scaled to its own peak, so states of any size can be compared.

Process:
    1. Transform the clean daily dataset into one row per state with 52 weekly values.
    2. Reduce the 52 weeks with PCA and choose the number of components.
    3. Group the states with K-Means, Ward hierarchical clustering and DBSCAN.
    4. Evaluate with Silhouette, Calinski-Harabasz, Davies-Bouldin and bootstrap stability,
       tune the number of clusters / eps, and compare the default models with the tuned ones.
       The final k is the most stable K-Means partition with at least 3 groups.

Run from the repository root (after src/u3_supervised_analysis.py created data/clean):
    python src/u4_unsupervised_analysis.py
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
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "all-states-history.csv"
CLEAN_FILE = ROOT / "data" / "clean" / "covid_states_clean.csv"
TRANSFORMED_DIR = ROOT / "data" / "transformed"
OUT_DIR = ROOT / "outputs" / "u4"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SEED: int = 42
FIRST_WEEK = pd.Timestamp("2020-03-02")  # Monday; 52 full weeks end on 2021-02-28
N_WEEKS: int = 52
K_RANGE: range = range(2, 9)
MIN_USEFUL_K: int = 3  # two groups only split the states by the date of their peak
VARIANCE_TARGET: float = 0.90
DBSCAN_MIN_SAMPLES: tuple[int, ...] = (3, 4, 5)
DBSCAN_MAX_NOISE: float = 0.30
BOOTSTRAP_RUNS: int = 30
BOOTSTRAP_SHARE: float = 0.8

# Tile-grid map of the 50 states + DC: (row, column).
TILE_GRID: dict[str, tuple[int, int]] = {
    "AK": (0, 0), "ME": (0, 10),
    "VT": (1, 9), "NH": (1, 10),
    "WA": (2, 0), "ID": (2, 1), "MT": (2, 2), "ND": (2, 3), "MN": (2, 4), "IL": (2, 5), "WI": (2, 6),
    "MI": (2, 7), "NY": (2, 8), "RI": (2, 9), "MA": (2, 10),
    "OR": (3, 0), "NV": (3, 1), "WY": (3, 2), "SD": (3, 3), "IA": (3, 4), "IN": (3, 5), "OH": (3, 6),
    "PA": (3, 7), "NJ": (3, 8), "CT": (3, 9),
    "CA": (4, 0), "UT": (4, 1), "CO": (4, 2), "NE": (4, 3), "MO": (4, 4), "KY": (4, 5), "WV": (4, 6),
    "VA": (4, 7), "MD": (4, 8), "DE": (4, 9),
    "AZ": (5, 1), "NM": (5, 2), "KS": (5, 3), "AR": (5, 4), "TN": (5, 5), "NC": (5, 6), "SC": (5, 7),
    "DC": (5, 8),
    "OK": (6, 3), "LA": (6, 4), "MS": (6, 5), "AL": (6, 6), "GA": (6, 7),
    "HI": (7, 0), "TX": (7, 3), "FL": (7, 8),
}

# Reference categorical palette (light mode), fixed order.
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a5cd1"
PALETTE: list[str] = [BLUE, ORANGE, AQUA, YELLOW, VIOLET, "#d6409f", "#5b8c1a", "#7a7975"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
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


@dataclass
class ClusteringResult:
    """Final partition of the states and the tables that justify it."""

    labels: pd.Series
    k_search: pd.DataFrame
    pca_search: pd.DataFrame
    dbscan_search: pd.DataFrame
    comparison: pd.DataFrame
    before_after: pd.DataFrame
    summary: dict


# --------------------------------------------------------------------------- data

def load_clean(clean_file: Path = CLEAN_FILE) -> pd.DataFrame:
    """Daily clean dataset produced in U3 (50 states + DC, negative corrections set to zero)."""
    return pd.read_csv(clean_file, parse_dates=["date"])


def build_weekly_profiles(clean: pd.DataFrame) -> pd.DataFrame:
    """One row per state: weekly new cases divided by the state's highest week (0 to 1)."""
    last_day = FIRST_WEEK + pd.Timedelta(weeks=N_WEEKS) - pd.Timedelta(days=1)
    period = clean[clean["date"].between(FIRST_WEEK, last_day)].copy()
    period["week"] = FIRST_WEEK + pd.to_timedelta((period["date"] - FIRST_WEEK).dt.days // 7 * 7, unit="D")
    weekly = period.pivot_table(index="state", columns="week", values="new_cases", aggfunc="sum", fill_value=0)
    weekly = weekly.reindex(columns=pd.date_range(FIRST_WEEK, periods=N_WEEKS, freq="7D"), fill_value=0)
    profiles = weekly.div(weekly.max(axis=1), axis=0)
    profiles.columns = [d.strftime("%Y-%m-%d") for d in profiles.columns]
    profiles.insert(0, "total_cases", weekly.sum(axis=1).astype(int))
    profiles.insert(1, "peak_week", weekly.idxmax(axis=1).dt.strftime("%Y-%m-%d"))
    return profiles.sort_index()


def week_columns(profiles: pd.DataFrame) -> list[str]:
    return [c for c in profiles.columns if c[:2] == "20"]


# --------------------------------------------------------------------------- evaluation

def internal_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette (higher), Calinski-Harabasz (higher) and Davies-Bouldin (lower); noise (-1) is excluded."""
    mask = labels != -1
    n_clusters = len(set(labels[mask]))
    if n_clusters < 2 or mask.sum() <= n_clusters:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    return {
        "silhouette": silhouette_score(X[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(X[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(X[mask], labels[mask]),
    }


def kmeans(k: int) -> KMeans:
    return KMeans(n_clusters=k, n_init=50, random_state=SEED)


def ward(k: int) -> AgglomerativeClustering:
    return AgglomerativeClustering(n_clusters=k, linkage="ward")


def search_pca_components(X: np.ndarray) -> pd.DataFrame:
    """Variance kept and best K-Means silhouette for each number of PCA components."""
    full = PCA(random_state=SEED).fit(X)
    cumulative = np.cumsum(full.explained_variance_ratio_)
    rows = []
    for n in [2, 3, 4, 5, 6, 8, 10, 15, 20, X.shape[1]]:
        Z = PCA(n_components=n, random_state=SEED).fit_transform(X) if n < X.shape[1] else X
        best = max((internal_scores(Z, kmeans(k).fit_predict(Z))["silhouette"], k) for k in K_RANGE)
        rows.append({"components": n, "variance": cumulative[min(n, len(cumulative)) - 1],
                     "best_k": best[1], "silhouette": best[0]})
    return pd.DataFrame(rows)


def bootstrap_stability(Z: np.ndarray, make_model, runs: int = BOOTSTRAP_RUNS) -> float:
    """Mean adjusted Rand index between the full partition and partitions fitted on 80 % samples."""
    full = make_model(SEED).fit_predict(Z)
    rng = np.random.default_rng(SEED)
    scores = []
    for run in range(runs):
        idx = rng.choice(len(Z), int(BOOTSTRAP_SHARE * len(Z)), replace=False)
        scores.append(adjusted_rand_score(full[idx], make_model(run).fit_predict(Z[idx])))
    return float(np.mean(scores))


def search_k(Z: np.ndarray) -> pd.DataFrame:
    """Internal metrics and bootstrap stability of K-Means and Ward for every candidate k."""
    rows = []
    for k in K_RANGE:
        for name, make_model in [
            ("K-Means", lambda seed, k=k: KMeans(n_clusters=k, n_init=10, random_state=seed)),
            ("Jerárquico (Ward)", lambda seed, k=k: ward(k)),
        ]:
            labels = (kmeans(k) if name == "K-Means" else ward(k)).fit_predict(Z)
            rows.append({"model": name, "k": k, **internal_scores(Z, labels),
                         "stability": bootstrap_stability(Z, make_model)})
    return pd.DataFrame(rows)


def choose_k(k_search: pd.DataFrame) -> int:
    """Most stable K-Means partition with at least MIN_USEFUL_K groups; silhouette breaks ties."""
    candidates = k_search[(k_search["model"] == "K-Means") & (k_search["k"] >= MIN_USEFUL_K)]
    best = candidates.sort_values(["stability", "silhouette"], ascending=False).iloc[0]
    return int(best["k"])


def kdistance_eps(Z: np.ndarray, min_samples: int) -> np.ndarray:
    """Sorted distance of every state to its min_samples-th neighbor (k-distance curve)."""
    distances, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    return np.sort(distances[:, -1])


def search_dbscan(Z: np.ndarray) -> pd.DataFrame:
    """DBSCAN grid: eps between the 10th and 90th percentile of each k-distance curve."""
    rows = []
    for min_samples in DBSCAN_MIN_SAMPLES:
        curve = kdistance_eps(Z, min_samples)
        for eps in np.round(np.quantile(curve, np.linspace(0.1, 0.9, 17)), 3):
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
            n_clusters = len(set(labels) - {-1})
            rows.append({"min_samples": min_samples, "eps": eps, "clusters": n_clusters,
                         "noise": int((labels == -1).sum()), **internal_scores(Z, labels)})
    return pd.DataFrame(rows)


def best_dbscan(search: pd.DataFrame, n_states: int) -> pd.Series:
    """Best silhouette among settings with 2+ clusters that leave at most 30 % of the states as noise."""
    valid = search[(search["clusters"] >= 2) & (search["noise"] <= DBSCAN_MAX_NOISE * n_states)]
    return valid.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).iloc[0]


def order_by_peak(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Rename clusters 1..k by the week in which their average curve peaks (earliest first).

    Ties are broken by the share of cases before June, so the group hit in spring comes first.
    """
    clusters = [c for c in np.unique(labels) if c != -1]

    def key(c: int) -> tuple[int, float]:
        curve = X[labels == c].mean(axis=0)
        return int(np.argmax(curve)), -float(curve[:13].sum() / curve.sum())

    mapping = {c: i + 1 for i, c in enumerate(sorted(clusters, key=key))}
    mapping[-1] = 0
    return np.array([mapping[c] for c in labels])


def run_clustering(profiles: pd.DataFrame) -> tuple[ClusteringResult, np.ndarray, PCA]:
    X = profiles[week_columns(profiles)].to_numpy()

    pca_search = search_pca_components(X)
    n_components = int(pca_search.loc[pca_search["variance"] >= VARIANCE_TARGET, "components"].iloc[0])
    pca = PCA(n_components=n_components, random_state=SEED).fit(X)
    Z = pca.transform(X)

    k_search = search_k(Z)
    best_k = choose_k(k_search)
    km_labels = kmeans(best_k).fit_predict(Z)
    ward_labels = ward(best_k).fit_predict(Z)

    db_search = search_dbscan(Z)
    db_best = best_dbscan(db_search, len(Z))
    db_eps, db_min = float(db_best["eps"]), int(db_best["min_samples"])
    db_labels = DBSCAN(eps=db_eps, min_samples=db_min).fit_predict(Z)

    def row(model: str, params: str, labels: np.ndarray) -> dict:
        return {"model": model, "params": params, "clusters": len(set(labels) - {-1}),
                "noise": int((labels == -1).sum()), **internal_scores(Z, labels),
                "ari_vs_kmeans": adjusted_rand_score(km_labels, labels)}

    comparison = pd.DataFrame([
        row("K-Means", f"k = {best_k}", km_labels),
        row("Jerárquico (Ward)", f"k = {best_k}", ward_labels),
        row("DBSCAN", f"eps = {db_eps:.2f}; min_samples = {db_min}", db_labels),
    ])

    # Before vs after: scikit-learn defaults on the 52 raw weeks vs the tuned models on the PCA
    # components. Both are scored on the 52 original weeks so the numbers are comparable.
    defaults = {
        "K-Means": (KMeans(random_state=SEED, n_init="auto").fit_predict(X), "k = 8 (por defecto)"),
        "Jerárquico (Ward)": (AgglomerativeClustering().fit_predict(X), "k = 2 (por defecto)"),
        "DBSCAN": (DBSCAN().fit_predict(X), "eps = 0.5; min_samples = 5 (por defecto)"),
    }
    tuned = {"K-Means": km_labels, "Jerárquico (Ward)": ward_labels, "DBSCAN": db_labels}
    tuned_params = dict(zip(comparison["model"], comparison["params"]))
    rows = []
    for model, (labels, params) in defaults.items():
        for stage, stage_labels, stage_params in [("Antes", labels, params),
                                                   ("Después", tuned[model], tuned_params[model])]:
            rows.append({"model": model, "stage": stage, "params": stage_params,
                         "clusters": len(set(stage_labels) - {-1}), "noise": int((stage_labels == -1).sum()),
                         **internal_scores(X, stage_labels)})
    before_after = pd.DataFrame(rows)

    final = order_by_peak(km_labels, X)
    silhouette_best_k = int(k_search[k_search["model"] == "K-Means"]
                            .sort_values("silhouette", ascending=False).iloc[0]["k"])
    summary = {
        "states": int(X.shape[0]),
        "weeks": int(X.shape[1]),
        "pca_components": n_components,
        "pca_variance": float(pca.explained_variance_ratio_.sum()),
        "pca_variance_by_component": [float(v) for v in pca.explained_variance_ratio_],
        "best_k": best_k,
        "silhouette_best_k": silhouette_best_k,
        "dbscan": {"eps": db_eps, "min_samples": db_min},
        "cluster_sizes": {int(c): int((final == c).sum()) for c in np.unique(final)},
    }
    labels = pd.Series(final, index=profiles.index, name="cluster")
    return ClusteringResult(labels, k_search, pca_search, db_search, comparison, before_after, summary), Z, pca


def describe_clusters(profiles: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Size, members, typical peak week and share of cases by season for each cluster."""
    weeks = week_columns(profiles)
    rows = []
    for cluster, group in profiles.groupby(labels):
        mean_curve = group[weeks].mean()
        shares = {}
        for name, start, end in [("mar_may", "2020-03-02", "2020-05-31"), ("jun_ago", "2020-06-01", "2020-08-31"),
                                 ("sep_nov", "2020-09-01", "2020-11-30"), ("dic_feb", "2020-12-01", "2021-02-28")]:
            cols = [w for w in weeks if start <= w <= end]
            shares[f"share_{name}"] = float(mean_curve[cols].sum() / mean_curve.sum())
        rows.append({"cluster": int(cluster), "states": len(group), "members": ", ".join(group.index),
                     "peak_week": mean_curve.idxmax(), **shares})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=200, facecolor="white")
    plt.close(fig)


def figure_workflow() -> None:
    steps = [
        ("1. Datos", "Dataset limpio\nde la U3\n(estado y día)"),
        ("2. Transformar", "52 semanas\npor estado,\nescala 0-1\n(pico = 1)"),
        ("3. Reducir", "PCA:\ncomponentes\ncon 90 % de\nla varianza"),
        ("4. Agrupar", "K-Means,\nWard y\nDBSCAN"),
        ("5. Evaluar", "Silhouette,\nCalinski-\nHarabasz,\nDavies-Bouldin"),
        ("6. Optimizar", "k de 2 a 8,\neps y\ncomponentes"),
        ("7. Interpretar", "Perfiles,\nmapa y\nmiembros de\ncada grupo"),
    ]
    width, step = 1.32, 1.5
    fig, ax = plt.subplots(figsize=(7.2, 2.0))
    ax.set_xlim(0, len(steps) * step)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for i, (title, text) in enumerate(steps):
        x = 0.06 + i * step
        color = BLUE if i < 3 else (ORANGE if i < 6 else AQUA)
        ax.add_patch(FancyBboxPatch((x, 0.06), width, 0.88, boxstyle="round,pad=0.0,rounding_size=0.08",
                                    facecolor="white", edgecolor=color, linewidth=2))
        ax.text(x + width / 2, 0.8, title, ha="center", va="center", fontsize=6.6, fontweight="bold", color=INK)
        ax.text(x + width / 2, 0.4, text, ha="center", va="center", fontsize=6.0, color=INK_2, linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + step + 0.01, 0.5), xytext=(x + width + 0.01, 0.5),
                        arrowprops={"arrowstyle": "-|>", "color": INK_2, "lw": 1.2})
    save(fig, "fig1_flujo_modelo.png")


def month_ticks(weeks: list[str]) -> tuple[list[int], list[str]]:
    months = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    positions, labels, seen = [], [], set()
    for i, w in enumerate(weeks):
        key = w[:7]
        if key not in seen:
            seen.add(key)
            positions.append(i)
            labels.append(f"{months[int(w[5:7]) - 1]}\n{w[2:4]}" if w[5:7] in ("01", "03") else months[int(w[5:7]) - 1])
    return positions, labels


def figure_heatmap(profiles: pd.DataFrame, labels: pd.Series) -> None:
    weeks = week_columns(profiles)
    peak_index = profiles[weeks].to_numpy().argmax(axis=1)
    order = sorted(profiles.index, key=lambda s: (labels[s], peak_index[profiles.index.get_loc(s)]))
    data = profiles.loc[order, weeks].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    image = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(order)), order, fontsize=6)
    positions, tick_labels = month_ticks(weeks)
    ax.set_xticks(positions, tick_labels, fontsize=7)
    boundaries = np.cumsum([sum(labels[s] == c for s in order) for c in sorted(labels.unique())])[:-1]
    for b in boundaries:
        ax.axhline(b - 0.5, color=ORANGE, linewidth=1.6)
    for c in sorted(labels.unique()):
        rows = [i for i, s in enumerate(order) if labels[s] == c]
        ax.text(len(weeks) + 0.6, np.mean(rows), f"Grupo {c}", va="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.spines[["left", "bottom"]].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.12)
    bar.set_label("Casos de la semana / semana más alta del estado")
    save(fig, "fig2_mapa_calor_perfiles.png")


def figure_pca_variance(pca_full: PCA, n_components: int) -> None:
    ratio = pca_full.explained_variance_ratio_[:12] * 100
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)[:12] * 100
    x = np.arange(1, len(ratio) + 1)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.bar(x, ratio, color=[BLUE if i <= n_components else GRID for i in x], label="Varianza de la componente")
    ax.plot(x, cumulative, color=ORANGE, marker="o", markersize=3.5, label="Varianza acumulada")
    ax.axhline(VARIANCE_TARGET * 100, color=INK_2, linestyle="--", linewidth=0.8)
    ax.text(12.45, VARIANCE_TARGET * 100 - 6, "meta: 90 %", va="center", ha="right", fontsize=7, color=INK_2)
    for xi, c in zip(x, cumulative):
        if xi <= n_components + 1:
            ax.text(xi - 0.12, c + 3, f"{c:.0f}", ha="right", fontsize=6.5, color=INK)
    ax.set_xticks(x)
    ax.set_xlabel("Componente principal")
    ax.set_ylabel("Varianza explicada (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="center right", fontsize=7)
    save(fig, "fig3_varianza_pca.png")


def figure_k_metrics(k_search: pd.DataFrame, best_k: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4))
    for ax, metric, title in [(axes[0, 0], "silhouette", "Silhouette (mayor es mejor)"),
                              (axes[0, 1], "calinski_harabasz", "Calinski-Harabasz (mayor es mejor)"),
                              (axes[1, 0], "davies_bouldin", "Davies-Bouldin (menor es mejor)"),
                              (axes[1, 1], "stability", "Estabilidad bootstrap, ARI (mayor es mejor)")]:
        for name, color in [("K-Means", BLUE), ("Jerárquico (Ward)", ORANGE)]:
            sub = k_search[k_search["model"] == name]
            ax.plot(sub["k"], sub[metric], marker="o", markersize=3.5, color=color, label=name)
        ax.axvline(best_k, color=INK_2, linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=7.5)
        ax.set_xticks(list(K_RANGE))
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("Número de grupos (k)")
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    fig.tight_layout()
    save(fig, "fig4_metricas_por_k.png")


def figure_dendrogram(Z: np.ndarray, states: list[str], best_k: int) -> None:
    tree = linkage(Z, method="ward")
    threshold = (tree[-best_k, 2] + tree[-best_k + 1, 2]) / 2
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    dendrogram(tree, labels=states, color_threshold=threshold, above_threshold_color=INK_2, ax=ax,
               leaf_font_size=6)
    ax.axhline(threshold, color=ORANGE, linestyle="--", linewidth=1)
    ax.text(len(states) * 10, threshold, f"  corte: {best_k} grupos", va="bottom", ha="right", fontsize=7, color=ORANGE)
    ax.set_ylabel("Distancia de Ward")
    ax.spines["bottom"].set_visible(False)
    save(fig, "fig5_dendrograma.png")


def figure_pca_scatter(Z: np.ndarray, labels: pd.Series, pca: PCA) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for c in sorted(labels.unique()):
        mask = (labels == c).to_numpy()
        ax.scatter(Z[mask, 0], Z[mask, 1], s=34, color=PALETTE[c - 1], edgecolor="white", linewidth=0.6,
                   label=f"Grupo {c}")
    for (x, y), state in zip(Z[:, :2], labels.index):
        ax.text(x + 0.012, y + 0.012, state, fontsize=6, color=INK_2)
    ratio = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"Componente 1 ({ratio[0]:.0f} % de la varianza)")
    ax.set_ylabel(f"Componente 2 ({ratio[1]:.0f} % de la varianza)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    save(fig, "fig6_grupos_pca.png")


def figure_cluster_profiles(profiles: pd.DataFrame, labels: pd.Series) -> None:
    weeks = week_columns(profiles)
    clusters = sorted(labels.unique())
    fig, axes = plt.subplots(1, len(clusters), figsize=(7.2, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(weeks))
    positions, tick_labels = month_ticks(weeks)
    for ax, c in zip(axes, clusters):
        group = profiles.loc[labels == c, weeks]
        for _, row in group.iterrows():
            ax.plot(x, row.to_numpy(), color=PALETTE[c - 1], alpha=0.18, linewidth=0.7)
        ax.plot(x, group.mean().to_numpy(), color=PALETTE[c - 1], linewidth=2)
        ax.set_title(f"Grupo {c} ({len(group)} estados)", fontsize=7.5)
        ax.set_xticks(positions[::3], [t.replace("\n", " ") for t in tick_labels[::3]], fontsize=6.3)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Casos semanales\n(pico del estado = 1)")
    fig.tight_layout()
    save(fig, "fig7_perfiles_por_grupo.png")


def figure_tile_map(labels: pd.Series, names: dict[int, str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for state, (row, col) in TILE_GRID.items():
        c = int(labels[state])
        ax.add_patch(plt.Rectangle((col, -row), 0.92, 0.92, facecolor=PALETTE[c - 1], edgecolor="white"))
        ax.text(col + 0.46, -row + 0.46, state, ha="center", va="center", fontsize=7.5, color="white",
                fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[c - 1]) for c in sorted(names)]
    ax.legend(handles, [f"Grupo {c}: {names[c]}" for c in sorted(names)], frameon=False, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    ax.set_xlim(-0.1, 11.1)
    ax.set_ylim(-7.2, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig8_mapa_grupos.png")


# --------------------------------------------------------------------------- main

MONTH_NAMES: list[str] = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                          "septiembre", "octubre", "noviembre", "diciembre"]


def cluster_name(cluster: pd.Series) -> str:
    """Short label from the month of the peak and any earlier wave (spring or summer)."""
    name = f"pico en {MONTH_NAMES[int(cluster['peak_week'][5:7]) - 1]}"
    if cluster["share_mar_may"] >= 0.10:
        return f"ola de primavera y {name}"
    if cluster["share_jun_ago"] >= 0.18:
        return f"ola de verano y {name}"
    return name


def main() -> None:
    for directory in (TRANSFORMED_DIR, FIG_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    raw_rows = len(pd.read_csv(RAW_FILE, usecols=["state"]))
    clean = load_clean()
    profiles = build_weekly_profiles(clean)

    result, Z, pca = run_clustering(profiles)
    clusters = describe_clusters(profiles, result.labels)
    clusters["name"] = clusters.apply(cluster_name, axis=1)
    names = dict(zip(clusters["cluster"].astype(int), clusters["name"]))

    transformed = profiles.copy()
    transformed.insert(0, "cluster", result.labels)
    transformed.reset_index().to_csv(TRANSFORMED_DIR / "covid_state_weekly_profiles.csv", index=False,
                                     float_format="%.4f")
    pd.DataFrame(Z, index=profiles.index, columns=[f"PC{i + 1}" for i in range(Z.shape[1])]) \
        .assign(cluster=result.labels).reset_index() \
        .to_csv(TRANSFORMED_DIR / "covid_state_pca_components.csv", index=False, float_format="%.4f")

    for name, table in [("pca_components_search", result.pca_search), ("k_search", result.k_search),
                        ("dbscan_search", result.dbscan_search), ("model_comparison", result.comparison),
                        ("before_after", result.before_after), ("cluster_summary", clusters)]:
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False, float_format="%.4f")

    pca_full = PCA(random_state=SEED).fit(profiles[week_columns(profiles)].to_numpy())
    figure_workflow()
    figure_heatmap(profiles, result.labels)
    figure_pca_variance(pca_full, result.summary["pca_components"])
    figure_k_metrics(result.k_search, result.summary["best_k"])
    figure_dendrogram(Z, list(profiles.index), result.summary["best_k"])
    figure_pca_scatter(Z, result.labels, pca)
    figure_cluster_profiles(profiles, result.labels)
    figure_tile_map(result.labels, names)

    summary = {
        "dataset": {"raw_rows": raw_rows, "clean_rows": len(clean), **{k: result.summary[k] for k in ("states", "weeks")},
                    "period": [FIRST_WEEK.strftime("%Y-%m-%d"),
                               (FIRST_WEEK + pd.Timedelta(weeks=N_WEEKS, days=-1)).strftime("%Y-%m-%d")]},
        **{k: v for k, v in result.summary.items() if k not in ("states", "weeks")},
        "before_after": result.before_after.to_dict(orient="records"),
        "comparison": result.comparison.to_dict(orient="records"),
        "clusters": clusters.to_dict(orient="records"),
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                         encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
