"""
Standalone descriptive PCA visualizations.

PCA here is a descriptive aid for understanding the structure of the
*numeric* feature space (e.g. "how many dimensions are needed to summarize
these variables, and which variables drive them").
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from data_report.generate_figures.primitives import declutter_radial_labels


@dataclass
class PCAResult:
    """
    Bundle the fitted PCA model with all data needed to render every plot.

    Attributes:
        pca (PCA): The fitted scikit-learn PCA object.
        components (np.ndarray): Sample projections, shape
            ``(n_samples, n_components)``.
        feature_names (List[str]): Names of the numeric features passed to
            ``run_pca``.
        loadings (np.ndarray): Scaled loading vectors, shape
            ``(n_features, n_components)``. Each column is the corresponding
            component vector scaled by the square root of its explained
            variance.
        recommended_n_components (int): Smallest number of components whose
            cumulative explained variance reaches the ``variance_threshold``
            passed to ``run_pca``.
    """

    pca: PCA
    components: np.ndarray
    feature_names: List[str]
    loadings: np.ndarray
    recommended_n_components: int

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Per-component explained variance fractions from the fitted PCA object."""
        return self.pca.explained_variance_ratio_


def run_pca(df: pd.DataFrame, features: List[str], variance_threshold: float = 0.9) -> PCAResult:
    """
    Standardize the numeric columns in ``features`` and fit PCA.

    Columns that are entirely missing are dropped silently rather than
    failing the whole analysis. Mean imputation is used for remaining
    missing values — a defensible default for descriptive visualization
    that preserves each column's mean and variance contribution without
    dropping rows.

    Args:
        df (pd.DataFrame): Source data. Only numeric columns in ``features``
            are used.
        features (List[str]): Names of the numeric columns to analyse.
        variance_threshold (float): Cumulative variance target used to
            compute ``recommended_n_components``.

    Returns:
        PCAResult: Fitted model and all projections needed for plotting.

    Raises:
        ValueError: If ``features`` contains non-numeric columns, all
            columns are entirely empty, or fewer than 2 samples or 2
            features remain after filtering.
    """
    numeric_df = df[features].select_dtypes(include="number")

    dropped = [f for f in features if f not in numeric_df.columns]
    if dropped:
        raise ValueError(
            f"run_pca received non-numeric feature(s) that cannot be used: {dropped}. "
            "Pass only numeric columns, or encode them before calling run_pca."
        )

    # Columns that are entirely missing carry no information to impute from or
    # to project (their variance is undefined). This is common in federated
    # health data, where a given site may not record a particular field at
    # all -- so we drop such columns rather than failing the whole analysis,
    # and only fail if nothing usable remains.
    empty_columns = numeric_df.columns[numeric_df.isna().all()].tolist()
    if empty_columns:
        numeric_df = numeric_df.drop(columns=empty_columns)

    if numeric_df.shape[1] == 0:
        raise ValueError(
            "run_pca requires at least one numeric feature with non-missing values "
            f"(all of {features} were either non-numeric or entirely empty)."
        )

    # Mean imputation is a simple, defensible default for descriptive PCA: it
    # preserves each column's mean/variance contribution without dropping rows,
    # which would shrink an already-numeric-only sample. It is not appropriate
    # for inferential analysis -- this module is visualization-only.
    X = numeric_df.fillna(numeric_df.mean())

    max_components = min(X.shape)
    if max_components < 2:
        raise ValueError(
            "PCA requires at least 2 samples and 2 numeric features; "
            f"got {X.shape[0]} samples and {X.shape[1]} features."
        )

    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=max_components)
    components = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    recommended_n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
    recommended_n_components = min(recommended_n_components, max_components)

    return PCAResult(
        pca=pca,
        components=components,
        feature_names=list(numeric_df.columns),
        loadings=loadings,
        recommended_n_components=recommended_n_components,
    )


def _pc_label(result: PCAResult, index: int) -> str:
    """Return a formatted principal component label including the explained variance percentage."""
    return f"PC{index + 1} ({result.explained_variance_ratio[index] * 100:.1f}%)"


def _with_target(components: np.ndarray, columns: List[str], df: pd.DataFrame,
                  target: Optional[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Attach a target column from ``df`` to a component DataFrame for colour-coding."""
    plot_df = pd.DataFrame(components, columns=columns, index=df.index)
    if target is None:
        return plot_df, None
    plot_df[target] = df[target]
    return plot_df, target


def save_explained_variance_plot(result: PCAResult, output_path: Path, figsize=(8, 5)):
    """
    Save per-component and cumulative explained variance with the recommended
    component count marked.

    This plot should drive the decision of how many components matter —
    separate from the fixed 2–3 dimensions the other plots use purely for
    visualization.

    Args:
        result (PCAResult): Fitted PCA result.
        output_path (Path): Destination path for the PNG file.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    ratios = result.explained_variance_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    k = result.recommended_n_components

    plt.figure(figsize=figsize)
    plt.bar(range(1, n + 1), ratios * 100, alpha=0.6, label="Per-component variance")
    plt.plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick", label="Cumulative variance")
    plt.axvline(
        k, color="gray", linestyle="--",
        label=f"{k} component{'s' if k != 1 else ''} reach {cumulative[k - 1] * 100:.1f}%",
    )
    plt.title("Explained Variance by Principal Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.xticks(range(1, n + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_scatter(result: PCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None, figsize=(8, 6)):
    """
    Save a 2D projection of samples onto the first two principal components.

    Args:
        result (PCAResult): Fitted PCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    plot_df, hue = _with_target(result.components[:, :2], ["PC1", "PC2"], df, target)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, alpha=0.8)
    plt.title("PCA Projection (first two components)")
    plt.xlabel(_pc_label(result, 0))
    plt.ylabel(_pc_label(result, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_pca_scatter_matrix(result: PCAResult, df: pd.DataFrame, output_path: Path,
                            n_dims: int = 3, target: Optional[str] = None):
    """
    Save a pairwise grid of the first ``n_dims`` principal components.

    A superset of the 2D scatter that also shows PC1×PC3 and PC2×PC3,
    useful for checking whether structure visible in the PC1/PC2 plane
    persists or is hidden along further components.

    Args:
        result (PCAResult): Fitted PCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        n_dims (int): Number of leading components to include in the grid.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
    """
    n_dims = min(n_dims, result.components.shape[1])
    pc_cols = [f"PC{i + 1}" for i in range(n_dims)]
    plot_df, hue = _with_target(result.components[:, :n_dims], pc_cols, df, target)

    grid = sns.pairplot(plot_df, vars=pc_cols, hue=hue, corner=True, plot_kws={"alpha": 0.7})
    grid.figure.suptitle("PCA Scatter Matrix", y=1.02)
    grid.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def save_pca_loadings_biplot(result: PCAResult, output_path: Path,
                             pc_x: int = 0, pc_y: int = 1,
                             top_n: Optional[int] = None, figsize=(8, 8)):
    """
    Save a biplot-style view of how strongly each original variable contributes to a component pair.

    The most directly interpretable PCA output: it ties abstract axes back
    to real variable names that report readers know. When there are more
    than 20 features, only the 20 with the largest loading magnitude are
    shown.

    Args:
        result (PCAResult): Fitted PCA result.
        output_path (Path): Destination path for the PNG file.
        pc_x (int): Zero-based index of the component plotted on the
            x-axis.
        pc_y (int): Zero-based index of the component plotted on the
            y-axis.
        top_n (int, optional): Explicit cap on the number of features
            shown. Overridden by the hard 20-feature limit when there are
            more than 20 features total.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    x = result.loadings[:, pc_x]
    y = result.loadings[:, pc_y]
    names = np.array(result.feature_names)

    _MAX_LABELS = 20
    truncated = False
    if len(names) > _MAX_LABELS:
        magnitude = np.hypot(x, y)
        keep = np.argsort(magnitude)[-_MAX_LABELS:]
        x, y, names = x[keep], y[keep], names[keep]
        truncated = True
    elif top_n is not None and top_n < len(names):
        magnitude = np.hypot(x, y)
        keep = np.argsort(magnitude)[-top_n:]
        x, y, names = x[keep], y[keep], names[keep]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    for xi, yi in zip(x, y):
        ax.annotate(
            "", xy=(xi, yi), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.7),
        )
    max_radius_factor = declutter_radial_labels(ax, x, y, names)

    # Labels for angularly-clustered arrows are staggered out to a larger
    # radius (see declutter_radial_labels) -- size the limit from the actual
    # radius used (plus room for the text itself) so those outer labels
    # stay inside the frame instead of floating past it.
    limit = max(np.abs(x).max(), np.abs(y).max()) * (max_radius_factor + 0.3)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    title = "PCA Loadings (Feature Contributions)"
    if truncated:
        title += f" — top {_MAX_LABELS} features"
    ax.set_title(title)
    ax.set_xlabel(_pc_label(result, pc_x))
    ax.set_ylabel(_pc_label(result, pc_y))
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_3d_html(result: PCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None):
    """
    Save an interactive 3D projection of samples to a standalone HTML file.

    A static 3D scatter is hard to read (fixed angle, occlusion) and easy
    to misjudge — an interactive export lets the reader rotate it
    themselves, which is far more useful for spotting real structure.

    Args:
        result (PCAResult): Fitted PCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the HTML file.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
    """
    pc_cols = ["PC1", "PC2", "PC3"]
    plot_df, color = _with_target(result.components[:, :3], pc_cols, df, target)

    fig = px.scatter_3d(
        plot_df, x="PC1", y="PC2", z="PC3", color=color,
        title="PCA 3D Projection (interactive)",
        labels={
            "PC1": _pc_label(result, 0),
            "PC2": _pc_label(result, 1),
            "PC3": _pc_label(result, 2),
        },
    )
    fig.write_html(str(output_path))


def save_pca_overview(result: PCAResult, df: pd.DataFrame, output_path: Path,
                      target: Optional[str] = None, figsize=(15, 4.5)):
    """
    Save a single composite figure: variance chart, 2D scatter, and top loadings.

    Intended as a one-glance summary panel for non-expert report readers.

    Args:
        result (PCAResult): Fitted PCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        target (str, optional): Column name in ``df`` used to colour-code
            the sample projection scatter.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ratios = result.explained_variance_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    axes[0].bar(range(1, n + 1), ratios * 100, alpha=0.6)
    axes[0].plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick")
    axes[0].set_title("Explained Variance")
    axes[0].set_xlabel("Component")
    axes[0].set_ylabel("Variance (%)")

    plot_df, hue = _with_target(result.components[:, :2], ["PC1", "PC2"], df, target)
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, alpha=0.8, ax=axes[1], legend=False)
    axes[1].set_title("Sample Projection")
    axes[1].set_xlabel(_pc_label(result, 0))
    axes[1].set_ylabel(_pc_label(result, 1))

    x, y = result.loadings[:, 0], result.loadings[:, 1]
    top = np.argsort(np.hypot(x, y))[-10:]
    top_x, top_y = x[top], y[top]
    top_names = [result.feature_names[i] for i in top]
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axvline(0, color="gray", linewidth=0.8)
    for xi, yi in zip(top_x, top_y):
        axes[2].annotate("", xy=(xi, yi), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color="steelblue", alpha=0.7))
    # A larger perp_step_frac than the standalone biplot's default -- this
    # panel is ~1/3 the width, so the same data-unit gap buys less pixel
    # space between fanned-out labels.
    max_radius_factor = declutter_radial_labels(
        axes[2], top_x, top_y, top_names, perp_step_frac=0.2,
    )
    # Annotation arrows don't participate in matplotlib's autoscale, so the
    # limits must be set explicitly from the plotted points (mirrors
    # save_pca_loadings_biplot), sized from the actual radius declutter_radial_labels
    # used (plus room for the text itself) so outer labels stay inside the frame.
    limit = max(np.abs(top_x).max(), np.abs(top_y).max()) * (max_radius_factor + 0.3)
    axes[2].set_xlim(-limit, limit)
    axes[2].set_ylim(-limit, limit)
    axes[2].set_title("Top Feature Contributions")
    axes[2].set_xlabel(_pc_label(result, 0))
    axes[2].set_ylabel(_pc_label(result, 1))

    fig.suptitle("PCA Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pca_outputs(df: pd.DataFrame, features: List[str], output_dir: Path,
                     target: Optional[str] = None) -> PCAResult:
    """
    Run PCA on ``features`` and save the full set of descriptive visualizations.

    Saves the scree plot, 2D scatter, and loadings biplot for PC1/PC2. When
    at least 3 components are available, also saves the scatter matrix,
    PC1/PC3 loadings biplot, and an interactive 3D HTML. Always saves an
    overview panel.

    Args:
        df (pd.DataFrame): Source data containing all columns in
            ``features``.
        features (List[str]): Names of the numeric columns to analyse.
        output_dir (Path): Directory where all output files are written.
        target (str, optional): Column name in ``df`` used to colour-code
            sample scatter plots. Silently ignored if not present in
            ``df``.

    Returns:
        PCAResult: The fitted PCA result, available for further downstream
            use.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target is not None and target not in df.columns:
        target = None

    result = run_pca(df, features)
    n_dims = result.components.shape[1]

    dropped = [f for f in features if f not in result.feature_names]
    if dropped:
        pd.DataFrame({"feature": dropped}).to_csv(output_dir / "excluded_columns.csv", index=False)

    save_explained_variance_plot(result, output_dir / "pca_explained_variance.png")
    save_pca_scatter(result, df, output_dir / "pca_scatter_2d.png", target=target)
    save_pca_loadings_biplot(result, output_dir / "pca_loadings_pc1_pc2.png", pc_x=0, pc_y=1)

    if n_dims >= 3:
        save_pca_scatter_matrix(result, df, output_dir / "pca_scatter_matrix.png", target=target)
        save_pca_loadings_biplot(result, output_dir / "pca_loadings_pc1_pc3.png", pc_x=0, pc_y=2)
        save_pca_3d_html(result, df, output_dir / "pca_scatter_3d.html", target=target)

    save_pca_overview(result, df, output_dir / "pca_overview.png", target=target)

    return result
