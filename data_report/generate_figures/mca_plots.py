"""
Standalone descriptive MCA (Multiple Correspondence Analysis) visualizations.

MCA is the categorical-data analog of PCA: instead of projecting numeric
variables into a reduced variance-maximizing space, it projects categorical
levels (and the samples that hold them) into a space that captures association
between categories. It is a more honest tool for healthcare categorical data
(symptoms, diagnoses, demographics) than coercing one-hot-encoded categories
through PCA's Euclidean/variance assumptions.

This module is a descriptive/exploratory aid only.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import plotly.express as px
import prince

from data_report.generate_figures.primitives import declutter_point_labels


@dataclass
class MCAResult:
    """
    Bundle the fitted MCA model with all data needed to render every plot.

    Attributes:
        mca (prince.MCA): The fitted prince MCA object.
        row_coordinates (np.ndarray): Sample projections, shape
            ``(n_samples, n_components)``.
        column_coordinates (pd.DataFrame): Category-level coordinates, shape
            ``(n_categories, n_components)``.
        column_variable (pd.Series): Maps each ``"variable__level"`` label
            back to its source variable name.
        feature_names (List[str]): Names of the categorical features passed
            to ``run_mca``.
        explained_inertia_ratio (np.ndarray): Per-dimension explained inertia
            fractions (sum to approximately 1).
        recommended_n_components (int): Smallest number of dimensions whose
            cumulative inertia reaches the ``variance_threshold`` passed to
            ``run_mca``.
    """

    mca: prince.MCA
    row_coordinates: np.ndarray
    column_coordinates: pd.DataFrame
    column_variable: pd.Series  # maps each "variable__level" label back to its source variable
    feature_names: List[str]
    explained_inertia_ratio: np.ndarray
    recommended_n_components: int


def _variable_for_label(label: str, feature_names: List[str], sep: str) -> str:
    """Return the source variable name for a ``"variable<sep>level"`` category label.

    Picks the longest (most specific) matching feature name, not the first
    one found -- a first-match search would mis-attribute every category of
    ``"site__region"`` to ``"site"`` if ``"site"`` happens to come first in
    ``feature_names`` and is itself a prefix of the other feature's name.
    """
    matches = [name for name in feature_names if label.startswith(f"{name}{sep}")]
    return max(matches, key=len) if matches else label


def run_mca(df: pd.DataFrame, features: List[str], variance_threshold: float = 0.9,
            max_levels_per_variable: int = 30) -> MCAResult:
    """
    Fit MCA on the categorical columns in ``features``.

    Columns with more than ``max_levels_per_variable`` distinct values are
    rejected: MCA one-hot-encodes every category, so high-cardinality columns
    (free-text fields, raw codes, identifiers) would dominate the geometry,
    blow up runtime, and make the category map unreadable.

    Columns that are entirely missing are dropped silently rather than
    failing the whole analysis (mirrors ``run_pca``'s handling of
    entirely-missing numeric columns), and only fail if fewer than 2 usable
    columns remain.

    ``recommended_n_components`` is the smallest number of dimensions whose
    cumulative explained inertia reaches ``variance_threshold``.
    Visualizations always use a fixed 2–3 dimensions regardless, since their
    job is to show structure, not to define a retained subspace.

    Args:
        df (pd.DataFrame): Source data. Only ``object``, ``category``, and
            ``bool`` columns in ``features`` are used.
        features (List[str]): Names of the categorical columns to analyse.
            Must contain at least two columns.
        variance_threshold (float): Cumulative inertia target used to
            compute ``recommended_n_components``.
        max_levels_per_variable (int): Maximum number of distinct values
            allowed per column; columns that exceed this limit raise an
            error.

    Returns:
        MCAResult: Fitted model and all coordinates needed for plotting.

    Raises:
        ValueError: If ``features`` contains non-categorical columns, fewer
            than two usable columns, or any column exceeds
            ``max_levels_per_variable``.
    """
    # Columns that are entirely missing have no categories to one-hot-encode
    # and would otherwise reach prince.MCA.fit() and fail unpredictably. This
    # mirrors run_pca's handling of entirely-missing numeric columns. Checked
    # on the raw requested columns *before* the dtype filter below, since an
    # all-NaN column can be inferred as a non-object dtype (e.g. float64),
    # which would otherwise misclassify it as "non-categorical" instead of
    # recognising it as simply empty.
    requested_df = df[features]
    empty_columns = [f for f in features if requested_df[f].isna().all()]
    usable_features = [f for f in features if f not in empty_columns]

    # "str" is deliberately excluded: pandas >= 2.x raises TypeError from
    # select_dtypes when "str" appears in `include` ("numpy string dtypes are
    # not allowed") -- "object" already covers plain Python str columns.
    categorical_df = df[usable_features].select_dtypes(include=["object", "category", "bool"])

    dropped = [f for f in usable_features if f not in categorical_df.columns]
    if dropped:
        raise ValueError(
            f"run_mca received non-categorical feature(s) that cannot be used: {dropped}. "
            "Pass only categorical/object/bool columns, or recode them before calling run_mca."
        )

    if categorical_df.shape[1] == 0:
        raise ValueError("run_mca requires at least one categorical feature.")
    if categorical_df.shape[1] < 2:
        raise ValueError(
            "MCA requires at least 2 categorical features to relate to each other; "
            f"got {categorical_df.shape[1]}."
        )

    high_cardinality = {
        col: int(categorical_df[col].nunique(dropna=True))
        for col in categorical_df.columns
        if categorical_df[col].nunique(dropna=True) > max_levels_per_variable
    }
    if high_cardinality:
        raise ValueError(
            f"Column(s) exceed max_levels_per_variable={max_levels_per_variable}: "
            f"{high_cardinality}. Exclude them or raise the limit explicitly -- "
            "high-cardinality columns make the MCA category map unreadable."
        )

    sep = "__"
    total_categories = sum(categorical_df[col].nunique(dropna=True) for col in categorical_df.columns)
    max_components = max(total_categories - categorical_df.shape[1], 1)

    mca = prince.MCA(
        n_components=max_components,
        one_hot_prefix_sep=sep,
        random_state=42,
        engine="sklearn",
    )
    mca = mca.fit(categorical_df)

    row_coordinates = mca.row_coordinates(categorical_df).to_numpy()
    column_coordinates = mca.column_coordinates(categorical_df)
    column_variable = pd.Series(
        [_variable_for_label(label, list(categorical_df.columns), sep) for label in column_coordinates.index],
        index=column_coordinates.index,
    )

    explained_inertia_ratio = np.asarray(mca.percentage_of_variance_) / 100.0
    cumulative = np.cumsum(explained_inertia_ratio)
    recommended_n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
    recommended_n_components = min(recommended_n_components, len(explained_inertia_ratio))

    return MCAResult(
        mca=mca,
        row_coordinates=row_coordinates,
        column_coordinates=column_coordinates,
        column_variable=column_variable,
        feature_names=list(categorical_df.columns),
        explained_inertia_ratio=explained_inertia_ratio,
        recommended_n_components=recommended_n_components,
    )


def _dim_label(result: MCAResult, index: int) -> str:
    """Return a formatted dimension label including the explained inertia percentage."""
    return f"Dim {index + 1} ({result.explained_inertia_ratio[index] * 100:.1f}%)"


def _with_target(coordinates: np.ndarray, columns: List[str], df: pd.DataFrame,
                  target: Optional[str]) -> tuple[pd.DataFrame, Optional[str]]:
    """Attach a target column from ``df`` to a coordinate DataFrame for colour-coding."""
    plot_df = pd.DataFrame(coordinates, columns=columns, index=df.index)
    if target is None:
        return plot_df, None
    plot_df[target] = df[target]
    return plot_df, target


def save_explained_inertia_plot(result: MCAResult, output_path: Path, figsize=(8, 5)):
    """
    Save per-dimension and cumulative explained inertia with the recommended
    dimension count marked.

    Inertia is MCA's analog of explained variance in PCA. The x-axis is
    limited to at most 12 evenly spaced ticks so labels do not collide when
    there are many dimensions.

    Args:
        result (MCAResult): Fitted MCA result.
        output_path (Path): Destination path for the PNG file.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    ratios = result.explained_inertia_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    k = result.recommended_n_components

    plt.figure(figsize=figsize)
    plt.bar(range(1, n + 1), ratios * 100, alpha=0.6, label="Per-dimension inertia")
    plt.plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick", label="Cumulative inertia")
    plt.axvline(
        k, color="gray", linestyle="--",
        label=f"{k} dimension{'s' if k != 1 else ''} reach {cumulative[k - 1] * 100:.1f}%",
    )
    plt.title("Explained Inertia by MCA Dimension")
    plt.xlabel("MCA Dimension")
    plt.ylabel("Explained Inertia (%)")
    # Limit x-axis ticks to at most 12 evenly spaced integers so labels don't
    # collide when there are many MCA dimensions (e.g. 100+).
    step = max(1, n // 12)
    plt.xticks(range(1, n + 1, step))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mca_row_scatter(result: MCAResult, df: pd.DataFrame, output_path: Path,
                         target: Optional[str] = None, figsize=(8, 6)):
    """
    Save a 2D projection of samples onto the first two MCA dimensions.

    Args:
        result (MCAResult): Fitted MCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    plot_df, hue = _with_target(result.row_coordinates[:, :2], ["Dim1", "Dim2"], df, target)

    plt.figure(figsize=figsize)
    sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue=hue, alpha=0.8)
    plt.title("MCA Projection (first two dimensions)")
    plt.xlabel(_dim_label(result, 0))
    plt.ylabel(_dim_label(result, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_mca_scatter_matrix(result: MCAResult, df: pd.DataFrame, output_path: Path,
                            n_dims: int = 3, target: Optional[str] = None):
    """
    Save a pairwise grid of the first ``n_dims`` row-coordinate dimensions.

    A superset of the 2D projection that also shows cross-dimension pairs
    (e.g. Dim1 × Dim3), useful for checking whether structure persists
    beyond the first two axes.

    Args:
        result (MCAResult): Fitted MCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        n_dims (int): Number of leading dimensions to include in the grid.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
    """
    n_dims = min(n_dims, result.row_coordinates.shape[1])
    dim_cols = [f"Dim{i + 1}" for i in range(n_dims)]
    plot_df, hue = _with_target(result.row_coordinates[:, :n_dims], dim_cols, df, target)

    grid = sns.pairplot(plot_df, vars=dim_cols, hue=hue, corner=True, plot_kws={"alpha": 0.7})
    grid.figure.suptitle("MCA Scatter Matrix", y=1.02)
    grid.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(grid.figure)


def save_mca_column_map(result: MCAResult, output_path: Path,
                        dim_x: int = 0, dim_y: int = 1, figsize=(9, 8)):
    """
    Save a map of category-level coordinates for a batch of source variables.

    When the total number of categories exceeds 20, only the 20 most
    distinctive labels (by distance from the origin) are annotated to
    prevent overlap.

    Args:
        result (MCAResult): Fitted MCA result; typically a subset produced
            by ``_subset_mca_result``.
        output_path (Path): Destination path for the PNG file.
        dim_x (int): Zero-based index of the MCA dimension to plot on the
            x-axis.
        dim_y (int): Zero-based index of the MCA dimension to plot on the
            y-axis.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    _MAX_LABELS = 20
    coords = result.column_coordinates
    x = coords.iloc[:, dim_x].to_numpy()
    y = coords.iloc[:, dim_y].to_numpy()
    variables = result.column_variable.to_numpy()
    labels = [idx.split("__", 1)[-1] if "__" in idx else idx for idx in coords.index]

    # If too many labels, keep only the most distinctive ones (furthest from origin)
    total = len(labels)
    if total > _MAX_LABELS:
        dist = np.hypot(x, y)
        top_idx = np.argsort(dist)[-_MAX_LABELS:]
        label_mask = np.zeros(total, dtype=bool)
        label_mask[top_idx] = True
    else:
        label_mask = np.ones(total, dtype=bool)

    unique_variables = list(dict.fromkeys(variables))
    palette = colormaps["tab10"].resampled(max(len(unique_variables), 1))
    color_for_variable = {var: palette(i) for i, var in enumerate(unique_variables)}

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    for var in unique_variables:
        mask = variables == var
        ax.scatter(x[mask], y[mask], color=color_for_variable[var], label=var, s=40, alpha=0.85)
    shown_x = x[label_mask]
    shown_y = y[label_mask]
    shown_labels = [l for l, show in zip(labels, label_mask) if show]
    declutter_point_labels(ax, shown_x, shown_y, shown_labels)

    title = "MCA Category Map (Column Coordinates)"
    if total > _MAX_LABELS:
        title += f" — top {_MAX_LABELS} of {total} categories shown"
    ax.set_title(title)
    ax.set_xlabel(_dim_label(result, dim_x))
    ax.set_ylabel(_dim_label(result, dim_y))
    ax.legend(title="Variable", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_mca_3d_html(result: MCAResult, df: pd.DataFrame, output_path: Path,
                     target: Optional[str] = None):
    """
    Save an interactive 3D projection of samples to a standalone HTML file.

    An interactive export lets readers rotate the scatter plot themselves,
    which is far more useful than a static 3D image for spotting real
    structure.

    Args:
        result (MCAResult): Fitted MCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the HTML file.
        target (str, optional): Column name in ``df`` used to colour-code
            points.
    """
    dim_cols = ["Dim1", "Dim2", "Dim3"]
    plot_df, color = _with_target(result.row_coordinates[:, :3], dim_cols, df, target)

    fig = px.scatter_3d(
        plot_df, x="Dim1", y="Dim2", z="Dim3", color=color,
        title="MCA 3D Projection (interactive)",
        labels={
            "Dim1": _dim_label(result, 0),
            "Dim2": _dim_label(result, 1),
            "Dim3": _dim_label(result, 2),
        },
    )
    fig.write_html(str(output_path))


def save_mca_overview(result: MCAResult, df: pd.DataFrame, output_path: Path,
                      target: Optional[str] = None, figsize=(15, 4.5)):
    """
    Save a single composite figure: inertia chart, row projection, and category map.

    Intended as a one-glance summary panel for non-expert report readers.

    Args:
        result (MCAResult): Fitted MCA result.
        df (pd.DataFrame): Source data; its index is used to align the
            target column when ``target`` is provided.
        output_path (Path): Destination path for the PNG file.
        target (str, optional): Column name in ``df`` used to colour-code
            the row projection scatter.
        figsize (tuple): Figure dimensions in inches as ``(width, height)``.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ratios = result.explained_inertia_ratio
    cumulative = np.cumsum(ratios)
    n = len(ratios)
    axes[0].bar(range(1, n + 1), ratios * 100, alpha=0.6)
    axes[0].plot(range(1, n + 1), cumulative * 100, marker="o", color="firebrick")
    axes[0].set_title("Explained Inertia")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Inertia (%)")
    step = max(1, n // 12)
    axes[0].set_xticks(range(1, n + 1, step))

    plot_df, hue = _with_target(result.row_coordinates[:, :2], ["Dim1", "Dim2"], df, target)
    sns.scatterplot(data=plot_df, x="Dim1", y="Dim2", hue=hue, alpha=0.8, ax=axes[1], legend=False)
    axes[1].set_title("Sample Projection")
    axes[1].set_xlabel(_dim_label(result, 0))
    axes[1].set_ylabel(_dim_label(result, 1))

    coords = result.column_coordinates
    x = coords.iloc[:, 0].to_numpy()
    y = coords.iloc[:, 1].to_numpy()
    labels = [idx.split("__", 1)[-1] if "__" in idx else idx for idx in coords.index]
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].axvline(0, color="gray", linewidth=0.8)
    axes[2].scatter(x, y, alpha=0.8, s=30, color="steelblue")
    # This panel is ~1/3 the width of the standalone category map, which
    # already caps at 20 labels -- with dozens of categories (common once a
    # dataset has several categorical variables) there simply isn't room to
    # label every point, however cleverly staggered, so only the most
    # distinctive (furthest from origin) categories are shown here.
    _OVERVIEW_MAX_LABELS = 15
    if len(labels) > _OVERVIEW_MAX_LABELS:
        top_idx = np.argsort(np.hypot(x, y))[-_OVERVIEW_MAX_LABELS:]
        declutter_point_labels(axes[2], x[top_idx], y[top_idx],
                                [labels[i] for i in top_idx], fontsize=7)
    else:
        declutter_point_labels(axes[2], x, y, labels, fontsize=7)
    axes[2].set_title("Category Map")
    axes[2].set_xlabel(_dim_label(result, 0))
    axes[2].set_ylabel(_dim_label(result, 1))

    fig.suptitle("MCA Overview")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _subset_mca_result(result: MCAResult, source_vars: List[str]) -> MCAResult:
    """Shallow copy of result with column_coordinates filtered to `source_vars`."""
    mask = result.column_variable.isin(source_vars)
    return MCAResult(
        mca=result.mca,
        row_coordinates=result.row_coordinates,
        column_coordinates=result.column_coordinates[mask],
        column_variable=result.column_variable[mask],
        feature_names=list(source_vars),
        explained_inertia_ratio=result.explained_inertia_ratio,
        recommended_n_components=result.recommended_n_components,
    )


def save_mca_outputs(df: pd.DataFrame, features: List[str], output_dir: Path,
                     target: Optional[str] = None) -> MCAResult:
    """
    Run MCA on ``features`` and save the full set of descriptive visualizations.

    Column maps are batched at 5 source variables per image so legend entries
    and category labels never pile up on a single unreadable chart. Also saves
    a 2D scatter, scatter matrix (when at least 3 dimensions are available),
    interactive 3D HTML, and an overview panel.

    Args:
        df (pd.DataFrame): Source data containing all columns in ``features``.
        features (List[str]): Names of the categorical columns to analyse.
        output_dir (Path): Directory where all output files are written.
        target (str, optional): Column name in ``df`` used to colour-code
            sample scatter plots. Silently ignored if not present in ``df``.

    Returns:
        MCAResult: The fitted MCA result, available for further downstream
            use.
    """
    _COLUMN_MAP_BATCH = 5

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if target is not None and target not in df.columns:
        target = None

    result = run_mca(df, features)
    n_dims = result.row_coordinates.shape[1]

    dropped = [f for f in features if f not in result.feature_names]
    if dropped:
        pd.DataFrame({"feature": dropped}).to_csv(output_dir / "excluded_columns.csv", index=False)

    save_explained_inertia_plot(result, output_dir / "mca_explained_inertia.png")
    save_mca_row_scatter(result, df, output_dir / "mca_row_scatter_2d.png", target=target)

    # Batched column maps: 5 source variables per image
    all_vars = result.feature_names
    n_batches = max(1, math.ceil(len(all_vars) / _COLUMN_MAP_BATCH))
    for b in range(n_batches):
        batch_vars = all_vars[b * _COLUMN_MAP_BATCH:(b + 1) * _COLUMN_MAP_BATCH]
        batch_result = _subset_mca_result(result, batch_vars)
        fname = f"mca_column_map_batch_{b + 1:02d}.png"
        save_mca_column_map(batch_result, output_dir / fname)

    if n_dims >= 3:
        save_mca_scatter_matrix(result, df, output_dir / "mca_scatter_matrix.png", target=target)
        save_mca_3d_html(result, df, output_dir / "mca_scatter_3d.html", target=target)

    save_mca_overview(result, df, output_dir / "mca_overview.png", target=target)

    return result
