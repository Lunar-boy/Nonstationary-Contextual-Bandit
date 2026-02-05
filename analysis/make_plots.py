import argparse
import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

FIGSIZE_THREE_PANEL = (7.0, 2.4)
FIGSIZE_THREE_PANEL_TALL = (7.0, 2.5)

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 8
LEGEND_SIZE = 8.5
LINE_WIDTH = 2.0
MARKER_SIZE = 5.5
BAND_ALPHA = 0.14
RAW_SCATTER_SIZE = 10
MEAN_MARKER_SIZE = 60
SAVEFIG_DPI = 320

PUB_STYLE = {
    "figure.figsize": FIGSIZE_THREE_PANEL,
    "font.size": LABEL_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "axes.linewidth": 0.8,
    "axes.titlepad": 4.0,
    "axes.labelpad": 4.0,
    "grid.alpha": 0.35,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "grid.color": "0.85",
}

COLORBLIND = [
    "#000000",  # black
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
]
ETA_COLORS = {
    0.0: "#0072B2",
    0.1: "#D55E00",
    0.3: "#009E73",
}
ETA_MARKERS = {
    0.0: "o",
    0.1: "s",
    0.3: "^",
}
COST_MARKERS = {
    0.05: "o",
    0.10: "s",
    0.20: "^",
}
FRONTIER_BUDGET_MARKERS = {
    0.05: "o",
    0.10: "s",
    0.20: "^",
}
FRONTIER_BUDGET_COLORS = {
    0.05: "#4C78A8",
    0.10: "#F58518",
    0.20: "#54A24B",
}
LINESTYLES = ["solid", "dashed", "dotted", "dashdot"]
LATEX_LINEBREAK = "\\\\"
FIXED_BUDGET_TICKS = [0.00, 0.05, 0.10, 0.20]
DELTA_OUTPUT_DIRNAME = "figures_ns_bandit"
SHOW_XERR = False
ANNOTATE_COST = False
ANNOTATE_OFFSETS = {
    0.0: (6, 3),
    0.1: (6, 8),
    0.3: (6, 13),
}


def set_pub_style() -> None:
    plt.rcParams.update(PUB_STYLE)
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=COLORBLIND)


def format_three_panel_axes(axs: Iterable[plt.Axes]) -> None:
    axs = list(axs)
    if len(axs) != 3:
        return
    for idx, ax in enumerate(axs):
        ax.tick_params(axis="y", labelleft=(idx == 0))
        if idx != 0:
            ax.set_ylabel("")


def add_global_legend(
    fig: plt.Figure,
    handles: List[plt.Line2D],
    labels: List[str],
    ncol: Optional[int] = None,
    bbox_to_anchor: Tuple[float, float] = (0.5, 1.08),
) -> None:
    if not handles:
        return
    if ncol is None:
        ncol = min(4, len(labels))
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=bbox_to_anchor,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
        borderaxespad=0.0,
    )
    legend.set_in_layout(True)


def load_results(results_dir: str, csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if csv_path is not None:
        csv_path = os.path.abspath(csv_path)
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Results file not found: {csv_path}")
        return pd.read_csv(csv_path), csv_path

    results_dir = os.path.abspath(results_dir)
    merged_path = os.path.join(results_dir, "nonstationary_bandit_results_merged.csv")
    if os.path.isfile(merged_path):
        return pd.read_csv(merged_path), merged_path

    seed_files = sorted(
        glob.glob(os.path.join(results_dir, "nonstationary_bandit_results_seed*.csv"))
    )
    if seed_files:
        dfs = [pd.read_csv(path) for path in seed_files]
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_csv(merged_path, index=False)
        return df_all, merged_path

    csv_path = os.path.join(results_dir, "nonstationary_bandit_results.csv")
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path), csv_path

    raise FileNotFoundError(
        "No results CSV found in results_dir. Expected merged, per-seed, or single CSV."
    )


def validate_columns(df: pd.DataFrame) -> None:
    required = {
        "M",
        "eta",
        "C_frac",
        "seed",
        "DynReg",
        "AvgReg",
        "Mismatch",
        "Stab",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def warn_missing_runs(df: pd.DataFrame, expected_seed_count: Optional[int]) -> None:
    seeds = sorted(df["seed"].dropna().unique().tolist())
    if expected_seed_count is not None:
        expected = set(range(expected_seed_count))
        present = set(seeds)
        missing = sorted(expected - present)
        extra = sorted(present - expected)

        if len(seeds) != expected_seed_count:
            print(
                f"WARNING: expected {expected_seed_count} seeds, found {len(seeds)}: {seeds}"
            )
        if missing:
            print(f"WARNING: missing seeds: {missing}")
        if extra:
            print(f"WARNING: extra seeds present: {extra}")

    grouped = df.groupby(["M", "eta", "C_frac"])["seed"].nunique()
    for (M, eta, cfrac), count in grouped.items():
        if expected_seed_count is not None and count < expected_seed_count:
            print(
                "WARNING: missing seeds for "
                f"M={M}, eta={eta}, C_frac={cfrac}: "
                f"{expected_seed_count - count} missing"
            )


def nice_bounds(values: np.ndarray, pad_frac: float = 0.06) -> Optional[Tuple[float, float]]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.0))
    if low == high:
        low -= 1.0
        high += 1.0
    pad = (high - low) * pad_frac
    low -= pad
    high += pad

    span = high - low
    if span <= 0:
        return low, high
    rough_step = span / 5.0
    magnitude = 10 ** np.floor(np.log10(rough_step))
    for mult in (1.0, 2.0, 5.0, 10.0):
        step = magnitude * mult
        if rough_step <= step:
            break

    low = np.floor(low / step) * step
    high = np.ceil(high / step) * step
    return float(low), float(high)


def apply_common_axis_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)


def aggregate_over_seeds(
    df: pd.DataFrame, group_cols: List[str], y_col: str
) -> pd.DataFrame:
    agg = (
        df.groupby(group_cols)[y_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"count": "n"})
    )
    return agg


def eta_styles(etas: Iterable[float]) -> Dict[float, Dict[str, str]]:
    styles: Dict[float, Dict[str, str]] = {}
    for idx, eta in enumerate(sorted(set(etas))):
        styles[eta] = {
            "linestyle": LINESTYLES[idx % len(LINESTYLES)],
        }
    return styles


def eta_color(eta: float) -> str:
    key = round(float(eta), 1)
    return ETA_COLORS.get(key, COLORBLIND[1])


def eta_marker(eta: float) -> str:
    key = round(float(eta), 1)
    return ETA_MARKERS.get(key, "o")


def cost_marker(cost: float) -> str:
    key = round(float(cost), 2)
    return COST_MARKERS.get(key, "o")


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(samples, alpha))
    upper = float(np.quantile(samples, 1.0 - alpha))
    return mean, lower, upper


def make_eta_legend_handles(
    etas: Iterable[float],
    style_map: Optional[Dict[float, Dict[str, str]]] = None,
    show_marker: bool = False,
) -> Tuple[List[plt.Line2D], List[str]]:
    handles: List[plt.Line2D] = []
    labels: List[str] = []
    for eta in etas:
        linestyle = "solid"
        if style_map is not None:
            linestyle = style_map[eta]["linestyle"]
        marker = eta_marker(eta) if show_marker else None
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=eta_color(eta),
                linewidth=LINE_WIDTH,
                linestyle=linestyle,
                marker=marker,
                markersize=MARKER_SIZE,
            )
        )
        labels.append(rf"$\eta={eta}$")
    return handles, labels


def make_cost_legend_handles(
    budgets: Iterable[float],
) -> Tuple[List[plt.Line2D], List[str]]:
    handles: List[plt.Line2D] = []
    labels: List[str] = []
    for cost in budgets:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=cost_marker(cost),
                linestyle="None",
                color="0.2",
                markerfacecolor="white",
                markeredgecolor="0.2",
                markersize=MARKER_SIZE + 1.0,
            )
        )
        labels.append(rf"$C/T={cost:.2f}$")
    return handles, labels


def add_panel_label(ax: plt.Axes, m_value: int) -> None:
    ax.set_title(rf"$M={m_value}$")


def plot_budget_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    n: np.ndarray,
    style: Dict[str, str],
    color: str,
    label: str,
    marker: str,
) -> None:
    ax.plot(
        x,
        y_mean,
        linestyle=style["linestyle"],
        marker=marker,
        color=color,
        label=label,
        linewidth=LINE_WIDTH,
        zorder=3,
    )

    mask = (n >= 2) & np.isfinite(y_std)
    if np.any(mask):
        y_std_plot = np.where(mask, y_std, 0.0)
        ax.fill_between(
            x,
            y_mean - y_std_plot,
            y_mean + y_std_plot,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0.0,
            zorder=2,
        )


def seed_jitter(seed: int, cfrac: float, scale: float = 0.003) -> float:
    value = np.sin(seed * 12.9898 + cfrac * 78.233) * 43758.5453
    frac = value - np.floor(value)
    return (frac - 0.5) * 2.0 * scale


def plot_metric_vs_budget(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    out_base: str,
    output_dir: str,
) -> List[str]:
    grouped = aggregate_over_seeds(df, ["M", "eta", "C_frac"], metric_col)
    grouped = grouped.sort_values(["M", "eta", "C_frac"])

    ms = sorted(grouped["M"].unique().tolist())
    etas = sorted(grouped["eta"].unique().tolist())
    style_map = eta_styles(etas)

    fig, axes = plt.subplots(
        1,
        len(ms),
        sharex=True,
        figsize=FIGSIZE_THREE_PANEL,
        constrained_layout=True,
    )
    if len(ms) == 1:
        axes = [axes]
    format_three_panel_axes(axes)

    y_values: List[float] = []
    for ax, M in zip(axes, ms):
        sub_m = grouped[grouped["M"] == M]
        for eta in etas:
            sub = sub_m[sub_m["eta"] == eta]
            if sub.empty:
                continue
            sub = sub.sort_values("C_frac")

            x = sub["C_frac"].to_numpy()
            y_mean = sub["mean"].to_numpy()
            y_std = sub["std"].to_numpy()
            n = sub["n"].to_numpy()

            label = rf"$\eta={eta}$"
            plot_budget_curve(
                ax=ax,
                x=x,
                y_mean=y_mean,
                y_std=y_std,
                n=n,
                style=style_map[eta],
                color=eta_color(eta),
                label=label,
                marker=eta_marker(eta),
            )
            mask = (n >= 2) & np.isfinite(y_std)
            if np.any(mask):
                y_values.extend((y_mean[mask] - y_std[mask]).tolist())
                y_values.extend((y_mean[mask] + y_std[mask]).tolist())
            else:
                y_values.extend(y_mean.tolist())

        ax.set_title(rf"$M = {M}$")
        ax.set_xticks(FIXED_BUDGET_TICKS)
        ax.set_xticklabels([f"{tick:.2f}" for tick in FIXED_BUDGET_TICKS])
        apply_common_axis_style(ax)

    ylim = nice_bounds(np.array(y_values))
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    fig.supxlabel("Budget C/T")
    fig.supylabel(ylabel)

    eta_handles, eta_labels = make_eta_legend_handles(etas, style_map=style_map)
    add_global_legend(
        fig,
        eta_handles,
        eta_labels,
        ncol=min(3, len(eta_labels)),
        bbox_to_anchor=(0.5, 1.12),
    )

    output_paths: List[str] = []
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{out_base}.{ext}")
        fig.savefig(
            out_path,
            bbox_inches="tight",
            dpi=SAVEFIG_DPI if ext == "png" else None,
        )
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def compute_paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[np.isclose(df["C_frac"], 0.0)].copy()
    teacher = df[~np.isclose(df["C_frac"], 0.0)].copy()

    baseline = baseline[["M", "eta", "seed", "Mismatch", "AvgReg", "Stab"]]
    merged = teacher.merge(
        baseline,
        on=["M", "eta", "seed"],
        how="left",
        suffixes=("", "_base"),
        indicator=True,
    )

    missing = merged[merged["_merge"] != "both"]
    if not missing.empty:
        print(
            "WARNING: missing baseline for some teacher runs; "
            f"skipping {len(missing)} pairs."
        )

    merged = merged[merged["_merge"] == "both"].copy()
    merged["DeltaMismatch"] = merged["Mismatch"] - merged["Mismatch_base"]
    merged["DeltaAvgReg"] = merged["AvgReg"] - merged["AvgReg_base"]
    merged["DeltaStab"] = merged["Stab"] - merged["Stab_base"]
    return merged


def summarize_sign_consistency(
    df: pd.DataFrame, delta_col: str, label: str
) -> None:
    grouped = df.groupby(["M", "eta", "C_frac"])[delta_col]
    print(f"\nPaired sign summary for {label} (count negative / total):")
    for (M, eta, cfrac), values in grouped:
        total = int(values.count())
        negative = int((values < 0.0).sum())
        print(f"  M={M}, eta={eta}, C_frac={cfrac:.2f}: {negative}/{total}")


def plot_delta_metric_vs_budget(
    df_delta: pd.DataFrame,
    delta_col: str,
    ylabel: str,
    out_base: str,
    output_dir: str,
) -> List[str]:
    ms = sorted(df_delta["M"].unique().tolist())
    etas = sorted(df_delta["eta"].unique().tolist())
    budgets = sorted(df_delta["C_frac"].unique().tolist())
    rng = np.random.default_rng(12345)

    fig, axes = plt.subplots(
        1,
        len(ms),
        sharex=True,
        sharey=True,
        figsize=FIGSIZE_THREE_PANEL_TALL,
        constrained_layout=True,
    )
    if len(ms) == 1:
        axes = [axes]
    format_three_panel_axes(axes)

    y_values: List[float] = []
    for ax, M in zip(axes, ms):
        raw_m = df_delta[df_delta["M"] == M]

        for eta in etas:
            raw = raw_m[raw_m["eta"] == eta]
            if raw.empty:
                continue

            means = []
            lowers = []
            uppers = []
            costs = []

            for cfrac in budgets:
                seed_vals = raw[raw["C_frac"] == cfrac]
                if seed_vals.empty:
                    continue
                values = seed_vals[delta_col].to_numpy()
                mean, low, high = bootstrap_mean_ci(values, rng)

                costs.append(cfrac)
                means.append(mean)
                lowers.append(low)
                uppers.append(high)

                xs = cfrac + seed_vals["seed"].apply(
                    lambda s: seed_jitter(int(s), float(cfrac))
                ).to_numpy()
                ax.scatter(
                    xs,
                    values,
                    s=RAW_SCATTER_SIZE,
                    color="0.3",
                    alpha=0.25,
                    marker=".",
                    linewidths=0.0,
                    zorder=1,
                )

            if not costs:
                continue

            costs_arr = np.array(costs, dtype=float)
            means_arr = np.array(means, dtype=float)
            lowers_arr = np.array(lowers, dtype=float)
            uppers_arr = np.array(uppers, dtype=float)

            ax.plot(
                costs_arr,
                means_arr,
                color=eta_color(eta),
                linewidth=LINE_WIDTH,
                label=rf"$\eta={eta}$",
                zorder=3,
            )
            ax.errorbar(
                costs_arr,
                means_arr,
                yerr=[means_arr - lowers_arr, uppers_arr - means_arr],
                fmt="none",
                ecolor=eta_color(eta),
                elinewidth=1.0,
                capsize=2.5,
                alpha=0.65,
                zorder=2,
            )
            for cfrac, mean in zip(costs_arr, means_arr):
                ax.scatter(
                    cfrac,
                    mean,
                    s=MEAN_MARKER_SIZE,
                    marker=cost_marker(cfrac),
                    facecolor="white",
                    edgecolor=eta_color(eta),
                    linewidths=1.2,
                    zorder=4,
                )

            y_values.extend(lowers_arr.tolist())
            y_values.extend(uppers_arr.tolist())

        ax.axhline(0.0, color="0.5", linewidth=0.9, alpha=0.7)
        add_panel_label(ax, M)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{tick:.2f}" for tick in budgets])
        apply_common_axis_style(ax)

    ylim = nice_bounds(np.array(y_values))
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    fig.supxlabel("Budget C/T")
    fig.supylabel(ylabel)

    eta_handles, eta_labels = make_eta_legend_handles(etas)
    cost_handles, cost_labels = make_cost_legend_handles(budgets)
    add_global_legend(
        fig,
        eta_handles + cost_handles,
        eta_labels + cost_labels,
        ncol=4,
        bbox_to_anchor=(0.5, 1.16),
    )

    output_paths: List[str] = []
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{out_base}.{ext}")
        fig.savefig(
            out_path,
            bbox_inches="tight",
            dpi=SAVEFIG_DPI if ext == "png" else None,
        )
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def plot_cov_ellipse(
    ax: plt.Axes,
    points: np.ndarray,
    color: str,
    n_std: float = 1.0,
    alpha: float = 0.1,
) -> bool:
    if points.shape[0] < 3:
        return False

    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        print("WARNING: covariance contains non-finite values; skipping ellipse.")
        return False

    cond = np.linalg.cond(cov)
    if not np.isfinite(cond) or cond > 1e6:
        print("WARNING: ill-conditioned covariance; skipping ellipse.")
        return False

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    if vals.max() <= 0.0:
        return False

    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    scale = np.nanmax(np.std(points, axis=0))
    width = 2.0 * n_std * np.sqrt(vals[0])
    height = 2.0 * n_std * np.sqrt(vals[1])
    if scale > 0.0 and max(width, height) > 6.0 * scale:
        print("WARNING: oversized ellipse skipped.")
        return False

    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_patch(ellipse)
    return True


def plot_regret_mismatch_frontier(
    df: pd.DataFrame, out_base: str, output_dir: str
) -> List[str]:
    ms = sorted(df["M"].unique().tolist())
    etas = sorted(df["eta"].unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(ms),
        sharey=True,
        figsize=FIGSIZE_THREE_PANEL,
        constrained_layout=True,
    )
    if len(ms) == 1:
        axes = [axes]
    format_three_panel_axes(axes)

    x_values: List[float] = []
    y_values: List[float] = []

    for ax, M in zip(axes, ms):
        sub_m = df[df["M"] == M]
        best_point: Optional[Tuple[float, float]] = None

        for eta in etas:
            sub_eta = sub_m[sub_m["eta"] == eta]
            if sub_eta.empty:
                continue

            for cfrac in sorted(sub_eta["C_frac"].unique().tolist()):
                sub = sub_eta[sub_eta["C_frac"] == cfrac]
                if sub.empty:
                    continue

                points = sub[["Mismatch", "AvgReg"]].to_numpy()
                mean_point = points.mean(axis=0)
                x_values.append(mean_point[0])
                y_values.append(mean_point[1])

                if points.shape[0] >= 3:
                    plot_cov_ellipse(
                        ax,
                        points,
                        color=eta_color(eta),
                        n_std=1.0,
                        alpha=0.08,
                    )

                ax.scatter(
                    mean_point[0],
                    mean_point[1],
                    color=eta_color(eta),
                    marker=eta_marker(eta),
                    s=36,
                    alpha=0.9,
                )

                if best_point is None or mean_point[1] < best_point[1]:
                    best_point = (mean_point[0], mean_point[1])

        if best_point is not None:
            ax.scatter(
                best_point[0],
                best_point[1],
                marker="*",
                s=75,
                facecolor="none",
                edgecolor=COLORBLIND[0],
                linewidths=1.0,
                zorder=5,
            )

        ax.set_title(rf"$M = {M}$")
        apply_common_axis_style(ax)

    fig.supxlabel("Mismatch")
    fig.supylabel("DynReg/K")

    xlim = nice_bounds(np.array(x_values))
    ylim = nice_bounds(np.array(y_values))
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    handles = []
    labels = []
    for eta in etas:
        handle = plt.Line2D(
            [0],
            [0],
            marker=eta_marker(eta),
            linestyle="None",
            color=eta_color(eta),
            label=rf"$\eta={eta}$",
        )
        handles.append(handle)
        labels.append(rf"$\eta={eta}$")

    add_global_legend(
        fig,
        handles,
        labels,
        ncol=min(3, len(labels)),
        bbox_to_anchor=(0.5, 1.12),
    )

    output_paths: List[str] = []
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{out_base}.{ext}")
        fig.savefig(
            out_path,
            bbox_inches="tight",
            dpi=SAVEFIG_DPI if ext == "png" else None,
        )
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def plot_paired_frontier_arrows(
    df: pd.DataFrame, out_base: str, output_dir: str
) -> List[str]:
    ms = sorted(df["M"].unique().tolist())
    etas = sorted(df["eta"].unique().tolist())
    budgets = sorted(df.loc[~np.isclose(df["C_frac"], 0.0), "C_frac"].unique().tolist())
    rng = np.random.default_rng(12345)

    teacher = df[~np.isclose(df["C_frac"], 0.0)].copy()

    fig, axes = plt.subplots(
        1,
        len(ms),
        sharey=True,
        figsize=FIGSIZE_THREE_PANEL_TALL,
        constrained_layout=True,
    )
    if len(ms) == 1:
        axes = [axes]
    format_three_panel_axes(axes)

    x_values: List[float] = []
    y_values: List[float] = []

    for ax, M in zip(axes, ms):
        sub_m = teacher[teacher["M"] == M]
        if sub_m.empty:
            continue

        for eta in etas:
            sub_eta = sub_m[sub_m["eta"] == eta]
            if sub_eta.empty:
                continue

            mean_xs = []
            mean_ys = []
            low_xs = []
            high_xs = []
            low_ys = []
            high_ys = []
            costs = []

            for cfrac in budgets:
                sub = sub_eta[np.isclose(sub_eta["C_frac"], cfrac)]
                if sub.empty:
                    continue

                x_vals = sub["Mismatch"].to_numpy()
                y_vals = sub["AvgReg"].to_numpy()

                ax.scatter(
                    x_vals,
                    y_vals,
                    s=RAW_SCATTER_SIZE,
                    color="0.3",
                    alpha=0.25,
                    marker=".",
                    linewidths=0.0,
                    zorder=1,
                )

                mean_x, low_x, high_x = bootstrap_mean_ci(x_vals, rng)
                mean_y, low_y, high_y = bootstrap_mean_ci(y_vals, rng)

                costs.append(cfrac)
                mean_xs.append(mean_x)
                mean_ys.append(mean_y)
                low_xs.append(low_x)
                high_xs.append(high_x)
                low_ys.append(low_y)
                high_ys.append(high_y)

            if not costs:
                continue

            mean_xs_arr = np.array(mean_xs, dtype=float)
            mean_ys_arr = np.array(mean_ys, dtype=float)
            order = np.argsort(costs)
            mean_xs_arr = mean_xs_arr[order]
            mean_ys_arr = mean_ys_arr[order]
            costs_arr = np.array(costs, dtype=float)[order]

            ax.plot(
                mean_xs_arr,
                mean_ys_arr,
                color=eta_color(eta),
                linewidth=LINE_WIDTH,
                label=rf"$\eta={eta}$",
                zorder=3,
            )

            for idx, cfrac in enumerate(costs_arr):
                ax.scatter(
                    mean_xs_arr[idx],
                    mean_ys_arr[idx],
                    s=MEAN_MARKER_SIZE,
                    marker=cost_marker(cfrac),
                    facecolor="white",
                    edgecolor=eta_color(eta),
                    linewidths=1.2,
                    zorder=4,
                )
                if ANNOTATE_COST:
                    offset = ANNOTATE_OFFSETS.get(round(float(eta), 1), (6, 3))
                    ax.annotate(
                        f"{cfrac:.2f}",
                        (mean_xs_arr[idx], mean_ys_arr[idx]),
                        textcoords="offset points",
                        xytext=offset,
                        fontsize=10,
                        color=eta_color(eta),
                        alpha=0.8,
                    )

            low_xs_arr = np.array(low_xs, dtype=float)[order]
            high_xs_arr = np.array(high_xs, dtype=float)[order]
            low_ys_arr = np.array(low_ys, dtype=float)[order]
            high_ys_arr = np.array(high_ys, dtype=float)[order]

            xerr = None
            if SHOW_XERR:
                xerr = [mean_xs_arr - low_xs_arr, high_xs_arr - mean_xs_arr]
            ax.errorbar(
                mean_xs_arr,
                mean_ys_arr,
                xerr=xerr,
                yerr=[mean_ys_arr - low_ys_arr, high_ys_arr - mean_ys_arr],
                fmt="none",
                ecolor=eta_color(eta),
                elinewidth=1.0,
                capsize=2,
                alpha=0.65,
                zorder=2,
            )

            x_values.extend(low_xs_arr.tolist())
            x_values.extend(high_xs_arr.tolist())
            y_values.extend(low_ys_arr.tolist())
            y_values.extend(high_ys_arr.tolist())

        add_panel_label(ax, M)
        apply_common_axis_style(ax)

    fig.supxlabel("Mismatch")
    fig.supylabel("DynReg/K")

    xlim = nice_bounds(np.array(x_values))
    ylim = nice_bounds(np.array(y_values))
    if xlim is not None:
        for ax in axes:
            ax.set_xlim(*xlim)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(*ylim)

    eta_handles, eta_labels = make_eta_legend_handles(etas)
    cost_handles, cost_labels = make_cost_legend_handles(budgets)
    add_global_legend(
        fig,
        eta_handles + cost_handles,
        eta_labels + cost_labels,
        ncol=4,
        bbox_to_anchor=(0.5, 1.16),
    )

    output_paths: List[str] = []
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{out_base}.{ext}")
        fig.savefig(
            out_path,
            bbox_inches="tight",
            dpi=SAVEFIG_DPI if ext == "png" else None,
        )
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def plot_paired_frontier_regret_vs_mismatch(
    df: pd.DataFrame, out_base: str, output_dir: str
) -> List[str]:
    ms = sorted(df["M"].unique().tolist())
    etas = sorted(df["eta"].unique().tolist())

    atol = 1e-5
    baseline_mask = np.isclose(df["C_frac"], 0.0, atol=atol)
    baseline = df[baseline_mask].copy()
    teacher = df[~baseline_mask].copy()

    budget_keys = sorted({round(float(c), 2) for c in teacher["C_frac"].unique().tolist()})
    if not budget_keys:
        raise ValueError("No teacher budgets found (C/T > 0) for paired frontier plot.")
    final_budget = max(budget_keys)

    nrows = len(etas)
    ncols = len(ms)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=True,
        sharey=True,
        figsize=(7.2, 5.8),
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    x_values: List[float] = []
    y_values: List[float] = []

    for row_idx, eta in enumerate(etas):
        for col_idx, M in enumerate(ms):
            ax = axes[row_idx, col_idx]
            ax.tick_params(labelleft=(col_idx == 0), labelbottom=(row_idx == nrows - 1))

            panel_base = baseline[(baseline["M"] == M) & (baseline["eta"] == eta)]
            panel_teacher = teacher[(teacher["M"] == M) & (teacher["eta"] == eta)]

            base_seeds = set(panel_base["seed"].tolist())
            teacher_seeds = set(panel_teacher["seed"].tolist())

            if not base_seeds or not teacher_seeds:
                apply_common_axis_style(ax)
                continue

            for budget in budget_keys:
                budget_mask = np.isclose(panel_teacher["C_frac"], budget, atol=atol)
                budget_seeds = set(panel_teacher[budget_mask]["seed"].tolist())
                missing = base_seeds - budget_seeds
                if missing:
                    print(
                        "WARNING: missing seeds for "
                        f"M={M}, eta={eta}, C/T={budget:.2f}: {len(missing)} missing"
                    )

            panel_base_idx = panel_base.set_index("seed")

            deltas_x: List[float] = []
            deltas_y: List[float] = []

            for seed, base_row in panel_base_idx.iterrows():
                seed_teacher = panel_teacher[panel_teacher["seed"] == seed]
                if seed_teacher.empty:
                    continue

                x0 = float(base_row["Mismatch"])
                y0 = float(base_row["AvgReg"])
                points = [(0.0, x0, y0)]

                for budget in budget_keys:
                    budget_rows = seed_teacher[
                        np.isclose(seed_teacher["C_frac"], budget, atol=atol)
                    ]
                    if budget_rows.empty:
                        continue
                    row = budget_rows.iloc[0]
                    points.append((budget, float(row["Mismatch"]), float(row["AvgReg"])))

                if len(points) < 2:
                    continue

                points = sorted(points, key=lambda item: item[0])
                xs = [p[1] for p in points]
                ys = [p[2] for p in points]

                ax.plot(xs, ys, color="0.5", linewidth=0.8, alpha=0.45, zorder=2)
                if len(xs) >= 2:
                    arrow = FancyArrowPatch(
                        (xs[-2], ys[-2]),
                        (xs[-1], ys[-1]),
                        arrowstyle="-|>",
                        mutation_scale=8,
                        linewidth=0.8,
                        color="0.5",
                        alpha=0.6,
                        zorder=2,
                    )
                    ax.add_patch(arrow)

                ax.scatter(
                    x0,
                    y0,
                    s=MARKER_SIZE * 9.0,
                    color="0.2",
                    edgecolor="0.2",
                    linewidths=0.4,
                    zorder=3,
                )

                for budget, x_val, y_val in points[1:]:
                    budget_key = round(float(budget), 2)
                    color = FRONTIER_BUDGET_COLORS.get(budget_key, "#4C78A8")
                    marker = FRONTIER_BUDGET_MARKERS.get(budget_key, "o")
                    ax.scatter(
                        x_val,
                        y_val,
                        s=MARKER_SIZE * 10.5,
                        marker=marker,
                        facecolor="white",
                        edgecolor=color,
                        linewidths=1.0,
                        zorder=4,
                    )

                x_values.extend(xs)
                y_values.extend(ys)

                if round(float(points[-1][0]), 2) == final_budget:
                    deltas_x.append(points[-1][1] - x0)
                    deltas_y.append(points[-1][2] - y0)

            if deltas_x:
                pct_improved = 100.0 * (np.array(deltas_x) < 0.0).mean()
                med_dx = float(np.median(deltas_x))
                med_dy = float(np.median(deltas_y))
                annotation = (
                    f"%$\\Delta x<0$: {pct_improved:.0f}%\n"
                    f"med $\\Delta x$={med_dx:.3f}\n"
                    f"med $\\Delta y$={med_dy:.3f}"
                )
                ax.text(
                    0.03,
                    0.97,
                    annotation,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=7,
                    color="0.4",
                )

            if row_idx == 0:
                ax.set_title(rf"$M={M}$")
            if col_idx == 0:
                ax.text(
                    -0.32,
                    0.5,
                    rf"$\eta={eta}$",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=TITLE_SIZE,
                    color="0.2",
                    clip_on=False,
                )

            apply_common_axis_style(ax)

    xlim = nice_bounds(np.array(x_values))
    ylim = nice_bounds(np.array(y_values))
    if xlim is not None:
        for ax in axes.ravel():
            ax.set_xlim(*xlim)
    if ylim is not None:
        for ax in axes.ravel():
            ax.set_ylim(*ylim)

    fig.supxlabel(r"Mismatch $\downarrow$")
    fig.supylabel(r"DynReg/K $\downarrow$")

    handles: List[plt.Line2D] = []
    labels: List[str] = []

    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="0.2",
            markerfacecolor="0.2",
            markeredgecolor="0.2",
            markersize=MARKER_SIZE,
        )
    )
    labels.append("C/T=0 (baseline)")

    for budget in budget_keys:
        budget_key = round(float(budget), 2)
        color = FRONTIER_BUDGET_COLORS.get(budget_key, "#4C78A8")
        marker = FRONTIER_BUDGET_MARKERS.get(budget_key, "o")
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markersize=MARKER_SIZE,
            )
        )
        labels.append(rf"C/T={budget_key:.2f}")

    handles.append(
        plt.Line2D(
            [0, 1],
            [0, 0],
            color="0.5",
            linewidth=0.9,
        )
    )
    labels.append("paired seed trajectory (baseline -> teacher)")

    add_global_legend(
        fig,
        handles,
        labels,
        ncol=3,
        bbox_to_anchor=(0.5, 1.09),
    )

    output_paths: List[str] = []
    for ext in ("pdf", "png"):
        out_path = os.path.join(output_dir, f"{out_base}.{ext}")
        fig.savefig(
            out_path,
            bbox_inches="tight",
            dpi=SAVEFIG_DPI if ext == "png" else None,
        )
        output_paths.append(out_path)

    plt.close(fig)
    return output_paths


def format_mean_std(mean: float, std: float, fmt: str) -> str:
    return f"{fmt.format(mean)} +/- {fmt.format(std)}"


def format_mean_std_tex(mean: float, std: float, fmt: str) -> str:
    return f"{fmt.format(mean)} \\pm {fmt.format(std)}"


def write_global_tradeoff_table(df: pd.DataFrame, output_dir: str) -> Tuple[str, str, str]:
    grouped = df.groupby("C_frac").agg(
        dynreg_mean=("DynReg", "mean"),
        dynreg_std=("DynReg", "std"),
        avgreg_mean=("AvgReg", "mean"),
        avgreg_std=("AvgReg", "std"),
        mismatch_mean=("Mismatch", "mean"),
        mismatch_std=("Mismatch", "std"),
        stab_mean=("Stab", "mean"),
        stab_std=("Stab", "std"),
    )
    grouped = grouped.reset_index().sort_values("C_frac")

    rows = []
    tex_rows = []
    for _, row in grouped.iterrows():
        cfrac = f"{row['C_frac']:.2f}"
        dynreg = format_mean_std(row["dynreg_mean"], row["dynreg_std"], "{:.2f}")
        avgreg = format_mean_std(row["avgreg_mean"], row["avgreg_std"], "{:.2e}")
        mismatch = format_mean_std(row["mismatch_mean"], row["mismatch_std"], "{:.3f}")
        stab = format_mean_std(row["stab_mean"], row["stab_std"], "{:.3f}")

        rows.append(
            {
                "C_frac": cfrac,
                "DynReg_K": dynreg,
                "DynReg_K/K": avgreg,
                "Mismatch_K": mismatch,
                "Stab_K": stab,
            }
        )

        dynreg_tex = format_mean_std_tex(row["dynreg_mean"], row["dynreg_std"], "{:.2f}")
        avgreg_tex = format_mean_std_tex(row["avgreg_mean"], row["avgreg_std"], "{:.2e}")
        mismatch_tex = format_mean_std_tex(
            row["mismatch_mean"], row["mismatch_std"], "{:.3f}"
        )
        stab_tex = format_mean_std_tex(row["stab_mean"], row["stab_std"], "{:.3f}")

        tex_rows.append(
            f"{cfrac} & {dynreg_tex} & {avgreg_tex} & {mismatch_tex} & {stab_tex} {LATEX_LINEBREAK}"
        )

    csv_path = os.path.join(output_dir, "tradeoff_summary_global.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    tex_lines = [
        "\\begin{tabular}{lcccc}",
        "\\hline",
        f"C/T & DynReg$_K$ & DynReg$_K/K$ & Mismatch$_K$ & Stab$_K$ {LATEX_LINEBREAK}",
        "\\hline",
        *tex_rows,
        "\\hline",
        "\\end{tabular}",
    ]
    tex_block = "\n".join(tex_lines)

    tex_path = os.path.join(output_dir, "tradeoff_summary_global.tex")
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write(tex_block)

    return csv_path, tex_path, tex_block


def write_appendix_figures_tex(output_dir: str) -> str:
    appendix_path = os.path.join(output_dir, "appendix_figures.tex")
    caption_delta = (
        "Paired changes relative to the no-teacher baseline ($C/T=0$) "
        "across budgets. Error bars denote bootstrap 95\\% confidence intervals over seeds, "
        "computed on paired differences relative to the no-teacher baseline (C_frac=0) using common random seeds."
    )
    caption_frontier = (
        "Paired seed trajectories in mismatch--dynamic regret space for each $(\\eta, M)$. "
        "Gray arrows connect the same seed from the baseline ($C/T=0$) through increasing budgets, "
        "with arrowheads at the largest budget. Hollow markers denote teacher budgets "
        "($C/T\\in\\{0.05, 0.10, 0.20\\}$), and filled dots denote the baseline. "
        "Here $\\Delta x$ and $\\Delta y$ are computed per seed as the change from baseline "
        "to the largest budget in mismatch ($x$) and dynamic regret ($y$), respectively."
    )

    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{DELTA_OUTPUT_DIRNAME}/appendix_delta_mismatch_vs_cost.pdf}}",
        f"\\caption{{{caption_delta}}}",
        "\\label{fig:appendix-delta-mismatch}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{DELTA_OUTPUT_DIRNAME}/appendix_delta_avgreg_vs_cost.pdf}}",
        f"\\caption{{{caption_delta}}}",
        "\\label{fig:appendix-delta-avgreg}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{DELTA_OUTPUT_DIRNAME}/appendix_delta_stab_vs_cost.pdf}}",
        f"\\caption{{{caption_delta}}}",
        "\\label{fig:appendix-delta-stab}",
        "\\end{figure}",
        "",
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{DELTA_OUTPUT_DIRNAME}/appendix_paired_regret_vs_mismatch.pdf}}",
        f"\\caption{{{caption_frontier}}}",
        "\\label{fig:appendix-paired-frontier}",
        "\\end{figure}",
    ]
    content = "\n".join(lines)
    with open(appendix_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return appendix_path


def write_paired_frontier_tex(output_dir: str) -> str:
    tex_path = os.path.join(output_dir, "appendix_paired_regret_vs_mismatch.tex")
    caption = (
        "Paired seed trajectories in mismatch--dynamic regret space for each $(\\eta, M)$. "
        "Gray arrows connect the same seed from the baseline ($C/T=0$) through increasing budgets, "
        "with arrowheads at the largest budget. Hollow markers denote teacher budgets "
        "($C/T\\in\\{0.05, 0.10, 0.20\\}$), and filled dots denote the baseline. "
        "Here $\\Delta x$ and $\\Delta y$ are computed per seed as the change from baseline "
        "to the largest budget in mismatch ($x$) and dynamic regret ($y$), respectively."
    )
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        f"\\includegraphics[width=\\linewidth]{{{DELTA_OUTPUT_DIRNAME}/appendix_paired_regret_vs_mismatch.pdf}}",
        f"\\caption{{{caption}}}",
        "\\label{fig:appendix-paired-frontier}",
        "\\end{figure}",
    ]
    with open(tex_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return tex_path


def generate_all(
    results_dir: str,
    output_dir: Optional[str] = None,
    expected_seed_count: Optional[int] = 5,
    csv_path: Optional[str] = None,
) -> Dict[str, List[str]]:
    output_dir = os.path.abspath(output_dir or results_dir)
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, DELTA_OUTPUT_DIRNAME)
    os.makedirs(figures_dir, exist_ok=True)

    df, csv_path = load_results(results_dir, csv_path=csv_path)
    validate_columns(df)
    warn_missing_runs(df, expected_seed_count)

    set_pub_style()

    outputs: Dict[str, List[str]] = {}

    outputs["figure_teaching_error_vs_budget"] = plot_metric_vs_budget(
        df,
        metric_col="Mismatch",
        ylabel="Mismatch",
        out_base="figure_teaching_error_vs_budget",
        output_dir=output_dir,
    )

    outputs["figure_avgreg_vs_budget"] = plot_metric_vs_budget(
        df,
        metric_col="AvgReg",
        ylabel="DynReg/K",
        out_base="figure_avgreg_vs_budget",
        output_dir=output_dir,
    )

    outputs["figure_stability_vs_budget"] = plot_metric_vs_budget(
        df,
        metric_col="Stab",
        ylabel="Stab.",
        out_base="figure_stability_vs_budget",
        output_dir=output_dir,
    )

    outputs["appendix_regret_vs_mismatch"] = plot_regret_mismatch_frontier(
        df,
        out_base="appendix_regret_vs_mismatch",
        output_dir=output_dir,
    )

    paired = compute_paired_deltas(df)
    outputs["appendix_delta_mismatch_vs_cost"] = plot_delta_metric_vs_budget(
        paired,
        delta_col="DeltaMismatch",
        ylabel=r"$\Delta$Mismatch",
        out_base="appendix_delta_mismatch_vs_cost",
        output_dir=figures_dir,
    )
    outputs["appendix_delta_avgreg_vs_cost"] = plot_delta_metric_vs_budget(
        paired,
        delta_col="DeltaAvgReg",
        ylabel=r"$\Delta$DynReg/K",
        out_base="appendix_delta_avgreg_vs_cost",
        output_dir=figures_dir,
    )
    outputs["appendix_delta_stab_vs_cost"] = plot_delta_metric_vs_budget(
        paired,
        delta_col="DeltaStab",
        ylabel=r"$\Delta$Stab.",
        out_base="appendix_delta_stab_vs_cost",
        output_dir=figures_dir,
    )
    outputs["appendix_paired_regret_vs_mismatch"] = plot_paired_frontier_regret_vs_mismatch(
        df,
        out_base="appendix_paired_regret_vs_mismatch",
        output_dir=figures_dir,
    )

    csv_out, tex_out, tex_block = write_global_tradeoff_table(df, output_dir)
    outputs["tradeoff_summary_global"] = [csv_out, tex_out]

    appendix_tex = write_appendix_figures_tex(figures_dir)
    outputs["appendix_figures_tex"] = [appendix_tex]
    paired_frontier_tex = write_paired_frontier_tex(figures_dir)
    outputs["appendix_paired_regret_vs_mismatch_tex"] = [paired_frontier_tex]

    summarize_sign_consistency(paired, "DeltaMismatch", "DeltaMismatch")
    summarize_sign_consistency(paired, "DeltaAvgReg", "DeltaAvgReg")
    summarize_sign_consistency(paired, "DeltaStab", "DeltaStab")

    print("\nLaTeX tabular block:\n")
    print(tex_block)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots and global summary table for bandit results."
    )
    parser.add_argument(
        "--results_dir",
        default=".",
        help="Directory containing results CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write figures and tables (defaults to results_dir).",
    )
    parser.add_argument(
        "--expected_seed_count",
        type=int,
        default=5,
        help="Expected number of distinct seeds (for warnings).",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="Optional explicit results CSV path.",
    )
    args = parser.parse_args()

    generate_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        expected_seed_count=args.expected_seed_count,
        csv_path=args.csv_path,
    )


if __name__ == "__main__":
    main()
