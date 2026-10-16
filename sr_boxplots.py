"""Subject-level boxplots and paired Wilcoxon tests for two distinct questions.

Input must have one row per subject × network × path × metric × atlas × scanner pair.
Question 1: MRR differs from Supersynth within MSN and within MIND.
Question 2: MSN differs from MIND within MRR and within Supersynth.
For each metric and question separately, FDR-BH correction is across 8 tests.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

INPUT = Path("/data/project/unity/Hajer/PhD/recon-any/KCL-HYPE/subject_topology_metrics_summary_fsaverage.csv")
#INPUT_FS5= Path("/data/project/unity/Hajer/PhD/recon-any/KCL-HYPE/subject_topology_metrics_summary_fsaverage5.csv")
OUT = Path("/data/project/unity/Hajer/PhD/paper/figures/boxplots")
N_SUBJECTS = 23
PATH_ORDER = ["MRR", "Supersynth"]
NETWORK_ORDER = ["MSN", "MIND"]
NETWORK_PALETTE = {"MSN": "#ab8f53", "MIND": "#c0392b"}
MRR_ALPHA = 0.40
SUPERSYNTH_ALPHA = 1.00
BOXPLOT_WIDTH = 0.65
# These match Seaborn's two-level hue dodge for width=0.65: +/- (0.65 / 4).
HUE_OFFSET = BOXPLOT_WIDTH / 4
POINT_JITTER = 0.10
POINT_SIZE = 2.8
METRIC_LABELS = {"r": "Pearson r", "CCC": "CCC", "ICC": "ICC(3,1)"}
ATLAS_LABELS = {
    "aparc_5feat": "$DK_{68}$",
    "aparc.a2009s_5feat": "$Destrieux_{148}$",
}
SCANNER_TITLES = {
    "GE-HFC": "$HF-ULF_{S1}$",
    "GE-HFE": "$HF-ULF_{S2}$",
    "HFC-HFE": "$ULF_{S1}-ULF_{S2}$",
}


def load_and_validate(input) -> pd.DataFrame:
    """Load the already subject-level long data and validate every condition."""
    data = pd.read_csv(input).copy()
    
    required = {"subject", "method", "path", "metric", "value", "atlas_nfeat", "scanner_pair"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    data["network"] = data["method"].astype(str).str.upper()
    data["path"] = data["path"].astype(str).str.upper().map(
        {"MRR": "MRR", "SUPERSYNTH": "Supersynth"}
    )
    if not data["network"].isin(NETWORK_ORDER).all():
        raise ValueError(f"Unexpected network values: {sorted(data['method'].unique())}")
    if data["path"].isna().any():
        raise ValueError("Path must contain only MRR and Supersynth.")

    cell_keys = ["subject", "network", "path", "metric", "atlas_nfeat", "scanner_pair"]
    if data.duplicated(cell_keys, keep=False).any():
        raise ValueError("Duplicate rows exist within a subject-level analytical cell.")

    counts = data.groupby(cell_keys[1:], observed=True)["subject"].nunique()
    print("Subjects per plotted/tested condition (every value must be 23):\n", counts.to_string())
    if not counts.eq(N_SUBJECTS).all():
        raise ValueError(f"Every condition must contain exactly {N_SUBJECTS} subjects.")
    return data


def paired_tests(
    data: pd.DataFrame,
    metric: str,
    compared_variable: str,
    level_a: str,
    level_b: str,
    group_cols: list[str],
    test_family: str,
) -> pd.DataFrame:
    """Run paired, two-sided Wilcoxon tests and FDR-BH within metric/family."""
    rows = []
    metric_data = data.loc[data["metric"] == metric]
    for labels, subset in metric_data.groupby(group_cols, sort=False, observed=True):
        wide = subset.pivot(index="subject", columns=compared_variable, values="value")
        wide = wide.reindex(columns=[level_a, level_b]).dropna()
        if len(wide) != N_SUBJECTS:
            raise ValueError(f"Expected {N_SUBJECTS} complete pairs for {labels}; got {len(wide)}.")

        difference = wide[level_b] - wide[level_a]
        statistic, p_value = wilcoxon(wide[level_b], wide[level_a], alternative="two-sided")
        rows.append({
            **dict(zip(group_cols, labels)),
            "metric": metric,
            "test_family": test_family,
            "comparison": f"{level_a} vs {level_b}",
            "compared_variable": compared_variable,
            "N_pairs": len(wide),
            f"mean_{level_b}_minus_{level_a}": difference.mean(),
            f"median_{level_b}_minus_{level_a}": difference.median(),
            "wilcoxon_statistic": statistic,
            "p_uncorrected": p_value,
        })

    results = pd.DataFrame(rows)
    results["p_fdr_bh_within_metric_and_family"] = multipletests(
        results["p_uncorrected"], method="fdr_bh"
    )[1]
    return results


def significance_label(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_bracket(ax, x1: float, x2: float, y: float, h: float, label: str, color: str = "black") -> None:
    """Draw an FDR-adjusted significance bracket using both figure styles."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=color, lw=0.9, clip_on=False)
    ax.text((x1 + x2) / 2, y + h, label, ha="center", va="bottom", fontsize=11, color=color)


def add_atlas_side_label(ax, atlas: str) -> None:
    ax.annotate(
        ATLAS_LABELS[atlas], xy=(-0.34, 0.5), xycoords="axes fraction",
        rotation=90, ha="center", va="center", fontsize=13,
    )


def add_network_brackets(ax, panel_stats: pd.DataFrame, ymin: float, span: float) -> None:
    """Network plot: compare MSN vs MIND at each fixed SR method."""
    for position, path in enumerate(PATH_ORDER):
        stat = panel_stats.loc[panel_stats["path"] == path].iloc[0]
        y = ymin + span * (1.08 + 0.13 * position)
        add_bracket(
            ax, position - HUE_OFFSET, position + HUE_OFFSET, y, span * 0.025,
            significance_label(stat["p_fdr_bh_within_metric_and_family"]),
        )


def draw_sr_method_panel(
    ax, panel: pd.DataFrame, panel_stats: pd.DataFrame, ymin: float, span: float
) -> None:
    """SR-method plot with Seaborn-matched geometry and custom network/path colors.

    Matplotlib is necessary here because box color depends jointly on network and
    path. Width, dodge offset, jitter, point size, line width, grid, and z-order
    deliberately match the Seaborn network-comparison panels.
    """
    offsets = {"MRR": -HUE_OFFSET, "Supersynth": HUE_OFFSET}
    box_width = BOXPLOT_WIDTH / len(PATH_ORDER)
    rng = np.random.default_rng(20260921)

    for x_center, network in enumerate(NETWORK_ORDER):
        base_color = NETWORK_PALETTE[network]
        for path in PATH_ORDER:
            x_position = x_center + offsets[path]
            values = panel.loc[
                (panel["network"] == network) & (panel["path"] == path), "value"
            ].to_numpy()
            alpha = MRR_ALPHA if path == "MRR" else SUPERSYNTH_ALPHA
            fill_color = to_rgba(base_color, alpha)

            # Seaborn-style box geometry; edge colour remains network-specific so
            # translucent MRR fills stay legible on the whitegrid background.
            ax.boxplot(
                values,
                positions=[x_position],
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                zorder=2,
                boxprops={"facecolor": fill_color, "edgecolor": base_color, "linewidth": 0.8},
                whiskerprops={"color": base_color, "linewidth": 0.8},
                capprops={"color": base_color, "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 0.9},
            )
            jitter = rng.uniform(-POINT_JITTER, POINT_JITTER, len(values))
            ax.scatter(
                x_position + jitter, values,
                s=POINT_SIZE ** 2, color=fill_color, edgecolors="none", zorder=3,
            )

        stat = panel_stats.loc[panel_stats["network"] == network].iloc[0]
        y = ymin + span * (1.08 + 0.13 * x_center)
        add_bracket(
            ax, x_center + offsets["MRR"], x_center + offsets["Supersynth"],
            y, span * 0.025,
            significance_label(stat["p_fdr_bh_within_metric_and_family"]),
        )

    ax.set_xticks(range(len(NETWORK_ORDER)))
    ax.set_xticklabels(NETWORK_ORDER)


def plot_metric(data: pd.DataFrame, results: pd.DataFrame, metric: str, question: str) -> None:
    """Create one consistently styled faceted figure for one scientific question."""
    atlases = [atlas for atlas in ATLAS_LABELS if atlas in data["atlas_nfeat"].unique()]
    pairs = sorted(data["scanner_pair"].unique())
    figure, axes = plt.subplots(
        len(atlases), len(pairs), figsize=(3.6 * len(pairs), 3.1 * len(atlases)),
        sharex=True, sharey=True, squeeze=False,
    )
    metric_values = data.loc[data["metric"] == metric, "value"]
    ymin, ymax = metric_values.min(), metric_values.max()
    span = max(ymax - ymin, 0.02)

    for row, atlas in enumerate(atlases):
        for col, scanner_pair in enumerate(pairs):
            ax = axes[row, col]
            panel = data.loc[
                (data["metric"] == metric)
                & (data["atlas_nfeat"] == atlas)
                & (data["scanner_pair"] == scanner_pair)
            ]
            panel_stats = results.loc[
                (results["atlas_nfeat"] == atlas)
                & (results["scanner_pair"] == scanner_pair)
            ]

            if question == "network":
                # Original layout: path on x-axis; network as hue.
                sns.boxplot(
                    data=panel, x="path", y="value", hue="network", order=PATH_ORDER,
                    hue_order=NETWORK_ORDER, palette=NETWORK_PALETTE, width=BOXPLOT_WIDTH,
                    showfliers=False, linewidth=0.8, ax=ax,
                )
                sns.stripplot(
                    data=panel, x="path", y="value", hue="network", order=PATH_ORDER,
                    hue_order=NETWORK_ORDER, palette=NETWORK_PALETTE, dodge=True,
                    jitter=POINT_JITTER, alpha=0.65, size=POINT_SIZE, legend=False, ax=ax,
                )
                legend = ax.get_legend()
                if legend is not None:
                    legend.remove()
                add_network_brackets(ax, panel_stats, ymin, span)
                x_label = "Super-resolution method"
            elif question == "sr_method":
                # Distinct layout: network on x-axis; paths adjacent within network.
                draw_sr_method_panel(ax, panel, panel_stats, ymin, span)
                x_label = "Network"
            else:
                raise ValueError("question must be 'sr_method' or 'network'.")

            ax.set_ylim(ymin - span * 0.05, ymin + span * 1.35)
            ax.set_title(SCANNER_TITLES.get(scanner_pair, scanner_pair) if row == 0 else "")
            ax.set_xlabel(x_label if row == len(atlases) - 1 else "")
            ax.set_ylabel(METRIC_LABELS.get(metric, metric) if col == 0 else "")
            if col == 0:
                add_atlas_side_label(ax, atlas)
            sns.despine(ax=ax)

    if question == "network":
        handles = [Patch(facecolor=NETWORK_PALETTE[name], label=name) for name in NETWORK_ORDER]
        figure.legend(handles=handles, title="Network", loc="upper right")
        title = f"Edge-level {METRIC_LABELS.get(metric, metric)}: MSN vs MIND within SR method"
        filename = f"{metric}_network_within_SRmethod.png"
    else:
        handles = [
            Patch(facecolor=to_rgba(NETWORK_PALETTE["MSN"], MRR_ALPHA), label="MSN — MRR"),
            Patch(facecolor=to_rgba(NETWORK_PALETTE["MSN"], SUPERSYNTH_ALPHA), label="MSN — Supersynth"),
            Patch(facecolor=to_rgba(NETWORK_PALETTE["MIND"], MRR_ALPHA), label="MIND — MRR"),
            Patch(facecolor=to_rgba(NETWORK_PALETTE["MIND"], SUPERSYNTH_ALPHA), label="MIND — Supersynth"),
        ]
        figure.legend(handles=handles, title="Network and SR method", loc="upper right")
        title = f"Edge-level {METRIC_LABELS.get(metric, metric)}: MRR vs Supersynth within network"
        filename = f"{metric}_SRmethod_within_network.png"

    figure.suptitle(title, y=1.03)
    figure.tight_layout()
    figure.savefig(OUT / filename, dpi=400, bbox_inches="tight")
    figure.savefig(OUT / filename.replace(".png", ".svg"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    data = load_and_validate(INPUT)
    #data_fs5 = load_and_validate(INPUT_FS5)
    
    #Merge data and data_fs5
    #data = pd.concat([data, data_fs5])
    #data = data[data["network"]=="mind"]
    all_sr_results = []
    all_network_results = []
    for metric in data["metric"].unique():
        sr_results = paired_tests(
            data=data, metric=metric, compared_variable="path",
            level_a="MRR", level_b="Supersynth",
            group_cols=["network", "atlas_nfeat", "scanner_pair"],
            test_family="SR_method_within_network"
        )
        all_sr_results.append(sr_results)
        plot_metric(data, sr_results, metric, question="sr_method")

        network_results = paired_tests(
            data=data, metric=metric, compared_variable="network",
            level_a="MSN", level_b="MIND",
            group_cols=["path", "atlas_nfeat", "scanner_pair"],
            test_family="network_within_SR_method",
        )
       
        
        all_network_results.append(network_results)
        
        plot_metric(data, network_results, metric, question="network")

    pd.concat(all_sr_results, ignore_index=True).to_csv(
        OUT / "paired_MRR_vs_Supersynth_wilcoxon.csv", index=False
    )
    pd.concat(all_network_results, ignore_index=True).to_csv(
        OUT / "paired_MSN_vs_MIND_wilcoxon.csv", index=False
    )


if __name__ == "__main__":
    main()
