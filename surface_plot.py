#!/usr/bin/env python3
"""
Plot parcel-wise agreement values on the Desikan-Killiany atlas using surfplot.

This script expects a CSV/TSV table with region names that already exactly match
Desikan-Killiany cortical labels, e.g.:
    lh_bankssts, rh_cuneus, lh_superiorfrontal, ...

Expected columns include at least:
    hf_label, lf_label, region, ccc
and optionally:
    subject

Example
-------
python plot_dk_ccc_surfplot.py \
    --input agreement_values.csv \
    --hf GE \
    --lf HFC \
    --metric ccc \
    --agg median \
    --output dk_ccc_GE_HFC.png

If nilearn cannot provide DK annotation files, you can pass them explicitly:
python plot_dk_ccc_surfplot.py \
    --input agreement_values.csv \
    --hf GE \
    --lf HFC \
    --metric ccc \
    --lh-annot $SUBJECTS_DIR/fsaverage5/label/lh.aparc.annot \
    --rh-annot $SUBJECTS_DIR/fsaverage5/label/rh.aparc.annot \
    --output dk_ccc_GE_HFC.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets
from surfplot import Plot
import os 

import matplotlib.colors as mcolors
import palettable.wesanderson as wesandersonp
from matplotlib.colors import LinearSegmentedColormap

wes_cmap = mcolors.LinearSegmentedColormap.from_list("IsleOfDogs3_4", wesandersonp.IsleOfDogs3_4.mpl_colors)


METRIC_MAP = {
    "pearson_r": "r",
    "ccc": "CCC",
    "icc3": "ICC",
    "metric":"metric"
}
CMAP="PuOr"
CMAP_MSN_MIND= {"MSN":wesandersonp.IsleOfDogs1_5.mpl_colormap, "MIND":wesandersonp.IsleOfDogs1_5.mpl_colormap}

def get_cmap(label):
    global CMAP_MSN_MIND
    
    base = wesandersonp.GrandBudapest1_4_r.hex_colors
    print(base)
    # assume: base = [pale, mid1, orange, brown, black]
    dark, mid_dark, mid, pale = base  # adjust if your order is different

    # --- MIND: all positive, warm, no black ---
    center = "#ffffff"  # low values

    # MIND: light → warm (no dark)
    mind_cmap = LinearSegmentedColormap.from_list(
    "mind_isle",
    [mid_dark, pale, mid],
    #[center, pale, mid],   # note: no brown, no black
    N=256,
    )

    # MSN: keep diverging as before
    msn_cmap = LinearSegmentedColormap.from_list(
        "msn_isle_div",
        [mid_dark, dark, center, pale, mid],
        N=256,
    )
    CMAP_MSN_MIND = {"MSN": msn_cmap, "MIND": mind_cmap}
    return CMAP_MSN_MIND.get(label, "PuOr")
    
def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or TSV automatically based on filename suffix."""
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def load_annot(annot_file: str | Path):
    """Load a FreeSurfer annotation file and return labels plus decoded names."""
    labels, _, names = nib.freesurfer.read_annot(str(annot_file))
    names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
    return labels, names


def build_vertex_map(roi_df: pd.DataFrame, annot_file: str | Path, hemi: str, metric: str) -> np.ndarray:
    """Map parcel-level values onto vertices for one hemisphere."""
    labels, names = load_annot(annot_file)
    vertex_data = np.full(labels.shape, np.nan, dtype=float)

    hemi_df = roi_df[roi_df["region"].str.startswith(f"{hemi}_")].copy()
    value_map = dict(zip(hemi_df["region"], hemi_df[metric]))

    for idx, name in enumerate(names):
        if name in {"unknown", "corpuscallosum"}:
            continue
        region_key = f"{hemi}_{name}"
        if region_key in value_map:
            vertex_data[labels == idx] = value_map[region_key]

    return vertex_data


def get_fsaverage_annotations(fsavg,atlas="aparc"):
    """Find left/right annotation files from nilearn fsaverage fetch results."""
    lh_candidates = [
        getattr(fsavg, f"annot_left", None),
        getattr(fsavg, f"{atlas}_left", None),
    ]
    rh_candidates = [
        getattr(fsavg, f"annot_right", None),
        getattr(fsavg, f"{atlas}_right", None),
    ]

    lh_annot = next((x for x in lh_candidates if x is not None), None)
    rh_annot = next((x for x in rh_candidates if x is not None), None)

    if lh_annot is None or rh_annot is None:
        raise RuntimeError(
            "Could not find fsaverage annotation files in nilearn's fetched dataset. "
            "Provide --lh-annot and --rh-annot explicitly, or try updating nilearn."
        )

    return lh_annot, rh_annot


def main():
    parser = argparse.ArgumentParser(description="Plot DK parcel values with surfplot")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV or TSV file")
    parser.add_argument("--hf", required=True, help="High-field label to filter, e.g. GE")
    parser.add_argument("--lf", required=True, help="Low-field label to filter, e.g. HFC")
    parser.add_argument("--nfeat", help="nfeat, e.g. aparc_5feat")
    parser.add_argument("--metric", default="ccc", help="Metric column to plot, e.g. ccc")
    parser.add_argument("--agg", default="median", choices=["mean", "median"], help="Aggregate across rows within region")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale minimum")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--title", default=None, help="Optional figure title")
    parser.add_argument("--mesh", default="fsaverage5", choices=["fsaverage5", "fsaverage"], help="Surface mesh to use")
    parser.add_argument('--input_type', default='supersynth', choices=['mrr', 'supersynth'], help='Type of input data')
    parser.add_argument("--lh-annot", type=Path, default="/Users/Hajer/unity/PhD/Adult/fsaverage5/label/lh.aparc.annot", help="Optional explicit path to lh.aparc.annot")
    parser.add_argument("--rh-annot", type=Path, default="/Users/Hajer/unity/PhD/Adult/fsaverage5/label/rh.aparc.annot", help="Optional explicit path to rh.aparc.annot")
    parser.add_argument("--output", required=True, type=Path, help="Output image path")
    parser.add_argument("--atlas", default="aparc", help="Atlas name, e.g. aparc")
    parser.add_argument("--method",default="mind",help="MSN or mind network")
    args = parser.parse_args()

    atlas_map = {
        "aparc.a2009s": "Destrieux",
        "aparc.DKTatlas": "DKT",
        "aparc": "Desikan-Killiany",
    }
    global CMAP
    
    CMAP = get_cmap(args.method)

    scanner_label_mapping={"GE": "$HF$", "HFC": "$ULF_{S1}$", "HFE": "$ULF_{S2}$", "3T": "$HF_{3T}$", "1.5T": "$HF_{1.5T}$", "Hyperfine": "$ULF$"}
    parcellation_text = f"Parcellation: {atlas_map.get(args.atlas, args.atlas)} | N-features: {args.nfeat} | Template: {args.mesh} | Scanner pair: {scanner_label_mapping.get(args.hf, args.hf)}-{scanner_label_mapping.get(args.lf, args.lf)}"
    print(parcellation_text)
    df = read_table(args.input)
    df = df[(df["path"] == args.input_type) & (df["network"] == args.method)]
    
    # DF AFTER FILTERING BY INPUT TYPE AND NETWORK:", df.shape)
    
    df = df[df["atlas_nfeat"] == f"{args.atlas}_{args.nfeat}"]
    print(f"Read {len(df)} rows from {args.input} with atlas_nfeat= {args.atlas}_{args.nfeat}")
    print("Minimum value of metric column:", df[args.metric].min())
    
    required = {"hf_label", "lf_label", "region", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    print(df.columns)
    
    df["scanner_pair"] = df["hf_label"] + "-" + df["lf_label"]
    print(df.scanner_pair.unique(), args.hf, args.lf)
    #return
    sub = df[df["scanner_pair"] == f"{args.hf}-{args.lf}"].copy()
    
    #sub = df[(df["hf_label"] == args.hf) & (df["lf_label"] == args.lf)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for hf_label={args.hf} and lf_label={args.lf}")

    sub = sub.dropna(subset=["region", args.metric]).copy()
    if sub.empty:
        raise ValueError("No rows remain after dropping missing region/metric values.")

    if args.agg == "median":
        roi_df = sub.groupby("region", as_index=False)[args.metric].median()
    else:
        roi_df = sub.groupby("region", as_index=False)[args.metric].mean()

    fsavg = datasets.fetch_surf_fsaverage(mesh=args.mesh)
    lh_annot = f"/Users/Hajer/unity/PhD/Adult/{args.mesh}/label/lh.{args.atlas}.annot"
    rh_annot = f"/Users/Hajer/unity/PhD/Adult/{args.mesh}/label/rh.{args.atlas}.annot"
    
    # if args.lh_annot is not None or args.rh_annot is not None:
    #     if args.lh_annot is None or args.rh_annot is None:
    #         raise ValueError("Provide both --lh-annot and --rh-annot, or neither.")
    #     lh_annot, rh_annot = args.lh_annot, args.rh_annot
    # else:
    #     lh_annot, rh_annot = get_fsaverage_annotations(fsavg)

    #print(roi_df)
    lh_data = build_vertex_map(roi_df, lh_annot, hemi="lh", metric=args.metric)
    rh_data = build_vertex_map(roi_df, rh_annot, hemi="rh", metric=args.metric)

    combined = np.concatenate([lh_data[np.isfinite(lh_data)], rh_data[np.isfinite(rh_data)]])
    if combined.size == 0:
        raise ValueError("No atlas parcels matched your region names. Ensure region names exactly match DK labels.")

    vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(combined, 5))
    vmax = args.vmax if args.vmax is not None else float(np.nanmax(combined))

    # combined_before_filtering = np.concatenate([
    #     build_vertex_map(df, lh_annot, hemi="lh", metric=args.metric),
    #     build_vertex_map(df, rh_annot, hemi="rh", metric=args.metric)
    # ])
    # vmin = args.vmin if args.vmin is not None else float(np.nanpercentile(combined_before_filtering, 5))
    # vmax = args.vmax if args.vmax is not None else float(np.nanmax(combined_before_filtering))
    
    p = Plot(
        fsavg.infl_left,
        fsavg.infl_right,
        layout= "grid", #grid #column
        views=["lateral", "medial"],
        size=(1000, 800),
        zoom=1.5,
        #size=(800, 1000),         # overall canvas
        # zoom=2.0, 
    )

    p.add_layer(
        {"left": lh_data, "right": rh_data},
        cmap=CMAP,  # Using a divergent palette
        color_range=(vmin, vmax),
        cbar=False,  # Change to False if not adding my own colorbar
    )

    
    fig = p.build()
    
    cbar = fig.axes[-1]  # Get the colorbar axis
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_xticks(ticks)
    
    cbar.set_xticklabels([f"{t:.3f}" for t in ticks])  # or .2f

    # 3) Optionally reduce label size and spacing
    cbar.tick_params(labelsize=8, pad=2)

    ####### if using my own colorbar, keep this, otherwise comment out
    # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # sm = mpl.cm.ScalarMappable(norm=norm, cmap=  "viridis")#wes_cmap)
    # sm.set_array([])

    # cbar = fig.colorbar(
    #     sm,
    #     ax=fig.axes,
    #     orientation="horizontal",
    #     fraction=0.03,
    #     pad=0.04,
    #     shrink=0.35,
    #     cmap= "viridis", #wes_cmap,
    #     extend="min",   # arrow on the low end
    # )
    #########
    
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])

    # create ONE colorbar
    cbar = fig.colorbar(
        sm,
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.03,
        pad=0.04,
        shrink=0.35,
        #extend="min",
    )

    # fewer, nicely formatted ticks
    cbar.locator = ticker.MaxNLocator(nbins=5)     # at most 5 ticks
    cbar.formatter = ticker.FormatStrFormatter("%.3f")
    cbar.update_ticks()                            # apply locator/formatter

    cbar.ax.set_title(args.metric.upper(), fontsize=10, pad=10)
    cbar.ax.tick_params(labelsize=8, pad=2)
    cbar.ax.set_title(args.metric.upper(), fontsize=10, pad=10)# Set the title on top in caps

    input_type = ''
    if "supersynth" in str(args.input_type):
        input_type = "SuperSynthetic/ISO-T2w"
    elif "mrr" in str(args.input_type):
        input_type = "MRR/ISO-T2w"
       
    if df["atlas_nfeat"].iloc[0] == f"{args.atlas}_5feat":
        nfeat_label = "N-features: 5"
    elif df["atlas_nfeat"].iloc[0] == f"{args.atlas}_4feat":
        nfeat_label = "N-features: 4" 
    fig.suptitle(f'{METRIC_MAP[args.metric]}', fontsize=16, y=0.98, fontweight='bold')
    fig.text(0.5, 0.9, args.method + " | " + input_type ,  ha="center", fontsize=14,fontweight='bold')
    fig.text(
    0.5, 0.2,
    parcellation_text,
    ha="center", fontsize=9, color="gray"
)
    #fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    
    svg_output = args.output.with_suffix(".svg")
    fig.savefig(os.path.join("/Users/Hajer/unity/PhD/Papers/1_Adult/illustrative_networks", svg_output)) #save as SVG
    #fig.savefig(args.output, dpi= 96 ,bbox_inches="tight") #save as PNG with 96 dpi for better quality in PowerPoint
    plt.close(fig)

    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
