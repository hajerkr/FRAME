"""
build_msn.py
============
Build Morphometric Similarity Networks (MSNs) from FreeSurfer recon-all-clinical
outputs. Supports DKTatlas and Destrieux (2009s) parcellations.

Compatible with paired HF / ULF MRI studies.

References:
    Seidlitz et al. (2018) Neuron
    Sebenius et al. (2023) Nature Neuroscience
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.io import savemat
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime
import pytz
from IPython.display import display
import re
import pandas as pd
import math
from pathlib import Path
import shutil
import flywheel
import subprocess
import logging
from nibabel.freesurfer.io import read_morph_data, read_annot

log = logging.getLogger(__name__)

ulf_filter, hf_filter  =None,  None

# Source zshrc and get the env var you want
def get_env_from_zshrc(var_name):
    command = f"source ~/.zshrc && echo ${var_name}"
    result = subprocess.run(['zsh', '-c', command], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Map parcellation name → stats file suffix
PARCELLATION_FILES = {
    "DKTatlas": "aparc.DKTatlas",   # Desikan-Killiany-Tourville atlas
    "2009s":    "aparc.a2009s",     # Destrieux atlas
    "aparc": "aparc",                 # FreeSurfer default (DK)
}

# The five canonical MSN features as they appear in FreeSurfer .stats headers
# MSN_FEATURES = [
#     "ThickAvg",   # Mean cortical thickness
#     "SurfArea",   # Surface area
#     "GrayVol",    # Gray matter volume
#     "MeanCurv",   # Mean curvature
#     "GausCurv",   # Gaussian curvature (proxy for sulcal depth / folding)
# ]

MSN_FEATURES =  []
MSN_FEATURES_DICT = {"5feat": ["SurfArea",   # Surface area
                        "GrayVol",    # Gray matter volume
                        "MeanCurv",   # Mean curvature
                        "SulcDepth",   # previously Gaussian curvature (proxy for sulcal depth / folding)]}
                        "ThickAvg",   # Mean cortical thickness
                        ], "4feat":
                        ["SurfArea",   # Surface area
                        "GrayVol",    # Gray matter volume
                        "MeanCurv",   # Mean curvature
                        "SulcDepth",   # previously Gaussian curvature (proxy for sulcal depth / folding)]}
                        ]
    
}

HEMISPHERES = ["lh", "rh"]


# ─────────────────────────────────────────────
# STEP 1: PARSE FreeSurfer .stats FILES
# ─────────────────────────────────────────────

def parse_stats_file(stats_path: str) -> pd.DataFrame:
    """
    Parse a FreeSurfer parcellation .stats file into a DataFrame.

    Parameters
    ----------
    stats_path : str
        Full path to the .stats file (e.g., lh.aparc.DKTatlas.stats)

    Returns
    -------
    pd.DataFrame
        Rows = regions, Columns = morphometric features + 'StructName'
    """
    rows = []
    col_names = None

    with open(stats_path, "r") as f:
        for line in f:
            line = line.strip()
            # Capture the column header line
            if line.startswith("# ColHeaders"):
                col_names = line.replace("# ColHeaders", "").split()
                continue
            # Skip all other comment lines
            if line.startswith("#") or len(line) == 0:
                continue
            # Parse data rows
            parts = line.split()
            rows.append(parts)

    if col_names is None or len(rows) == 0:
        raise ValueError(f"Could not parse stats file: {stats_path}")

    df = pd.DataFrame(rows, columns=col_names)

    # Convert numeric columns
    for col in df.columns:
        if col != "StructName":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def extract_regional_features(
    subject_dir: str,
    parcellation: str = "DKTatlas",
    features: list = MSN_FEATURES,
) -> pd.DataFrame:
    """
    Extract regional morphometric features for both hemispheres.

    Parameters
    ----------
    subject_dir : str
        Path to a single subject's FreeSurfer output directory.
    parcellation : str
        One of 'DKTatlas' or '2009s'.
    features : list
        List of feature column names to extract.

    Returns
    -------
    pd.DataFrame
        Rows = regions (prefixed lh_ / rh_), Columns = features.
    """
    stats_suffix = PARCELLATION_FILES[parcellation]
    all_regions = []

    for hemi in HEMISPHERES:
        stats_file = os.path.join(
            subject_dir, "stats", f"{hemi}.{stats_suffix}.stats"
        )
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Stats file not found: {stats_file}")

        df = parse_stats_file(stats_file)

        # Keep only requested features + region name
        available = [f for f in features if f in df.columns]
        missing = set(features) - set(available)
        if missing and "SulcDepth" not in missing:
            print(f"  [WARNING] Missing features in {stats_file}: {missing}")

        df_sub = df[["StructName"] + available].copy()
        df_sub["region"] = hemi + "_" + df_sub["StructName"]
        df_sub = df_sub.drop(columns=["StructName"]).set_index("region")
        all_regions.append(df_sub)

    #region_df = pd.concat(all_regions, axis=0)
    #print("Columns and index at extract regional features: ", region_df.columns,region_df.index)
    return pd.concat(all_regions, axis=0)

##### ADD SULCAL DEPTH ####

def region_sulc(subject_dir: str,
    parcellation: str = "DKTatlas",
) -> pd.DataFrame: 
    
    surf_dir = os.path.join(subject_dir, "surf")
    label_dir = os.path.join(subject_dir, "label")
    stats_suffix = PARCELLATION_FILES[parcellation]
    
    rows = []
    for hemi in HEMISPHERES:
        # 1) vertex-wise sulc
        sulc_path = os.path.join(surf_dir, f"{hemi}.sulc")
        sulc = read_morph_data(sulc_path)  # shape (n_vertices,)

        # 2) annotation: labels per vertex + names
        annot_path = os.path.join(label_dir, f"{hemi}.{stats_suffix}.annot")
        labels, ctab, names = read_annot(annot_path, orig_ids=True)

        # mapping: numeric label -> name string
        lut = dict(zip(ctab[:, -1], [n.decode("utf-8") for n in names]))

        
        for lab, name in lut.items():
            mask = labels == lab
            if not mask.any():
                continue
            # exclude unknown / medial wall if desired
            if name.lower() in ("unknown", "corpuscallosum") or "Medial" in name:
                continue
            mean_sulc = float(np.mean(sulc[mask]))
            rows.append({"region": f"{hemi}_{name}", "SulcDepth": mean_sulc})

    return pd.DataFrame(rows)

# ─────────────────────────────────────────────
# STEP 2: BUILD FEATURE MATRIX ACROSS SUBJECTS
# ─────────────────────────────────────────────

def build_cohort_feature_matrix(
    subjects_dir: str,
    subject_list: list,
    parcellation: str = "DKTatlas",
    features: list = MSN_FEATURES

) -> dict:
    """
    Build a {region × feature} matrix for each subject.

    Parameters
    ----------
    subjects_dir : str
        Root FreeSurfer subjects directory.
    subject_list : list
        List of subject IDs (subdirectory names).
    parcellation : str
        Parcellation to use.
    features : list
        Features to extract.

    Returns
    -------
    dict
        {subject_id: pd.DataFrame(regions × features)}
    """
    subject_data = {}

    for sub in subject_list:
        sub_dir = os.path.join(subjects_dir, sub)
        try:
            df_4feat = extract_regional_features(sub_dir, parcellation, features)
            #Compute sulcal depth dataframe, and merge on region
            df_sd =  region_sulc(sub_dir, parcellation).set_index("region")        
            #df = pd.merge(df_4feat,df_sd, on='region')
            df = df_4feat.join(df_sd, how="inner")
            subject_data[sub.split('_')[0]] = df
            print(f"  [OK] {sub}: {df.shape[0]} regions, {df.shape[1]} features {df.columns[1:]}")
            
        except Exception as e:
            print(f"  [SKIP] {sub}: {e}")

    return subject_data


# ─────────────────────────────────────────────
# STEP 3: COMPUTE MSN FOR A SINGLE SUBJECT
# ─────────────────────────────────────────────

def compute_msn(
    region_feature_df: pd.DataFrame,
    features: list = MSN_FEATURES,
) -> pd.DataFrame:
    """
    Compute a Morphometric Similarity Network (MSN) for a single subject.

    Method (Seidlitz et al. 2018 / Sebenius et al. 2023):
    1. Extract the R × F feature matrix (R regions, F features).
    2. Z-score each feature column across regions.
    3. Compute pairwise Pearson correlation between all region vectors.
       The resulting R × R matrix is the MSN.

    Parameters
    ----------
    region_feature_df : pd.DataFrame
        Rows = regions, Columns = morphometric features.
    features : list
        Subset of features to use.

    Returns
    -------
    pd.DataFrame
        R × R symmetric MSN (Pearson correlation matrix).
    """
    # Select and drop NaN rows
    available_features = [f for f in features if f in region_feature_df.columns]
    mat = region_feature_df[available_features].dropna()

    if mat.shape[0] < 2:
        raise ValueError("Too few valid regions to compute MSN.")

    # Z-score each feature across regions (column-wise)
    mat_z = mat.apply(zscore, axis=0, ddof=1)

    # Pairwise Pearson correlation between region vectors (row-wise)
    # Result shape: (n_regions × n_regions)
    msn = mat_z.T.corr(method="pearson")  # .T so regions are columns

    return msn


def compute_all_msns(
    subject_data: dict,
    features: list = MSN_FEATURES,
) -> dict:
    """
    Compute MSNs for all subjects.

    Parameters
    ----------
    subject_data : dict
        {subject_id: region_feature_df}
    features : list
        Features to use.

    Returns
    -------
    dict
        {subject_id: MSN DataFrame}
    """
    msns = {}
    for sub, df in subject_data.items():
        try:
            msn = compute_msn(df, features)
            msns[sub] = msn
        except Exception as e:
            print(f"  [SKIP MSN] {sub}: {e}")
    print(f"\nComputed MSNs for {len(msns)} subjects.")
    #print(msns)
    return msns


# ─────────────────────────────────────────────
# STEP 4: GROUP-LEVEL MSN (MEAN ACROSS SUBJECTS)
# ─────────────────────────────────────────────

def compute_group_msn(msns: dict) -> pd.DataFrame:
    """
    Compute the group-mean MSN by averaging individual MSN matrices.

    Parameters
    ----------
    msns : dict
        {subject_id: MSN DataFrame}

    Returns
    -------
    pd.DataFrame
        Group-mean MSN.
    """
    msn_list = list(msns.values())
    # Align on common regions
    regions = msn_list[0].index
    stacked = np.stack([m.reindex(index=regions, columns=regions).values
                        for m in msn_list], axis=0)
    group_mean = np.nanmean(stacked, axis=0)
    return pd.DataFrame(group_mean, index=regions, columns=regions)


# ─────────────────────────────────────────────
# STEP 5: RELIABILITY — ICC / PEARSON BETWEEN PAIRED SCANS
# ─────────────────────────────────────────────

def msn_edge_correlation(msn1: pd.DataFrame, msn2: pd.DataFrame) -> float:
    """
    Compute Pearson correlation between upper-triangle edges of two MSNs.
    Used to assess reliability between paired HF and ULF scans.

    Parameters
    ----------
    msn1, msn2 : pd.DataFrame
        Two MSN matrices with matching region labels.

    Returns
    -------
    float
        Pearson r between vectorised upper-triangle edges.
    """
    # Align
    regions = msn1.index.intersection(msn2.index)
    m1 = msn1.reindex(index=regions, columns=regions).values
    m2 = msn2.reindex(index=regions, columns=regions).values

    # Upper triangle (excluding diagonal)
    idx = np.triu_indices(len(regions), k=1)
    v1 = m1[idx]
    v2 = m2[idx]

    mask = np.isfinite(v1) & np.isfinite(v2)
    r = np.corrcoef(v1[mask], v2[mask])[0, 1]
    return float(r)


def compute_paired_reliability(
    msns_hf: dict,
    msns_ulf: dict,
    paired_ids: list,
) -> pd.DataFrame:
    """
    Compute edge-wise Pearson correlation between paired HF and ULF MSNs.

    Parameters
    ----------
    msns_hf : dict
        {subject_id: MSN} for high-field scans.
    msns_ulf : dict
        {subject_id: MSN} for ultra-low-field scans.
    paired_ids : list
        Subject IDs present in both dictionaries.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [subject_id, pearson_r].
    """
    results = []
    for sub in paired_ids:
        if sub in msns_hf and sub in msns_ulf:
            r = msn_edge_correlation(msns_hf[sub], msns_ulf[sub])
            results.append({"subject_id": sub, "pearson_r": r})
        else:
            print(f"  [SKIP] {sub} not found in both HF and ULF dictionaries.")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# STEP 6: WEIGHTED NODAL DEGREE
# ─────────────────────────────────────────────

def weighted_degree(msn: pd.DataFrame) -> pd.Series:
    """
    Compute weighted nodal degree (mean edge weight per node).

    Parameters
    ----------
    msn : pd.DataFrame
        R × R MSN matrix.

    Returns
    -------
    pd.Series
        Weighted degree per region.
    """
    # Exclude self-connections (diagonal)
    mat = msn.values.copy()
    np.fill_diagonal(mat, np.nan)
    return pd.Series(np.nanmean(mat, axis=1), index=msn.index)


# ─────────────────────────────────────────────
# STEP 7: SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_msn(msn: pd.DataFrame, out_path: str, fmt: str = "csv"):
    """
    Save an MSN matrix to disk.

    Parameters
    ----------
    msn : pd.DataFrame
    out_path : str
        Output file path (without extension if fmt='mat').
    fmt : str
        'csv' or 'mat' (MATLAB format).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        #print("Columns and index ", msn.columns, msn.index)
        msn.to_csv(out_path)
    elif fmt == "mat":
        savemat(out_path, {"MSN": msn.values,
                           "regions": list(msn.index)})
    else:
        raise ValueError(f"Unknown format: {fmt}")


def save_all_msns(msns: dict, output_dir: str, fmt: str = "csv", hf_or_ulf: str = "HF"):
    """Save all subject MSNs to output_dir/{subject_id}_{hf_or_ulf}_msn.csv"""
    for sub, msn in msns.items():
        out_path = os.path.join(output_dir, f"{sub}_{hf_or_ulf}_msn.{fmt}")
        #print(out_path)
        save_msn(msn, out_path, fmt=fmt)
    print(f"Saved {len(msns)} MSNs to {output_dir}")


# ─────────────────────────────────────────────
# STEP 8: VISUALISATION
# ─────────────────────────────────────────────

def plot_msn(msn: pd.DataFrame, title: str = "MSN", ax=None, cmap: str = "RdBu_r"):
    """Plot an MSN as a heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        msn,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(title)
    return ax


def plot_group_comparison(
    group_msn_hf: pd.DataFrame,
    group_msn_ulf: pd.DataFrame,
    output_path: str = None,
):
    """Side-by-side comparison of group-mean HF and ULF MSNs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_msn(group_msn_hf,  title=f"Group MSN — High-Field ({hf_filter})",      ax=axes[0])
    plot_msn(group_msn_ulf, title=f"Group MSN —  Ultra-Low-Field ({ulf_filter})", ax=axes[1])

    # Difference matrix
    diff = group_msn_hf - group_msn_ulf
    sns.heatmap(
        diff,
        ax=axes[2],
        cmap="RdBu_r",
        center=0,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "HF − ULF"},
    )
    axes[2].set_title("Difference (HF − ULF)")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    #plt.show()


def plot_reliability(reliability_df: pd.DataFrame, output_path: str = None):
    """Histogram of per-subject edge-wise Pearson r (HF vs ULF)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(reliability_df["pearson_r"], bins=20, color="steelblue", edgecolor="white")
    ax.axvline(reliability_df["pearson_r"].mean(), color="red",
               linestyle="--", label=f"Mean r = {reliability_df['pearson_r'].mean():.3f}")
    ax.set_xlabel(f"Edge-wise Pearson r (HF vs ULF)")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"MSN Reliability: HF ({hf_filter}) vs ULF ({ulf_filter})")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    #plt.show()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_msn_pipeline(
    subjects_dir_hf: str,
    subjects_dir_ulf: str,
    paired_subject_ids: list,
    parcellation: str = "DKTatlas",   # or "2009s"
    features: list = MSN_FEATURES,
    output_dir: str = "./msn_outputs",
    save_format: str = "csv"
):
    """
    Full MSN pipeline for paired HF / ULF data.

    Parameters
    ----------
    subjects_dir_hf : str
        FreeSurfer subjects directory for high-field scans.
    subjects_dir_ulf : str
        FreeSurfer subjects directory for ultra-low-field scans.
    paired_subject_ids : list
        Subject IDs present in both directories.
    parcellation : str
        'DKTatlas' or '2009s'.
    features : list
        Morphometric features to use.
    output_dir : str
        Root directory for all outputs.
    save_format : str
        'csv' or 'mat'.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  MSN Pipeline | Parcellation: {parcellation}")
    print(f"  Features: {features}")
    print(f"{'='*60}\n")

    # --- Extract features ---
    print("[1/6] Extracting features — High Field")
    paired_subject_ids_hf = [s for s in paired_subject_ids if hf_filter in s]
    data_hf = build_cohort_feature_matrix(
        subjects_dir_hf, paired_subject_ids_hf, parcellation, features)

    print("\n[2/6] Extracting features — Ultra Low Field")
    paired_subject_ids_lf = [s for s in paired_subject_ids if ulf_filter in s]
    data_ulf = build_cohort_feature_matrix(
        subjects_dir_ulf, paired_subject_ids_lf, parcellation, features)

    # --- Compute MSNs ---
    print("\n[3/6] Computing MSNs — High Field")
    msns_hf = compute_all_msns(data_hf, features)

    print("\n[4/6] Computing MSNs — Ultra Low Field")
    msns_ulf = compute_all_msns(data_ulf, features)

    # --- Save individual MSNs ---
    save_all_msns(msns_hf,  os.path.join(output_dir), save_format,hf_filter)
    save_all_msns(msns_ulf, os.path.join(output_dir), save_format, ulf_filter)

    # --- Group-mean MSNs ---
    print("\n[5/6] Computing group-mean MSNs")
    group_hf  = compute_group_msn(msns_hf)
    group_ulf = compute_group_msn(msns_ulf)

    save_msn(group_hf,  os.path.join(output_dir,  f"group_msn_{hf_filter}.{save_format}"),  save_format)
    save_msn(group_ulf, os.path.join(output_dir , f"group_msn_{ulf_filter}.{save_format}"), save_format)

    # --- Reliability ---
    print("\n[6/6] Computing HF vs ULF reliability")
    valid_ids = [s for s in msns_hf ]#and s in msns_ulf]
    reliability = compute_paired_reliability(msns_hf, msns_ulf, valid_ids)
    reliability.to_csv(os.path.join(output_dir, f"msn_reliability_{hf_filter}_{ulf_filter}.csv"), index=False)

    print(f"\n  Mean edge-wise Pearson r (HF vs ULF): "
          f"{reliability['pearson_r'].mean():.4f} ± "
          f"{reliability['pearson_r'].std():.4f}")

    with open(os.path.join(output_dir, f"msn_reliability_{hf_filter}_{ulf_filter}.txt"), "w") as f:
        f.write(f"Mean edge-wise Pearson r ({hf_filter} vs {ulf_filter} - {len(features)}) : "
                f"{reliability['pearson_r'].mean():.4f} ± "
                f"{reliability['pearson_r'].std():.4f}\n")

    # --- Weighted degree ---
    deg_hf  = pd.DataFrame({s: weighted_degree(m) for s, m in msns_hf.items()})
    deg_ulf = pd.DataFrame({s: weighted_degree(m) for s, m in msns_ulf.items()})
    deg_hf.to_csv(os.path.join(output_dir, f"{hf_filter}_weighted_degree.csv"))
    deg_ulf.to_csv(os.path.join(output_dir, f"{ulf_filter}_weighted_degree.csv"))

    # --- Plots ---
    plot_group_comparison(
        group_hf, group_ulf,
        output_path=os.path.join(output_dir, f"group_msn_comparison_{hf_filter}_{ulf_filter}_{parcellation}.png")
    )
    plot_reliability(
        reliability,
        output_path=os.path.join(output_dir, f"msn_reliability_{hf_filter}_{ulf_filter}_{parcellation}.png")
    )

    print(f"\nAll outputs saved to: {output_dir}")
    return msns_hf, msns_ulf, group_hf, group_ulf, reliability


#Function to download zip from analysis, unzip 
def download_session_data(session, analysis_filter, input_gear, version_input_gear, work_dir, log):
    session = session.reload()  # Ensure we have the latest data
    analyses = session.analyses
    recon_all = [a for a in analyses if input_gear in a.label and analysis_filter in a.label and a.job.get('state')== "complete" and a.gear_info.name == input_gear and a.gear_info.version == version_input_gear]
    #recon_all = [a for a in analyses if ('recon-all' in a.label or 'freesurfer' in a.label) and a.job.get('state')== "complete"]
    if recon_all == []:
        log.warning(f"[SKIPPING] recon-all not found for {session.subject.label}/{session.label}.")
        exit(1)
    #Check if amongst the output files of recon-all, there is a file ending in .zip
    log.info(f'Container to be used:  {recon_all[-1].label} ')
    recon_all_outputs = recon_all[-1].files
    # if there are multiple recon-all analyses, take the last one (most recent)
    if len(recon_all) > 1:
        log.warning(f"[WARNING] Multiple recon-all analyses found for {session.subject.label}/{session.label}. Using the most recent one.")
        
    recon_all = recon_all[-1]
    #if 'freesurfer' in recon_all.gear_info.name :
    #    features = ['CT','MC','SD','SA','Vol']

    zip_files = [f for f in recon_all_outputs if f.name.endswith('.zip')]
    
    if not zip_files:
        log.warning(f"[SKIPPING] recon-all output .zip not found for {session.subject.label}/{session.label}.")
        exit(1)

    # Download the .zip file to the work_dir
    zip_file_path = os.path.join(work_dir, zip_files[0].name)
    if not os.path.exists(work_dir):
        os.makedirs(os.path.join(work_dir))
                    
    zip_files[0].download(zip_file_path)  #UNCOMMENT BEFORE PUSHING
    # Unzip the file to a directory named sub-ses under work_dir
    #make directory if it doesn't exist
    surf_dir = os.path.join(work_dir, input_gear, session.label, f"{session.subject.label}_{session.label}")
    # if not os.path.exists(surf_dir):
    #     os.makedirs(os.path.join(config["work_dir"], f"{sub}-{ses}"))
    #     shutil.unpack_archive(zip_file_path, surf_dir)
    #     # Remove the .zip file after unzipping
    #     os.remove(zip_file_path)

    os.makedirs(surf_dir, exist_ok=True)
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.unpack_archive(zip_file_path, tmpdir)
        
        # find the *deepest* directory that contains your data
        # in your case it's something like .../flywheel/v0/work/sub_HYPE01_T2w_mrr
        candidate = None
        for root, dirs, files in os.walk(tmpdir):
            if f"{session.subject.label}_{session.label}" in os.path.basename(root) or "stats" in dirs:
                candidate = root
                break

        if candidate is None:
            raise RuntimeError("Could not find expected folder inside zip")

        # move everything from candidate → surf_dir
        for item in os.listdir(candidate):
            try:

                shutil.move(os.path.join(candidate, item), surf_dir)
            except Exception as e:
                log.warning(f"Could not move {item} to {surf_dir}: {e}")
        # Remove all contents under /flywheel/v0/work/sub-HYPE01-ses-HFE/flywheel/v0/work/
        #shutil.rmtree(os.path.join(surf_dir, 'flywheel'))
    # Remove the .zip file after unzipping


    os.remove(zip_file_path)
    #Delete everything but the .stats files
    for root, dirs, files in os.walk(surf_dir):
        for name in files:
            if not ".stats" in name:
                try:
                    os.remove(os.path.join(root, name))
                except Exception as e:
                    log.warning(f"Could not remove file {name}: {e}")
        for name in dirs:
            try:
                if not "stats" in name:
                    shutil.rmtree(os.path.join(root, name))
                    print(f"Removed directory: {os.path.join(root, name)}")
            except Exception as e:
                log.warning(f"Could not remove directory {name}: {e}")

    print(surf_dir)
    #List contents of surf_dir
    for root, dirs, files in os.walk(surf_dir):
        for name in files:
            log.info(f"File: {os.path.join(root, name)}")
        for name in dirs:
            log.info(f"Directory: {os.path.join(root, name)}")
# ─────────────────────────────────────────────
# EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    
    #get_env_from_zshrc('FW_CLI_API_KEY')

    def list_of_strings(arg):
        return arg.split(',')

    #Get project from argument passed
    parser = argparse.ArgumentParser(description="Build MSNs from FreeSurfer outputs in Flywheel")
    parser.add_argument("--project", type=str, required=True, help="Flywheel project label (e.g., 'UCT-Khula-Hyperfine')")
    parser.add_argument("--workdir", type=str, default="./msn_workdir", help="Local directory for intermediate files")
    parser.add_argument("--analysis_filter", type=str, default="recon-all-clinical", help="String to filter analyses (default: 'recon-all-clinical')")
    parser.add_argument("--input_gear", type=str, default="recon-all-clinical", help="Gear name to filter analyses (default: 'recon-all-clinical')")
    parser.add_argument("--version", type=str, default="0.1", help="Version string of input gear")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for MSNs and plots")

    parser.add_argument("--hf_filter", type=str, default="HF", help="List of strings to identify high-field scans in session labels (default: ['HF'])")
    parser.add_argument("--ulf_filter", type=str, default="ULF", help="String to identify ultra-low-field scans in session labels (default: 'ULF')")
    #parser.add_argument("--parc", type=str, default="aparc", choices=["DKTatlas", "2009s","aparc"], help="Parcellation to use (default: 'aparc')")
    parser.add_argument("--update", action="store_true", help="Whether to update existing outputs (default: False)")
    args = parser.parse_args()

    project = args.project
    work_dir = args.workdir
    analysis_filter = args.analysis_filter
    input_gear = args.input_gear
    version_input_gear = args.version

    
    hf_filter = args.hf_filter
    ulf_filter = args.ulf_filter
    #parcellation = args.parc
    update = args.update
    
    
    api_key = os.environ.get('FW_CLI_API_KEY') 
    # Connect to your Flywheel instance
    fw = flywheel.Client(api_key=api_key)
    #Connect to Flywheel
    #fw_in = flywheel.Client(api_key=api_key_in)
    display(f"User: {fw.get_current_user().firstname} {fw.get_current_user().lastname}")
    fw_project = fw.projects.find_first(f'label={project}')
    if fw_project is None:
        raise ValueError(f"Project not found: {project}")
    else:
        print(f"Found project: {fw_project.label} (ID: {fw_project.id})")
        fw_project = fw_project.reload()
        
    if args.update:
    
    
            for session in fw_project.sessions():
                print(f"Processing session: {session.label} (ID: {session.id})")
                # Example filter for recon-all-clinical outputs
                download_session_data(session, analysis_filter, input_gear,version_input_gear, work_dir, log)

    # --- Edit these paths ---
    HF_SUBJECTS_DIR  = f"{work_dir}/{hf_filter}"
    ULF_SUBJECTS_DIR = f"{work_dir}/{ulf_filter}"
    OUTPUT_DIR       = args.outdir #f"/Users/Hajer/unity/PhD/Adult/{project}/MSN"

    # Subject IDs (must match subdirectory names in both HF and ULF dirs)
    SUBJECT_IDS = [
        session.subject.label+"_"+session.label for session in fw_project.sessions() 
        if "PHANTOM" not in session.subject.label.upper()
    ]

    #Filter SUBJECT_IDS to only include those that have both hf_filter and ulf_filter in their session label
    SUBJECT_IDS = [s for s in SUBJECT_IDS if hf_filter in s or ulf_filter in s]

    #For 4feat and 5feat, run the pipeline:
    for n, feat in MSN_FEATURES_DICT.items():
        MSN_FEATURES = feat
        # Run for DKatlas
        msns_hf, msns_ulf, grp_hf, grp_ulf, rel = run_msn_pipeline(
            subjects_dir_hf=HF_SUBJECTS_DIR,
            subjects_dir_ulf=ULF_SUBJECTS_DIR,
            paired_subject_ids=SUBJECT_IDS,
            parcellation="aparc",
            features=MSN_FEATURES,
            output_dir=os.path.join(OUTPUT_DIR, "aparc",n),
            save_format="csv"
        )
        
        # Run for DKTatlas
        # msns_hf, msns_ulf, grp_hf, grp_ulf, rel = run_msn_pipeline(
        #     subjects_dir_hf=HF_SUBJECTS_DIR,
        #     subjects_dir_ulf=ULF_SUBJECTS_DIR,
        #     paired_subject_ids=SUBJECT_IDS,
        #     parcellation="DKTatlas",
        #     features=MSN_FEATURES,
        #     output_dir=os.path.join(OUTPUT_DIR, "aparc.DKTatlas",n),
        #     save_format="csv"
        # )

        # Run for 2009s (Destrieux)
        msns_hf_2009s, msns_ulf_2009s, grp_hf_2009s, grp_ulf_2009s, rel_2009s = run_msn_pipeline(
            subjects_dir_hf=HF_SUBJECTS_DIR,
            subjects_dir_ulf=ULF_SUBJECTS_DIR,
            paired_subject_ids=SUBJECT_IDS,
            parcellation="2009s",
            features=MSN_FEATURES,
            output_dir=os.path.join(OUTPUT_DIR, "aparc.a2009s",n),
            save_format="csv"
    )