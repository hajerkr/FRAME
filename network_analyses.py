#!/usr/bin/env python
"""
mind_analysis.py

Compares MIND (Morphometric INverse Divergence) structural networks between
a high-field (HF) and a low-field (LF) acquisition, per subject.

For each edge (ROI pair) it computes, across subjects:
    - Pearson's r
    - Lin's concordance correlation coefficient (CCC)
    - ICC(3,1) (two-way mixed effects, single measures, consistency)

It also computes subject identifiability (differential identifiability,
Amico & Goni 2018) from the HF vs LF edge vectors.

Expected input
--------------
A directory of per-subject, per-session MIND network CSVs, one matrix per
file, with ROI names as both the row index and the column header, e.g.:

    ,lh_bankssts,lh_caudalanteriorcingulate,...
    lh_bankssts,1.0,0.42,...
    lh_caudalanteriorcingulate,0.42,1.0,...

Filenames are assumed to be "-"-delimited, with the subject ID and session
label at fixed positions (configurable via --subj-part-index and
--session-part-index), e.g.:

    sub-HYPE001_ses-01-desc-mind-GE.csv
                                  ^ session label (default: part index 3)

Usage
-----
    python mind_analysis.py \\
        --mind-dir /path/to/mind_networks \\
        --hf-label GE \\
        --lf-label HFE \\
        --output-dir /path/to/results
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
# -----------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge-wise HF vs LF MIND network comparison "
        "(Pearson r, CCC, ICC) and subject identifiability."
    )
    parser.add_argument(
        "--mind-dir",
        required=True,
        help="Directory containing per-subject, per-session MIND network CSVs.",
    )
    parser.add_argument(
        "--hf-label",
        required=True,
        help="Session label identifying the high-field scans (e.g. 'GE').",
    )
    parser.add_argument(
        "--lf-label",
        required=True,
        help="Session label identifying the low-field scans (e.g. 'HFE').",
    )
    parser.add_argument(
        "--output-dir",
        default="mind_analysis_output",
        help="Directory to write result CSVs to (default: ./mind_analysis_output).",
    )
    parser.add_argument(
        "--subj-part-index",
        type=int,
        default=1,
        help="Index of the subject ID when the filename is split on '-' "
        "(default: 1).",
    )
    parser.add_argument(
        "--session-part-index",
        type=int,
        default=3,
        help="Index of the session label when the filename is split on '-' "
        "(default: 3).",
    )
    parser.add_argument(
        "--zscore",
        action="store_true",
        help="Z-score edges (per-edge, across subjects) before computing "
        "the identifiability matrix.",
    )
    
    parser.add_argument(
        "--suffix",
        help="suffix to add at the end of the output files to indicate which input was used to compute"
    )
    parser.add_argument(
        "--template",
        default="",
        help="Optional surface template label (e.g. 'fsaverage', 'fsaverage5'), "
        "stored as a metadata column in the master CSVs so runs across "
        "templates/atlases/features/label-pairs can be told apart after "
        "concatenation.",
    )
    
    # parser.add_argument(
    #     "--overwrite",
    #     default=False,
    #     help="overwrite master dfs there such identifiability summary, subject topology summary ...",
    # )
    return parser.parse_args()


# -----------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------
def load_networks(mind_dir, subj_idx, session_idx):
    """
    Load all MIND network CSVs in `mind_dir` into a nested dict:
        {subject_id: {session_label: DataFrame(ROI x ROI)}}

    ROI names are kept (row index == column names) so that networks with
    differing ROI sets (e.g. from DKT atlas parcellation dropouts) can be
    aligned rather than silently misaligned.
    """
    files = sorted(glob(os.path.join(mind_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in ", mind_dir )

    subj_networks = {}
    for f in files:
        base = os.path.basename(f).replace(".csv", "")
        #parts = base.split("-")
        parts = base.split("_")
        try:
            # subj = parts[1]
            # session = parts[3]
            
            subj = parts[0]
            session = parts[1]
            if "-" in subj and "-" in session:
                subj = parts[0].split('-')[1]
                session = parts[1].split('-')[1]

        except IndexError:
            print(f"  Skipping {base}: not enough '-'-delimited parts "
                  f"for subj-part-index={subj_idx}, session-part-index={session_idx}")
            continue

        df = pd.read_csv(f, index_col=0)
        subj_networks.setdefault(subj, {})[session] = df

    return subj_networks


def align_rois(df_hf, df_lf):
    """
    Restrict two ROI x ROI DataFrames to their common ROI set (intersection),
    in a consistent order, to guard against atlas/region-labelling
    inconsistencies across subjects or sessions.
    """
    #Make columns and index the same
    # print(df_hf.shape, df_lf.shape,len(set(df_hf.index)), len(set(df_lf.index)))
    #print index
    #If the first column is not the same as the index, set the index to the first column
   
    if not df_hf.index.equals(df_hf.columns):
        #If length of index is not 
        print(len(df_hf.index), len(df_hf.columns))
        df_hf.index = df_hf.columns
    if not df_lf.index.equals(df_lf.columns):
        df_lf.index = df_lf.columns
    #df_hf.index = df_hf.columns
    #df_lf.index = df_lf.columns
    common_rois = [r for r in df_hf.index if r in set(df_lf.index)]
    return df_hf.loc[common_rois, common_rois], df_lf.loc[common_rois, common_rois]


def get_hf_lf_pairs(subj_networks, hf_label, lf_label):
    """
    For each subject with both HF and LF sessions present, align ROIs and
    return {subject_id: (hf_df, lf_df)}. Subjects missing either session
    are skipped (reported).
    """
    pairs = {}
    for subj, scans in subj_networks.items():
        if hf_label not in scans or lf_label not in scans:
            print(f"  Subject {subj}: missing {hf_label} and/or {lf_label}; skipping.")
            continue
        print(subj)
        hf_df, lf_df = align_rois(scans[hf_label], scans[lf_label])
        if hf_df.shape[0] < 2:
            print(f"  Subject {subj}: fewer than 2 common ROIs after alignment; skipping.")
            continue
        pairs[subj] = (hf_df, lf_df)
    return pairs


# -----------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------
def upper_triangle_vector(matrix):
    """Vectorize the upper triangle (excluding diagonal) of a square array."""
    arr = np.asarray(matrix)
    iu = np.triu_indices_from(arr, k=1)
    return arr[iu]



def subject_topology_corrs(subj_networks, scanner_pairs):
    rows = []
    for subj, scans in subj_networks.items():
        for s1, s2 in scanner_pairs:
            if s1 not in scans or s2 not in scans:
                continue
            A = scans[s1].values
            B = scans[s2].values
            #print(A.values, A.shape)
            v1 = A[np.triu_indices_from(A, k=1)]
            v2 = B[np.triu_indices_from(B, k=1)]
            mask = np.isfinite(v1) & np.isfinite(v2)
            if mask.sum() > 3:
                r, _ = pearsonr(v1[mask], v2[mask])
                ccc = lin_ccc(v1[mask], v2[mask])
                rows.append({'subject': subj, 'pair': f'{s1}-{s2}', 'r': r, 'ccc': ccc})
    return pd.DataFrame(rows)





def lin_ccc(x, y):
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x, ddof=1), np.std(y, ddof=1)
    rho, _ = pearsonr(x, y)
    return (2 * rho * sigma_x * sigma_y) / (sigma_x**2 + sigma_y**2 + (mu_x - mu_y) ** 2)


def icc_3_1(data):
    """
    ICC(3,1): two-way mixed effects, single measures, consistency.

    Parameters
    ----------
    data : array-like, shape (n_subjects, n_raters)
        Here "raters" = the two sessions (HF, LF).

    Returns
    -------
    float
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    if n < 2 or k < 2:
        return np.nan

    grand_mean = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)

    ss_total = ((data - grand_mean) ** 2).sum()
    ss_rows = k * ((row_means - grand_mean) ** 2).sum()
    ss_cols = n * ((col_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_rows + (k - 1) * ms_error
    if denom == 0:
        return np.nan
    return (ms_rows - ms_error) / denom


def build_edge_dataframe(hf_lf_pairs):
    """
    Long-format DataFrame with one row per (subject, edge): columns
    subject, ROI_i, ROI_j, hf_value, lf_value.
    """
    rows = []
    for subj, (hf_df, lf_df) in hf_lf_pairs.items():
        rois = hf_df.index.tolist()
        hf_vals = hf_df.values
        lf_vals = lf_df.values
        n = len(rois)
        for i in range(n):
            for j in range(i + 1, n):
                rows.append({
                    "subject": subj,
                    "ROI_i": rois[i],
                    "ROI_j": rois[j],
                    "hf_value": hf_vals[i, j],
                    "lf_value": lf_vals[i, j],
                })
    return pd.DataFrame(rows)


def compute_edge_metrics(edge_df):
    """
    For each edge (ROI_i, ROI_j), compute Pearson r, CCC and ICC(3,1)
    across subjects, from the long-format edge DataFrame.
    """
    results = []
    for (roi_i, roi_j), group in edge_df.groupby(["ROI_i", "ROI_j"]):
        hf = group["hf_value"].to_numpy()
        lf = group["lf_value"].to_numpy()
        mask = np.isfinite(hf) & np.isfinite(lf)
        hf, lf = hf[mask], lf[mask]

        if len(hf) < 2 or np.std(hf) == 0 or np.std(lf) == 0:
            r, ccc, icc = np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(hf, lf)
            ccc = lin_ccc(hf, lf)
            icc = icc_3_1(np.column_stack([hf, lf]))

        results.append({
            "ROI_i": roi_i, "ROI_j": roi_j,
            "pearson_r": r, "ccc": ccc, "icc": icc, "n_subjects": len(hf),
        })
    return pd.DataFrame(results)

def compute_subject_topology_metrics(hf_lf_pairs, hf_label, lf_label):
    """
    For each subject, compare their whole HF vs LF connectivity vector
    (all edges at once):
        - Pearson's r
        - Lin's CCC
        - ICC(3,1), with p-value and 95% CI, via pingouin. Here the
          "targets" are the individual edges and the "raters" are the two
          sessions (HF, LF) -- i.e. this asks whether the subject's
          edge-to-edge pattern of values is consistent between sessions.
 
    This answers a different question than compute_edge_metrics(): "does
    this subject's overall network pattern reproduce across field
    strengths", rather than "is this specific edge reliable across
    subjects".
    """
    rows = []
    for subj, (hf_df, lf_df) in hf_lf_pairs.items():
        v_hf = upper_triangle_vector(hf_df.values)
        v_lf = upper_triangle_vector(lf_df.values)
        mask = np.isfinite(v_hf) & np.isfinite(v_lf)
        v_hf, v_lf = v_hf[mask], v_lf[mask]
        n_edges = len(v_hf)
 
        if n_edges < 4 or np.std(v_hf) == 0 or np.std(v_lf) == 0:
            r, ccc = np.nan, np.nan
            icc3, icc3_p, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan
        else:
            r, _ = pearsonr(v_hf, v_lf)
            ccc = lin_ccc(v_hf, v_lf)
 
            long_df = pd.DataFrame({
                "edge": np.tile(np.arange(n_edges), 2),
                "session": [hf_label] * n_edges + [lf_label] * n_edges,
                "value": np.concatenate([v_hf, v_lf]),
            })
            icc_table = pg.intraclass_corr(
                data=long_df, targets="edge", raters="session", ratings="value"
            )
            # pingouin labels ICC(3,1) (two-way mixed, consistency, single
            # measures) as "ICC(C,1)" in current versions, "ICC3" in older
            # ones -- and the CI column as "CI95" or "CI95%" depending on
            # version, so handle both.
            icc3_row = icc_table[icc_table["Type"].isin(["ICC3", "ICC(C,1)"])].iloc[0]
            ci_col = "CI95%" if "CI95%" in icc_table.columns else "CI95"
            icc3 = icc3_row["ICC"]
            icc3_p = icc3_row["pval"]
            ci_low, ci_high = icc3_row[ci_col]
 
        rows.append({
            "subject": subj,
            "pearson_r": r,
            "ccc": ccc,
            "icc3": icc3,
            "icc3_pval": icc3_p,
            "icc3_ci_low": ci_low,
            "icc3_ci_high": ci_high,
            "n_edges": n_edges,
        })
    return pd.DataFrame(rows)

def compute_region_level_metrics(hf_lf_pairs, hf_label, lf_label):
    rows = []
    for subj, (hf_df, lf_df) in hf_lf_pairs.items():
        #print(hf_df.columns,lf_df.columns)
        #print(hf_df["lh_caudalmiddlefrontal"],lf_df["lh_caudalmiddlefrontal"])
        for region in set(list(hf_df.columns)+list(lf_df.columns)):
            v_hf_region , v_lf_region = hf_df[region], lf_df[region]
            mask = np.isfinite(v_hf_region) & np.isfinite(v_lf_region)
            v_hf_region, v_lf_region = v_hf_region[mask], v_lf_region[mask]
            n_nodes = len(v_hf_region)
           
            if n_nodes < 4 or np.std(v_hf_region) == 0 or np.std(v_lf_region) == 0:
                r, ccc = np.nan, np.nan
                icc3, icc3_p, ci_low, ci_high = np.nan, np.nan, np.nan, np.nan
            else:
                r, _ = pearsonr(v_hf_region, v_lf_region)
                ccc = lin_ccc(v_hf_region, v_lf_region)
    
                long_df = pd.DataFrame({
                    "node": np.tile(np.arange(n_nodes), 2),
                    "session": [hf_label] * n_nodes + [lf_label] * n_nodes,
                    "value": np.concatenate([v_hf_region, v_lf_region]),
                })
                icc_table = pg.intraclass_corr(
                    data=long_df, targets="node", raters="session", ratings="value"
                )
                # pingouin labels ICC(3,1) (two-way mixed, consistency, single
                # measures) as "ICC(C,1)" in current versions, "ICC3" in older
                # ones -- and the CI column as "CI95" or "CI95%" depending on
                # version, so handle both.
                icc3_row = icc_table[icc_table["Type"].isin(["ICC3", "ICC(C,1)"])].iloc[0]
                ci_col = "CI95%" if "CI95%" in icc_table.columns else "CI95"
                icc3 = icc3_row["ICC"]
                icc3_p = icc3_row["pval"]
                ci_low, ci_high = icc3_row[ci_col]
    
            rows.append({
                "region": region,
                "subject": subj,
                "pearson_r": r,
                "ccc": ccc,
                "icc3": icc3,
                "icc3_pval": icc3_p,
                "icc3_ci_low": ci_low,
                "icc3_ci_high": ci_high,
                "n_nodes": n_nodes,
            })
    return pd.DataFrame(rows)


 #### NOT BEING USED - CONNECTIVITY COLLAPSED BY AVERAGE THE WHOLE UPPER TRIANGLE ######   
def compute_subject_level_icc(hf_lf_pairs, hf_label, lf_label):
    """
    Reduce each subject/session to a single scalar (mean connectivity
    across all edges), then compute ICC(3,1) across subjects between HF
    and LF using pingouin's `intraclass_corr`.
 
    Returns
    -------
    scores_df : DataFrame with columns subject, session, mean_connectivity
    icc_table : DataFrame, pingouin's full ICC results table (ICC1..ICC3k)
    """
    rows = []
    for subj, (hf_df, lf_df) in hf_lf_pairs.items():
        hf_mean = np.nanmean(upper_triangle_vector(hf_df.values))
        lf_mean = np.nanmean(upper_triangle_vector(lf_df.values))
        rows.append({"subject": subj, "session": hf_label, "mean_connectivity": hf_mean})
        rows.append({"subject": subj, "session": lf_label, "mean_connectivity": lf_mean})
    scores_df = pd.DataFrame(rows)
 
    icc_table = pg.intraclass_corr(
        data=scores_df,
        targets="subject",
        raters="session",
        ratings="mean_connectivity",
    )
    return scores_df, icc_table

def build_identifiability_matrix(hf_lf_pairs, zscore=False):
    """
    Differential identifiability (Amico & Goni, 2018).

    Returns
    -------
    A : (N, N) array
        Pearson correlation matrix between every subject's HF network and
        every subject's LF network.
    subjects : list
        Subject order corresponding to rows/columns of A.
    I_self, I_others, I_diff : float
        Mean of diagonal (self), mean of off-diagonal (others), and
        100 * (I_self - I_others).
    """
    subjects = list(hf_lf_pairs.keys())
    for i, s in enumerate(subjects):
        vec = upper_triangle_vector(hf_lf_pairs[s][0])
        print(i, s, vec.shape)
    
    subject_roi_sets = {}
    for s in subjects:
        df_hf, df_lf = hf_lf_pairs[s]
        df_hf, df_lf = align_rois(df_hf, df_lf)
        subject_roi_sets[s] = set(df_hf.index)

    # 2) intersection across all subjects
    from functools import reduce
    global_rois = sorted(reduce(set.intersection, subject_roi_sets.values()))

    print("Number of global ROIs:", len(global_rois))  # should be 62 here
    
    X = np.vstack([upper_triangle_vector(hf_lf_pairs[s][0].loc[global_rois, global_rois]) for s in subjects])  # HF
    Y = np.vstack([upper_triangle_vector(hf_lf_pairs[s][1].loc[global_rois, global_rois]) for s in subjects])  # LF

    if zscore:
        X = _zscore_columns(X)
        Y = _zscore_columns(Y)

    n = len(subjects)
    A = np.empty((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            r, _ = pearsonr(X[i], Y[j])
            A[i, j] = r

    diag_vals = np.diag(A)
    I_self = diag_vals.mean()

    off_mask = ~np.eye(n, dtype=bool)
    I_others = A[off_mask].mean()
    I_diff = 100.0 * (I_self - I_others)

    return A, subjects, I_self, I_others, I_diff


def _zscore_columns(M):
    mean = M.mean(axis=0, keepdims=True)
    std = M.std(axis=0, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (M - mean) / std


def append_to_master_csv(df, path):
    """
    Append `df` to the CSV at `path`, creating it if it doesn't exist yet.
    Reads any existing content first so repeated calls across a sweep
    (e.g. one call per atlas/feature/label-pair combination) accumulate
    into a single file instead of overwriting each other.

    If a 'subject' column is present, it's forced to string dtype on both
    read and write -- otherwise pandas infers zero-padded numeric-looking
    IDs (e.g. "001") as int64 on reread and silently strips the padding,
    which would corrupt subject matching in later analysis.
    """
    if "subject" in df.columns:
        df = df.copy()
        df["subject"] = df["subject"].astype(str)

    if os.path.exists(path):
        dtype_map = {"subject": str} if "subject" in df.columns else None
        existing = pd.read_csv(path, dtype=dtype_map)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_csv(path, index=False)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("################",args.suffix, "################")
    print(f"Loading networks from {args.mind_dir} ...")
    subj_networks = load_networks(args.mind_dir, args.subj_part_index, args.session_part_index)
    print(f"  Found {len(subj_networks)} subjects.")

    print(f"Pairing {args.hf_label} (HF) vs {args.lf_label} (LF) sessions ...")
    hf_lf_pairs = get_hf_lf_pairs(subj_networks, args.hf_label, args.lf_label)
    print(f"  {len(hf_lf_pairs)} subjects have both sessions.")
    if len(hf_lf_pairs) < 2:
        raise RuntimeError("Need at least 2 subjects with both HF and LF sessions.")

    # ---- Edge-wise Pearson r / CCC / ICC ----
    #Question: 
    # Given N subjects' HF and LF values at that one connection, 
    # how correlated/concordant are they? This gives me a reliability map : one r/CCC/ICC PER EDGE
    # I am not interested in this question because I am looking at the reliability of patterns and specificity per subject


    #print("Building edge-level table ...")
    #edge_df = build_edge_dataframe(hf_lf_pairs)

    #print("Computing edge-wise Pearson r, CCC, ICC(3,1) ...")
    #edge_metrics = compute_edge_metrics(edge_df)
    # edge_metrics_path = os.path.join(args.output_dir, f"edge_metrics_{args.suffix}.csv")
    # edge_metrics.to_csv(edge_metrics_path, index=False)
    # print(f"  Saved edge-wise metrics -> {edge_metrics_path}")

    #summary = edge_metrics[["pearson_r", "ccc", "icc"]].agg(["mean", "std", "median"])
    #summary_path = os.path.join(args.output_dir, f"edge_metrics_summary_{args.hf_label}_{args.lf_label}.csv")
    #summary.to_csv(summary_path)
    #print("Edge-wise metric summary (across all edges):")
    #print(summary)
    #print(f"  Saved summary -> {summary_path}")
    
    # ----- Subject-level topology correlations ----
    #print("Computing subject-level topology correlations ...")
    #Question: Taking a subject's entire vectorized connectivity vector (thousands of edges) at HF and at LF, 
    # how similar is the whole network pattern? That gives you one r/CCC PER SUBJECT.
    # This question answers: how reliable are ULF-derived networks?
    print("Computing subject-level topology metrics (r, CCC, ICC3) ...")
    topology_metrics = compute_subject_topology_metrics(hf_lf_pairs, args.hf_label, args.lf_label)

    # Metadata columns so this is identifiable once combined across runs
    # (bare --suffix isn't unique per label-pair, so per-run filenames and
    # the master CSV both need hf/lf explicitly).
    topology_metrics.insert(0, "template", args.template)
    topology_metrics.insert(1, "atlas_nfeat", args.suffix)
    topology_metrics.insert(2, "hf_label", args.hf_label)
    topology_metrics.insert(3, "lf_label", args.lf_label)

    topology_path = os.path.join(
        args.output_dir, f"subject_topology_metrics_{args.suffix}_{args.hf_label}_{args.lf_label}.csv"
    )
    #topology_metrics.to_csv(topology_path, index=False)
    #print(f"  Saved subject-level topology metrics -> {topology_path}")
    print(topology_metrics[["pearson_r", "ccc", "icc3"]].agg(["mean", "std", "median"]))

    #Reporting the proportion of subjects that had icc > 0.75, and those where p-val is significant
    prop_icc_gt_0_75 = (topology_metrics["icc3"] > 0.75).mean()
    prop_p_lt_0_05   = (topology_metrics["icc3_pval"] < 0.05).mean()
    print("Prop. subjects where ICC3>0.75", prop_icc_gt_0_75)
    print("Prop. subjects where p<0.05", prop_p_lt_0_05)

    #Reporting min and max val of p
    p_min = topology_metrics["icc3_pval"].min()
    p_max = topology_metrics["icc3_pval"].max()

    print("p_min: ", p_min, "p_max: ", p_max)

    master_topology_path = os.path.join(args.output_dir, "all_subject_topology_metrics.csv")
    append_to_master_csv(topology_metrics, master_topology_path)
    print(f"  Appended to master topology file -> {master_topology_path}")

    #scanner_pairs = [(args.hf_label,args.lf_label)]
    #df_topology = subject_topology_corrs(subj_networks, scanner_pairs)

    #summary_topology = (df_topology.groupby('pair')
    #    .agg(mean_r=('r', lambda x: round(x.mean(), 3)), 
    #        std_r=('r', lambda x: round(x.std(), 3)),
    #        mean_ccc=('ccc', lambda x: round(x.mean(), 3)), 
    #        std_ccc=('ccc', lambda x: round(x.std(), 3)), 
    #        n=('ccc', 'size'))
    #    .reset_index())
    

    #Save summary topology
    #summary_topology_path = os.path.join(args.output_dir, f"subject_topology_summary_{args.suffix}.csv")
    #Read existing file and append new data if it exists
    #if os.path.exists(summary_topology_path):
    #    existing_summary = pd.read_csv(summary_topology_path)
    #    summary_topology = pd.concat([existing_summary, summary_topology], ignore_index=True)
    #summary_topology.to_csv(summary_topology_path, index=False)
    #print(f"  Saved subject-level topology correlation summary -> {summary_topology_path}")
    #print("Subject-level topology correlation summary (across all subjects):")
    #print(summary_topology)
    #print(f"  Saved summary -> {summary_topology_path}")

        

    print("Computing region-level topology metrics (r, CCC, ICC3) ...")
    region_topology_metrics = compute_region_level_metrics(hf_lf_pairs, args.hf_label, args.lf_label)
    region_topology_metrics.insert(0, "template", args.template)
    region_topology_metrics.insert(1, "atlas_nfeat", args.suffix)
    region_topology_metrics.insert(2, "hf_label", args.hf_label)
    region_topology_metrics.insert(3, "lf_label", args.lf_label)
    master_region_topology_path = os.path.join(args.output_dir, "region_topology_metrics.csv")
    append_to_master_csv(region_topology_metrics, master_region_topology_path)

    # ---- Subject identifiability ----
    print("Computing subject identifiability matrix ...")
    A, subjects, I_self, I_others, I_diff = build_identifiability_matrix(
        hf_lf_pairs, zscore=args.zscore
    )
    ident_df = pd.DataFrame(A, index=subjects, columns=subjects)
    # NOTE: previously this filename only included `suffix`, so the matrix
    # from one label-pair silently overwrote another's for the same
    # atlas/feature. hf/lf are now included.
    ident_path = os.path.join(
        args.output_dir, f"identifiability_matrix_{args.suffix}_{args.hf_label}_{args.lf_label}.csv"
    )
    
    #Plot the identifiability matrix
    A_tanh = np.arctanh(A)

    cmap = ["magma", "viridis"]
    # Suppose you have multiple (s1, s2) pairs or multiple conditions
    conditions = [False]   # extend this list when needed

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a gridspec inside the subplot to shrink the colorbar
    cax = ax.inset_axes([1.02, 0.1, 0.03, 0.8])  # [x0, y0, width, height]

    sns.heatmap(
        A_tanh,
        fmt=".2f",
        cmap="magma_r",
        cbar_ax=cax,
        xticklabels=subjects,
        yticklabels=subjects,
        square=True,
        cbar_kws={"label": "tanh(Pearson r)"},
        ax=ax
    )
    cax.set_ylabel("tanh(Pearson r)", fontsize=16)

    ax.set_title(
        fr"Subject Identifiability ({args.hf_label} vs {args.lf_label})",
        fontsize=20,
        weight='bold'
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)

    # Add metrics under the plot
    ax.text(
        0.5, -0.15,
        (
            r"$I_{\text{self}}$ = " + f"{I_self:.4f},  "
            r"$I_{\text{others}}$ = " + f"{I_others:.4f},  "
            r"$I_{\text{diff}}$ = " + f"{I_diff:.4f}"
        ),
        ha='center',
        va='center',
        transform=ax.transAxes,
        fontsize=14,
        weight='bold'
    )

    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/network_identifiability_grid_{args.suffix}_{args.hf_label}_{args.lf_label}.png",  bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)

    ident_df.to_csv(ident_path)
    print(f"  Saved identifiability matrix -> {ident_path}")
    print(f"  I_self={I_self:.3f}, I_others={I_others:.3f}, I_diff={I_diff:.3f}")

    ident_summary_row = pd.DataFrame([{
        "template": args.template,
        "mesh_atlas": args.suffix,
        "hf_label": args.hf_label,
        "lf_label": args.lf_label,
        "n_subjects": len(subjects),
        "I_self": I_self,
        "I_others": I_others,
        "I_diff": I_diff,
    }])

    #ident_summary_path = os.path.join(args.output_dir, f"identifiability_summary_{args.suffix}.csv")
    #append_to_master_csv(ident_summary_row, ident_summary_path)
    #print(f"  Saved identifiability summary -> {ident_summary_path}")

    master_ident_path = os.path.join(args.output_dir, "all_identifiability_summary.csv")
    append_to_master_csv(ident_summary_row, master_ident_path)
    print(f"  Appended to master identifiability summary -> {master_ident_path}")

    print("Done.")


if __name__ == "__main__":
    main()