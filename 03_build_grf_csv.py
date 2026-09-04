"""
03_build_grf_csv.py — Concatenate per-window GRF .mot files into one CSV.

Reads the window manifest produced by 02_run_grf_simulation.py, selects only
converged windows, concatenates them (sorted by time, duplicates removed),
exports a single CSV, a coverage sidecar JSON, and optionally plots the 6
ground reaction force components.

Prerequisites:
    - 02_run_grf_simulation.py has already been run, producing
      window_manifest_<trial_name>.json under OpenSimData/Dynamics/<trial_name>/.

Output:
    - grf_df_<trial_name>_<session_id>.csv in dataFolder
    - grf_df_<trial_name>_<session_id>_coverage.json in dataFolder
    - (Optional) 2x3 matplotlib plot window

Usage:
    python 03_build_grf_csv.py [--session-uuid UUID] [--trial-name TRIAL]
                               [--no-plots]

    All flags optional — bare invocation uses the defaults in the config
    block below (session ab7eb7cf-817d-4035-a30b-ee68773906cb, trial Suhasno_1).
"""

import argparse
import os
import json
import sys
import numpy as np
import pandas as pd

# Configuration — module-level defaults, overridable via CLI flags
# (--session-uuid / --trial-name / --no-plots). See parse_args().

# Raw 36-char OpenCap session UUID (no "OpenCapData_" prefix).
session_uuid = "ab7eb7cf-817d-4035-a30b-ee68773906cb"

# Prefixed folder name — must match what Files 1-2 use.
session_id = "OpenCapData_" + session_uuid

# Name of the trial whose GRF files should be concatenated.
trial_name = "Suhasno_1"

# Path to the data folder (parent of the OpenCapData_* session folder).
dataFolder = os.path.dirname(os.path.abspath(__file__))

# Set to True to show a 2x3 matplotlib plot of the 6 GRF force components.
make_plots = True

# Force components checked by the all-zero window warning.
GRF_labels = [
    "ground_force_right_vx", "ground_force_right_vy", "ground_force_right_vz",
    "ground_force_left_vx",  "ground_force_left_vy",  "ground_force_left_vz",
]


def get_mot_time_range(mot_file_path):
    """Read a .mot file and return the min and max time values."""
    with open(mot_file_path, "r") as f:
        for line in f:
            if "endheader" in line:
                break
        times = []
        for line in f:
            if line.strip() == "":
                continue
            try:
                times.append(float(line.split()[0]))
            except ValueError:
                continue
        if times:
            return min(times), max(times)
        return None, None


def read_mot_file_to_df(filepath):
    """Read a .mot file into a pandas DataFrame."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if "endheader" in line:
            header_end = i
            break

    df = pd.read_csv(filepath, sep="\t", skiprows=header_end + 1).dropna(
        axis=1, how="all"
    )
    return df


def concatenate_grf_files(file_paths):
    """Concatenate multiple GRF .mot files, sorting by time and removing
    duplicate timestamps (which can appear at window boundaries).
    """
    all_grf_data = []
    for f_path in file_paths:
        try:
            df = read_mot_file_to_df(f_path)
            all_grf_data.append(df)
        except Exception as e:
            print(f"  Error reading {f_path}: {e}")
            continue

    if not all_grf_data:
        print("No GRF data found!")
        return pd.DataFrame()

    concatenated_df = pd.concat(all_grf_data, ignore_index=True)
    concatenated_df = concatenated_df.sort_values("time").drop_duplicates(
        subset=["time"]
    )
    return concatenated_df


def compute_coverage(converged_windows):
    """Build valid_ranges and gap_ranges from a sorted list of converged windows.

    Each window entry has 'time_start', 'time_end'. Returns two lists of
    [start, end] pairs.
    """
    sorted_ws = sorted(converged_windows, key=lambda w: w["time_start"])
    valid_ranges = []
    for w in sorted_ws:
        if valid_ranges and w["time_start"] <= valid_ranges[-1][1] + 1e-6:
            valid_ranges[-1][1] = max(valid_ranges[-1][1], w["time_end"])
        else:
            valid_ranges.append([w["time_start"], w["time_end"]])

    return valid_ranges


def compute_gaps(valid_ranges, kin_start, kin_end):
    """Return gap_ranges covering every part of [kin_start, kin_end] not in
    valid_ranges."""
    gaps = []
    cursor = kin_start
    for vs, ve in valid_ranges:
        if vs > cursor + 1e-6:
            gaps.append([cursor, vs])
        cursor = max(cursor, ve)
    if cursor < kin_end - 1e-6:
        gaps.append([cursor, kin_end])
    return gaps


def plot_grf(grf_df, trial_name):
    """2x3 matplotlib grid of the 6 GRF force components vs time."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if grf_df.empty or not all(lbl in grf_df.columns for lbl in GRF_labels):
        print("Cannot plot: missing expected GRF columns in DataFrame.")
        print(f"Available columns: {grf_df.columns.tolist()}")
        return

    time_col = grf_df["time"]
    linewidth = 2
    fontsizeLabel = 14
    fontsizeTitle = 14
    fontsizeTicks = 14
    fontsizeSubTitle = 16
    colors = sns.color_palette("colorblind", 1)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(
        f"Ground Reaction Forces - {trial_name}",
        fontsize=fontsizeSubTitle,
        fontweight="bold",
    )

    for i, ax in enumerate(axs.flat):
        if i < len(GRF_labels):
            ax.plot(time_col, grf_df[GRF_labels[i]], c=colors[0], linewidth=linewidth)
            ax.set_title(GRF_labels[i], fontsize=fontsizeTitle, fontweight="bold")

    for ax in axs[-1, :]:
        ax.set_xlabel("Time (s)", fontsize=fontsizeLabel, fontweight="bold")
    for ax in axs[:, 0]:
        ax.set_ylabel("Force (N)", fontsize=fontsizeLabel, fontweight="bold")

    for ax in axs.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=fontsizeTicks)
        ax.grid(True)

    fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.85)
    plt.show()


def parse_args(argv=None):
    """Parse CLI overrides for the config block. Returns an argparse
    Namespace whose defaults are the module-level config values, so a bare
    invocation behaves exactly like the pre-parameterized script."""
    parser = argparse.ArgumentParser(
        description="Concatenate converged per-window GRF .mot files into one "
                    "CSV (from the window manifest of "
                    "02_run_grf_simulation.py).")
    parser.add_argument("--session-uuid", default=session_uuid,
                        help="Raw 36-char OpenCap session UUID (no "
                             "'OpenCapData_' prefix). [default: %(default)s]")
    parser.add_argument("--trial-name", default=trial_name,
                        help="Trial whose GRF windows should be concatenated. "
                             "[default: %(default)s]")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the 2x3 GRF plot window.")
    return parser.parse_args(argv)


def main():
    global session_id, trial_name, make_plots

    args = parse_args()
    session_id = "OpenCapData_" + args.session_uuid
    trial_name = args.trial_name
    make_plots = not args.no_plots

    manifest_path = os.path.join(
        dataFolder, session_id, "OpenSimData", "Dynamics", trial_name,
        f"window_manifest_{trial_name}.json",
    )

    if not os.path.exists(manifest_path):
        print(f"ERROR: No window manifest found: {manifest_path}")
        print("Run 02_run_grf_simulation.py first to produce the manifest.")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    windows = manifest.get("windows", [])
    converged = [w for w in windows if w.get("converged") and w.get("grf_resultant_path")]

    if not converged:
        print("No converged windows in manifest — nothing to concatenate.")
        return

    # Check each converged window's file still exists on disk. Keep the window
    # entry paired with its resolved path — a window whose file has gone missing
    # is dropped from both, so it can neither be misattributed in the warnings
    # below nor counted towards coverage.
    valid_windows = []
    for w in converged:
        path = w["grf_resultant_path"]
        if os.path.isabs(path):
            full_path = path
        else:
            # Manifests written by 02 store paths relative to the session
            # folder, which keeps them valid across machines.
            full_path = os.path.join(dataFolder, session_id, path)
        if os.path.exists(full_path):
            valid_windows.append((w, full_path))
        else:
            print(f"WARNING: converged window {w['index']} GRF file missing: "
                  f"{full_path}")

    if not valid_windows:
        print("No converged window GRF files found on disk.")
        return

    valid_paths = [path for _, path in valid_windows]

    print(f"Using {len(valid_paths)} converged window(s) from manifest:")
    for p in valid_paths:
        print(f"  {os.path.basename(p)}")

    # All-zero window warning.
    for w, path in valid_windows:
        df_w = read_mot_file_to_df(path)
        available = [c for c in GRF_labels if c in df_w.columns]
        if available and (df_w[available].abs().max() < 1e-3).all():
            print(f"WARNING: window {w['index']} appears all-zero — "
                  f"may be a degenerate no-contact solution.")

    # Concatenate.
    grf_df = concatenate_grf_files(valid_paths)
    print(f"\nConcatenated DataFrame: {grf_df.shape[0]} rows x "
          f"{grf_df.shape[1]} columns")

    if grf_df.empty:
        print("No data to export.")
        return

    # Coverage analysis.
    pathTrial = os.path.join(
        dataFolder, session_id, "OpenSimData", "Kinematics",
        f"{trial_name}.mot",
    )
    kin_start, kin_end = get_mot_time_range(pathTrial)
    if kin_start is None:
        kin_start = grf_df["time"].min()
        kin_end = grf_df["time"].max()

    valid_ranges = compute_coverage([w for w, _ in valid_windows])
    gaps = compute_gaps(valid_ranges, kin_start, kin_end)

    total_valid = sum(ve - vs for vs, ve in valid_ranges)
    total_kin = kin_end - kin_start
    pct = 100 * total_valid / total_kin if total_kin > 0 else 0

    print(f"\nTrial {trial_name}: kinematics [{kin_start:.2f}, {kin_end:.2f}] s")
    if valid_ranges:
        ranges_str = ", ".join(f"[{vs:.2f}, {ve:.2f}]" for vs, ve in valid_ranges)
        print(f"GRF coverage: {ranges_str} s  ({total_valid:.2f}s / {total_kin:.2f}s = {pct:.1f}%)")
    if gaps:
        gaps_str = ", ".join(f"[{gs:.2f}, {ge:.2f}]" for gs, ge in gaps)
        print(f"GAP: {gaps_str} s")

    # Export CSV.
    output_path = os.path.join(
        dataFolder, f"grf_df_{trial_name}_{session_id}.csv"
    )
    grf_df.to_csv(output_path, index=False)
    print(f"CSV saved: {output_path}")

    # Coverage sidecar.
    coverage = {
        "trial_name": trial_name,
        "kinematics_time_range": [kin_start, kin_end],
        "valid_ranges": valid_ranges,
        "gap_ranges": gaps,
    }
    sidecar_path = os.path.join(
        dataFolder,
        f"grf_df_{trial_name}_{session_id}_coverage.json",
    )
    with open(sidecar_path, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"Coverage sidecar saved: {sidecar_path}")

    if make_plots:
        plot_grf(grf_df, trial_name)


if __name__ == "__main__":
    main()
