"""
03_build_grf_csv.py - Concatenate per-window GRF .mot files into one CSV.

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

    All flags optional - bare invocation uses the defaults in the config
    block below (session ab7eb7cf-817d-4035-a30b-ee68773906cb, trial Suhasno_1).
"""

import argparse
import json
import os

import pandas as pd

import pipeline_io

# Configuration - module-level defaults, overridable via CLI flags. These are
# read once, by parse_args(), and never reassigned.

# Raw 36-char OpenCap session UUID (no "OpenCapData_" prefix).
DEFAULT_SESSION_UUID = "ab7eb7cf-817d-4035-a30b-ee68773906cb"

# Name of the trial whose GRF files should be concatenated.
DEFAULT_TRIAL_NAME = "Suhasno_1"

# Path to the data folder (parent of the OpenCapData_* session folder).
DATA_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Force components checked by the all-zero window warning.
GRF_LABELS = [
    "ground_force_right_vx", "ground_force_right_vy", "ground_force_right_vz",
    "ground_force_left_vx",  "ground_force_left_vy",  "ground_force_left_vz",
]

# Tolerance for treating window boundaries as touching, in seconds.
_TIME_EPS = 1e-6


def load_converged_windows(manifest, dataFolder, session_id):
    """Read every converged window's GRF file, once.

    Returns a list of ``(window_entry, DataFrame)`` pairs. A window whose file
    has gone missing or cannot be parsed is dropped from the list entirely, so
    it can neither be misattributed in the warnings below nor counted towards
    coverage. Reading here rather than in each consumer means each .mot is
    parsed exactly once.
    """
    session_root = pipeline_io.session_dir(dataFolder, session_id)
    loaded = []
    for w in manifest.get("windows", []):
        if not (w.get("converged") and w.get("grf_resultant_path")):
            continue
        path = w["grf_resultant_path"]
        # Manifests written by 02 store paths relative to the session folder,
        # which keeps them valid across machines.
        full_path = path if os.path.isabs(path) else os.path.join(
            session_root, path)
        if not os.path.exists(full_path):
            print(f"WARNING: converged window {w['index']} GRF file missing: "
                  f"{full_path}")
            continue
        try:
            loaded.append((w, pipeline_io.read_mot(full_path)))
        except Exception as e:
            print(f"WARNING: window {w['index']} GRF file unreadable "
                  f"({full_path}): {e}")
    return loaded


def warn_all_zero(loaded):
    """Flag windows whose force components are all ~zero (no-contact solves)."""
    for w, df in loaded:
        available = [c for c in GRF_LABELS if c in df.columns]
        if available and (df[available].abs().max() < 1e-3).all():
            print(f"WARNING: window {w['index']} appears all-zero - "
                  f"may be a degenerate no-contact solution.")


def concatenate_grf_frames(frames):
    """Concatenate per-window GRF frames, sorting by time and dropping the
    duplicate timestamps that appear at window boundaries."""
    if not frames:
        return pd.DataFrame()
    return (pd.concat(frames, ignore_index=True)
              .sort_values("time")
              .drop_duplicates(subset=["time"]))


def merge_time_ranges(windows):
    """Collapse window [time_start, time_end] spans into contiguous ranges.

    Returns a list of ``[start, end]`` pairs, sorted, with touching or
    overlapping windows merged into one.
    """
    ranges = []
    for w in sorted(windows, key=lambda w: w["time_start"]):
        if ranges and w["time_start"] <= ranges[-1][1] + _TIME_EPS:
            ranges[-1][1] = max(ranges[-1][1], w["time_end"])
        else:
            ranges.append([w["time_start"], w["time_end"]])
    return ranges


def compute_gaps(valid_ranges, kin_start, kin_end):
    """Return the parts of [kin_start, kin_end] not covered by valid_ranges."""
    gaps = []
    cursor = kin_start
    for vs, ve in valid_ranges:
        if vs > cursor + _TIME_EPS:
            gaps.append([cursor, vs])
        cursor = max(cursor, ve)
    if cursor < kin_end - _TIME_EPS:
        gaps.append([cursor, kin_end])
    return gaps


def plot_grf(grf_df, trial_name):
    """2x3 matplotlib grid of the 6 GRF force components vs time."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if grf_df.empty or not all(lbl in grf_df.columns for lbl in GRF_LABELS):
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
        if i < len(GRF_LABELS):
            ax.plot(time_col, grf_df[GRF_LABELS[i]], c=colors[0],
                    linewidth=linewidth)
            ax.set_title(GRF_LABELS[i], fontsize=fontsizeTitle,
                         fontweight="bold")

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
    parser.add_argument("--session-uuid", default=DEFAULT_SESSION_UUID,
                        help="Raw 36-char OpenCap session UUID (no "
                             "'OpenCapData_' prefix). [default: %(default)s]")
    parser.add_argument("--trial-name", default=DEFAULT_TRIAL_NAME,
                        help="Trial whose GRF windows should be concatenated. "
                             "[default: %(default)s]")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip the 2x3 GRF plot window.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    session_id = "OpenCapData_" + args.session_uuid
    trial_name = args.trial_name

    manifest_file = pipeline_io.manifest_path(
        DATA_FOLDER, session_id, trial_name)
    if not os.path.exists(manifest_file):
        print(f"ERROR: No window manifest found: {manifest_file}")
        print("Run 02_run_grf_simulation.py first to produce the manifest.")
        return

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    loaded = load_converged_windows(manifest, DATA_FOLDER, session_id)
    if not loaded:
        print("No converged window GRF files found on disk - "
              "nothing to concatenate.")
        return

    print(f"Using {len(loaded)} converged window(s) from manifest:")
    for w, _ in loaded:
        print(f"  window {w['index']}: [{w['time_start']:.2f}, "
              f"{w['time_end']:.2f}]")

    warn_all_zero(loaded)

    grf_df = concatenate_grf_frames([df for _, df in loaded])
    print(f"\nConcatenated DataFrame: {grf_df.shape[0]} rows x "
          f"{grf_df.shape[1]} columns")
    if grf_df.empty:
        print("No data to export.")
        return

    # Coverage analysis, against the kinematics the windows were cut from.
    kin_start, kin_end = pipeline_io.mot_time_range(
        pipeline_io.kinematics_mot(DATA_FOLDER, session_id, trial_name))
    if kin_start is None:
        kin_start = float(grf_df["time"].min())
        kin_end = float(grf_df["time"].max())

    valid_ranges = merge_time_ranges([w for w, _ in loaded])
    gaps = compute_gaps(valid_ranges, kin_start, kin_end)

    total_valid = sum(ve - vs for vs, ve in valid_ranges)
    total_kin = kin_end - kin_start
    pct = 100 * total_valid / total_kin if total_kin > 0 else 0

    print(f"\nTrial {trial_name}: kinematics [{kin_start:.2f}, {kin_end:.2f}] s")
    if valid_ranges:
        ranges_str = ", ".join(f"[{vs:.2f}, {ve:.2f}]" for vs, ve in valid_ranges)
        print(f"GRF coverage: {ranges_str} s  "
              f"({total_valid:.2f}s / {total_kin:.2f}s = {pct:.1f}%)")
    if gaps:
        gaps_str = ", ".join(f"[{gs:.2f}, {ge:.2f}]" for gs, ge in gaps)
        print(f"GAP: {gaps_str} s")

    stem = os.path.join(DATA_FOLDER, f"grf_df_{trial_name}_{session_id}")

    grf_df.to_csv(f"{stem}.csv", index=False)
    print(f"CSV saved: {stem}.csv")

    with open(f"{stem}_coverage.json", "w") as f:
        json.dump({
            "trial_name": trial_name,
            "kinematics_time_range": [kin_start, kin_end],
            "valid_ranges": valid_ranges,
            "gap_ranges": gaps,
        }, f, indent=2)
    print(f"Coverage sidecar saved: {stem}_coverage.json")

    if not args.no_plots:
        plot_grf(grf_df, trial_name)


if __name__ == "__main__":
    main()
