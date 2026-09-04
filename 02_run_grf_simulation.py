"""
02_run_grf_simulation.py

Run muscle-driven GRF simulation per 1-second window.

Splits a trial's kinematics into 1-second sliding windows, runs the
OpenSimAD trajectory optimization for each window sequentially, and
writes per-window GRF_resultant_*.mot files.

Prerequisites:
    - 01_download_session.py has already been run (or session data is
      otherwise present on disk under dataFolder).
    - .env with API_TOKEN exists (not used for download here, but
      processInputsOpenSimAD reads session metadata).

Output:
    - OpenSimData/Dynamics/<trial_name>/GRF_resultant_*_window_*.mot files
    - OpenSimData/Dynamics/<trial_name>/window_manifest_<trial_name>.json
    - Stale pre-existing files archived to _archive_<timestamp>/

Usage:
    python 02_run_grf_simulation.py [--session-uuid UUID] [--trial-name TRIAL]
        [--motion-type TYPE] [--contact-side SIDE] [--treadmill-speed M/S]
        [--repetition N] [--only-missing]

    All flags optional — bare invocation uses the defaults in the config
    block below (session ab7eb7cf-817d-4035-a30b-ee68773906cb, trial Suhasno_1).

Runtime note:
    5-15 minutes per 1-second window. Windows are solved in parallel, sized to
    available RAM (spare ~1 GB, ~2 GB per worker). Three things are shared by
    every window and so must not be built concurrently:
      - the C++ external function, built once serially here (warm-up) to avoid
        concurrent build collisions on repo-global scratch paths;
      - the muscle-tendon / polynomial caches in the session Model folder, which
        run_tracking builds on first use. Those are guarded by an inter-process
        lock (parallel_config.SharedPrepLock) rather than by the warm-up,
        because which polynomial variant a window needs depends on that window's
        own range of motion, so no single warm-up window can prime them all;
      - the optimaltrajectories.npy aggregate, merged serially after the pool.
"""

import argparse
import os
import sys
import json
import time
import shutil
import numpy as np
from datetime import datetime, timezone

# Configuration — module-level defaults, overridable via CLI flags
# (--session-uuid / --trial-name / --motion-type / --contact-side /
# --treadmill-speed / --repetition). See parse_args().

# Raw 36-char OpenCap session UUID OpenCapData_*
session_uuid = "ab7eb7cf-817d-4035-a30b-ee68773906cb"
session_id = "OpenCapData_" + session_uuid

# Name of the downloaded trial to simulate
trial_name = "Suhasno_1"

# Motion type: "walking", "running", "squats", "sit_to_stand", etc.
motion_type = "walking"

# Contact side: "all", "left", or "right".
contact_side = "all"

# Treadmill speed in m/s (0 = overground).
treadmill_speed = 0

# Repetition index for squats/STS (None = not segmented by rep).
repetition = None

# Time range to analyse. Set both to None to auto-detect from the .mot file.
start_time = None
end_time = None

# Set to True to solve the optimal control problem.
solveProblem = True
# Set to True to analyse results of the solved problem. If you already solved
# the problem previously, set the above to False.
analyzeResults = True

# Path to the data folder where OpenCapData_* sessions are.
dataFolder = os.path.dirname(os.path.abspath(__file__))

# Behaviour: if True, re-run only windows that previously failed.
# Set from the command line via --only-missing.
only_missing = False


def build_windows(start_time, end_time, step=1.0, min_last_duration=0.5):
    """Build 1-second sliding windows, merging a trailing window < 0.5 s.

    Returns a list of [start, end] pairs.
    """
    starts = np.arange(start_time, end_time, step)
    windows = [[float(s), min(float(s) + step, end_time)] for s in starts]

    if len(windows) > 1:
        last_dur = windows[-1][1] - windows[-1][0]
        if 0 < last_dur < min_last_duration:
            windows[-2][1] = windows[-1][1]
            windows.pop()

    return windows


def archive_existing_outputs(dyn_dir):
    """Move pre-existing output files to a timestamped archive subfolder."""
    if not os.path.isdir(dyn_dir):
        return

    existing = [
        f for f in os.listdir(dyn_dir)
        if os.path.isfile(os.path.join(dyn_dir, f))
        and not f.startswith("_archive_")
    ]
    if not existing:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = os.path.join(dyn_dir, f"_archive_{ts}")
    os.makedirs(archive_dir, exist_ok=True)

    print(f"Archiving {len(existing)} pre-existing file(s) -> {archive_dir}/")
    for f in existing:
        shutil.move(os.path.join(dyn_dir, f), os.path.join(archive_dir, f))


def load_manifest(manifest_path):
    """Load an existing window manifest, or None if not found."""
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            return json.load(f)
    return None


def parse_args(argv=None):
    """Parse CLI overrides for the config block. Returns an argparse
    Namespace whose defaults are the module-level config values, so a bare
    invocation behaves exactly like the pre-parameterized script."""
    parser = argparse.ArgumentParser(
        description="Run muscle-driven GRF simulation per 1-second window, "
                    "windows solved in parallel (RAM-sized worker pool).")
    parser.add_argument("--session-uuid", default=session_uuid,
                        help="Raw 36-char OpenCap session UUID (no "
                             "'OpenCapData_' prefix). [default: %(default)s]")
    parser.add_argument("--trial-name", default=trial_name,
                        help="Trial (kinematics .mot stem) to simulate. "
                             "[default: %(default)s]")
    parser.add_argument("--motion-type", default=motion_type,
                        help="Motion type: walking / running / squats / "
                             "sit_to_stand. [default: %(default)s]")
    parser.add_argument("--contact-side", default=contact_side,
                        choices=["all", "left", "right"],
                        help="Contact side. [default: %(default)s]")
    parser.add_argument("--treadmill-speed", type=float, default=treadmill_speed,
                        help="Treadmill speed in m/s (0 = overground). "
                             "[default: %(default)s]")
    parser.add_argument("--repetition", type=int, default=repetition,
                        help="Repetition index for squats/STS. "
                             "[default: %(default)s]")
    parser.add_argument("--only-missing", action="store_true",
                        help="Re-run only windows that previously failed "
                             "(skips converged ones per window_manifest).")
    return parser.parse_args(argv)


def main():
    global session_uuid, session_id, trial_name
    global motion_type, contact_side, treadmill_speed, repetition, only_missing

    args = parse_args()
    session_uuid = args.session_uuid
    session_id = "OpenCapData_" + session_uuid
    trial_name = args.trial_name
    motion_type = args.motion_type
    contact_side = args.contact_side
    treadmill_speed = args.treadmill_speed
    repetition = args.repetition
    only_missing = args.only_missing

    # Path setup
    baseDir = os.path.dirname(os.path.abspath(__file__))
    opensimADDir = os.path.join(baseDir, "UtilsDynamicSimulations", "OpenSimAD")
    modifiedUtilsDir = os.path.join(baseDir, "modified_utils")
    sys.path.insert(0, opensimADDir)
    sys.path.insert(0, baseDir)
    # modified_utils holds the modified copies of mainOpenSimAD.py,
    # utilsOpenSimAD.py, grf_prediction.py, and simulation.py (shared-trajectory
    # race guard, Windows build fix, convergence checks). It is inserted FIRST
    # so the imports below resolve to these modified copies. Sibling modules
    # (settingsOpenSimAD.py, utils.py, utilsProcessing.py) are not copied here
    # and keep resolving from their original locations.
    sys.path.insert(0, modifiedUtilsDir)

    from grf_prediction import get_mot_time_range, process_single_window
    from utilsOpenSimAD import processInputsOpenSimAD
    from parallel_config import compute_worker_count, merge_optimaltrajectories
    import multiprocessing
    import psutil

    dyn_dir = os.path.join(
        dataFolder, session_id, "OpenSimData", "Dynamics", trial_name
    )
    pathTrial = os.path.join(
        dataFolder, session_id, "OpenSimData", "Kinematics", f"{trial_name}.mot"
    )

    _start = start_time
    _end = end_time
    if _start is None or _end is None:
        detected_start, detected_end = get_mot_time_range(pathTrial)
        if _start is None:
            _start = detected_start
        if _end is None:
            _end = detected_end
        print(f"Auto-detected time range: [{_start}, {_end}]")
    else:
        print(f"Using provided time range: [{_start}, {_end}]")

    if _start is None or _end is None:
        raise ValueError(
            "Could not determine time range. "
            "Set start_time / end_time or check that the .mot file exists."
        )

    windows = build_windows(_start, _end)
    print(f"\nTotal windows: {len(windows)}")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i}: [{ws:.2f}, {we:.2f}]  ({we - ws:.2f} s)")

    # --only-missing: skip windows already marked converged in a prior manifest.
    manifest_path = os.path.join(dyn_dir, f"window_manifest_{trial_name}.json")
    skip_indices = set()
    if only_missing:
        prev = load_manifest(manifest_path)
        if prev:
            for w in prev.get("windows", []):
                if w.get("converged", False):
                    skip_indices.add(w["index"])
            if skip_indices:
                print(f"\n--only-missing: skipping {len(skip_indices)} "
                      f"already-converged window(s): {sorted(skip_indices)}")
        else:
            print("\n--only-missing: no prior manifest found — running all windows.")

    # Archive stale outputs before the first window runs (unless --only-missing).
    if not only_missing:
        archive_existing_outputs(dyn_dir)

    # Windows that will actually run this invocation.
    run_list = [(i, w) for i, w in enumerate(windows) if i not in skip_indices]
    if not run_list:
        print("\nNothing to run (all windows already converged).")
        return

    # --- Serial warm-up: build the C++ external function ONCE before any
    # parallelism. buildExternalFunction writes to repo-global scratch paths
    # (utilsOpenSimAD.py:1681, :1800), so concurrent first builds corrupt each
    # other (serverside.md, Section 2). One serial processInputsOpenSimAD call
    # builds it, or early-returns if already built; every worker then only READS
    # the cached per-session function. This also primes the adjusted model and
    # contact geometry. It does NOT prime the muscle-tendon parameters or the
    # polynomial coefficients — those are built inside run_tracking and are
    # serialised by SharedPrepLock instead; see the module docstring.
    warm_i, (warm_s, warm_e) = run_list[0]
    print(f"\nWarm-up: building/loading external function via "
          f"processInputsOpenSimAD on window {warm_i} "
          f"[{warm_s:.2f}, {warm_e:.2f}] ...")
    processInputsOpenSimAD(
        baseDir, dataFolder, session_id, trial_name, motion_type,
        [warm_s, warm_e], repetition, treadmill_speed, contact_side,
        use_local_data=True,
    )
    print("Warm-up complete — external function is built and cached.")

    # --- Size the worker pool from available RAM (spare ~1 GB, ~2 GB/worker).
    available = psutil.virtual_memory().available
    workers = compute_worker_count(
        available_bytes=available,
        cpu_count=os.cpu_count() or 1,
        num_windows=len(run_list),
    )
    print(f"\nAvailable RAM: {available / 1024**3:.1f} GB  ->  {workers} "
          f"parallel worker(s) (reserve 1 GB, ~2 GB/worker, "
          f"{os.cpu_count()} CPUs, {len(run_list)} windows to run).")

    # --- Prevent concurrent workers from racing on the shared aggregate file.
    # Each worker writes only its per-case optimaltrajectories_<case>.npy; the
    # shared optimaltrajectories.npy is rebuilt serially after the pool.
    os.environ["OPENSIMAD_SKIP_SHARED_TRAJ"] = "1"

    tasks = [
        (
            baseDir, dataFolder, session_id, trial_name, motion_type,
            [win_start, win_end], repetition, treadmill_speed, contact_side,
            solveProblem, analyzeResults, trial_name, i,
        )
        for i, (win_start, win_end) in run_list
    ]

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.starmap(process_single_window, tasks)

    # --- Rebuild the shared aggregate once, serially (race-free).
    merged = merge_optimaltrajectories(dyn_dir, trial_name)
    if merged:
        print(f"Merged shared aggregate: {merged}")

    # --- Build manifest rows (read per-window stats for return status).
    # GRF paths are stored relative to the session folder so a manifest stays
    # valid if the repo is moved or cloned somewhere else; 03 resolves them
    # against its own dataFolder. Absolute is only a fallback for the odd case
    # of a path that isn't under the session folder at all.
    session_root = os.path.join(dataFolder, session_id)
    window_results = []
    for (i, (win_start, win_end)), (grf_path, _) in zip(run_list, results):
        current_case = f"{trial_name}_window_{i}"
        stats_path = os.path.join(dyn_dir, f"stats_{current_case}.npy")
        converged = bool(grf_path)
        return_status = "unknown"
        if os.path.exists(stats_path):
            try:
                s = np.load(stats_path, allow_pickle=True).item()
                return_status = s.get("return_status", "unknown")
                converged = s.get("success", False) and bool(grf_path)
            except Exception:
                pass
        manifest_grf_path = grf_path
        if grf_path:
            try:
                rel = os.path.relpath(grf_path, session_root)
                if not rel.startswith(os.pardir):
                    manifest_grf_path = rel.replace(os.sep, "/")
            except ValueError:
                pass  # different drive on Windows: keep the absolute path
        window_results.append({
            "index": i,
            "time_start": win_start,
            "time_end": win_end,
            "converged": converged,
            "ipopt_return_status": return_status,
            "grf_resultant_path": manifest_grf_path,
            "solved_at": datetime.now(timezone.utc).isoformat(),
        })
        status = "OK" if converged else "FAIL"
        print(f"  Window {i}: {status}  -> "
              f"{os.path.basename(grf_path) if grf_path else 'N/A'}")

    # Per-window status table.
    print(f"\n{'=' * 70}")
    print(f"{'Idx':>4s}  {'Start':>7s}  {'End':>7s}  {'Converged':>9s}  "
          f"{'Return Status':>25s}")
    print(f"{'-'*4}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*25}")
    for wr in window_results:
        print(f"{wr['index']:4d}  {wr['time_start']:7.2f}  {wr['time_end']:7.2f}  "
              f"{str(wr['converged']):>9s}  {wr['ipopt_return_status']:>25s}")
    print(f"{'=' * 70}")

    num_ok = sum(1 for w in window_results if w["converged"])
    num_fail = len(window_results) - num_ok
    print(f"Summary: {num_ok}/{len(window_results)} converged, {num_fail} failed.")

    # Under --only-missing, merge new rows with the prior manifest so previously
    # converged (skipped) windows are preserved in the manifest.
    if only_missing:
        prev = load_manifest(manifest_path)
        if prev:
            by_idx = {w["index"]: w for w in prev.get("windows", [])}
            for wr in window_results:
                by_idx[wr["index"]] = wr
            window_results = [by_idx[k] for k in sorted(by_idx)]

    # Write manifest.
    manifest = {
        "trial_name": trial_name,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "windows": window_results,
    }
    os.makedirs(dyn_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_path}")


if __name__ == "__main__":
    main()
