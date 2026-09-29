"""
02_run_grf_simulation.py

Run muscle-driven GRF simulation per 1-second window.

Splits a trial's kinematics into 1-second sliding windows, solves the OpenSimAD
trajectory optimization for each window in parallel, and writes per-window
GRF_resultant_*.mot files plus a manifest describing what converged.

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
        [--only-missing] [--max-workers N]

    All flags optional - bare invocation uses the defaults in the config
    block below (session ab7eb7cf-817d-4035-a30b-ee68773906cb, trial Suhasno_1).
    --repetition is rejected: repetition segmentation replaces every window's
    interval with that one repetition, so it cannot be combined with windowing.

    RAM reserve / worker cap:
    --max-workers N caps the RAM/CPU-derived worker pool at N (default:
    unset, no cap). The GRF_RESERVE_MIB environment variable (an integer
    number of mebibytes) overrides how much RAM is kept free instead of
    parallel_config.DEFAULT_RESERVE_BYTES; unset or invalid falls back to
    that default (invalid values print a warning first).

Runtime note:
    5-15 minutes per 1-second window. Windows are solved in parallel, sized to
    available RAM (spare ~1 GB, ~2 GB per worker). The results must be the
    same as a sequential run of the same windows (grf_prediction_linear.py), so
    everything one window leaves on disk for the next is built before the pool:
      - A serial prep pass calls processInputsOpenSimAD and
        run_tracking(prepOnly=True) for every window, in index order. Its first
        call builds the C++ external function, whose build writes to
        repo-global scratch paths and so must never run concurrently. The pass
        as a whole builds the muscle-tendon / dummy-motion / polynomial caches
        in the session Model folder in exactly the order a sequential run
        would. Which cache a window creates depends on that window's own range
        of motion, so first-come-first-served inside the pool could let a
        different window own a cache than in the sequential run.
      - Workers then only read those caches. SharedPrepLock
        (sharedPrepLockOpenSimAD) still guards the cache regions as a safety
        net, and serialises the joint-reaction analyses, which write scratch
        files into the results folder all windows share.
    The shared optimaltrajectories.npy is never written by a worker - each
    writes only its own per-case file - and is rebuilt here after the pool.
"""

import argparse
import json
import multiprocessing
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone

from pipeline_io import build_windows  # noqa: F401  (also used by tests)
from parallel_config import DEFAULT_RESERVE_BYTES, compute_worker_count

# Configuration - module-level defaults, overridable via CLI flags. These are
# read once, by parse_args(), and never reassigned.

# Raw 36-char OpenCap session UUID (no "OpenCapData_" prefix).
DEFAULT_SESSION_UUID = "ab7eb7cf-817d-4035-a30b-ee68773906cb"

# Name of the downloaded trial to simulate.
DEFAULT_TRIAL_NAME = "Suhasno_1"

# Motion type: "walking", "running", "squats", "sit_to_stand", etc.
DEFAULT_MOTION_TYPE = "walking"

# Contact side: "all", "left", or "right".
DEFAULT_CONTACT_SIDE = "all"

# Treadmill speed in m/s (0 = overground).
DEFAULT_TREADMILL_SPEED = 0

# Repetition index for squats/STS. Must stay None: see parse_args.
DEFAULT_REPETITION = None

# Time range to analyse. Both None = auto-detect from the kinematics .mot.
START_TIME = None
END_TIME = None

# Solve the optimal control problem / analyse its results. Set SOLVE_PROBLEM to
# False and ANALYZE_RESULTS to True to post-process an already-solved problem.
SOLVE_PROBLEM = True
ANALYZE_RESULTS = True

# Path to the data folder where OpenCapData_* sessions are.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = BASE_DIR


def _reserve_bytes_from_env():
    """RAM reserve (bytes) for the worker pool, from the GRF_RESERVE_MIB
    environment variable (an integer number of mebibytes), falling back to
    parallel_config.DEFAULT_RESERVE_BYTES.

    Unset or empty: falls back silently (this is the normal case). Set but
    not a valid positive integer: also falls back, but prints a warning,
    since that is a misconfiguration rather than an intentional omission.
    """
    raw = os.environ.get("GRF_RESERVE_MIB")
    if raw is None or raw.strip() == "":
        return DEFAULT_RESERVE_BYTES

    try:
        mib = int(raw.strip())
        if mib <= 0:
            raise ValueError("must be positive")
    except ValueError:
        print(f"WARNING: GRF_RESERVE_MIB={raw!r} is not a valid positive "
              f"integer (mebibytes); falling back to the default "
              f"{DEFAULT_RESERVE_BYTES // 1024**2} MiB reserve.")
        return DEFAULT_RESERVE_BYTES

    return mib * 1024**2


def _effective_worker_count(available_bytes, cpu_count, num_windows,
                             reserve_bytes, max_workers=None):
    """compute_worker_count's RAM/CPU-sized pool, additionally capped by
    ``max_workers`` (an operator-supplied ceiling, e.g. --max-workers).

    ``compute_worker_count`` itself is untouched - this only adds one more
    ``min(...)`` term on top of what it returns.
    """
    workers = compute_worker_count(
        available_bytes=available_bytes,
        cpu_count=cpu_count,
        num_windows=num_windows,
        reserve_bytes=reserve_bytes,
    )
    if max_workers is not None:
        workers = min(workers, max_workers)
    return workers


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


def load_manifest(manifest_file):
    """Load an existing window manifest, or None if not found or unreadable."""
    try:
        with open(manifest_file, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def resolve_time_range(kinematics_mot, start_time, end_time):
    """Fill in whichever of start/end was not configured, from the .mot file."""
    from pipeline_io import mot_time_range

    if start_time is not None and end_time is not None:
        print(f"Using provided time range: [{start_time}, {end_time}]")
        return start_time, end_time

    detected_start, detected_end = mot_time_range(kinematics_mot)
    start = detected_start if start_time is None else start_time
    end = detected_end if end_time is None else end_time
    print(f"Auto-detected time range: [{start}, {end}]")

    if start is None or end is None:
        raise ValueError(
            f"Could not determine time range from {kinematics_mot}. "
            "Set START_TIME / END_TIME or check that the .mot file exists."
        )
    return start, end


def run_pool(tasks, workers):
    """Solve every window, keeping whatever finishes if the pool breaks.

    ``pool.starmap`` would collect all results or raise; a single OOM-killed
    worker would then take down the merge and the manifest write with it,
    discarding hours of solving that already succeeded and already wrote its
    files. Streaming results back as they land means a broken pool costs only
    the windows that had not reported yet.
    """
    from grf_prediction import solve_window_task

    results = {}
    try:
        with multiprocessing.Pool(processes=workers) as pool:
            for res in pool.imap_unordered(solve_window_task, tasks):
                results[res.index] = res
                print(f"  Window {res.index}: "
                      f"{'OK' if res.converged else 'FAIL'} "
                      f"({len(results)}/{len(tasks)} reported)")
    except Exception:
        traceback.print_exc()
        print(f"\nWARNING: the worker pool failed after "
              f"{len(results)}/{len(tasks)} window(s) reported. "
              f"Recording those; the rest are marked failed.")
    return results


def manifest_row(result, session_root):
    """One manifest entry, straight from the worker's own verdict.

    The GRF path is stored relative to the session folder so a manifest stays
    valid if the repo is moved or cloned somewhere else; 03 resolves it against
    its own dataFolder. Absolute is only a fallback for the odd case of a path
    that isn't under the session folder at all.
    """
    grf_path = result.grf_path
    if grf_path:
        try:
            rel = os.path.relpath(grf_path, session_root)
            if not rel.startswith(os.pardir):
                grf_path = rel.replace(os.sep, "/")
        except ValueError:
            pass  # different drive on Windows: keep the absolute path

    return {
        "index": result.index,
        "time_start": result.time_start,
        "time_end": result.time_end,
        "converged": result.converged,
        "ipopt_return_status": result.return_status,
        "failure_reason": result.failure_reason,
        "grf_resultant_path": grf_path,
        "solved_at": datetime.now(timezone.utc).isoformat(),
        "mean_vertical_grf": result.mean_vertical_grf,
        "peak_vertical_grf": result.peak_vertical_grf,
        "min_clearance_m": result.min_clearance_m,
        "frac_frames_within_5mm": result.frac_frames_within_5mm,
        "dedrift_method": result.dedrift_method,
        "dynamics_consistency_residual_N": result.dynamics_consistency_residual_N,
        "flip_count": result.flip_count,
    }


def print_summary(rows):
    """Per-window status table plus a converged/failed tally."""
    divider = "=" * 70
    print(f"\n{divider}")
    print(f"{'Idx':>4s}  {'Start':>7s}  {'End':>7s}  {'Converged':>9s}  "
          f"{'Return Status':>25s}")
    print(f"{'-' * 4}  {'-' * 7}  {'-' * 7}  {'-' * 9}  {'-' * 25}")
    for row in rows:
        print(f"{row['index']:4d}  {row['time_start']:7.2f}  "
              f"{row['time_end']:7.2f}  {str(row['converged']):>9s}  "
              f"{row['ipopt_return_status']:>25s}")
    print(divider)

    num_ok = sum(1 for r in rows if r["converged"])
    print(f"Summary: {num_ok}/{len(rows)} converged, {len(rows) - num_ok} failed.")


def parse_args(argv=None):
    """Parse CLI overrides for the config block. Returns an argparse
    Namespace whose defaults are the module-level config values, so a bare
    invocation behaves exactly like the pre-parameterized script."""
    parser = argparse.ArgumentParser(
        description="Run muscle-driven GRF simulation per 1-second window, "
                    "windows solved in parallel (RAM-sized worker pool).")
    parser.add_argument("--session-uuid", default=DEFAULT_SESSION_UUID,
                        help="Raw 36-char OpenCap session UUID (no "
                             "'OpenCapData_' prefix). [default: %(default)s]")
    parser.add_argument("--trial-name", default=DEFAULT_TRIAL_NAME,
                        help="Trial (kinematics .mot stem) to simulate. "
                             "[default: %(default)s]")
    parser.add_argument("--motion-type", default=DEFAULT_MOTION_TYPE,
                        help="Motion type: walking / running / squats / "
                             "sit_to_stand. [default: %(default)s]")
    parser.add_argument("--contact-side", default=DEFAULT_CONTACT_SIDE,
                        choices=["all", "left", "right"],
                        help="Contact side. [default: %(default)s]")
    parser.add_argument("--treadmill-speed", type=float,
                        default=DEFAULT_TREADMILL_SPEED,
                        help="Treadmill speed in m/s (0 = overground). "
                             "[default: %(default)s]")
    parser.add_argument("--repetition", type=int, default=DEFAULT_REPETITION,
                        help="Not supported by the windowed pipeline; passing "
                             "it exits with an error.")
    parser.add_argument("--only-missing", action="store_true",
                        help="Re-run only windows that previously failed "
                             "(skips converged ones per window_manifest).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Cap the RAM/CPU-derived worker pool size at "
                             "this many concurrent windows. [default: "
                             "unset - no cap beyond RAM/CPU sizing]")
    args = parser.parse_args(argv)

    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be >= 1.")

    # Rejected before anything touches the disk. processInputsOpenSimAD would
    # replace every window's interval with times_window[repetition], so all
    # workers would solve the same interval, and run_tracking would write to
    # Dynamics/<trial>_rep<N>, which this script never reads.
    if args.repetition is not None:
        parser.error(
            "--repetition is not supported by the windowed pipeline: "
            "repetition segmentation replaces every window's interval with "
            "that one repetition and writes results to "
            "Dynamics/<trial>_rep<N>, which this script never reads. "
            "Omit --repetition.")
    return args


def main(argv=None):
    args = parse_args(argv)

    # Pin each solve to one thread BEFORE any worker is forked or spawned:
    # OpenMP and MKL read this when their runtime loads, so setting it inside
    # the worker (as an earlier version did) was already too late. Without it,
    # `workers` solves each spawn `cpu_count` threads and thrash the machine.
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # modified_utils used to shadow mainOpenSimAD/utilsOpenSimAD via sys.path
    # ordering; its changes now live in the vendored copies, so plain appends
    # suffice and import order no longer decides which code runs.
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(
        BASE_DIR, "UtilsDynamicSimulations", "OpenSimAD"))

    import psutil
    import pipeline_io
    from pipeline_io import TrialSpec, WindowResult
    from parallel_config import merge_optimaltrajectories
    from utilsOpenSimAD import processInputsOpenSimAD
    from kinematicsQC import run_qc_pass, verify_qc_marker
    from mainOpenSimAD import run_tracking

    session_id = "OpenCapData_" + args.session_uuid
    spec = TrialSpec(
        baseDir=BASE_DIR,
        dataFolder=DATA_FOLDER,
        session_id=session_id,
        trial_name=args.trial_name,
        motion_type=args.motion_type,
        repetition=args.repetition,
        treadmill_speed=args.treadmill_speed,
        contact_side=args.contact_side,
        solve_problem=SOLVE_PROBLEM,
        analyze_results=ANALYZE_RESULTS,
    )
    dyn_dir = spec.dynamics_dir
    manifest_file = pipeline_io.manifest_path(
        DATA_FOLDER, session_id, spec.trial_name)

    # --- Trial-scope QC pass, before any windows are cut. The QC pass needs
    # the contact model, which only exists after processInputsOpenSimAD runs,
    # so a trial-scope call must happen first; and it must run before
    # resolve_time_range/build_windows so the trimmed trailing frame is gone
    # before windows are cut from the (possibly shorter) QC'd kinematics. The
    # window argument here only selects `settings` (irrelevant at this call
    # site); its `timeInterval` is discarded in favor of the real windows
    # built below from the post-QC kinematics.
    raw_start, raw_end = pipeline_io.mot_time_range(spec.kinematics_mot)
    trial_settings = processInputsOpenSimAD(
        spec.baseDir, spec.dataFolder, spec.session_id, spec.trial_name,
        spec.motion_type, [raw_start, raw_end], spec.repetition,
        spec.treadmill_speed, spec.contact_side, use_local_data=True)
    run_qc_pass(spec.dataFolder, spec.session_id, spec.trial_name,
                OpenSimModel=trial_settings['OpenSimModel'])
    qc = verify_qc_marker(spec.dataFolder, spec.session_id, spec.trial_name)
    qc_marker = {'qc_version': qc['qc_version'],
                 'kinematics_sha256': qc['kinematics_sha256']}

    start, end = resolve_time_range(spec.kinematics_mot, START_TIME, END_TIME)
    windows = build_windows(start, end)
    print(f"\nTotal windows: {len(windows)}")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i}: [{ws:.2f}, {we:.2f}]  ({we - ws:.2f} s)")

    # --only-missing: skip windows already marked converged in a prior manifest.
    # A prior manifest's rows are only reusable when they were solved against
    # the SAME kinematics this run is about to use (qc_marker match): rows
    # solved against different kinematics cannot be merged into one CSV, so a
    # marker mismatch invalidates the whole prior manifest rather than being
    # merged window-by-window.
    previous = load_manifest(manifest_file) if args.only_missing else None
    skip_indices = set()
    if args.only_missing:
        if previous and previous.get("qc_marker") == qc_marker:
            skip_indices = {w["index"] for w in previous.get("windows", [])
                            if w.get("converged")}
            if skip_indices:
                print(f"\n--only-missing: skipping {len(skip_indices)} "
                      f"already-converged window(s): {sorted(skip_indices)}")
        elif previous:
            print("\n--only-missing: kinematics changed under the prior "
                  "manifest (qc_marker mismatch) - re-solving every window.")
            previous = None
        else:
            print("\n--only-missing: no prior manifest found - running all windows.")
    else:
        # Archive stale outputs before the first window runs.
        archive_existing_outputs(dyn_dir)

    run_list = [(i, w) for i, w in enumerate(windows) if i not in skip_indices]
    if not run_list:
        print("\nNothing to run (all windows already converged).")
        return

    # --- Serial prep pass, before any parallelism, over EVERY window in index
    # order (including ones --only-missing skips). This does on disk exactly
    # what a sequential run does before each window's solve:
    #   - the first processInputsOpenSimAD call builds the adjusted model, the
    #     contact model and the C++ external function (repo-global scratch
    #     paths: concurrent builds corrupt each other); later calls return
    #     early;
    #   - run_tracking(prepOnly=True) builds the muscle-tendon parameters, the
    #     trial's adjusted dummy motion and the polynomial data if a window
    #     needs them and they are missing, then returns before the problem is
    #     formulated.
    # A window's solve never writes these caches, so after this pass every
    # worker finds exactly what the sequential run would have found.
    print(f"\nPrep pass: building shared inputs serially for {len(windows)} "
          f"window(s) in index order ...")
    for i, (win_start, win_end) in enumerate(windows):
        settings = processInputsOpenSimAD(
            spec.baseDir, spec.dataFolder, spec.session_id, spec.trial_name,
            spec.motion_type, [win_start, win_end], spec.repetition,
            spec.treadmill_speed, spec.contact_side, use_local_data=True,
        )
        run_tracking(
            spec.baseDir, spec.dataFolder, spec.session_id, settings,
            case=pipeline_io.case_name(spec.trial_name, i),
            solveProblem=False, analyzeResults=False, prepOnly=True,
        )
    print("Prep pass complete - external function and model caches are built.")

    # --- Size the worker pool from available RAM (reserve configurable via
    # the GRF_RESERVE_MIB env var, ~2 GB/worker), then cap with --max-workers
    # if the operator gave one.
    available = psutil.virtual_memory().available
    cpu_count = os.cpu_count() or 1
    reserve_bytes = _reserve_bytes_from_env()
    workers = _effective_worker_count(
        available_bytes=available,
        cpu_count=cpu_count,
        num_windows=len(run_list),
        reserve_bytes=reserve_bytes,
        max_workers=args.max_workers,
    )
    cap_note = ("" if args.max_workers is None
                else f", capped at {args.max_workers} by --max-workers")
    print(f"\nAvailable RAM: {available / 1024**3:.1f} GB  ->  {workers} "
          f"parallel worker(s) (reserve {reserve_bytes / 1024**3:.2f} GB "
          f"[GRF_RESERVE_MIB], ~2 GB/worker, {cpu_count} CPUs, "
          f"{len(run_list)} windows to run{cap_note}).\n")

    tasks = [(spec, i, window) for i, window in run_list]
    results = run_pool(tasks, workers)

    # --- Rebuild the shared aggregate once, serially (race-free).
    merged = merge_optimaltrajectories(dyn_dir, spec.trial_name)
    if merged:
        print(f"Merged shared aggregate: {merged}")

    # Any window the pool never reported on is a failure, not a missing row.
    for i, (win_start, win_end) in run_list:
        results.setdefault(i, WindowResult(
            i, win_start, win_end, converged=False, return_status="unknown",
            failure_reason="worker never reported (pool failure?)"))

    session_root = pipeline_io.session_dir(DATA_FOLDER, session_id)
    rows = [manifest_row(results[i], session_root) for i, _ in run_list]
    print_summary(rows)

    # Under --only-missing, merge new rows with the prior manifest so previously
    # converged (skipped) windows are preserved.
    if args.only_missing and previous:
        by_idx = {w["index"]: w for w in previous.get("windows", [])}
        by_idx.update({r["index"]: r for r in rows})
        rows = [by_idx[k] for k in sorted(by_idx)]

    # Thresholds are copied out of trial_settings into the manifest (rather
    # than read from settingsOpenSimAD by 03) so that 03_build_grf_csv.py
    # stays free of OpenSim/CasADi imports.
    thresholds = {
        key: trial_settings.get(key, default)
        for key, default in (
            ("peak_vertical_grf_bw_zero_max", 0.02),
            ("mean_vertical_grf_bw_min", 0.20),
            ("frac_frames_within_5mm_min", 0.20),
            ("peak_vertical_grf_bw_max", 2.0),
            ("clearance_gate_m", 0.005),
            ("dynamics_residual_bw_fraction", 0.05),
        )
    }

    manifest = {
        "trial_name": spec.trial_name,
        "session_id": session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "windows": rows,
        "qc_marker": qc_marker,
        "mass_kg": trial_settings["mass_kg"],
        "thresholds": thresholds,
    }
    os.makedirs(dyn_dir, exist_ok=True)
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written: {manifest_file}")


if __name__ == "__main__":
    main()
