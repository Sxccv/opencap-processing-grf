"""Solve one time window of a trial with the OpenSimAD tracking problem.

This module is a library, not an entry point. `02_run_grf_simulation.py` owns
the orchestration: it sizes the worker pool to available RAM, builds the C++
external function once before any parallelism, and merges the shared trajectory
aggregate afterwards. A second orchestrator used to live here; it had none of
those safeguards, so running it reproduced exactly the races the pool was built
to avoid. It was removed — drive the pipeline through 02.
"""
import os
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_OPENSIM_AD_DIR = os.path.join(_HERE, "UtilsDynamicSimulations", "OpenSimAD")
for _p in (_HERE, _OPENSIM_AD_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

import pipeline_io
from pipeline_io import TrialSpec, WindowResult  # noqa: F401  (re-exported)
from utilsOpenSimAD import processInputsOpenSimAD
from mainOpenSimAD import run_tracking


def solve_window(spec, index, time_window):
    """Run the tracking problem for one window and report what came of it.

    Never raises: a window that blows up is one failed row in the manifest, not
    a dead pool. The traceback is printed because a five-to-fifteen minute solve
    is too expensive to debug from a one-line message.
    """
    case = pipeline_io.case_name(spec.trial_name, index)
    win_start, win_end = time_window
    print(f"Processing window {index}: [{win_start:.2f}, {win_end:.2f}] "
          f"with case: {case}")

    dyn_dir = spec.dynamics_dir
    started_at = time.time()

    try:
        settings = processInputsOpenSimAD(
            spec.baseDir, spec.dataFolder, spec.session_id, spec.trial_name,
            spec.motion_type, list(time_window), spec.repetition,
            spec.treadmill_speed, spec.contact_side, use_local_data=True,
        )
        run_tracking(
            spec.baseDir, spec.dataFolder, spec.session_id, settings,
            case=case,
            solveProblem=spec.solve_problem,
            analyzeResults=spec.analyze_results,
        )
    except Exception:
        print(f"FAIL [{case}]: exception during solve")
        traceback.print_exc()
        return WindowResult(index, win_start, win_end, converged=False,
                            return_status="exception",
                            failure_reason="exception during solve")

    return _inspect_outputs(spec, index, time_window, case, dyn_dir, started_at)


def _inspect_outputs(spec, index, time_window, case, dyn_dir, started_at):
    """Decide whether this window actually converged, from what it left on disk.

    run_tracking writes stats_<case>.npy whether or not IPOPT converged, so the
    stats file's `success` flag is the authority. The GRF file is then checked
    for existence *and* for being newer than this run, so a leftover file from a
    previous invocation cannot be mistaken for a fresh solve.
    """
    win_start, win_end = time_window
    stats_file = pipeline_io.stats_path(dyn_dir, case)
    grf_file = pipeline_io.grf_resultant_path(dyn_dir, spec.trial_name, case)

    traj_file = pipeline_io.trajectories_path(dyn_dir, case)
    if not os.path.exists(traj_file):
        print(f"Warning: optimal trajectories .npy not found for window "
              f"{index}. Expected at {traj_file}")
        traj_file = None

    def failed(reason, status="unknown"):
        print(f"FAIL [{case}]: {reason}")
        return WindowResult(index, win_start, win_end, converged=False,
                            return_status=status, trajectories_path=traj_file,
                            failure_reason=reason)

    if not os.path.exists(stats_file):
        return failed(f"no stats file — solver may not have run. "
                      f"Expected: {stats_file}")

    try:
        stats = np.load(stats_file, allow_pickle=True).item()
    except Exception as e:
        return failed(f"stats file unreadable: {e}")

    status = stats.get("return_status", "unknown")
    if not stats.get("success", False):
        return failed(f"IPOPT did not converge (return_status={status})", status)
    if not os.path.exists(grf_file):
        return failed(f"converged but no GRF file found. Expected: {grf_file}",
                      status)
    if os.path.getmtime(grf_file) < started_at:
        return failed("GRF file predates this run (stale file from a prior "
                      "invocation — archived or deleted)", status)

    return WindowResult(index, win_start, win_end, converged=True,
                        return_status=status, grf_path=grf_file,
                        trajectories_path=traj_file)


def solve_window_task(task):
    """Unpack a ``(spec, index, time_window)`` tuple.

    ``Pool.imap_unordered`` passes one argument, and results must stream back as
    they finish so a later worker death cannot discard the windows that already
    succeeded.
    """
    spec, index, time_window = task
    return solve_window(spec, index, time_window)
