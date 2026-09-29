"""Solve one time window of a trial with the OpenSimAD tracking problem.

This module is a library, not an entry point. `02_run_grf_simulation.py` owns
the orchestration: it sizes the worker pool to available RAM, builds the C++
external function once before any parallelism, and merges the shared trajectory
aggregate afterwards. A second orchestrator used to live here; it had none of
those safeguards, so running it reproduced exactly the races the pool was built
to avoid. It was removed — drive the pipeline through 02.
"""
import json
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
import dynamicsConsistency


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

    return _inspect_outputs(spec, index, time_window, case, dyn_dir, started_at,
                            settings)


def _inspect_outputs(spec, index, time_window, case, dyn_dir, started_at,
                     settings):
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

    def failed(reason, status="unknown", **metrics):
        print(f"FAIL [{case}]: {reason}")
        return WindowResult(index, win_start, win_end, converged=False,
                            return_status=status, trajectories_path=traj_file,
                            failure_reason=reason, **metrics)

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

    # Metrics below are all "nice to have" for the manifest: a failure to
    # compute any of them must never turn a genuinely converged window into a
    # failure, so each block is independently wrapped and just prints a
    # traceback on error, leaving that metric (and only that metric) at None.
    metrics = {}

    try:
        grf_df = pipeline_io.read_mot(grf_file)
        vy = (grf_df["ground_force_right_vy"] + grf_df["ground_force_left_vy"])
        metrics["mean_vertical_grf"] = float(vy.mean())
        metrics["peak_vertical_grf"] = float(vy.abs().max())
    except Exception:
        print(f"Warning [{case}]: failed to compute GRF metrics")
        traceback.print_exc()

    try:
        qc_path = pipeline_io.qc_sidecar_path(
            spec.dataFolder, spec.session_id, spec.trial_name)
        with open(qc_path, "r") as f:
            qc = json.load(f)
        per_window = next(
            (w for w in qc.get("per_window", []) if w.get("index") == index),
            None)
        if per_window is not None:
            metrics["min_clearance_m"] = per_window.get("min_clearance_m")
            metrics["frac_frames_within_5mm"] = per_window.get(
                "frac_frames_within_5mm")
        metrics["dedrift_method"] = qc.get("dedrift", {}).get("method")
        metrics["flip_count"] = len(
            qc.get("flip_detection", {}).get("multi_coordinate", {})
            .get("frame_indices", []))
    except Exception:
        print(f"Warning [{case}]: failed to compute QC sidecar metrics")
        traceback.print_exc()

    residual_N = None
    body_weight_N = None
    if traj_file is not None:
        # A missing/unreadable trajectories file leaves the residual None and
        # skips the F2 gate below rather than failing the window.
        try:
            model_name = settings["OpenSimModel"]
            model_path = os.path.join(
                pipeline_io.session_dir(spec.dataFolder, spec.session_id),
                "OpenSimData", "Model",
                f"{model_name}_scaled_adjusted_contacts.osim")
            dyn_result = dynamicsConsistency.window_mean_residual(
                model_path, traj_file, case)
            residual_N = dyn_result["residual_N"]
            body_weight_N = dyn_result["body_weight_N"]
            metrics["dynamics_consistency_residual_N"] = residual_N
        except Exception:
            print(f"Warning [{case}]: failed to compute dynamics consistency "
                  f"residual")
            traceback.print_exc()

    # F2 gate: the pelvis is a floating base with no reserve actuators, so
    # a window whose exported GRF cannot even balance its own exported
    # kinematics (sum(GRF) far from m*(a_com+g)) is not physically usable no
    # matter what IPOPT's return_status says. On the 2026-09-15 run,
    # physically grounded windows measured 0.1-0.8% of body weight while
    # ungrounded ones measured 12-62%, so 5% sits between the two
    # populations with roughly 6x margin on each side. This only fires on a
    # residual actually computed above -- never on a missing one.
    if residual_N is not None and body_weight_N is not None:
        threshold_frac = settings.get("dynamics_residual_bw_fraction", 0.05)
        threshold_N = threshold_frac * body_weight_N
        if abs(residual_N) > threshold_N:
            return failed(
                f"dynamics consistency residual {residual_N:.1f} N exceeds "
                f"{threshold_frac * 100:.1f}% of body weight "
                f"({threshold_N:.1f} N) -- overriding IPOPT "
                f"return_status={status}",
                status, **metrics)

    return WindowResult(index, win_start, win_end, converged=True,
                        return_status=status, grf_path=grf_file,
                        trajectories_path=traj_file, **metrics)


def solve_window_task(task):
    """Unpack a ``(spec, index, time_window)`` tuple.

    ``Pool.imap_unordered`` passes one argument, and results must stream back as
    they finish so a later worker death cannot discard the windows that already
    succeeded.
    """
    spec, index, time_window = task
    return solve_window(spec, index, time_window)
