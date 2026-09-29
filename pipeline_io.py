"""Where the windowed-GRF pipeline's files live, and how to read them.

Every path the pipeline builds is constructed here, so `02_run_grf_simulation.py`
(which writes them), `03_build_grf_csv.py` (which reads them) and
`grf_prediction.py` (which checks them) cannot drift on the layout.

Deliberately not part of ``utils.py`` as that module imports ``opensim`` and the
OpenCap API client, and nothing in this pipeline's post-processing should need
either just to read a time column.
"""
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# %% Windowing.

def build_windows(start_time, end_time, step=1.0, min_last_duration=0.5):
    """Build 1-second sliding windows, merging a trailing window < 0.5 s.

    Shared by `02_run_grf_simulation.py` and `grf_prediction_linear.py` so the
    parallel pipeline and its sequential reference solve exactly the same
    intervals. Returns a list of [start, end] pairs.
    """
    starts = np.arange(start_time, end_time, step)
    windows = [[float(s), min(float(s) + step, end_time)] for s in starts]

    if len(windows) > 1:
        last_dur = windows[-1][1] - windows[-1][0]
        if 0 < last_dur < min_last_duration:
            windows[-2][1] = windows[-1][1]
            windows.pop()

    return windows


# %% Session folder layout.

def session_dir(dataFolder, session_id):
    """The ``OpenCapData_<uuid>`` folder itself."""
    return os.path.join(dataFolder, session_id)


def dynamics_dir(dataFolder, session_id, trial_name):
    """Where run_tracking writes a trial's GRF, stats and trajectory files."""
    return os.path.join(session_dir(dataFolder, session_id),
                        "OpenSimData", "Dynamics", trial_name)


def kinematics_mot(dataFolder, session_id, trial_name):
    """The trial's IK output — the pipeline's input, and its time range."""
    return os.path.join(session_dir(dataFolder, session_id),
                        "OpenSimData", "Kinematics", f"{trial_name}.mot")


def markers_trc(dataFolder, session_id, trial_name):
    """The trial's marker data — the QC pass's shared-time-base check."""
    return os.path.join(session_dir(dataFolder, session_id),
                        "MarkerData", f"{trial_name}.trc")


def kinematics_raw_mot(dataFolder, session_id, trial_name):
    """The pre-QC kinematics, preserved by the QC pass before it rewrites."""
    return os.path.join(session_dir(dataFolder, session_id),
                        "OpenSimData", "Kinematics", f"{trial_name}_raw.mot")


def qc_sidecar_path(dataFolder, session_id, trial_name):
    """The QC pass's sidecar, read back by `verify_qc_marker`."""
    return os.path.join(session_dir(dataFolder, session_id),
                        "OpenSimData", "Kinematics", f"{trial_name}_qc.json")


def kinematics_activations_path(dyn_dir, trial_name, case):
    return os.path.join(dyn_dir, f"kinematics_activations_{trial_name}_{case}.mot")


def manifest_path(dataFolder, session_id, trial_name):
    """The window manifest 02 writes and 03 consumes."""
    return os.path.join(dynamics_dir(dataFolder, session_id, trial_name),
                        f"window_manifest_{trial_name}.json")


def case_name(trial_name, window_index):
    """The ``case`` string identifying one window's outputs on disk."""
    return f"{trial_name}_window_{window_index}"


def grf_resultant_path(dyn_dir, trial_name, case):
    return os.path.join(dyn_dir, f"GRF_resultant_{trial_name}_{case}.mot")


def stats_path(dyn_dir, case):
    return os.path.join(dyn_dir, f"stats_{case}.npy")


def trajectories_path(dyn_dir, case):
    return os.path.join(dyn_dir, f"optimaltrajectories_{case}.npy")


# %% Reading .mot files.

def read_mot(path):
    """Read a ``.mot`` file into a DataFrame, dropping all-empty columns.

    Everything up to and including the ``endheader`` line is skipped; the line
    after it is the tab-separated column header.
    """
    with open(path, "r") as f:
        header_end = 0
        for i, line in enumerate(f):
            if "endheader" in line:
                header_end = i
                break

    return pd.read_csv(path, sep="\t", skiprows=header_end + 1).dropna(
        axis=1, how="all"
    )


def mot_time_range(path):
    """Return ``(first_time, last_time)`` from a ``.mot`` file.

    Returns ``(None, None)`` when the file has no usable time column, which the
    callers treat as "time range could not be determined".
    """
    try:
        times = read_mot(path)["time"]
    except (OSError, KeyError, ValueError, pd.errors.ParserError):
        return None, None
    if times.empty:
        return None, None
    return float(times.min()), float(times.max())


# %% The records that travel between the orchestrator and its workers.

@dataclass(frozen=True)
class TrialSpec:
    """Everything about a solve that is the same for every window of a trial.

    Frozen and picklable so one instance can be handed to every pool worker
    instead of re-listing thirteen positional arguments per window.
    """
    baseDir: str
    dataFolder: str
    session_id: str
    trial_name: str
    motion_type: str
    repetition: Optional[int] = None
    treadmill_speed: float = 0
    contact_side: str = "all"
    solve_problem: bool = True
    analyze_results: bool = True

    @property
    def dynamics_dir(self):
        return dynamics_dir(self.dataFolder, self.session_id, self.trial_name)

    @property
    def kinematics_mot(self):
        return kinematics_mot(self.dataFolder, self.session_id, self.trial_name)


@dataclass(frozen=True)
class WindowResult:
    """The verdict on one window, decided once, by the worker that ran it.

    The orchestrator writes these straight into the manifest. It deliberately
    does not re-derive `converged` from the stats file: that check needs the
    time the solve started, to tell a fresh GRF file from a leftover one, and
    only the worker knows that.
    """
    index: int
    time_start: float
    time_end: float
    converged: bool
    return_status: str
    grf_path: Optional[str] = None
    trajectories_path: Optional[str] = None
    failure_reason: Optional[str] = None
    mean_vertical_grf: Optional[float] = None
    peak_vertical_grf: Optional[float] = None
    min_clearance_m: Optional[float] = None
    frac_frames_within_5mm: Optional[float] = None
    dedrift_method: Optional[str] = None
    dynamics_consistency_residual_N: Optional[float] = None
    flip_count: Optional[int] = None
