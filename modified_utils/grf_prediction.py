# %% Directories, paths, and imports. Nothing should need to be changed.
import os
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
import argparse

# ---- Path bootstrap: this copy lives in <repo>/modified_utils/, directly
# under the repo root. Derive paths from this file (not the CWD) so the
# script works from any working directory. This dir is kept first so the
# modified copies of mainOpenSimAD.py / utilsOpenSimAD.py win; the repo root
# (utils.py, utilsProcessing.py) and the upstream OpenSimAD dir
# (settingsOpenSimAD.py & friends) are added for everything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
baseDir = os.path.dirname(_HERE)
opensimADDir = os.path.join(baseDir, "UtilsDynamicSimulations", "OpenSimAD")
for _p in (baseDir, opensimADDir):
    if _p not in sys.path:
        sys.path.append(_p)
if _HERE in sys.path:
    sys.path.remove(_HERE)
sys.path.insert(0, _HERE)

from utilsOpenSimAD import processInputsOpenSimAD, plotResultsOpenSimAD
from mainOpenSimAD import run_tracking


def get_mot_time_range(mot_file_path):
    with open(mot_file_path, "r") as f:
        # Skip header
        for line in f:
            if "endheader" in line:
                break
        # Read the rest as data
        times = []
        for line in f:
            if line.strip() == "":
                continue
            time_str = line.split()[0]
            try:
                times.append(float(time_str))
            except ValueError:
                continue
        if times:
            return min(times), max(times)
        else:
            return None, None


# List storing paths of GRF files
grf_file_paths = []


def process_single_window(
    baseDir,
    dataFolder,
    session_id,
    trial_name,
    motion_type,
    current_time_window,
    repetition,
    treadmill_speed,
    contact_side,
    solveProblem,
    analyzeResults,
    case_prefix,
    window_index,
):

    current_case = f"{case_prefix}_window_{window_index}"
    print(f"Processing window: {current_time_window} with case: {current_case}")

    # Rnvironment variable to prevent nested parallelism issues with joblib/loky
    os.environ["OMP_NUM_THREADS"] = "1"

    run_started_at = time.time()
    grf_file_path = None
    optimaltrajectories_file_path = None

    try:
        # %% Setup.
        settings = processInputsOpenSimAD(
            baseDir,
            dataFolder,
            session_id,
            trial_name,
            motion_type,
            current_time_window,
            repetition,
            treadmill_speed,
            contact_side,
            use_local_data=True,
        )

        # %% Simulation.
        run_tracking(
            baseDir,
            dataFolder,
            session_id,
            settings,
            case=current_case,
            solveProblem=solveProblem,
            analyzeResults=analyzeResults,
        )

        # Convergence check require stats file + success flag + fresh GRF file.
        # run_tracking writes stats_<case>.npy regardless of convergence.
        # Check stats['success'] to prevent stale GRF resultant files being 
        # mistakenly considered as successful solves
        dyn_dir = os.path.join(
            dataFolder, session_id, "OpenSimData", "Dynamics", trial_name
        )
        stats_path = os.path.join(dyn_dir, f"stats_{current_case}.npy")
        potential_grf_path = os.path.join(
            dyn_dir, f"GRF_resultant_{trial_name}_{current_case}.mot"
        )

        converged = False
        return_status = "unknown"

        if not os.path.exists(stats_path):
            print(
                f"FAIL [{current_case}]: no stats file — solver may not have run. "
                f"Expected: {stats_path}"
            )
        else:
            stats = np.load(stats_path, allow_pickle=True).item()
            return_status = stats.get("return_status", "unknown")
            if not stats.get("success", False):
                print(
                    f"FAIL [{current_case}]: IPOPT did not converge "
                    f"(return_status={return_status})"
                )
            elif not os.path.exists(potential_grf_path):
                print(
                    f"FAIL [{current_case}]: converged but no GRF file found. "
                    f"Expected: {potential_grf_path}"
                )
            elif os.path.getmtime(potential_grf_path) < run_started_at:
                print(
                    f"FAIL [{current_case}]: GRF file predates this run "
                    f"(stale file from prior invocation — archived or deleted)."
                )
            else:
                converged = True

        if converged:
            grf_file_path = potential_grf_path

        optimaltrajectories_file_path = os.path.join(
            dyn_dir, f"optimaltrajectories_{current_case}.npy"
        )
        if not os.path.exists(optimaltrajectories_file_path):
            print(
                f"Warning: Optimal trajectories NPY file not found for "
                f"window {current_time_window}. Expected at {optimaltrajectories_file_path}"
            )
            optimaltrajectories_file_path = None

    except Exception as e:
        print(
            f"Error processing window {current_time_window} for trial {trial_name}: {e}"
        )
        grf_file_path = None
        optimaltrajectories_file_path = None

    return grf_file_path, optimaltrajectories_file_path


def main():
    parser = argparse.ArgumentParser(
        description="Process OpenCap motion data in windows."
    )
    parser.add_argument(
        "--session_id", type=str, required=True, help="The ID of the OpenCap session."
    )
    parser.add_argument(
        "--trial_name",
        type=str,
        required=True,
        help="The name of the trial to simulate.",
    )
    parser.add_argument(
        "--start_time", type=float, help="Start time of the analysis window."
    )
    parser.add_argument(
        "--end_time", type=float, help="End time of the analysis window."
    )

    args = parser.parse_args()

    # User Inputs
    session_id = args.session_id
    trial_name = args.trial_name
    session_type = "overground"  # Options are 'overground' and 'treadmill'.
    motion_type = "walking"
    if not "repetition" in locals():
        repetition = None
    if not "treadmill_speed" in locals():
        treadmill_speed = 0
    if not "contact_side" in locals():
        contact_side = "all"
    # Set to True to solve the optimal control problem.
    solveProblem = True
    # Set to True to analyze the results of the optimal control problem. If the problem
    # was already solved and only the analyzing/processing of the results is needed,
    # set solveProblem to False and run this script with analyzeResults set to True.
    # Mainly for additional post-processing.
    analyzeResults = True

    # Path to where the data will be downloaded.
    dataFolder = baseDir  # Set dataFolder to the base directory

    sessionFolder = os.path.join(dataFolder, session_id)
    pathTrial = os.path.join(
        sessionFolder, "OpenSimData", "Kinematics", trial_name + ".mot"
    )

    # Use provided start_time and end_time, or determine from mot file if not provided
    if args.start_time is not None:
        start_time = args.start_time
    else:
        start_time, _ = get_mot_time_range(pathTrial)

    if args.end_time is not None:
        end_time = args.end_time
    else:
        _, end_time = get_mot_time_range(pathTrial)

    if start_time is None or end_time is None:
        print(
            "Error: Could not determine time range for analysis. Please provide --start_time and --end_time or ensure the .mot file exists and is valid."
        )
        return

    starts = np.arange(start_time, end_time, 1.0)
    windows = [[s, min(s + 1.0, end_time)] for s in starts]

    if len(windows) > 1:
        last_dur = windows[-1][1] - windows[-1][0]
        if 0 < last_dur < 0.5:
            windows[-2][1] = windows[-1][1]
            windows.pop()

    # Process windows in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(
            process_single_window,
            [
                (
                    baseDir,
                    dataFolder,
                    session_id,
                    trial_name,
                    motion_type,
                    [win_start, win_end],
                    repetition,
                    treadmill_speed,
                    contact_side,
                    solveProblem,
                    analyzeResults,
                    trial_name,
                    i,
                )
                for i, (win_start, win_end) in enumerate(windows)
            ],
        )

    grf_file_paths = [r[0] for r in results if r[0] is not None]

    # %% Summary Report
    num_total_windows = len(windows)
    num_successful_windows = len(grf_file_paths)
    num_failed_windows = num_total_windows - num_successful_windows

    print("\n--- Processing Summary ---")
    print(f"Total windows attempted: {num_total_windows}")
    print(f"Successfully processed windows: {num_successful_windows}")
    print(f"Failed windows: {num_failed_windows}")
    print("------------------------")


if __name__ == "__main__":
    main()
