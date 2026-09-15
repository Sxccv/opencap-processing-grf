# OpenCap Processing

This repository enables the post-processing of human movement kinematics collected using [OpenCap](opencap.ai). You can run kinematic analyses, download multiple sessions using scripting, and run muscle-driven simulations to estimate kinetics.

## Publication
More information is available in our [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462): <br> <br>
Uhlrich SD*, Falisse A*, Kidzinski L*, Ko M, Chaudhari AS, Hicks JL, Delp SL, 2022. OpenCap: Human movement dynamics from smartphone videos. PLoS Comput Biol 19(10): e1011462. https://doi.org/10.1371/journal.pcbi.1011462. *contributed equally <br> <br>
Archived code base corresponding to publication: https://zenodo.org/record/7419973

## Install requirements
### General
1. Install [Anaconda](https://www.anaconda.com/)
1. Open Anaconda prompt
2. Create environment (python 3.11 recommended): `conda create -n opencap-processing python=3.11`
3. Activate environment: `conda activate opencap-processing`
4. Install OpenSim: `conda install -c opensim-org opensim=4.5=py311np123`
    - Test that OpenSim was successfully installed:
        - Start python: `python`
        - Import OpenSim: `import opensim`
            - If you don't get any error message at this point, you should be good to go.
        - You can also double check which version you installed : `opensim.GetVersion()`
        - Exit python: `quit()`
    - Visit this [webpage](https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53116061/Conda+Package) for more details about the OpenSim conda package.
5. (Optional): Install an IDE such as Spyder: `conda install spyder`
6. Clone the repository to your machine: 
    - Navigate to the directory where you want to download the code: eg. `cd Documents`. Make sure there are no spaces in this path.
    - Clone the repository: `git clone https://github.com/stanfordnmbl/opencap-processing.git`
    - Navigate to the directory: `cd opencap-processing`
7. Install required packages: `python -m pip install -r requirements.txt`
8. Run `python createAuthenticationEnvFile.py`
    - An environment variable (`.env` file) will be saved after authenticating.    
    
### Muscle-driven simulations
1. **Windows only**: Install [Visual Studio](https://visualstudio.microsoft.com/downloads/)
    - The Community variant is sufficient and is free for everyone.
    - During the installation, select the *workload Desktop Development with C++*.
    - The code was tested with the 2017, 2019, and 2022 Community editions.
2. **Linux only**: Install OpenBLAS libraries
    - `sudo apt-get install libopenblas-base`
3. **MacOs**: As of now this workflow is supported only on x86_64 architecture. If you intend to use this workflow use x86_64 conda environment even on apple silicon (arm64 is not supported yet). 
    
## Examples
- Run `example.py` for examples of how to run kinematic analyses
- Run `example_kinetics.py` for examples of how to generate muscle-driven simulations
- Moco
    - The [Moco folder](https://github.com/stanfordnmbl/opencap-processing/tree/main/Moco) contains examples for generating muscle-driven simulations using [OpenSim Moco](https://opensim-org.github.io/opensim-moco-site/). 

## Download OpenCap data

### Using Colab
- Open `batchDownload.ipynb` in Colab and follow the instructions
    - You do not need to follow the install requirements above.

### Locally
- Follow the install requirements above
- Open `batchDownload.py` and follow the instructions

# Windowed-GRF Solving

Estimating ground reaction forces for a whole trial in one optimal control problem is slow and fragile: a single hard second can stop the solve and leave you with nothing. This workflow splits a trial's kinematics into 1-second windows, solves each one independently in a worker pool sized to available RAM, and records what converged. A window that fails costs you that window, not the run.

Three scripts run in order. Each is independently re-runnable.

## Prerequisites

- The full install above, including the **Muscle-driven simulations** section — this workflow builds and runs the OpenSimAD external function.
- `python -m pip install -r requirements.txt` (adds `psutil` for RAM sizing and `pytest`).
- `.env` with `API_TOKEN` present. Run `python createAuthenticationEnvFile.py` once if you have not.

## 1. Download the session

`01_download_session.py` has no command-line flags. Edit the three values at the top of the file:

```python
session_uuid = "ab7eb7cf-817d-4035-a30b-ee68773906cb"   # raw 36-char UUID, no "OpenCapData_" prefix
trial_name   = "Suhasno_1"
dataFolder   = os.path.dirname(os.path.abspath(__file__))
```

```bash
python 01_download_session.py
```

Downloads marker data, IK results, the model and metadata into `OpenCapData_<uuid>/`. Skips videos and calibration images, so it is much lighter than a full `download_session`. It is idempotent: if the trial `.mot` already exists it prints a message and exits without downloading. Delete the session folder to force a re-download.

## 2. Solve the windows

```bash
python 02_run_grf_simulation.py
```

A bare invocation uses the defaults in the config block at the top of the file. Every one of them has a flag:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--session-uuid` | Raw 36-char UUID, no `OpenCapData_` prefix | `ab7eb7cf-…` |
| `--trial-name` | Trial (kinematics `.mot` stem) to simulate | `Suhasno_1` |
| `--motion-type` | `walking` / `running` / `squats` / `sit_to_stand` | `walking` |
| `--contact-side` | `all`, `left` or `right` | `all` |
| `--treadmill-speed` | m/s; `0` means overground | `0` |
| `--repetition` | Not supported by the windowed pipeline; passing it exits with an error | none |
| `--only-missing` | Re-run only windows that did not converge last time | off |

```bash
python 02_run_grf_simulation.py --trial-name Suhasno_2 --motion-type running
```

**What it does, in order.** It reads the trial's time range from the kinematics `.mot` and cuts it into 1-second windows, merging a trailing window shorter than 0.5 s into the one before it. Unless `--only-missing` is set, it moves any pre-existing output into `_archive_<timestamp>/` so a fresh run cannot be confused by stale files. It then runs a **serial prep pass**: for every window, in index order, it calls `processInputsOpenSimAD` and `run_tracking(..., prepOnly=True)`. The first call builds the C++ external function (`buildExternalFunction` writes to repo-global scratch paths, so concurrent first builds corrupt each other). The pass as a whole builds the muscle-tendon parameter, dummy-motion and polynomial caches in the session `Model/` folder in the same order a sequential run (`grf_prediction_linear.py`) would. Only then does it start the pool.

**How many windows run at once.** `available_RAM − 1 GB reserve`, divided by 2 GB per worker, capped by the CPU count and by the number of windows, and never below 1. An IPOPT solve for one window plateaus around 1.7–2.0 GB. The chosen worker count is printed before the pool starts. `OMP_NUM_THREADS=1` is set in the parent process, so each solve stays single-threaded rather than every worker spawning a thread per core.

Which cache a window creates depends on that window's own range of motion. If the workers built them first-come-first-served, a different window could own a cache than in a sequential run, so the prep pass builds them all before any worker starts; workers only load them. The cross-process file lock (`UtilsDynamicSimulations/OpenSimAD/sharedPrepLockOpenSimAD.py`) stays around those regions as a safety net. It also serialises the joint-reaction analyses (`computeKAM` / `computeMCF`), which write scratch files into the `Dynamics/<trial>/` folder that every window shares.

**Runtime: 5–15 minutes per window.** A 7-second trial is 7 windows.

**Outputs**, under `OpenCapData_<uuid>/OpenSimData/Dynamics/<trial_name>/`:

- `GRF_resultant_<trial>_<trial>_window_<i>.mot` — one per converged window
- `stats_<trial>_window_<i>.npy` — IPOPT statistics, written whether or not the solve converged
- `optimaltrajectories_<trial>_window_<i>.npy` — per-window trajectories
- `optimaltrajectories.npy` — the aggregate, merged serially after the pool
- `window_manifest_<trial>.json` — what converged, the IPOPT return status, and a `failure_reason` for anything that did not

A window counts as converged only if its stats file says `success`, the GRF file exists, **and** that file is newer than the run that claimed it.

**Resuming.** If some windows failed, fix what caused it and re-run with `--only-missing`. Converged windows are skipped with their manifest rows preserved and nothing archived.

```bash
python 02_run_grf_simulation.py --only-missing
```

**Solving a single window** — useful for a first smoke test as a full run takes a lot of time.

```python
START_TIME = 0.0
END_TIME   = 1.0
```

### 3. Build the CSV

```bash
python 03_build_grf_csv.py
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--session-uuid` | Must match what `02` used | `ab7eb7cf-…` |
| `--trial-name` | Must match what `02` used | `Suhasno_1` |
| `--no-plots` | Skip the 2×3 matplotlib plot window | show plots |

Reads the manifest and takes only converged windows and concatenates them sorted by time with duplicate timestamps at window boundaries dropped written to `dataFolder`:

- `grf_df_<trial>_<session_id>.csv`
- `grf_df_<trial>_<session_id>_coverage.json` — which time ranges have GRF and which are gaps

It warns about any window whose force components are all near zero, which usually means a degenerate no-contact solution, and any manifest entry file that has since gone missing. Coverage is reported against the full kinematics range.

```
Trial Suhasno_1: kinematics [0.00, 7.30] s
GRF coverage: [0.00, 6.00] s  (6.00s / 7.30s = 82.2%)
GAP: [6.00, 7.30] s
```

## Supporting modules

| File | Role |
| --- | --- |
| `pipeline_io.py` | Session folder layout, `.mot` reading, and the `TrialSpec` / `WindowResult` records passed between the orchestrator and its workers |
| `parallel_config.py` | Worker-count arithmetic and the serial merge of the trajectory aggregate |
| `grf_prediction.py` | `solve_window` — runs one window and decides whether it converged. A library, not an entry point |
| `UtilsDynamicSimulations/OpenSimAD/sharedPrepLockOpenSimAD.py` | The cross-process lock guarding `run_tracking`'s one-time model caches |
