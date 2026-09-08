"""
01_download_session.py — Download OpenCap session kinematics data.

Downloads marker data, IK results (.mot), model, and metadata for a single
trial. Skips videos and calibration images (lighter than download_session).

Prerequisites:
    - .env file with API_TOKEN (and optionally API_URL) exists in the repo
      root. Run createAuthenticationEnvFile.py once if you haven't already.

Output:
    - OpenCapData_<session_uuid>/ dir under dataFolder, ready for File 2.

Usage:
    python 01_download_session.py
"""

import os

# Edit these three values before running.

# Raw 36-char OpenCap session UUID (no "OpenCapData_" prefix).
session_uuid = "ab7eb7cf-817d-4035-a30b-ee68773906cb"

# Name of the trial to download (must match an OpenCap trial name).
trial_name = "Suhasno_1"

# Where to place the downloaded session folder.
# Default: the directory this script is in.
dataFolder = os.path.dirname(os.path.abspath(__file__))


def main():
    # Build the prefixed session-id folder name that downstream scripts expect.
    # download_kinematics writes directly into its `folder` argument without
    # adding any prefix, so we pass the already-prefixed path as `folder`.
    session_id = "OpenCapData_" + session_uuid
    session_folder = os.path.join(dataFolder, session_id)

    # Idempotency check: skip download if the trial .mot already exists.
    pathTrial = os.path.join(
        session_folder, "OpenSimData", "Kinematics", f"{trial_name}.mot"
    )

    if os.path.exists(pathTrial):
        print(f"Trial data already exists at: {pathTrial}")
        print("Skipping download. Delete the session folder to force re-download.")
        return

    # Late imports so config errors surface before touching utils / API.
    from utils import download_kinematics

    print(f"Downloading session {session_uuid} -> {session_folder}")
    print(f"Trial filter: {trial_name}")

    download_kinematics(
        session_uuid,
        folder=session_folder,
        trialNames=[trial_name],
        use_local_data=False,
    )

    # Confirm the download.
    if os.path.exists(pathTrial):
        print(f"Download complete: {pathTrial}")
    else:
        print(
            "WARNING: Trial .mot not found after download. "
            "Check that the trial name matches the OpenCap session exactly."
        )


if __name__ == "__main__":
    main()
