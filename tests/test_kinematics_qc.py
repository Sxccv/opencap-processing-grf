"""Tests for the OpenSim-dependent kinematics QC pass and the bounds hash.

Everything here needs OpenSim (and, for the `_hash_bounds` case, CasADi via
`mainOpenSimAD`), so the whole module is skipped with `pytest.importorskip`
when those are unavailable -- this keeps the fast tier (test_windowed_grf.py)
free of any OpenSim import. Run with the opensim-env interpreter:

    opensim-env/python.exe -m pytest tests/test_kinematics_qc.py -q

Everything below uses `tmp_path` and small synthetic fixtures. The real
440-frame QC pass and the real session folder are never touched.
"""
import json
import os
import shutil
import sys

import pytest

opensim = pytest.importorskip("opensim")
pytest.importorskip("casadi")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "UtilsDynamicSimulations", "OpenSimAD"))

import pipeline_io
from kinematicsQC import (KinematicsQCError, QC_VERSION, _sha256_file,
                          run_qc_pass, verify_qc_marker)
from mainOpenSimAD import _hash_bounds

SESSION_ID = "OpenCapData_x"
TRIAL_NAME = "T"
OPENSIM_MODEL = "LaiUhlrich2022"


def _make_session_dirs(tmp_path):
    """Bare session skeleton: Kinematics/, Model/ and MarkerData/ folders."""
    dataFolder = str(tmp_path)
    session_root = pipeline_io.session_dir(dataFolder, SESSION_ID)
    os.makedirs(os.path.join(session_root, "OpenSimData", "Kinematics"))
    os.makedirs(os.path.join(session_root, "OpenSimData", "Model"))
    os.makedirs(os.path.join(session_root, "MarkerData"))
    return dataFolder


def _write_mot(path, n_rows):
    with open(path, "w") as f:
        f.write("Coordinates\nversion=1\n")
        f.write(f"nRows={n_rows}\nnColumns=2\ninDegrees=yes\nendheader\n")
        f.write("time\tpelvis_ty\n")
        for i in range(n_rows):
            f.write(f"{i * 0.01}\t0.9\n")


def _write_trc(path, n_rows):
    with open(path, "w") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tT.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
               "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"100.0\t100.0\t{n_rows}\t1\tmm\t100.0\t1\t{n_rows}\n")
        f.write("Frame#\tTime\tM1\t\t\n")
        f.write("\t\tX1\tY1\tZ1\n")
        f.write("\n")
        for i in range(n_rows):
            f.write(f"{i + 1}\t{i * 0.01}\t0.0\t0.0\t0.0\n")


def _build_contact_model(model_path):
    """A minimal-but-valid .osim with one SmoothSphereHalfSpaceForce contact.

    Just enough for `run_qc_pass`'s Step 2 (contact model load + sphere
    discovery) to succeed, so the shared-time-base check downstream (Step 4)
    is actually reachable without a real musculoskeletal model.
    """
    model = opensim.Model()
    model.setName("test")
    body = opensim.Body("body1", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    model.addBody(body)
    joint = opensim.FreeJoint("joint1", model.getGround(), opensim.Vec3(0),
                              opensim.Vec3(0), body, opensim.Vec3(0), opensim.Vec3(0))
    model.addJoint(joint)
    half_space = opensim.ContactHalfSpace(
        opensim.Vec3(0), opensim.Vec3(0, 0, -1.5707963267948966),
        model.getGround(), "ground_contact")
    model.addContactGeometry(half_space)
    sphere = opensim.ContactSphere(0.02, opensim.Vec3(0, 0, 0), body, "contact_sphere")
    model.addContactGeometry(sphere)
    force = opensim.SmoothSphereHalfSpaceForce("contact_force", sphere, half_space)
    model.addForce(force)
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(model_path)


# %% verify_qc_marker

class TestVerifyQcMarker:
    def test_raises_when_sidecar_is_missing(self, tmp_path):
        dataFolder = _make_session_dirs(tmp_path)
        _write_mot(pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME), 5)

        with pytest.raises(KinematicsQCError, match="no QC sidecar"):
            verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME)

    def test_raises_when_sha256_does_not_match(self, tmp_path):
        dataFolder = _make_session_dirs(tmp_path)
        _write_mot(pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME), 5)
        sidecar_path = pipeline_io.qc_sidecar_path(dataFolder, SESSION_ID, TRIAL_NAME)
        with open(sidecar_path, "w") as f:
            json.dump({"qc_version": QC_VERSION,
                      "kinematics_sha256": "0" * 64}, f)

        with pytest.raises(KinematicsQCError, match="SHA-256 mismatch"):
            verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME)

    def test_returns_sidecar_when_version_and_hash_match(self, tmp_path):
        dataFolder = _make_session_dirs(tmp_path)
        mot_path = pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME)
        _write_mot(mot_path, 5)
        actual_hash = _sha256_file(mot_path)
        sidecar = {"qc_version": QC_VERSION, "kinematics_sha256": actual_hash,
                  "trial_name": TRIAL_NAME}
        with open(pipeline_io.qc_sidecar_path(dataFolder, SESSION_ID, TRIAL_NAME), "w") as f:
            json.dump(sidecar, f)

        result = verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME)
        assert result == sidecar


# %% run_qc_pass

class TestRunQcPass:
    def test_raises_a_clear_message_when_contact_model_is_absent(self, tmp_path):
        dataFolder = _make_session_dirs(tmp_path)
        _write_mot(pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME), 5)

        with pytest.raises(KinematicsQCError, match="contact model not found"):
            run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME, OpenSimModel=OPENSIM_MODEL)

    def test_raises_when_kinematics_and_markers_have_different_frame_counts(
            self, tmp_path):
        """Shared-time-base check (Step 4): a synthetic-but-valid contact
        model is enough to reach it without a real musculoskeletal model."""
        dataFolder = _make_session_dirs(tmp_path)
        model_path = os.path.join(
            pipeline_io.session_dir(dataFolder, SESSION_ID), "OpenSimData",
            "Model", f"{OPENSIM_MODEL}_scaled_adjusted_contacts.osim")
        _build_contact_model(model_path)

        _write_mot(pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME), 5)
        _write_trc(pipeline_io.markers_trc(dataFolder, SESSION_ID, TRIAL_NAME), 4)

        with pytest.raises(KinematicsQCError, match="frame count mismatch"):
            run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME, OpenSimModel=OPENSIM_MODEL)


# %% mainOpenSimAD._hash_bounds

class TestHashBounds:
    def test_same_key_regardless_of_dict_insertion_order(self):
        a = {"hip_flexion_r": {"lower": -10.0, "upper": 10.0},
            "knee_angle_r": {"lower": -5.0, "upper": 90.0}}
        # Same content, different insertion order at both levels.
        b = {"knee_angle_r": {"upper": 90.0, "lower": -5.0},
            "hip_flexion_r": {"upper": 10.0, "lower": -10.0}}
        assert _hash_bounds(a) == _hash_bounds(b)

    def test_different_key_when_a_value_changes(self):
        a = {"hip_flexion_r": {"lower": -10.0, "upper": 10.0}}
        b = {"hip_flexion_r": {"lower": -10.0, "upper": 10.5}}
        assert _hash_bounds(a) != _hash_bounds(b)
