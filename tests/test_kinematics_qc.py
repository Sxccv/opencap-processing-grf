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

import numpy as np
import pytest

opensim = pytest.importorskip("opensim")
pytest.importorskip("casadi")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "UtilsDynamicSimulations", "OpenSimAD"))

import pipeline_io
from kinematicsQC import (KinematicsQCError, QC_VERSION, UPPER_BODY, _sha256_file,
                          _step4_constant, run_qc_pass, verify_qc_marker)
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


def _build_pelvis_model(model_path, radius=0.02):
    """One body on a FreeJoint whose translations are named pelvis_t{x,y,z},
    with one contact sphere at the body origin: the lowest clearance is then
    exactly ``pelvis_ty - radius``, so the de-drift's effect can be read
    straight off the written ``pelvis_ty``."""
    model = opensim.Model()
    model.setName("pelvis_test")
    body = opensim.Body("pelvis", 1.0, opensim.Vec3(0), opensim.Inertia(1, 1, 1))
    model.addBody(body)
    joint = opensim.FreeJoint("ground_pelvis", model.getGround(), opensim.Vec3(0),
                              opensim.Vec3(0), body, opensim.Vec3(0), opensim.Vec3(0))
    for i, name in zip((3, 4, 5), ("pelvis_tx", "pelvis_ty", "pelvis_tz")):
        joint.upd_coordinates(i).setName(name)
    model.addJoint(joint)
    half_space = opensim.ContactHalfSpace(
        opensim.Vec3(0), opensim.Vec3(0, 0, -1.5707963267948966),
        model.getGround(), "ground_contact")
    model.addContactGeometry(half_space)
    sphere = opensim.ContactSphere(radius, opensim.Vec3(0), body, "contact_sphere")
    model.addContactGeometry(sphere)
    model.addForce(opensim.SmoothSphereHalfSpaceForce("contact_force", sphere, half_space))
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(model_path)


def _drifting_session(tmp_path, n_rows=300, rate=100.0, drift_m_s=0.01):
    """A complete synthetic trial: contact model, metadata, markers, and a
    .mot whose pelvis_ty rises linearly at `drift_m_s` from sphere-on-floor.
    Returns (dataFolder, pelvis_ty as written)."""
    dataFolder = _make_session_dirs(tmp_path)
    root = pipeline_io.session_dir(dataFolder, SESSION_ID)
    _build_pelvis_model(os.path.join(
        root, "OpenSimData", "Model", f"{OPENSIM_MODEL}_scaled_adjusted_contacts.osim"))
    with open(os.path.join(root, "sessionMetadata.yaml"), "w") as f:
        f.write("mass_kg: 70.0\n")
    t = np.arange(n_rows) / rate
    pelvis_tx = 1.2 * t
    pelvis_ty = 0.02 + drift_m_s * t
    with open(pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME), "w") as f:
        f.write("Coordinates\nversion=1\n")
        # The flip detector reads the upper-body columns; zeros are no flip.
        f.write(f"nRows={n_rows}\nnColumns={4 + len(UPPER_BODY)}\n"
                "inDegrees=yes\nendheader\n")
        f.write("\t".join(["time", "pelvis_tx", "pelvis_ty", "pelvis_tz"]
                          + UPPER_BODY) + "\n")
        zeros = "\t0.0" * len(UPPER_BODY)
        for i in range(n_rows):
            f.write(f"{t[i]:.6f}\t{pelvis_tx[i]:.8f}\t{pelvis_ty[i]:.8f}\t0.0"
                    f"{zeros}\n")
    with open(pipeline_io.markers_trc(dataFolder, SESSION_ID, TRIAL_NAME), "w") as f:
        f.write("PathFileType\t4\t(X/Y/Z)\tT.trc\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
               "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{rate}\t{rate}\t{n_rows}\t1\tm\t{rate}\t1\t{n_rows}\n")
        f.write("Frame#\tTime\tM1\t\t\n")
        f.write("\t\tX1\tY1\tZ1\n")
        f.write("\n")
        for i in range(n_rows):
            f.write(f"{i + 1}\t{t[i]:.6f}\t{pelvis_tx[i]:.8f}\t1.0\t0.0\n")
    return dataFolder, pelvis_ty


def _written_pelvis_ty(dataFolder):
    table = opensim.TimeSeriesTable(
        pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME))
    labels = list(table.getColumnLabels())
    return table.getMatrix().to_numpy()[:, labels.index("pelvis_ty")]


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


# %% run_qc_pass drift_correction switch

class TestDriftCorrectionSwitch:
    def test_rejects_an_unknown_value(self, tmp_path):
        dataFolder, _ = _drifting_session(tmp_path)
        with pytest.raises(ValueError, match="drift_correction"):
            run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                        OpenSimModel=OPENSIM_MODEL, drift_correction="linear")

    def test_dedrift_removes_the_trend(self, tmp_path):
        dataFolder, raw_ty = _drifting_session(tmp_path)
        sidecar = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                              OpenSimModel=OPENSIM_MODEL)
        ty = _written_pelvis_ty(dataFolder)
        assert sidecar["drift_correction"] == "dedrift"
        assert sidecar["dedrift"]["applied"] is True
        # 3 cm of rise over the trial is reduced to well under 1 cm.
        assert np.ptp(raw_ty) == pytest.approx(0.03, abs=1e-3)
        assert np.ptp(ty) < 0.005

    def test_none_records_the_fit_but_leaves_pelvis_ty_unchanged(self, tmp_path):
        dataFolder, raw_ty = _drifting_session(tmp_path)
        sidecar = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                              OpenSimModel=OPENSIM_MODEL, drift_correction="none")
        assert sidecar["drift_correction"] == "none"
        assert sidecar["dedrift"]["applied"] is False
        assert sidecar["dedrift"]["time_fit"]["slope_m_per_s"] == \
            pytest.approx(0.01, abs=1e-3)
        np.testing.assert_allclose(_written_pelvis_ty(dataFolder), raw_ty, atol=1e-6)
        assert all(w["dedrift_offset_m"] == 0.0 for w in sidecar["per_window"]
                   if w["dedrift_offset_m"] is not None)

    def test_switching_arms_restarts_from_the_raw_kinematics(self, tmp_path):
        """dedrift -> none -> dedrift: each run reads <trial>_raw.mot, the raw
        copy is never overwritten, and the two dedrift runs agree."""
        dataFolder, raw_ty = _drifting_session(tmp_path)
        raw_path = pipeline_io.kinematics_raw_mot(dataFolder, SESSION_ID, TRIAL_NAME)

        first = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME, OpenSimModel=OPENSIM_MODEL)
        ty_dedrift = _written_pelvis_ty(dataFolder)
        raw_sha = _sha256_file(raw_path)
        assert first["kinematics_source"] == "mot"

        off = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                          OpenSimModel=OPENSIM_MODEL, drift_correction="none")
        assert off["kinematics_source"] == "raw_mot"
        np.testing.assert_allclose(_written_pelvis_ty(dataFolder), raw_ty, atol=1e-6)
        assert _sha256_file(raw_path) == raw_sha

        again = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME, OpenSimModel=OPENSIM_MODEL)
        assert again["kinematics_source"] == "raw_mot"
        np.testing.assert_allclose(_written_pelvis_ty(dataFolder), ty_dedrift, atol=1e-9)
        assert _sha256_file(raw_path) == raw_sha

    def test_same_setting_twice_is_a_no_op(self, tmp_path):
        dataFolder, _ = _drifting_session(tmp_path)
        first = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                            OpenSimModel=OPENSIM_MODEL, drift_correction="none")
        second = run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                             OpenSimModel=OPENSIM_MODEL, drift_correction="none")
        assert second["generated_at"] == first["generated_at"]

    def test_verify_rejects_a_sidecar_from_the_other_arm(self, tmp_path):
        dataFolder, _ = _drifting_session(tmp_path)
        run_qc_pass(dataFolder, SESSION_ID, TRIAL_NAME,
                    OpenSimModel=OPENSIM_MODEL, drift_correction="none")
        verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME, drift_correction="none")
        with pytest.raises(KinematicsQCError, match="drift_correction='none'"):
            verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME,
                             drift_correction="dedrift")

    def test_sidecar_without_the_key_reads_as_dedrift(self, tmp_path):
        """Sidecars written before the switch existed always de-drifted."""
        dataFolder = _make_session_dirs(tmp_path)
        mot_path = pipeline_io.kinematics_mot(dataFolder, SESSION_ID, TRIAL_NAME)
        _write_mot(mot_path, 5)
        with open(pipeline_io.qc_sidecar_path(dataFolder, SESSION_ID, TRIAL_NAME), "w") as f:
            json.dump({"qc_version": QC_VERSION,
                      "kinematics_sha256": _sha256_file(mot_path)}, f)
        verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME, drift_correction="dedrift")
        with pytest.raises(KinematicsQCError, match="drift_correction"):
            verify_qc_marker(dataFolder, SESSION_ID, TRIAL_NAME, drift_correction="none")


# %% kinematicsQC._step4_constant (GRF_FIX_PLAN.md A1)

class TestStep4Constant:
    def test_all_bins_below_floor_gives_zero_constant(self):
        """A trial whose step-3 bin residuals are all negative (every bin
        already sits below the floor) must be left alone: the max(0, ...)
        guard should actually engage, not just floor at 0.005 always."""
        bin_residuals_after_step3 = np.array([-0.02, -0.01, -0.03, -0.005])
        assert _step4_constant(bin_residuals_after_step3) == 0.0

    def test_worst_positive_residual_shifts_it_five_mm_below_floor(self):
        """worst residual = +4 mm -> constant = 4mm + 5mm = 9mm, so the
        highest bin ends up 5 mm below the floor."""
        bin_residuals_after_step3 = np.array([-0.01, 0.004, -0.002, 0.001])
        assert _step4_constant(bin_residuals_after_step3) == \
            pytest.approx(0.009, abs=1e-9)


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
