"""Tests for the F2 dynamics-consistency gate: the number that can override an
IPOPT `Solve_Succeeded` and mark a window as failed, and the gate itself.

Part 1 covers ``dynamicsConsistency.window_mean_residual`` (the arithmetic).
Part 2 covers the gate in ``grf_prediction._inspect_outputs`` that decides
whether a computed residual actually fails a window.

Both parts need OpenSim (``window_mean_residual`` builds/loads an
``opensim.Model``), so the whole module is skipped via `pytest.importorskip`
when it is unavailable -- same as `tests/test_kinematics_qc.py`. Part 2 also
needs CasADi: `grf_prediction.py` unconditionally imports `mainOpenSimAD`
(which imports `casadi`) and `utilsOpenSimAD` (which imports `opensim`,
`casadi`, `matplotlib`, etc.) at module level, purely to get `run_tracking`
and `processInputsOpenSimAD` -- so merely *importing* `grf_prediction` to
reach `_inspect_outputs` drags in the whole heavy stack even though the F2
gate itself needs neither. That means this file cannot join the fast tier
(`tests/test_windowed_grf.py`) the way its "one behaviour per test" style
would otherwise suggest; see the report for more on this.

Run with the opensim-env interpreter:

    opensim-env/python.exe -m pytest tests/test_dynamics_consistency.py -q
"""
import os
import sys
import time

import numpy as np
import pytest

opensim = pytest.importorskip("opensim")
pytest.importorskip("casadi")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "UtilsDynamicSimulations", "OpenSimAD"))

import pipeline_io
import dynamicsConsistency
import grf_prediction


# %% Shared fixtures / helpers

def _build_contact_model(model_path):
    """A minimal-but-valid .osim with one SmoothSphereHalfSpaceForce contact.

    Copied from `tests/test_kinematics_qc.py::_build_contact_model` (there is
    no conftest.py in this repo, so each test module fixes its own setup) --
    a single 1 kg body on a FreeJoint is enough to exercise
    `model.getTotalMass`, `model.getCoordinateSet` and
    `model.calcMassCenterVelocity` without a real musculoskeletal model.
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


def _find_vertical_translation_dof(model_path, names):
    """Which coordinate index is the pure world-Y translation DOF.

    Found empirically (unit speed in, check which component of
    `calcMassCenterVelocity` comes back as 1.0) rather than assumed from
    joint-construction order, so this doesn't silently break if a future
    OpenSim version orders `FreeJoint` coordinates differently. Body1's
    center of mass is at the body origin (`Vec3(0)`, see
    `_build_contact_model`), so with all rotational coordinates held at zero
    this DOF's speed *is* the COM's vertical speed, exactly -- which is the
    physical fact the fixtures below rely on.
    """
    model = opensim.Model(model_path)
    state = model.initSystem()
    cs = model.getCoordinateSet()
    zeros = np.zeros(len(names))
    for i in range(len(names)):
        u = np.array(zeros)
        u[i] = 1.0
        vy = dynamicsConsistency.com_velocity_y(model, state, cs, names, zeros, u)
        if abs(vy - 1.0) < 1e-9:
            return i
    raise AssertionError(
        "no coordinate in the test model produced a unit vertical COM "
        "velocity response -- fixture assumption is broken")


def _write_trajectories_npy(path, case, *, coordinates, time_arr, coordinate_values,
                            coordinate_speeds, grf_labels, grf):
    raw = {case: {
        "coordinates": list(coordinates),
        "time": np.asarray(time_arr, dtype=float),
        "coordinate_values": np.asarray(coordinate_values, dtype=float),
        "coordinate_speeds": np.asarray(coordinate_speeds, dtype=float),
        "GRF": np.asarray(grf, dtype=float),
        "GRF_labels": list(grf_labels),
    }}
    np.save(path, raw)


def _build_dynamics_fixture(tmp_path, case, *, time_arr, q_mid, u_mid,
                            q_start, u_start, q_end, u_end,
                            right_vy, left_vy):
    """A model + `optimaltrajectories_<case>.npy` pair with one DOF (the
    vertical translation one) driven by the given start/mid/end q, u values,
    and the given GRF vertical components.
    """
    model_path = str(tmp_path / f"model_{case}.osim")
    _build_contact_model(model_path)

    model = opensim.Model(model_path)
    model.initSystem()
    cs = model.getCoordinateSet()
    names = [cs.get(i).getName() for i in range(cs.getSize())]
    y_idx = _find_vertical_translation_dof(model_path, names)

    n = len(names)
    q = np.zeros((n, 3))
    u = np.zeros((n, 3))
    q[y_idx] = [q_start, q_mid, q_end]
    u[y_idx] = [u_start, u_mid, u_end]

    grf_labels = ["ground_force_right_vy", "ground_force_left_vy"]
    grf = np.array([right_vy, left_vy])

    traj_path = str(tmp_path / f"optimaltrajectories_{case}.npy")
    _write_trajectories_npy(traj_path, case, coordinates=names, time_arr=time_arr,
                            coordinate_values=q, coordinate_speeds=u,
                            grf_labels=grf_labels, grf=grf)
    return model_path, traj_path


# %% dynamicsConsistency.G

class TestModuleConstants:
    def test_gravity_constant_is_standard_gravity(self):
        assert dynamicsConsistency.G == 9.80665


# %% dynamicsConsistency.window_mean_residual -- arithmetic

class TestWindowMeanResidualArithmetic:
    CASE = "case0"

    def test_returns_hand_computed_residual_from_two_endpoint_states(self, tmp_path):
        # Irregular time stamps (0.0, 0.4, 1.0): only time[0]/time[-1] should
        # matter for T. Interior q/u values (3.3, 99.0) are deliberately
        # absurd so a bug that used them would blow the hand-computed
        # comparison out of the water.
        model_path, traj_path = _build_dynamics_fixture(
            tmp_path, self.CASE,
            time_arr=[0.0, 0.4, 1.0],
            q_start=0.9, q_mid=3.3, q_end=1.0,
            u_start=0.2, u_mid=99.0, u_end=1.4,
            right_vy=[4.0, 5.0, 6.0], left_vy=[5.0, 6.0, 8.0])

        result = dynamicsConsistency.window_mean_residual(
            model_path, traj_path, self.CASE)

        # Hand computation, independent of the module under test.
        mass = 1.0  # single 1 kg body; ground contributes no mass
        G = 9.80665
        vcom_y_start, vcom_y_end = 0.2, 1.4  # == u_start, u_end (see fixture)
        T = 1.0 - 0.0
        required = mass * G + mass * (vcom_y_end - vcom_y_start) / T
        exported = float(np.mean([4.0 + 5.0, 5.0 + 6.0, 6.0 + 8.0]))
        body_weight = mass * G
        residual = exported - required
        residual_pct = 100.0 * residual / body_weight

        assert result["mass_kg"] == pytest.approx(mass)
        assert result["body_weight_N"] == pytest.approx(body_weight)
        assert result["required_mean_vertical_N"] == pytest.approx(required)
        assert result["exported_mean_vertical_N"] == pytest.approx(exported)
        assert result["residual_N"] == pytest.approx(residual)
        assert result["residual_pct_bw"] == pytest.approx(residual_pct)

    def test_interior_time_samples_do_not_affect_the_result(self, tmp_path):
        """Pins the docstring's central claim: mean(a_com) is derived from
        just the two endpoint states, with no numerical differentiation, so
        an interior q/u/time sample must be free to be garbage."""
        model_path_a, traj_path_a = _build_dynamics_fixture(
            tmp_path, "case_a",
            time_arr=[0.0, 0.4, 1.0],
            q_start=0.9, q_mid=3.3, q_end=1.0,
            u_start=0.2, u_mid=99.0, u_end=1.4,
            right_vy=[4.0, 5.0, 6.0], left_vy=[5.0, 6.0, 8.0])
        model_path_b, traj_path_b = _build_dynamics_fixture(
            tmp_path, "case_b",
            time_arr=[0.0, 0.81, 1.0],       # different interior time
            q_start=0.9, q_mid=-40.0, q_end=1.0,   # different interior q
            u_start=0.2, u_mid=-500.0, u_end=1.4,  # different interior u
            right_vy=[4.0, -1234.0, 6.0], left_vy=[5.0, 9999.0, 8.0])

        result_a = dynamicsConsistency.window_mean_residual(
            model_path_a, traj_path_a, "case_a")
        result_b = dynamicsConsistency.window_mean_residual(
            model_path_b, traj_path_b, "case_b")

        assert result_a["required_mean_vertical_N"] == \
            pytest.approx(result_b["required_mean_vertical_N"])
        # Exported means differ here because GRF is a straight .mean() over
        # every sample, interior included (that one IS sensitive to interior
        # values, by design) -- only the required/kinematic side is endpoint-only.


class TestWindowMeanResidualRaisesOnNonPositiveDuration:
    def test_raises_value_error_when_last_time_is_not_after_first_time(self, tmp_path):
        case = "case_bad_time"
        model_path, traj_path = _build_dynamics_fixture(
            tmp_path, case,
            time_arr=[1.0, 0.5],  # T = 0.5 - 1.0 = -0.5 <= 0
            q_start=0.0, q_mid=0.0, q_end=0.0,
            u_start=0.0, u_mid=0.0, u_end=0.0,
            right_vy=[0.0, 0.0], left_vy=[0.0, 0.0])
        # Trim the 3-column fixture down to 2 columns to match time_arr.
        raw = np.load(traj_path, allow_pickle=True).item()
        data = raw[case]
        data["coordinate_values"] = data["coordinate_values"][:, :2]
        data["coordinate_speeds"] = data["coordinate_speeds"][:, :2]
        data["GRF"] = data["GRF"][:, :2]
        np.save(traj_path, raw)

        with pytest.raises(ValueError, match="non-positive duration"):
            dynamicsConsistency.window_mean_residual(model_path, traj_path, case)


# %% Unit contract: coordinate_values/speeds are already in radians

class TestUnitContract:
    def test_window_mean_residual_never_calls_np_deg2rad(self, tmp_path, monkeypatch):
        """`coordinate_values`/`coordinate_speeds` in the trajectories .npy
        are already in radians (module docstring, lines 35-39); applying
        `np.deg2rad` to them would silently corrupt the residual. Pin this by
        making `np.deg2rad` explode if `window_mean_residual` ever calls it."""
        def _boom(*args, **kwargs):
            raise AssertionError(
                "window_mean_residual must not call np.deg2rad -- "
                "coordinate_values/coordinate_speeds are already radians")
        monkeypatch.setattr(dynamicsConsistency.np, "deg2rad", _boom)

        model_path, traj_path = _build_dynamics_fixture(
            tmp_path, "case_units",
            time_arr=[0.0, 0.4, 1.0],
            q_start=0.9, q_mid=3.3, q_end=1.0,
            u_start=0.2, u_mid=99.0, u_end=1.4,
            right_vy=[4.0, 5.0, 6.0], left_vy=[5.0, 6.0, 8.0])

        # Must not raise.
        dynamicsConsistency.window_mean_residual(model_path, traj_path, "case_units")


# %% grf_prediction._inspect_outputs -- the F2 gate

def _make_converged_window(tmp_path, *, write_traj_file, index=0):
    """A window whose stats file says IPOPT converged and whose GRF file is
    fresh -- i.e. everything up to (but not including) the F2 gate at
    grf_prediction.py:177 is already satisfied. Returns the args
    `_inspect_outputs` needs, ready to call.
    """
    spec = pipeline_io.TrialSpec(
        baseDir=str(tmp_path), dataFolder=str(tmp_path),
        session_id="OpenCapData_x", trial_name="T", motion_type="walking")
    dyn_dir = spec.dynamics_dir
    os.makedirs(dyn_dir, exist_ok=True)
    case = pipeline_io.case_name(spec.trial_name, index)

    started_at = time.time() - 10.0  # safely before any file this test writes

    stats_file = pipeline_io.stats_path(dyn_dir, case)
    np.save(stats_file, {"success": True, "return_status": "Solve_Succeeded"})

    grf_file = pipeline_io.grf_resultant_path(dyn_dir, spec.trial_name, case)
    with open(grf_file, "w") as f:
        f.write("dummy\n")

    if write_traj_file:
        traj_file = pipeline_io.trajectories_path(dyn_dir, case)
        with open(traj_file, "w") as f:
            f.write("dummy\n")

    settings = {"OpenSimModel": "TestModel"}
    time_window = (0.0, 1.0)
    return spec, index, time_window, case, dyn_dir, started_at, settings


class TestF2GateDoesNotFailOnMissingResidual:
    """The regression that would be most damaging and least visible: a
    missing/unreadable trajectories file must leave residual_N as None and
    must NOT fail an otherwise-converged window (grf_prediction.py:148-167,
    177)."""

    def test_a_missing_trajectories_file_does_not_fail_a_converged_window(
            self, tmp_path):
        args = _make_converged_window(tmp_path, write_traj_file=False)

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is True
        assert result.trajectories_path is None
        assert result.dynamics_consistency_residual_N is None
        assert result.failure_reason is None

    def test_an_unreadable_trajectories_file_does_not_fail_a_converged_window(
            self, tmp_path, monkeypatch):
        args = _make_converged_window(tmp_path, write_traj_file=True)

        def _boom(*a, **kw):
            raise OSError("corrupt trajectories file")
        monkeypatch.setattr(grf_prediction.dynamicsConsistency,
                            "window_mean_residual", _boom)

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is True
        assert result.dynamics_consistency_residual_N is None
        assert result.failure_reason is None


class TestF2GateWithinThreshold:
    def test_residual_inside_threshold_leaves_window_converged(
            self, tmp_path, monkeypatch):
        args = _make_converged_window(tmp_path, write_traj_file=True)
        monkeypatch.setattr(
            grf_prediction.dynamicsConsistency, "window_mean_residual",
            lambda *a, **kw: {"residual_N": 2.0, "body_weight_N": 700.0})

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is True
        assert result.dynamics_consistency_residual_N == 2.0
        assert result.failure_reason is None

    def test_residual_exactly_at_threshold_does_not_fail(
            self, tmp_path, monkeypatch):
        """The gate condition is `abs(residual_N) > threshold_N` (strictly
        greater); a residual equal to the threshold must not fail."""
        args = _make_converged_window(tmp_path, write_traj_file=True)
        # default fraction 0.05 * 700 N = 35 N threshold
        monkeypatch.setattr(
            grf_prediction.dynamicsConsistency, "window_mean_residual",
            lambda *a, **kw: {"residual_N": 35.0, "body_weight_N": 700.0})

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is True


class TestF2GateOutsideThreshold:
    def test_residual_outside_threshold_fails_window_despite_ipopt_success(
            self, tmp_path, monkeypatch):
        args = _make_converged_window(tmp_path, write_traj_file=True)
        # default fraction 0.05 * 700 N = 35 N threshold; 100 N > 35 N
        monkeypatch.setattr(
            grf_prediction.dynamicsConsistency, "window_mean_residual",
            lambda *a, **kw: {"residual_N": 100.0, "body_weight_N": 700.0})

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is False
        assert result.return_status == "Solve_Succeeded"  # IPOPT said success...
        assert "dynamics consistency residual" in result.failure_reason
        assert result.dynamics_consistency_residual_N == 100.0

    def test_negative_residual_outside_threshold_also_fails(
            self, tmp_path, monkeypatch):
        """The gate compares abs(residual_N), so a large negative residual
        must fail just like a large positive one."""
        args = _make_converged_window(tmp_path, write_traj_file=True)
        monkeypatch.setattr(
            grf_prediction.dynamicsConsistency, "window_mean_residual",
            lambda *a, **kw: {"residual_N": -100.0, "body_weight_N": 700.0})

        result = grf_prediction._inspect_outputs(*args)

        assert result.converged is False
        assert result.dynamics_consistency_residual_N == -100.0

    def test_threshold_fraction_is_read_from_settings_not_hardcoded(
            self, tmp_path, monkeypatch):
        args = _make_converged_window(tmp_path, write_traj_file=True)
        spec, index, time_window, case, dyn_dir, started_at, settings = args
        settings["dynamics_residual_bw_fraction"] = 0.5  # threshold -> 350 N
        monkeypatch.setattr(
            grf_prediction.dynamicsConsistency, "window_mean_residual",
            lambda *a, **kw: {"residual_N": 100.0, "body_weight_N": 700.0})

        # Same 100 N residual that failed the window under the default 0.05
        # fraction (see test above) must now pass under a wider 0.5 fraction.
        result = grf_prediction._inspect_outputs(
            spec, index, time_window, case, dyn_dir, started_at, settings)

        assert result.converged is True
