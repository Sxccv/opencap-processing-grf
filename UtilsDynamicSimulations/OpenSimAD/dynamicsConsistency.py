"""Checks an exported window's GRF against what its own exported states require.

The pelvis is a floating base with no reserve actuators, so the skeletal
dynamics force ``sum(GRF) = m*(a_com + g)``. Over a window, ``mean(a_com)`` is
exactly ``(v_com(T) - v_com(0)) / T`` -- no numerical differentiation involved
-- so the mean required vertical force follows from just two endpoint state
evaluations:

    mean_required_vy = mass*G + mass*(vcom_y_end - vcom_y_start) / T

This is the production port of the validated harness script
``grf_fix_experiments/t10_dynamics_consistency.py`` (outside this repo), which
found this window-mean, differentiation-free residual to be the metric that
cleanly separates physically-grounded windows from ones where the feet leave
the ground plane. A per-frame residual estimator was tried and rejected: its
own noise floor measured 50 N median / 140 N max even on a perfectly healthy
window, which would false-positive on good windows. Do not substitute it here.

``Model.calcMassCenterAcceleration`` was also tried and rejected:
``SimTK::State`` in these bindings has no ``setUDot``, and realizing to
``Stage::Acceleration`` would make Simbody solve forward dynamics from the
model's own forces -- a different question than "what do the *exported*
kinematics require".

Source of truth: this module reads ``optimaltrajectories_<case>.npy``, the
NLP's own decision variables, rather than ``kinematics_activations_*.mot``.
The states file has no speed columns at all (checked against
``mainOpenSimAD.py``'s writer, which emits only coordinate values in degrees
plus muscle activations), so an earlier version of this module reconstructed
the two endpoint speeds needed to realize ``v_com(0)`` and ``v_com(T)`` by
spline-differentiating the window's own ``q``. The trajectories ``.npy``
instead carries ``coordinate_speeds`` directly as the solver's own decision
variable, so no differentiation of any kind is needed here any more.

Units: values in the trajectories ``.npy`` (``coordinate_values``,
``coordinate_speeds``) are ALREADY IN RADIANS (and rad/s) -- do NOT apply
``np.deg2rad`` to them, unlike the ``.mot`` files, which carry
``inDegrees=yes``. ``pelvis_tx``/``pelvis_ty``/``pelvis_tz`` are metres in
both representations.
"""
import os

import numpy as np

G = 9.80665


def com_velocity_y(model, state, coordinate_set, names, q_col, u_col):
    """Exact COM vertical velocity: set every coordinate value AND speed, realize.

    ``names`` is the coordinate name list; ``q_col``/``u_col`` are 1-D arrays
    (radians / metres, and rad/s / m/s respectively) aligned with ``names``.
    """
    if len(names) != len(q_col) or len(names) != len(u_col):
        raise ValueError(
            f"names/q_col/u_col length mismatch: {len(names)}, {len(q_col)}, "
            f"{len(u_col)}"
        )
    for i, name in enumerate(names):
        co = coordinate_set.get(name)
        co.setValue(state, float(q_col[i]), False)
        co.setSpeedValue(state, float(u_col[i]))
    model.assemble(state)
    model.realizeVelocity(state)
    return model.calcMassCenterVelocity(state).get(1)


def _coordinate_names_in_model(model, candidate_names, trajectories_path):
    """Intersect the trajectories' coordinate name list with the model's own.

    The trajectories ``.npy`` carries every coordinate the NLP tracked; only
    names the model itself has a ``Coordinate`` for can be applied to it.
    """
    cs = model.getCoordinateSet()
    model_names = {cs.get(i).getName() for i in range(cs.getSize())}
    names = [c for c in candidate_names if c in model_names]
    if not names:
        raise ValueError(
            "no coordinate names in the trajectories file matched any "
            f"coordinate in the model's CoordinateSet ({trajectories_path})"
        )
    return names


def _load_case_trajectories(trajectories_path, case):
    """Load and validate one case's dict out of an ``optimaltrajectories_*.npy``.

    Raises a clear exception if the file, the case key, or a required array
    is missing.
    """
    if not os.path.exists(trajectories_path):
        raise FileNotFoundError(
            f"trajectories file not found: {trajectories_path}"
        )

    raw = np.load(trajectories_path, allow_pickle=True).item()
    if case not in raw:
        raise KeyError(
            f"case {case!r} not found in {trajectories_path} "
            f"(keys present: {list(raw.keys())})"
        )
    data = raw[case]

    required = ("coordinates", "time", "coordinate_values",
                "coordinate_speeds", "GRF", "GRF_labels")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"case {case!r} in {trajectories_path} is missing required "
            f"key(s) {missing} (keys present: {list(data.keys())})"
        )
    return data


def window_mean_residual(model_path, trajectories_path, case):
    """Compare a window's exported vertical GRF to what its states require.

    ``trajectories_path`` is an ``optimaltrajectories_<case>.npy`` file (the
    NLP's own decision variables) and ``case`` is the case name whose entry
    should be read out of it.

    Returns a dict: mass_kg, body_weight_N, exported_mean_vertical_N,
    required_mean_vertical_N, residual_N, residual_pct_bw.

    Every failure mode (missing file, missing case key, missing array) raises
    a clear exception rather than returning a silently wrong number.
    """
    import opensim

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found: {model_path}")

    data = _load_case_trajectories(trajectories_path, case)

    names_all = list(data["coordinates"])
    time = np.asarray(data["time"], dtype=float).reshape(-1)
    q_all = np.asarray(data["coordinate_values"], dtype=float)
    u_all = np.asarray(data["coordinate_speeds"], dtype=float)
    GRF = np.asarray(data["GRF"], dtype=float)
    GRF_labels = list(data["GRF_labels"])

    if q_all.shape[0] != len(names_all) or u_all.shape[0] != len(names_all):
        raise ValueError(
            f"coordinate_values/coordinate_speeds row count does not match "
            f"len(coordinates) ({q_all.shape[0]}, {u_all.shape[0]}, "
            f"{len(names_all)}) in {trajectories_path} case {case!r}"
        )
    if q_all.shape[1] != len(time) or u_all.shape[1] != len(time):
        raise ValueError(
            f"coordinate_values/coordinate_speeds column count does not "
            f"match len(time) ({q_all.shape[1]}, {u_all.shape[1]}, "
            f"{len(time)}) in {trajectories_path} case {case!r}"
        )
    if len(time) < 2:
        raise ValueError(
            f"trajectories file {trajectories_path} case {case!r} has fewer "
            f"than 2 time points ({len(time)}) -- cannot compute an endpoint "
            "residual"
        )

    for label in ("ground_force_right_vy", "ground_force_left_vy"):
        if label not in GRF_labels:
            raise KeyError(
                f"required GRF label {label!r} not found in {trajectories_path} "
                f"case {case!r} (labels present: {GRF_labels})"
            )

    opensim.Logger.setLevelString("error")
    model = opensim.Model(model_path)
    state = model.initSystem()
    cs = model.getCoordinateSet()

    names = _coordinate_names_in_model(model, names_all, trajectories_path)
    name_idx = [names_all.index(n) for n in names]

    # Values are already in radians / metres / rad/s / m/s -- no unit
    # conversion, unlike the .mot-reading code path this replaced.
    q_start = q_all[name_idx, 0]
    u_start = u_all[name_idx, 0]
    q_end = q_all[name_idx, -1]
    u_end = u_all[name_idx, -1]

    right_vy = GRF[GRF_labels.index("ground_force_right_vy"), :]
    left_vy = GRF[GRF_labels.index("ground_force_left_vy"), :]
    exported_vy = right_vy + left_vy

    mass = model.getTotalMass(state)
    body_weight_N = mass * G

    vcom_y_start = com_velocity_y(model, state, cs, names, q_start, u_start)
    vcom_y_end = com_velocity_y(model, state, cs, names, q_end, u_end)

    T = float(time[-1] - time[0])
    if T <= 0:
        raise ValueError(
            f"trajectories file {trajectories_path} case {case!r} has "
            f"non-positive duration T={T}"
        )

    required_mean_vertical_N = mass * G + mass * (vcom_y_end - vcom_y_start) / T
    exported_mean_vertical_N = float(exported_vy.mean())
    residual_N = exported_mean_vertical_N - required_mean_vertical_N
    residual_pct_bw = 100.0 * residual_N / body_weight_N

    return {
        "mass_kg": float(mass),
        "body_weight_N": float(body_weight_N),
        "exported_mean_vertical_N": exported_mean_vertical_N,
        "required_mean_vertical_N": float(required_mean_vertical_N),
        "residual_N": float(residual_N),
        "residual_pct_bw": float(residual_pct_bw),
    }


def residual_against_export(model_path, trajectories_path, case):
    """Convenience wrapper: returns (residual_N, residual_pct_bw)."""
    result = window_mean_residual(model_path, trajectories_path, case)
    return result["residual_N"], result["residual_pct_bw"]
