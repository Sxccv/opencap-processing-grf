"""Pre-flight kinematics QC pass, run once per trial before any solve window
is built.

Three things happen here, in order:

1. A corrupt trailing (or leading) frame -- a whole-body teleport that shows
   up as an implausible median marker speed -- is trimmed from both the
   kinematics and the marker data. Interior glitch frames are recorded but
   never repaired.
2. An upward vertical drift is removed from ``pelvis_ty`` so the contact
   spheres actually reach the floor. Without this, the smooth contact force
   is zero (and its gradient vanishes) in later windows of a trial, and the
   solver reports near-zero ground reaction force for a subject who is, in
   fact, standing on the floor.
3. Upper-body IK branch flips and horizontal drift are measured and recorded
   -- never corrected -- so later pipeline stages can decide what to do with
   them.

The de-drift trend is fit to the *minimum* lowest-contact-sphere clearance in
each 1 s bin, because that minimum is a stance instant (foot flat on a flat
floor), which should read zero. A bin that is airborne for its *entire*
duration has no such stance instant: its minimum is a flight height, not
floor contact. Fitting a trend line to a flight height would read genuine
airborne motion as drift -- but a straight line cannot absorb a jump's
parabolic excursion, so a jumping trial leaves a large residual and is
already rejected by the linear-fit residual selector (`RESIDUAL_LIMIT_M`)
before it ever reaches a fully-airborne bin. The genuinely dangerous case is
the `per_window` fallback, which subtracts each window's own minimum and
*will* silently absorb real vertical excursion. `run_qc_pass` therefore
checks for a fully airborne bin (against the *corrected* clearance, after
de-drift has been applied) only when the per-window method was selected, and
raises rather than guessing in that case. Bins that are airborne on the
*uncorrected* data are normal: a drifted trial can sit centimeters above the
gate for many bins, and that is exactly what the trend fit exists to
correct -- it is not, by itself, evidence of a jump.

HONEST LIMITATION: this pass has been validated only on walking trials. When
the linear (`time` or `x`) method is selected, it cannot reliably distinguish
a jumping trial from a drifted one -- the residual selector is the only
protection against a jump in that case, and a jump whose residual happens to
fall under `RESIDUAL_LIMIT_M` would not be caught.

Ported (not imported -- the production pipeline may not depend on that
directory) from the vetted prototypes in
``grf_fix_experiments/{dedrift.py, t7_foot_ground_contact.py,
t9_dedrift_methods.py, t1_kinematics_glitch.py, t6_arm_flip_grf_impact.py}``.

This file rewrites ``<trial>.mot`` in place; the pre-QC file is preserved as
``<trial>_raw.mot`` and a ``<trial>_qc.json`` sidecar records what was done,
so later stages (and `verify_qc_marker`) can confirm the QC pass actually ran
against the kinematics file they are about to read.
"""
import os
import json
import shutil
import hashlib
import datetime

import numpy as np
import pandas as pd
import opensim
from scipy.interpolate import InterpolatedUnivariateSpline

from utils import import_metadata, numpy_to_storage
from utilsOpenSimAD import filterDataFrame
from pipeline_io import (kinematics_mot, kinematics_raw_mot, qc_sidecar_path,
                         markers_trc, session_dir, build_windows)

QC_VERSION = 1

# Coordinates common.UPPER_BODY names for the arm-flip experiments; kept as a
# literal here so this module does not depend on grf_fix_experiments/common.py.
UPPER_BODY = [
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l"]

CLEARANCE_GATE_M = 0.005
RESIDUAL_LIMIT_M = 0.02
FLIP_THRESHOLD_DEG = 20.0
MULTI_COORD_STEP_DEG = 25.0
MULTI_COORD_MIN_COUNT = 3
GRAVITY_M_S2 = 9.80665


class KinematicsQCError(RuntimeError):
    """Raised when a trial's kinematics fail a pre-flight QC check."""


# %% Small helpers.

def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _read_mot(path):
    """(header, data) for a .mot file -- same idiom as
    ``adjustBoundsAndDummyMotion`` in utilsOpenSimAD.py (~line 2410): column
    labels + getMatrix().to_numpy() + getIndependentColumn() for time.
    """
    table = opensim.TimeSeriesTable(path)
    labels = list(table.getColumnLabels())
    matrix = table.getMatrix().to_numpy()
    times = np.array(table.getIndependentColumn())
    header = ['time'] + labels
    data = np.column_stack([times, matrix])
    return header, data


def _read_trc(path):
    """(marker_names, times, xyz[frames, markers, 3]). Ports common.read_trc."""
    lines = open(path).read().splitlines()
    names = [n for n in lines[3].split('\t') if n.strip()][2:]
    rows = [[float(x) for x in l.split('\t') if x.strip() != '']
            for l in lines[6:] if l.strip()]
    d = np.array(rows)
    return names, d[:, 1], d[:, 2:].reshape(len(d), -1, 3)


def _contact_spheres(model):
    """(name, body, location, radius) for every SmoothSphereHalfSpaceForce.

    Ports t7_foot_ground_contact.contact_spheres.
    """
    out = []
    fs = model.getForceSet()
    for i in range(fs.getSize()):
        f = fs.get(i)
        if f.getConcreteClassName() != 'SmoothSphereHalfSpaceForce':
            continue
        obj = opensim.SmoothSphereHalfSpaceForce.safeDownCast(f)
        frame = obj.getConnectee('sphere')
        sphere = opensim.ContactSphere.safeDownCast(frame)
        loc = sphere.get_location()
        out.append({
            'force': f.getName(),
            'body': sphere.getFrame().getName(),
            'location': np.array([loc.get(0), loc.get(1), loc.get(2)]),
            'radius': float(sphere.getRadius()),
        })
    return out


def _sphere_geometry(model, state, spheres, header, data):
    """Lowest sphere clearance and each sphere's ground-frame (x, z), per frame.

    Ports dedrift.lowest_sphere_height / t7_foot_ground_contact.sphere_heights,
    extended to also return horizontal positions for the A5 drift check so
    that the expensive part (an assemble + realizePosition per frame) only
    has to run once and serves both the de-drift trend and the horizontal
    drift measurement.
    """
    cs = model.getCoordinateSet()
    known = {cs.get(i).getName() for i in range(cs.getSize())}
    cols = [(c, header.index(c)) for c in header[1:] if c in known]
    trans = {'pelvis_tx', 'pelvis_ty', 'pelvis_tz'}
    bodies = {s['body']: model.getBodySet().get(s['body']) for s in spheres}

    n, ns = len(data), len(spheres)
    low = np.zeros(n)
    xz = np.zeros((n, ns, 2))
    for k in range(n):
        for name, j in cols:
            v = data[k, j]
            cs.get(name).setValue(state, v if name in trans else np.deg2rad(v),
                                  False)
        model.assemble(state)
        model.realizePosition(state)
        hs = np.zeros(ns)
        for si, s in enumerate(spheres):
            p = bodies[s['body']].findStationLocationInGround(
                state, opensim.Vec3(*s['location']))
            hs[si] = p.get(1) - s['radius']
            xz[k, si, 0] = p.get(0)
            xz[k, si, 1] = p.get(2)
        low[k] = hs.min()
    return low, xz


def _glitch_frames(times, xyz):
    """Frames where the whole body moves implausibly fast.

    Ports t1_kinematics_glitch.detect_glitches, with the threshold widened
    from a flat 5 m/s to ``min(5.0, 10 * trial_median_speed)`` so a trial
    whose median speed is itself close to 5 m/s does not get an
    effective threshold indistinguishable from "never trips".
    """
    dt = np.median(np.diff(times))
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=2) / dt
    med = np.median(step, axis=1)
    trial_median = float(np.median(med))
    threshold = min(5.0, 10.0 * trial_median)
    flagged = np.where(med > threshold)[0] + 1
    return flagged, trial_median, threshold


def _humerus_flip_times(header, data):
    """Frames where a humerus reorients more than FLIP_THRESHOLD_DEG in one
    step, per side. Ports t6_arm_flip_grf_impact.flip_times.
    """
    def rot(axis, ang):
        a = np.array(axis, float)
        a = a / np.linalg.norm(a)
        c, s = np.cos(ang), np.sin(ang)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + s * K + (1 - c) * K @ K

    out = {}
    for side in ('l', 'r'):
        jf = header.index('arm_flex_' + side)
        ja = header.index('arm_add_' + side)
        jr = header.index('arm_rot_' + side)
        angs = []
        prev = None
        for k in range(len(data)):
            R = (rot([0, 0, 1], np.deg2rad(data[k, jf]))
                 @ rot([-1, 0, 0], np.deg2rad(data[k, ja]))
                 @ rot([0, -1, 0], np.deg2rad(data[k, jr])))
            if prev is not None:
                rel = prev.T @ R
                angs.append(np.degrees(np.arccos(
                    np.clip((np.trace(rel) - 1) / 2, -1, 1))))
            prev = R
        angs = np.array(angs)
        idx = np.where(angs > FLIP_THRESHOLD_DEG)[0] + 1
        out[side] = {
            'frame_indices': [int(i) for i in idx],
            'times_s': [float(data[i, 0]) for i in idx],
            'max_deg': float(angs.max()) if len(angs) else 0.0,
            'median_deg': float(np.median(angs)) if len(angs) else 0.0,
        }
    return out


def _multi_coordinate_flip_frames(header, data):
    """Frames where 3+ upper-body coordinates each step > 25 deg at once."""
    cols = [c for c in UPPER_BODY if c in header]
    if not cols:
        return {'frame_indices': [], 'times_s': [], 'counts': []}
    idxs = [header.index(c) for c in cols]
    sub = data[:, idxs]
    steps = np.abs(np.diff(sub, axis=0))
    counts = (steps > MULTI_COORD_STEP_DEG).sum(axis=1)
    flagged = np.where(counts >= MULTI_COORD_MIN_COUNT)[0] + 1
    return {
        'frame_indices': [int(i) for i in flagged],
        'times_s': [float(data[i, 0]) for i in flagged],
        'counts': [int(counts[i - 1]) for i in flagged],
    }


def _per_window_offsets(times, low, windows):
    """One constant shift per window: that window's own minimum clearance.

    Ports t9_dedrift_methods.per_window_offsets.
    """
    shift = np.zeros(len(times))
    offsets = []
    for a, b in windows:
        m = (times >= a) & (times <= b)
        off = float(low[m].min()) if m.any() else 0.0
        shift[m] = off
        offsets.append(off)
    return shift, offsets


def _residuals_after(times, low, shift, windows):
    """Ports t9_dedrift_methods.residuals_after."""
    corrected = low - shift
    out = []
    for a, b in windows:
        m = (times >= a) & (times <= b)
        out.append(float(corrected[m].min()) if m.any() else 0.0)
    return out


def _window_value_at(t, windows, offsets):
    for (a, b), off in zip(windows, offsets):
        if a <= t <= b:
            return off
    return offsets[-1] if offsets else 0.0


def _contiguous_runs(mask):
    """[(start_idx, end_idx), ...] (inclusive) for each run of True in mask."""
    runs = []
    n = len(mask)
    k = 0
    while k < n:
        if mask[k]:
            start = k
            while k < n and mask[k]:
                k += 1
            runs.append((start, k - 1))
        else:
            k += 1
    return runs


# %% Public API.

def run_qc_pass(dataFolder, session_id, trial_name,
                OpenSimModel="LaiUhlrich2022", bin_seconds=1.0):
    """Run (or skip, if already current) the pre-flight kinematics QC pass.

    Idempotent: if ``<trial>_qc.json`` already records ``qc_version ==
    QC_VERSION`` and a ``kinematics_sha256`` matching the current
    ``<trial>.mot``, that sidecar is returned unchanged and nothing else
    happens.

    Otherwise, in order: requires the trial's `{OpenSimModel}
    _scaled_adjusted_contacts.osim` contact model (raises if it is missing --
    this pass does not build it); reads the trial's `.mot` and `.trc` and
    asserts they share one time base; trims a leading/trailing kinematics
    glitch (B1); computes the lowest contact-sphere clearance per frame on a
    6 Hz-filtered copy; bins it into 1 s bins anchored at the trial's own
    first time (a bin whose minimum sits above the ground gate on this
    *uncorrected* data is normal drifted stance, not rejected here -- see the
    module docstring); selects a de-drift method (a linear fit against time,
    else a linear fit against `pelvis_tx`, else a per-window constant) and
    applies it plus a small mandatory positive constant to `pelvis_ty`; if
    the per-window fallback was selected, rejects the trial when any 1 s bin
    is fully airborne with respect to the *corrected* clearance (see the
    module docstring for why only the per-window method is gated this way);
    detects (but does not correct) upper-body IK flips and horizontal drift;
    and finally rewrites `<trial>.mot` in place, preserving the pre-QC file
    as `<trial>_raw.mot` and recording a SHA-256 marker of the rewritten file
    in the sidecar.

    HONEST LIMITATION: validated only on walking trials. When the linear
    (`time`/`x`) method is selected, this pass cannot reliably detect a
    jumping trial -- the residual selector (`RESIDUAL_LIMIT_M`) is the only
    protection in that case.

    Returns the sidecar dict.
    """
    mot_path = kinematics_mot(dataFolder, session_id, trial_name)
    sidecar_path = qc_sidecar_path(dataFolder, session_id, trial_name)

    # --- Step 1: idempotence check, first, before anything else. ---
    if os.path.exists(sidecar_path) and os.path.exists(mot_path):
        try:
            with open(sidecar_path) as f:
                existing = json.load(f)
        except (OSError, ValueError):
            existing = None
        if (existing is not None
                and existing.get('qc_version') == QC_VERSION
                and existing.get('kinematics_sha256') == _sha256_file(mot_path)):
            return existing

    # --- Step 2: require the contact model. ---
    model_path = os.path.join(session_dir(dataFolder, session_id), 'OpenSimData',
                              'Model', f'{OpenSimModel}_scaled_adjusted_contacts.osim')
    if not os.path.exists(model_path):
        raise KinematicsQCError(
            f"trial {trial_name}: contact model not found at {model_path}. "
            "processInputsOpenSimAD must be called at trial scope before the "
            "kinematics QC pass can run -- this pass does not build the "
            "contacts model itself.")

    opensim.Logger.setLevelString('error')
    model = opensim.Model(model_path)
    state = model.initSystem()
    spheres = _contact_spheres(model)
    if not spheres:
        raise KinematicsQCError(
            f"trial {trial_name}: contact model {model_path} has no "
            "SmoothSphereHalfSpaceForce contact spheres.")

    # --- Step 3: read the .mot. ---
    if not os.path.exists(mot_path):
        raise KinematicsQCError(
            f"trial {trial_name}: kinematics file not found at {mot_path}.")
    header, data = _read_mot(mot_path)
    mot_times = data[:, 0]

    # --- Step 4: read the .trc and assert the shared time base. ---
    trc_path = markers_trc(dataFolder, session_id, trial_name)
    if not os.path.exists(trc_path):
        raise KinematicsQCError(
            f"trial {trial_name}: marker file not found at {trc_path}.")
    _marker_names, trc_times, xyz = _read_trc(trc_path)
    if len(trc_times) != len(mot_times):
        raise KinematicsQCError(
            f"trial {trial_name}: frame count mismatch between kinematics "
            f"({len(mot_times)} rows in {mot_path}) and markers "
            f"({len(trc_times)} rows in {trc_path}); they must share the "
            "same time base.")
    frame_rate = 1.0 / np.median(np.diff(mot_times))
    if not np.allclose(trc_times, mot_times, atol=0.25 / frame_rate):
        raise KinematicsQCError(
            f"trial {trial_name}: kinematics and marker time stamps disagree "
            f"by more than a quarter frame at {frame_rate:.3f} Hz "
            f"({mot_path} vs {trc_path}).")

    # --- Step 5: B1 glitch detection and trimming. ---
    flagged, trial_median_speed, eff_threshold = _glitch_frames(mot_times, xyz)
    n_source = len(mot_times)
    bad = set(int(i) for i in flagged)
    trailing = 0
    k = n_source - 1
    while k >= 0 and k in bad:
        trailing += 1
        k -= 1
    leading = 0
    k = 0
    while k < n_source and k in bad:
        leading += 1
        k += 1
    if leading + trailing >= n_source:
        raise KinematicsQCError(
            f"trial {trial_name}: every frame was flagged as a kinematics "
            "glitch; nothing usable remains after trimming leading/trailing "
            "frames.")
    interior = sorted(i for i in bad if leading <= i < n_source - trailing)

    data = data[leading:n_source - trailing]
    times = data[:, 0]

    # --- Step 6: lowest contact-sphere clearance per frame, 6 Hz filtered
    # copy for the trend fit only; `data` (written back) stays unfiltered. ---
    filt_df = filterDataFrame(pd.DataFrame(data=data, columns=header),
                              cutoff_frequency=6)
    filt_data = filt_df.to_numpy()
    low_filt, sphere_xz = _sphere_geometry(model, state, spheres, header, filt_data)

    # --- Step 7 + Step 8: 1 s bins anchored at this trial's own first time.
    # A bin whose minimum sits above the ground gate here is normal drifted
    # stance data (the uncorrected trial can sit centimeters above the floor
    # by design) and is still recorded and used for the trend fit; it is NOT
    # rejected here. See the module docstring and the real airborne guard
    # after de-drift correction (below) for the actual jump protection. ---
    edges = np.arange(times[0], times[-1] + bin_seconds, bin_seconds)
    # At a walking cadence of roughly 1 Hz, a bin has to span close to its
    # full nominal duration to be trusted to contain a near-flat-foot stance
    # instant -- the module's stated precondition for treating a bin's
    # minimum as a floor measurement. The trailing (or leading) bin is
    # commonly a short stub containing only the remainder of the trial; a
    # stub's minimum is a swing-phase height instead, which is
    # systematically too high, and folding it into the trend fit / selector
    # / step-4 constant biases all three toward that too-high value.
    frame_rate = 1.0 / np.median(np.diff(times))
    min_bin_coverage_fraction = 0.5
    min_bin_samples_coverage = min_bin_coverage_fraction * bin_seconds * frame_rate
    bin_times, bin_minima, bin_idx = [], [], []
    bins_rejected_short = []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (times >= a) & (times < b)
        if mask.sum() < 3:
            continue
        local_idx = np.where(mask)[0]
        if mask.sum() < min_bin_samples_coverage:
            bins_rejected_short.append(
                [float(times[local_idx[0]]), float(times[local_idx[-1]])])
            continue
        argmin_local = int(local_idx[np.argmin(low_filt[local_idx])])
        bmin = float(low_filt[argmin_local])
        bin_idx.append(argmin_local)
        bin_times.append(float(times[argmin_local]))
        bin_minima.append(bmin)
    bin_times = np.array(bin_times)
    bin_minima = np.array(bin_minima)
    bin_idx = np.array(bin_idx, dtype=int)

    if len(bin_times) < 2:
        raise KinematicsQCError(
            f"trial {trial_name}: fewer than two usable 1 s bins "
            f"({len(bin_times)}) after trimming; cannot fit a de-drift trend.")

    # --- Step 9: de-drift method selection, on step-3 (pre-constant) residuals. ---
    pelvis_tx_col = header.index('pelvis_tx')
    pelvis_ty_col = header.index('pelvis_ty')
    pelvis_tx_full = data[:, pelvis_tx_col]
    pelvis_tx_at_bins = pelvis_tx_full[bin_idx]

    slope_t, intercept_t = np.polyfit(bin_times, bin_minima, 1)
    resid_t = bin_minima - (slope_t * bin_times + intercept_t)
    worst_t = float(np.max(np.abs(resid_t)))

    slope_x, intercept_x = np.polyfit(pelvis_tx_at_bins, bin_minima, 1)
    resid_x = bin_minima - (slope_x * pelvis_tx_at_bins + intercept_x)
    worst_x = float(np.max(np.abs(resid_x)))

    windows = build_windows(float(times[0]), float(times[-1]), step=bin_seconds)
    pw_shift_full, pw_offsets = _per_window_offsets(times, low_filt, windows)
    pw_residuals = _residuals_after(times, low_filt, pw_shift_full, windows)

    if worst_t <= RESIDUAL_LIMIT_M:
        method = 'time'
    elif worst_x <= RESIDUAL_LIMIT_M:
        method = 'x'
    else:
        method = 'per_window'

    metadata_path = os.path.join(session_dir(dataFolder, session_id),
                                 'sessionMetadata.yaml')
    metadata = import_metadata(metadata_path)
    mass_kg = float(metadata['mass_kg'])

    accel_cost_rms = accel_cost_max = None
    if method == 'x':
        spline = InterpolatedUnivariateSpline(times, pelvis_tx_full, k=3)
        accel = spline.derivative(n=2)(times)
        force = slope_x * mass_kg * accel
        accel_cost_rms = float(np.sqrt(np.mean(force ** 2)))
        accel_cost_max = float(np.abs(force).max())

    if method == 'time':
        value_full = slope_t * times + intercept_t
        value_bins = slope_t * bin_times + intercept_t
    elif method == 'x':
        value_full = slope_x * pelvis_tx_full + intercept_x
        value_bins = slope_x * pelvis_tx_at_bins + intercept_x
    else:
        value_full = pw_shift_full
        value_bins = np.array([_window_value_at(t, windows, pw_offsets)
                               for t in bin_times])

    bin_residuals_after_step3 = bin_minima - value_bins
    positive = bin_residuals_after_step3[bin_residuals_after_step3 > 0]
    worst_positive = float(positive.max()) if len(positive) else 0.0

    # --- Step 10: mandatory constant, subtract (trend + shift) from pelvis_ty. ---
    step4_constant = max(0.0, worst_positive + 0.005)
    bin_residuals_after_step4 = bin_residuals_after_step3 - step4_constant

    shift_per_frame = value_full + step4_constant
    data[:, pelvis_ty_col] = data[:, pelvis_ty_col] - shift_per_frame
    low_corrected = low_filt - shift_per_frame

    # Diagnostic airborne spans, computed from the CORRECTED clearance (not
    # the raw/uncorrected data -- see the module docstring). During normal
    # walking, swing phases will show up here as numerous short spans; that
    # is expected and is not itself a fault.
    airborne_runs = _contiguous_runs(low_corrected > CLEARANCE_GATE_M)
    airborne_spans_s = [[float(times[s]), float(times[e])] for s, e in airborne_runs]

    # --- Real airborne guard: gates the per-window fallback ONLY. ---
    # A fully airborne 1 s bin (every frame's corrected clearance above the
    # gate) means there is no stance instant in that bin at all -- its
    # "minimum" is a flight height. The linear (time/x) methods cannot
    # silently absorb that: a straight line can't fit a jump's parabolic
    # excursion, so a jump already blows the RESIDUAL_LIMIT_M residual gate
    # above and is rejected there. The per_window fallback has no such
    # protection -- it subtracts each window's own minimum outright -- so it
    # alone is checked here. HONEST LIMITATION: this pass is validated only
    # on walking trials; when the linear method is selected, a jump that
    # happens to leave a residual under RESIDUAL_LIMIT_M would not be caught
    # by anything in this module.
    fully_airborne_bins = []
    for a, b in windows:
        mask = (times >= a) & (times <= b)
        if mask.any() and np.all(low_corrected[mask] > CLEARANCE_GATE_M):
            fully_airborne_bins.append([float(a), float(b)])

    if method == 'per_window' and fully_airborne_bins:
        a0, b0 = fully_airborne_bins[0]
        raise KinematicsQCError(
            f"trial {trial_name}: the per-window de-drift fallback was "
            f"selected because the linear fit's worst residual exceeded the "
            f"{RESIDUAL_LIMIT_M} m limit (time fit worst residual "
            f"{worst_t:.4f} m, pelvis_tx fit worst residual {worst_x:.4f} "
            f"m), but bin [{a0:.3f}, {b0:.3f}] s is fully airborne with "
            f"respect to the corrected clearance (every frame in that bin "
            f"is more than {CLEARANCE_GATE_M} m above the ground). "
            "Combining a fully airborne bin with a per-window offset would "
            "subtract genuine vertical motion rather than drift.")

    per_window_list = []
    for i, (a, b) in enumerate(windows):
        mask = (times >= a) & (times <= b)
        if not mask.any():
            per_window_list.append({
                'index': i, 'time_start': float(a), 'time_end': float(b),
                'min_clearance_m': None, 'frac_frames_within_5mm': None,
                'dedrift_offset_m': None,
            })
            continue
        per_window_list.append({
            'index': i,
            'time_start': float(a),
            'time_end': float(b),
            'min_clearance_m': float(low_corrected[mask].min()),
            'frac_frames_within_5mm': float(
                np.mean(low_corrected[mask] <= CLEARANCE_GATE_M)),
            'dedrift_offset_m': float(np.mean(shift_per_frame[mask])),
        })

    # --- Step 11: D1 flip detection, record only. ---
    flip_detection = {
        'humerus': _humerus_flip_times(header, data),
        'humerus_threshold_deg': FLIP_THRESHOLD_DEG,
        'multi_coordinate': _multi_coordinate_flip_frames(header, data),
        'multi_coordinate_threshold_deg': MULTI_COORD_STEP_DEG,
        'multi_coordinate_min_count': MULTI_COORD_MIN_COUNT,
    }

    # --- Step 12: A5 horizontal drift. ---
    stance_mask = low_corrected <= CLEARANCE_GATE_M
    speeds = []
    for si in range(len(spheres)):
        for k in range(len(times) - 1):
            if stance_mask[k] and stance_mask[k + 1]:
                dt_k = times[k + 1] - times[k]
                if dt_k > 0:
                    d = sphere_xz[k + 1, si] - sphere_xz[k, si]
                    speeds.append(np.linalg.norm(d) / dt_k)
    speeds = np.array(speeds)
    tilt_rad = float(np.arctan(slope_x))
    horizontal_drift = {
        'mean_speed_m_s': float(speeds.mean()) if len(speeds) else 0.0,
        'max_speed_m_s': float(speeds.max()) if len(speeds) else 0.0,
        'n_stance_frame_pairs': int(len(speeds)),
        'tilt_angle_deg_estimate': float(np.degrees(tilt_rad)),
        'gravity_bias_N_estimate': float(mass_kg * GRAVITY_M_S2 * np.sin(tilt_rad)),
    }

    # --- Step 13: write outputs, in order. ---
    raw_path = kinematics_raw_mot(dataFolder, session_id, trial_name)
    shutil.copy2(mot_path, raw_path)

    numpy_to_storage(header, data, mot_path, datatype='IK')

    kinematics_sha256 = _sha256_file(mot_path)

    sidecar = {
        'qc_version': QC_VERSION,
        'trial_name': trial_name,
        'kinematics_sha256': kinematics_sha256,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_frame_count': int(n_source),
        'trimmed_frame_count': int(leading + trailing),
        'glitch_frames_trimmed': {'leading': int(leading), 'trailing': int(trailing)},
        'glitch_frames_interior': [int(i) for i in interior],
        'glitch_detector': {
            'abs_threshold_m_s': 5.0,
            'relative_multiplier': 10.0,
            'trial_median_speed_m_s': trial_median_speed,
            'effective_threshold_m_s': eff_threshold,
        },
        'dedrift': {
            'method': method,
            'bin_seconds': float(bin_seconds),
            'bin_origin_s': float(times[0]),
            'time_fit': {
                'slope_m_per_s': float(slope_t),
                'intercept_m': float(intercept_t),
                'worst_abs_residual_m': worst_t,
            },
            'x_fit': {
                'slope_dimensionless': float(slope_x),
                'intercept_m': float(intercept_x),
                'worst_abs_residual_m': worst_x,
                'acceleration_cost_N_rms': accel_cost_rms,
                'acceleration_cost_N_max': accel_cost_max,
            },
            'per_window_fallback': {
                'offsets_m': [float(v) for v in pw_offsets],
                'residuals_m': [float(v) for v in pw_residuals],
            },
            'selector_threshold_m': RESIDUAL_LIMIT_M,
            'step4_constant_m': float(step4_constant),
            'bin_min_coverage_fraction': min_bin_coverage_fraction,
            'bins_rejected_short': bins_rejected_short,
            'bin_times_s': [float(v) for v in bin_times],
            'bin_minima_before_m': [float(v) for v in bin_minima],
            'bin_residuals_after_step3_m': [float(v) for v in bin_residuals_after_step3],
            'bin_residuals_after_step4_m': [float(v) for v in bin_residuals_after_step4],
        },
        'airborne_spans_s': airborne_spans_s,
        'airborne_guard': {
            'gate_m': CLEARANCE_GATE_M,
            'checked_against': 'corrected_clearance',
            'fully_airborne_bins': fully_airborne_bins,
        },
        'per_frame_lowest_sphere_clearance_m': [float(v) for v in low_corrected],
        'per_window': per_window_list,
        'flip_detection': flip_detection,
        'horizontal_drift': horizontal_drift,
        'mass_kg': mass_kg,
    }
    sidecar = _jsonable(sidecar)

    with open(sidecar_path, 'w') as f:
        json.dump(sidecar, f, indent=2)

    # Self-check: the rewritten file must still be readable by OpenSim.
    try:
        opensim.TimeSeriesTable(mot_path)
    except Exception as e:
        raise KinematicsQCError(
            f"trial {trial_name}: rewritten kinematics file at {mot_path} "
            f"failed to re-open as an OpenSim TimeSeriesTable: {e}")

    return sidecar


def verify_qc_marker(dataFolder, session_id, trial_name):
    """Raise KinematicsQCError unless `<trial>_qc.json` matches `<trial>.mot`.

    Checks that the sidecar exists, its `qc_version` equals `QC_VERSION`, and
    the SHA-256 it recorded still matches the kinematics file on disk -- i.e.
    that the file a later pipeline stage is about to read is exactly the one
    the QC pass produced, not a stale copy or an un-QC'd original. Returns
    the sidecar dict on success.
    """
    sidecar_path = qc_sidecar_path(dataFolder, session_id, trial_name)
    if not os.path.exists(sidecar_path):
        raise KinematicsQCError(
            f"trial {trial_name}: no QC sidecar at {sidecar_path}; "
            "run_qc_pass must be run before this trial can be solved.")

    with open(sidecar_path) as f:
        sidecar = json.load(f)

    if sidecar.get('qc_version') != QC_VERSION:
        raise KinematicsQCError(
            f"trial {trial_name}: QC sidecar at {sidecar_path} was written "
            f"by qc_version {sidecar.get('qc_version')!r}, expected "
            f"{QC_VERSION}; re-run run_qc_pass.")

    mot_path = kinematics_mot(dataFolder, session_id, trial_name)
    if not os.path.exists(mot_path):
        raise KinematicsQCError(
            f"trial {trial_name}: kinematics file not found at {mot_path}.")

    actual_sha256 = _sha256_file(mot_path)
    if actual_sha256 != sidecar.get('kinematics_sha256'):
        raise KinematicsQCError(
            f"trial {trial_name}: {mot_path} has changed since the QC pass "
            "ran (SHA-256 mismatch with the sidecar); re-run run_qc_pass.")

    return sidecar
