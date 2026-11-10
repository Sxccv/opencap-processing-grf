"""Does the per-camera vertical error explain the floor slope?

1. Read the triangulated video keypoints from the OpenCap TRC, undo OpenCap's
   vertical offset (settings verticalOffset) and its backWall rotation, giving
   board-frame coordinates (mm) exactly as OpenCap triangulated them.
2. Project them into every camera with the stored pickle pose -> 2D observations
   consistent with OpenCap's calibration.
3. Re-triangulate (linear DLT, undistorted normalised coords):
     a) with the pickle poses (sanity: must reproduce the TRC),
     b) with each camera's pose corrected so that its board-defined vertical matches
        the room-line vertical from vertical_vp.py. The correction rotates the camera
        about the board centre, which leaves the board's own image unchanged (the
        direction single-image PnP cannot see).
4. Map back through OpenCap's rotation and fit the stance-floor slope along X.
If 3b removes the slope, the measured calibration error accounts for it.
"""
import os, sys, glob, pickle
import numpy as np, cv2, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vertical_vp import opencap_rotation, board_points
import vertical_vp

FEET = ["RHeel", "LHeel", "RBigToe", "LBigToe", "RSmallToe", "LSmallToe"]


def read_trc(path):
    lines = open(path).read().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n]
    data = np.genfromtxt(lines[5:], delimiter="\t")
    data = data[~np.isnan(data[:, 1])]
    return data[:, 1], {n: data[:, 2 + 3 * i: 5 + 3 * i] for i, n in enumerate(names)}


def stance_slope(t, m, vmax=0.15):
    P, L = [], []
    for name in FEET:
        p = m[name][:-1]
        v = np.linalg.norm(np.gradient(p, t[:-1], axis=0), axis=1)
        idx = np.where(v < vmax)[0]
        P.append(p[idx]); L += [name] * len(idx)
    P, L = np.vstack(P), np.array(L)
    A = np.column_stack([P[:, 0]] + [(L == n).astype(float) for n in sorted(set(L))])
    c, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    return np.degrees(np.arctan(c[0]))


def rot_between(a, b):
    """Minimal rotation taking unit a to unit b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def triangulate(Ps, xs):
    """Linear DLT. Ps: list of 3x4 (normalised), xs: (ncam, npts, 2)."""
    n = xs.shape[1]
    out = np.zeros((n, 3))
    for k in range(n):
        A = []
        for P, x in zip(Ps, xs[:, k]):
            A.append(x[0] * P[2] - P[0]); A.append(x[1] * P[2] - P[1])
        _, _, Vt = np.linalg.svd(np.array(A))
        X = Vt[-1]; out[k] = X[:3] / X[3]
    return out


def main(sess, out_dir):
    meta = yaml.safe_load(open(os.path.join(sess, "sessionMetadata.yaml")))
    cams = sorted(int(d[3:]) for d in os.listdir(os.path.join(sess, "Videos")) if d.startswith("Cam"))
    params = {c: pickle.load(open(os.path.join(sess, "Videos", f"Cam{c}", "cameraIntrinsicsExtrinsics.pickle"), "rb")) for c in cams}
    upside = np.asarray(params[cams[0]]["rotation"])[1, 1] < 0
    M = opencap_rotation(upside)
    obj = board_points(meta)
    b = obj.mean(0)                                   # board centre, mm

    # per-camera room vertical, via vertical_vp's pipeline (captured from its internals)
    ups = {}
    import io, contextlib
    orig = vertical_vp.main
    # re-run the detector but keep the direction: small inline copy of the core steps
    for c in cams:
        P = params[c]
        K = np.asarray(P["intrinsicMat"], float); dist = np.asarray(P["distortion"], float).ravel()
        R = np.asarray(P["rotation"], float)
        frame, _ = vertical_vp.load_frame(sess, c, "auto")
        d = room_vertical(frame, K, dist, R, np.asarray(P["translation"], float).reshape(3, 1), obj)
        ups[c] = d

    poses_pk, poses_fix = [], []
    for c in cams:
        P = params[c]
        R = np.asarray(P["rotation"], float); t = np.asarray(P["translation"], float).reshape(3)
        g_board_true = M.T @ np.array([0.0, 1.0, 0.0])        # hypothesis: true up = board up
        g_i = R.T @ ups[c]                                    # true up seen through this camera's pickle pose
        Dw = rot_between(g_i, g_board_true)                   # world rotation about b
        R_true = R @ Dw.T
        t_true = t + R @ b - R @ Dw.T @ b
        poses_pk.append(np.hstack([R, t.reshape(3, 1)]))
        poses_fix.append(np.hstack([R_true, t_true.reshape(3, 1)]))
        ang = np.degrees(np.arccos(np.clip(g_i @ g_board_true, -1, 1)))
        print(f"  Cam{c}: correction {ang:.2f} deg")

    settings_dir = os.path.join(sess, "MarkerData", "Settings")
    for trc in sorted(glob.glob(os.path.join(sess, "MarkerData", "*.trc"))):
        name = os.path.splitext(os.path.basename(trc))[0]
        if name == "neutral":
            continue
        st = yaml.safe_load(open(os.path.join(settings_dir, f"settings_{name}.yaml")))
        voff = float(st["verticalOffset"])
        t, m = read_trc(trc)
        keys = FEET
        X_os = np.stack([m[k] for k in keys], 1)              # (frames, k, 3) in m, OpenCap frame
        X_os_raw = X_os.copy(); X_os_raw[..., 1] += voff        # undo offset
        Xb = (X_os_raw.reshape(-1, 3) @ M) * 1000.0           # M^T applied row-wise -> board mm
        xs = []
        for Pm in poses_pk:
            Xc = Xb @ Pm[:, :3].T + Pm[:, 3]
            xs.append(Xc[:, :2] / Xc[:, 2:3])
        xs = np.array(xs)
        def to_os(Xb_):
            Y = (Xb_ / 1000.0) @ M.T
            Y[:, 1] -= voff
            return Y.reshape(X_os.shape)
        rec_pk = to_os(triangulate(poses_pk, xs))
        rec_fx = to_os(triangulate(poses_fix, xs))
        m_pk = {k: rec_pk[:, i] for i, k in enumerate(keys)}
        m_fx = {k: rec_fx[:, i] for i, k in enumerate(keys)}
        s0 = stance_slope(t, m)
        s_pk = stance_slope(t, m_pk)
        s_fx = stance_slope(t, m_fx)
        print(f"  {name:12s} slope original TRC {s0:+.2f} | re-triangulated, pickle poses {s_pk:+.2f} | corrected poses {s_fx:+.2f} deg"
              f"  (max |diff| pickle-retri vs TRC {1000*np.abs(rec_pk - X_os).max():.2f} mm)")


def room_vertical(frame, K, dist, R, t, obj):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_pts = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(-1, 2)
    und = cv2.undistort(gray, K, dist)
    segs = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV).detect(und)[0].reshape(-1, 4)
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    L = np.hypot(dx, dy)
    keep = (L > vertical_vp.MINLEN) & (np.degrees(np.arctan2(np.abs(dx), np.abs(dy))) < 15)
    x0, y0 = img_pts.min(0) - 40; x1, y1 = img_pts.max(0) + 40
    mid = (segs[:, :2] + segs[:, 2:]) / 2
    keep &= ~((mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1))
    S, Lw = segs[keep], L[keep]
    Kinv = np.linalg.inv(K)
    nrm = lambda p: (np.column_stack([p, np.ones(len(p))]) @ Kinv.T)[:, :2]
    Sn = np.column_stack([nrm(S[:, :2]), nrm(S[:, 2:])])
    _, lall = vertical_vp.vanishing_direction(Sn, np.ones(len(Sn)))
    rng = np.random.default_rng(0)
    best, best_in = -1, None
    for _ in range(3000):
        i, j = rng.choice(len(Sn), 2, replace=False)
        d = np.cross(lall[i], lall[j]); d /= np.linalg.norm(d)
        inl = vertical_vp.angular_residual(lall, d) < vertical_vp.THR
        if Lw[inl].sum() > best:
            best, best_in = Lw[inl].sum(), inl
    d, _ = vertical_vp.vanishing_direction(Sn[best_in], Lw[best_in])
    for _ in range(3):
        inl = vertical_vp.angular_residual(lall, d) < vertical_vp.THR
        d, _ = vertical_vp.vanishing_direction(Sn[inl], Lw[inl])
    return -d if d[1] > 0 else d


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print("==", os.path.basename(s.rstrip("/\\"))[12:20])
        main(s, None)
