"""Independent vertical per camera from the room's vertical lines, compared
with the vertical OpenCap takes from the wall checkerboard.

For each camera of a session:
  1. load K, distortion, R, t from cameraIntrinsicsExtrinsics.pickle
  2. take one full-resolution frame (calibration.mov if present, else neutral_sync.mp4)
  3. check the pickle reprojects the board corners onto the board (overlay saved)
  4. undistort, detect line segments (LSD), keep long near-vertical ones
  5. RANSAC + least squares vanishing direction d (camera frame) of those lines
  6. express d in the board frame, push it through OpenCap's own backWall
     rotation (main.py:334-341, trc.rotate order), and report the floor slope
     dy/dx that a floor perpendicular to d would show in the OpenCap TRC.

Usage: vertical_vp.py <session_dir> <out_dir> [frame_source]
"""
import os, sys, pickle
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

rng = np.random.default_rng(0)
THR = float(os.environ.get("VP_THR", "0.15"))
MINLEN = float(os.environ.get("VP_MINLEN", "40"))


def load_frame(sess, cam, source):
    base = os.path.join(sess, "Videos", f"Cam{cam}", "InputMedia")
    cands = []
    if source in ("auto", "calibration"):
        cands.append((os.path.join(base, "calibration", "calibration.mov"), "calibration"))
    if source in ("auto", "neutral"):
        cands.append((os.path.join(base, "neutral", "neutral_sync.mp4"), "neutral"))
    for path, tag in cands:
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # OpenCap samples the end of the calibration video (vidLength - 0.3 s)
        idx = max(n - 10, 0) if tag == "calibration" else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if ok:
            return frame, tag
    return None, None


def board_points(meta):
    w = meta["checkerBoard"]["black2BlackCornersWidth_n"]
    h = meta["checkerBoard"]["black2BlackCornersHeight_n"]
    sq = meta["checkerBoard"]["squareSideLength_mm"]
    obj = np.zeros((w * h, 3), np.float32)
    obj[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * sq   # utilsChecker.generate3Dgrid
    return obj


def vanishing_direction(segs_n, w):
    """Least-squares direction d minimising sum w (l.d)^2, l = normalised line in camera coords."""
    p1 = np.column_stack([segs_n[:, 0:2], np.ones(len(segs_n))])
    p2 = np.column_stack([segs_n[:, 2:4], np.ones(len(segs_n))])
    l = np.cross(p1, p2)
    l /= np.linalg.norm(l, axis=1, keepdims=True)
    M = (l * w[:, None]).T @ l
    vals, vecs = np.linalg.eigh(M)
    return vecs[:, 0], l


def angular_residual(l, d):
    # angle between the segment's interpretation plane and direction d
    return np.degrees(np.abs(np.arcsin(np.clip(l @ d, -1, 1))))


def opencap_rotation(upside_down):
    if upside_down:
        seq = [("y", -90)]
    else:
        seq = [("y", 90), ("z", 180)]
    M = np.eye(3)
    for ax, ang in seq:          # trc.rotate applies them in dict order
        M = Rot.from_euler(ax, ang, degrees=True).as_matrix() @ M
    return M


def main(sess, out, source="auto"):
    import yaml
    meta = yaml.safe_load(open(os.path.join(sess, "sessionMetadata.yaml")))
    obj = board_points(meta)
    tag_s = os.path.basename(sess.rstrip("/\\"))[12:20]
    cams = sorted(int(d[3:]) for d in os.listdir(os.path.join(sess, "Videos")) if d.startswith("Cam"))
    params = {c: pickle.load(open(os.path.join(sess, "Videos", f"Cam{c}", "cameraIntrinsicsExtrinsics.pickle"), "rb")) for c in cams}
    upside_down = np.asarray(params[cams[0]]["rotation"])[1, 1] < 0     # utilsChecker.isCheckerboardUpsideDown
    M = opencap_rotation(upside_down)
    print(f"== {tag_s} placement={meta['checkerBoard']['placement']} upsideDown={upside_down}")
    results = []
    for c in cams:
        P = params[c]
        K = np.asarray(P["intrinsicMat"], float)
        dist = np.asarray(P["distortion"], float).ravel()
        R = np.asarray(P["rotation"], float)
        t = np.asarray(P["translation"], float).reshape(3, 1)
        frame, ftag = load_frame(sess, c, source)
        if frame is None:
            print(f"  Cam{c}: no frame"); continue
        # 3. reprojection check
        img_pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(R)[0], t, K, dist)
        img_pts = img_pts.reshape(-1, 2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 4. undistort + LSD
        und = cv2.undistort(gray, K, dist)
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
        segs = lsd.detect(und)[0]
        segs = segs.reshape(-1, 4) if segs is not None else np.zeros((0, 4))
        dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        length = np.hypot(dx, dy)
        ang_from_vert = np.degrees(np.arctan2(np.abs(dx), np.abs(dy)))
        keep = (length > MINLEN) & (ang_from_vert < 15)
        # exclude the board itself (its edges would bias the answer toward the board's own frame)
        x0, y0 = img_pts.min(0) - 40; x1, y1 = img_pts.max(0) + 40
        mid = (segs[:, 0:2] + segs[:, 2:4]) / 2
        on_board = (mid[:, 0] > x0) & (mid[:, 0] < x1) & (mid[:, 1] > y0) & (mid[:, 1] < y1)
        keep &= ~on_board
        S, Lw = segs[keep], length[keep]
        Kinv = np.linalg.inv(K)
        def norm_pts(p):
            ph = np.column_stack([p, np.ones(len(p))]) @ Kinv.T
            return ph[:, :2]
        Sn = np.column_stack([norm_pts(S[:, 0:2]), norm_pts(S[:, 2:4])])
        # 5. RANSAC on pairs, then weighted LS on inliers
        best, best_in = None, None
        _, lall = vanishing_direction(Sn, np.ones(len(Sn)))
        for _ in range(3000):
            i, j = rng.choice(len(Sn), 2, replace=False)
            d = np.cross(lall[i], lall[j]); d /= np.linalg.norm(d)
            inl = angular_residual(lall, d) < THR
            score = Lw[inl].sum()
            if best is None or score > best:
                best, best_in = score, inl
        d, _ = vanishing_direction(Sn[best_in], Lw[best_in])
        for _ in range(3):   # re-select inliers around refined solution
            inl = angular_residual(lall, d) < THR
            d, _ = vanishing_direction(Sn[inl], Lw[inl])
        if d[1] > 0:
            d = -d                               # point up (camera y is down)
        # bootstrap over inlier segments
        idx = np.where(inl)[0]
        boot = []
        for _ in range(500):
            b = rng.choice(idx, len(idx), replace=True)
            db, _ = vanishing_direction(Sn[b], Lw[b])
            if db[1] > 0: db = -db
            boot.append(db)
        boot = np.array(boot)
        # 6. compare with board vertical and predict OpenCap floor slope
        g_board = R.T @ d                        # true up in board coords
        n_os = M @ g_board                       # true up in OpenCap/OpenSim coords
        slope = -n_os[0] / n_os[1]               # dy/dx of a floor perpendicular to true up
        slopes_b = []
        for db in boot:
            nb = M @ (R.T @ db); slopes_b.append(-nb[0] / nb[1])
        lat = -n_os[2] / n_os[1]                 # dy/dz
        board_up_cam = R @ (M.T @ np.array([0, 1, 0]))   # OpenCap's +Y in camera coords
        ang = np.degrees(np.arccos(np.clip(board_up_cam @ d, -1, 1)))
        cam_pitch = np.degrees(np.arcsin(np.clip(-d[2], -1, 1)))  # camera optical axis elevation rel. true horizontal
        res = dict(cam=c, frame=ftag, n_seg=int(inl.sum()), seg_len=float(Lw[inl].sum()),
                   slope_deg=float(np.degrees(np.arctan(slope))),
                   slope_sd=float(np.degrees(np.std(np.arctan(slopes_b)))),
                   lateral_deg=float(np.degrees(np.arctan(lat))),
                   angle_board_vs_true=float(ang), cam_pitch=float(cam_pitch))
        results.append(res)
        print(f"  Cam{c} [{ftag}] inlier segs {res['n_seg']:3d} (total {res['seg_len']:.0f} px) | "
              f"board-up vs room-vertical {ang:.2f} deg | predicted floor slope along X "
              f"{res['slope_deg']:+.2f} +/- {res['slope_sd']:.2f} deg, along Z {res['lateral_deg']:+.2f} deg | camera elevation {cam_pitch:+.1f} deg")
        # overlays
        vis = cv2.cvtColor(und, cv2.COLOR_GRAY2BGR)
        for s_, k in zip(segs[keep], inl):
            cv2.line(vis, tuple(map(int, s_[:2])), tuple(map(int, s_[2:])), (0, 255, 0) if k else (0, 0, 255), 2)
        for p in img_pts:
            cv2.circle(vis, tuple(map(int, p)), 3, (255, 0, 255), -1)
        cv2.imwrite(os.path.join(out, f"vp_{tag_s}_cam{c}_{ftag}.jpg"), cv2.resize(vis, (360, 640)))
        crop = frame[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)].copy()
        for p in img_pts:
            cv2.circle(crop, (int(p[0] - max(x0, 0)), int(p[1] - max(y0, 0))), 2, (255, 0, 255), -1)
        cv2.imwrite(os.path.join(out, f"reproj_{tag_s}_cam{c}_{ftag}.png"), cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))
    if results:
        s = np.array([r["slope_deg"] for r in results])
        print(f"  mean predicted slope over cameras {s.mean():+.2f} deg (spread {s.min():+.2f} .. {s.max():+.2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "auto")
