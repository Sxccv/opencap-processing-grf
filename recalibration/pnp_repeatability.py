"""Frame-by-frame repeat of OpenCap's extrinsic solve on a static calibration video.

Replicates utilsChecker.calcExtrinsics: 4x INTER_AREA upsample, findChessboardCorners
(ADAPTIVE_THRESH), cornerSubPix (11,11), /4, solvePnPGeneric(SOLVEPNP_IPPE) -> two
solutions. The board and cameras are static, so every frame should give the same pose;
the scatter is the repeatability of OpenCap's single-image estimate.

For each frame, the solution closest to the stored pickle pose is kept, and the
board-defined vertical is converted to the OpenCap-frame floor slope dy/dx it implies
against a fixed reference vertical (the room-line vanishing direction from
vertical_vp.py, recomputed here on the last frame). Only the frame-to-frame scatter and
the offset from the pickle matter here, not the reference itself.

Also reports the Cramer-Rao-style standard deviation of the board's tilt about its
horizontal axis for corner noise sigma, from the PnP Jacobian at the pickle pose.
"""
import os, sys, pickle
import numpy as np, cv2, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vertical_vp import board_points, opencap_rotation


def corners_like_opencap(gray, dims, up=4):
    big = cv2.resize(gray, (gray.shape[1] * up, gray.shape[0] * up), interpolation=cv2.INTER_AREA)
    ret, c = cv2.findChessboardCorners(big, dims, cv2.CALIB_CB_ADAPTIVE_THRESH)
    if not ret:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(big, c, (11, 11), (-1, -1), crit) / up


def rot_angle(Ra, Rb):
    return np.degrees(np.linalg.norm(cv2.Rodrigues(Ra @ Rb.T)[0]))


def tilt_about_board_x(R_ref, R):
    """Signed rotation (deg) of the board frame about its own x (horizontal) axis, R vs R_ref."""
    dR = R_ref.T @ R          # board-frame relative rotation
    rv = cv2.Rodrigues(dR)[0].ravel()
    return np.degrees(rv)     # components about board x, y, z


def main(sess, sigma_px=(0.1, 0.25, 0.5)):
    meta = yaml.safe_load(open(os.path.join(sess, "sessionMetadata.yaml")))
    obj = board_points(meta)
    dims = (meta["checkerBoard"]["black2BlackCornersWidth_n"], meta["checkerBoard"]["black2BlackCornersHeight_n"])
    tag = os.path.basename(sess.rstrip("/\\"))[12:20]
    print(f"== {tag}")
    for cam in range(3):
        vid = os.path.join(sess, "Videos", f"Cam{cam}", "InputMedia", "calibration", "calibration.mov")
        if not os.path.exists(vid):
            print(f"  Cam{cam}: no calibration video"); continue
        P = pickle.load(open(os.path.join(sess, "Videos", f"Cam{cam}", "cameraIntrinsicsExtrinsics.pickle"), "rb"))
        K = np.asarray(P["intrinsicMat"], float); dist = np.asarray(P["distortion"], float).ravel()
        Rp = np.asarray(P["rotation"], float); tp = np.asarray(P["translation"], float).reshape(3)
        cap = cv2.VideoCapture(vid)
        rows, gap, sizes = [], [], []
        STEP = int(os.environ.get("PNP_STEP", "1")); fi = -1
        while True:
            ok, fr = cap.read(); fi += 1
            if not ok:
                break
            if fi % STEP:
                continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            c = corners_like_opencap(g, dims)
            if c is None:
                rows.append(None); continue
            n, rvecs, tvecs, err = cv2.solvePnPGeneric(obj, c, K, dist, flags=cv2.SOLVEPNP_IPPE)
            Rs = [cv2.Rodrigues(r)[0] for r in rvecs]
            k = int(np.argmin([rot_angle(R, Rp) for R in Rs]))
            rows.append(Rs[k])
            if n == 2:
                gap.append(rot_angle(Rs[0], Rs[1]))
            pts = c.reshape(-1, 2); sizes.append(np.ptp(pts[:, 0]))
        ok = [r for r in rows if r is not None]
        if not ok:
            print(f"  Cam{cam}: board not found in any frame"); continue
        comps = np.array([tilt_about_board_x(Rp, R) for R in ok])   # deg about board x,y,z
        dist_to_pickle = np.array([rot_angle(R, Rp) for R in ok])
        # Jacobian-based SD of tilt about board x at the pickle pose
        rv0 = cv2.Rodrigues(Rp)[0]
        _, J = cv2.projectPoints(obj, rv0, tp.reshape(3, 1), K, dist)
        Jp = J[:, :6]                                   # d(uv)/d(rvec, tvec)
        cov = np.linalg.inv(Jp.T @ Jp)                  # per unit pixel variance
        # map rvec (camera-frame rotation vector) covariance to board-x tilt: delta_board = Rp^T delta_cam
        Rb = Rp.T
        cov_board = Rb @ cov[:3, :3] @ Rb.T
        sd_unit = np.degrees(np.sqrt(np.diag(cov_board)))
        print(f"  Cam{cam}: board found in {len(ok)}/{len(rows)} frames, inner-corner span {np.mean(sizes):.0f} px wide; "
              f"IPPE two-solution gap {np.median(gap):.1f} deg (median)")
        print(f"     frame-to-frame SD of board rotation about board x/y/z: "
              f"{comps[:,0].std():.2f} / {comps[:,1].std():.2f} / {comps[:,2].std():.2f} deg; "
              f"range about x {comps[:,0].min():+.2f}..{comps[:,0].max():+.2f}; max angle from pickle {dist_to_pickle.max():.2f} deg")
        print("     Jacobian SD about board x/y/z per corner-noise sigma: " +
              "; ".join(f"{s} px -> {s*sd_unit[0]:.2f}/{s*sd_unit[1]:.2f}/{s*sd_unit[2]:.2f} deg" for s in sigma_px))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        main(s)
