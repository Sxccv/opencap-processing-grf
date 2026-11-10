"""Stance-floor slope per trial and pooled per session, for every OpenCap session given."""
import sys, glob, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stance_plane import read_trc, stance_points

def fit_x(P, L):
    names = sorted(set(L))
    A = np.column_stack([P[:, 0]] + [(L == n).astype(float) for n in names])
    c, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    r = P[:, 1] - A @ c
    return c[0], np.sqrt(np.mean(r**2))

for sess in sys.argv[1:]:
    trcs = sorted(f for f in glob.glob(os.path.join(sess, "MarkerData", "*.trc"))
                  if not f.endswith("neutral.trc"))
    print("=====", os.path.basename(sess.rstrip("/\\")))
    allP, allL = [], []
    for f in trcs:
        t, m = read_trc(f)
        P, L = stance_points(t, m, 0.15)
        hip = m["midHip"][:-1]
        if len(P) < 20:
            print(f"  {os.path.basename(f):14s} too few stance points"); continue
        s, rms = fit_x(P, L)
        print(f"  {os.path.basename(f):14s} n={len(P):4d} hip x {hip[0,0]:+.2f}->{hip[-1,0]:+.2f} z {hip[0,2]:+.2f}->{hip[-1,2]:+.2f}  "
              f"slope {s:+.4f} ({np.degrees(np.arctan(s)):+.2f} deg) rms {1000*rms:.1f} mm")
        allP.append(P); allL.append(L)
    if allP:
        s, rms = fit_x(np.vstack(allP), np.concatenate(allL))
        print(f"  POOLED (shared plane) slope {np.degrees(np.arctan(s)):+.2f} deg rms {1000*rms:.1f} mm")
