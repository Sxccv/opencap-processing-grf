"""Fit the floor implied by stance foot markers in OpenCap TRC files.

If the drift is a rigid tilt of the world frame (or a real incline), stance
markers from every trial lie on one tilted plane with small residuals.
"""
import sys, numpy as np

FEET = ["r_calc_study", "L_calc_study", "r_toe_study", "L_toe_study",
        "r_5meta_study", "L_5meta_study"]

def read_trc(path):
    lines = open(path).read().splitlines()
    names = [n for n in lines[3].split("\t")[2:] if n]
    data = np.genfromtxt(lines[5:], delimiter="\t")
    data = data[~np.isnan(data[:, 1])]
    t = data[:, 1]
    return t, {n: data[:, 2 + 3*i: 5 + 3*i] for i, n in enumerate(names)}

def stance_points(t, m, vmax):
    pts, labels = [], []
    for name in FEET:
        p = m[name][:-1]                      # drop trailing glitch frame
        v = np.linalg.norm(np.gradient(p, t[:-1], axis=0), axis=1)
        idx = np.where(v < vmax)[0]
        pts.append(p[idx]); labels += [name] * len(idx)
    return np.vstack(pts), np.array(labels)

def fit(P, labels, quad=False):
    # per-marker offsets (calc/toe/5meta sit at different heights) + shared plane
    names = sorted(set(labels))
    cols = [P[:, 0], P[:, 2]]
    if quad: cols += [P[:, 0]**2]
    cols += [(labels == n).astype(float) for n in names]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    res = P[:, 1] - A @ coef
    return coef, res

def main():
    vmax = float(sys.argv[1]); files = sys.argv[2:]
    allP, allL, allT = [], [], []
    for f in files:
        t, m = read_trc(f)
        P, L = stance_points(t, m, vmax)
        c, r = fit(P, L)
        hip = m["midHip"][:-1]
        print(f"{f.split('/')[-1]:14s} n={len(P):4d} x {hip[0,0]:+.2f}->{hip[-1,0]:+.2f} z {hip[0,2]:+.2f}->{hip[-1,2]:+.2f} "
              f"| slope dy/dx {c[0]:+.4f} ({np.degrees(np.arctan(c[0])):+.2f} deg)  dy/dz {c[1]:+.4f} ({np.degrees(np.arctan(c[1])):+.2f} deg)  rms {1000*np.sqrt(np.mean(r**2)):.1f} mm")
        allP.append(P); allL.append(L); allT.append(np.full(len(P), len(allT)))
    P, L = np.vstack(allP), np.concatenate(allL)
    for quad in (False, True):
        c, r = fit(P, L, quad)
        extra = f" x^2 {c[2]:+.4f}" if quad else ""
        print(f"POOLED quad={quad}: dy/dx {c[0]:+.4f} ({np.degrees(np.arctan(c[0])):+.2f} deg) dy/dz {c[1]:+.4f} ({np.degrees(np.arctan(c[1])):+.2f} deg){extra} rms {1000*np.sqrt(np.mean(r**2)):.1f} mm  p95|res| {1000*np.percentile(abs(r),95):.1f} mm")

if __name__ == "__main__":
    main()
