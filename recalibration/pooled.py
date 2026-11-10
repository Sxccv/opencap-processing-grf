import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from stance_plane import read_trc, stance_points
vmax = 0.15
P, L, T = [], [], []
for k, f in enumerate(sys.argv[2:]):
    t, m = read_trc(f); p, l = stance_points(t, m, vmax)
    P.append(p); L.append(l); T.append(np.full(len(p), k))
P, L, T = np.vstack(P), np.concatenate(L), np.concatenate(T)
names = sorted(set(L))
def run(cols, tag):
    A = np.column_stack(cols); c, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None); r = P[:, 1] - A @ c
    print(f"{tag:42s} dy/dx {c[0]:+.4f} ({np.degrees(np.arctan(c[0])):+.2f} deg)  rms {1000*np.sqrt(np.mean(r**2)):.1f} mm"); return c, r
off = [(L == n).astype(float) for n in names]
run([P[:, 0]] + off, "shared plane, linear in x")
run([P[:, 0], P[:, 0]**2] + off, "shared plane, quadratic in x")
run([P[:, 0]] + off + [(T == k).astype(float) for k in range(1, 4)], "linear in x + per-trial offset")
c, r = run([P[:, 0]] + off, "residual by x-band (shared linear):")
for a in np.arange(0.5, 4.0, 0.5):
    m = (P[:, 0] >= a) & (P[:, 0] < a + 0.5)
    if m.any(): print(f"   x {a:.1f}-{a+0.5:.1f} m  n={m.sum():4d}  mean res {1000*r[m].mean():+5.1f} mm")
