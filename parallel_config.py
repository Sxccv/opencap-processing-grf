"""Pure helpers for RAM-sized multi-window parallelism.

This module must stay importable in milliseconds so it
can be unit-tested without loading CasADi/OpenSim.

The cross-process lock that guards ``run_tracking``'s one-time caches lives in
``UtilsDynamicSimulations/OpenSimAD/sharedPrepLockOpenSimAD.py``, next to its
only caller, so the solver package does not import upward into this one.
"""
import glob
import os

import numpy as np

# Measured IPOPT solve plateau is ~1.7-2.0 GB.
DEFAULT_PER_WORKER_BYTES = 2 * 1024**3   # 2 GiB budget per concurrent window
DEFAULT_RESERVE_BYTES = 1 * 1024**3      # keep ~1 GB free (user requirement)


def compute_worker_count(
    available_bytes,
    cpu_count,
    num_windows,
    reserve_bytes=DEFAULT_RESERVE_BYTES,
    per_worker_bytes=DEFAULT_PER_WORKER_BYTES,
):
    """Number of windows to solve concurrently.

    Leave ``reserve_bytes`` of RAM free, give each worker ``per_worker_bytes``,
    and never exceed the CPU count or the number of windows. Always >= 1.
    """
    usable = available_bytes - reserve_bytes
    by_ram = int(usable // per_worker_bytes) if usable > 0 else 0
    return max(1, min(by_ram, cpu_count, num_windows))


def merge_optimaltrajectories(dyn_dir, trial_name):
    """Build the shared optimaltrajectories.npy from the per-case files.

    ``run_tracking`` only ever writes ``optimaltrajectories_<case>.npy``; the
    shared aggregate that ``plotResultsOpenSimAD`` and ``utilsKineticsOpenSimAD``
    read is derived from those, here, serially — so concurrent windows never
    contend for it. Returns the written aggregate path, or None when there are
    no per-case files.

    A per-case file that cannot be read is reported and skipped rather than
    taking down the whole merge: every other window's hours of solving are
    still worth recording.
    """
    pattern = os.path.join(dyn_dir, f"optimaltrajectories_{trial_name}_window_*.npy")
    per_case_files = sorted(glob.glob(pattern))
    if not per_case_files:
        return None

    aggregate = {}
    prefix, suffix = "optimaltrajectories_", ".npy"
    for path in per_case_files:
        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception as e:
            print(f"WARNING: skipping unreadable {os.path.basename(path)}: {e}")
            continue
        # "<trial>_window_<i>" — the case this file is authoritative for. Older
        # files may also carry copies of neighbouring cases; take only our own
        # so a stale copy cannot clobber its neighbour's real entry.
        case = os.path.basename(path)[len(prefix):-len(suffix)]
        if case in data:
            aggregate[case] = data[case]
        else:
            aggregate.update(data)  # unexpected shape: keep whatever is there

    if not aggregate:
        return None

    out_path = os.path.join(dyn_dir, "optimaltrajectories.npy")
    np.save(out_path, aggregate)
    return out_path
