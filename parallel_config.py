"""Pure helpers for RAM-sized multi-window parallelism.

No solver imports here — this module must stay importable in milliseconds so it
can be unit-tested without loading CasADi/OpenSim.
"""
import glob
import os
import time

import numpy as np

# Measured IPOPT solve plateau is ~1.7-2.0 GB (serverside.md, Section 3).
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
    """Rebuild the shared optimaltrajectories.npy from per-window files.

    Called serially by the orchestrator after the parallel pool finishes, so the
    single shared write never races. Each per-window file is keyed by its own
    ``case``; we copy only that case's entry to avoid a stale in-file copy of a
    neighbour clobbering the neighbour's authoritative entry. Returns the written
    aggregate path, or None when there are no per-window files.
    """
    pattern = os.path.join(dyn_dir, f"optimaltrajectories_{trial_name}_window_*.npy")
    per_case_files = sorted(glob.glob(pattern))
    if not per_case_files:
        return None

    aggregate = {}
    prefix, suffix = "optimaltrajectories_", ".npy"
    for path in per_case_files:
        data = np.load(path, allow_pickle=True).item()
        case = os.path.basename(path)[len(prefix):-len(suffix)]  # "<trial>_window_<i>"
        if case in data:
            aggregate[case] = data[case]
        else:
            aggregate.update(data)  # unexpected shape: keep whatever is there

    out_path = os.path.join(dyn_dir, "optimaltrajectories.npy")
    np.save(out_path, aggregate)
    return out_path


# Fitting muscle-tendon polynomials takes minutes; a waiter must outlast it.
DEFAULT_LOCK_TIMEOUT = 3600.0     # give up waiting after 1 h and proceed anyway
DEFAULT_LOCK_STALE_AFTER = 1800.0  # treat a 30-min-old lock as crashed and break it


class SharedPrepLock:
    """Advisory cross-process lock for one-time writes into a shared folder.

    ``run_tracking`` builds several caches into the session's Model folder the
    first time a trial is solved: ``*_mtParameters_{l,r}.npy``,
    ``data4PolynomialFitting_*.npy``, ``*_polynomial_{l,r}_*.npy`` and the
    adjusted ``dummy_motion_<trial>.mot``. Every window recomputes them if they
    are absent, so under the parallel pool all workers would build them at once:
    they overwrite each other's files, each spawns its own ``joblib`` pool
    (oversubscribing the machine by workers x cores), and the fitting routine
    deletes *every* ``motion4MA_*`` scratch file in the folder when it finishes,
    including the ones its neighbours are still reading.

    Serialising that region means the first worker builds the caches while the
    others wait, then they all load from disk. The lock is a plain O_EXCL file,
    so it works across processes on every platform and costs one create+unlink
    when nothing is contended (the serial case).

    A lock older than ``stale_after`` is assumed to belong to a crashed worker
    and is broken; waiting longer than ``timeout`` gives up and proceeds
    unlocked, which is no worse than not locking at all.
    """

    def __init__(self, lock_dir, name="shared_prep",
                 timeout=DEFAULT_LOCK_TIMEOUT,
                 stale_after=DEFAULT_LOCK_STALE_AFTER,
                 poll_interval=0.5):
        self.lock_path = os.path.join(lock_dir, f".{name}.lock")
        self.timeout = timeout
        self.stale_after = stale_after
        self.poll_interval = poll_interval
        self._token = None

    def acquire(self):
        """Take the lock. Returns True if held, False if it gave up waiting."""
        token = f"{os.getpid()} {time.time()}".encode()
        deadline = time.time() + self.timeout
        while True:
            try:
                os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, token)
                finally:
                    os.close(fd)
                self._token = token
                return True
            except FileExistsError:
                pass

            # Break a lock left behind by a worker that died mid-build.
            try:
                if time.time() - os.path.getmtime(self.lock_path) > self.stale_after:
                    print(f"Breaking stale prep lock: {self.lock_path}")
                    os.unlink(self.lock_path)
                    continue
            except OSError:
                continue  # holder released it between our check and the stat

            if time.time() >= deadline:
                print(f"WARNING: timed out waiting for {self.lock_path}; "
                      f"proceeding without it.")
                return False
            time.sleep(self.poll_interval)

    def release(self):
        """Drop the lock, but only if this process still owns it."""
        if self._token is None:
            return
        try:
            # Read and close before unlinking: Windows refuses to remove a file
            # that still has an open handle.
            with open(self.lock_path, "rb") as f:
                owned = f.read() == self._token
            if owned:
                os.unlink(self.lock_path)
        except OSError:
            pass  # already gone, or broken as stale and retaken by someone else
        finally:
            self._token = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
