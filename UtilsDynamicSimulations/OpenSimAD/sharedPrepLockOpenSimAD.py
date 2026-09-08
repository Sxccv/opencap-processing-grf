"""Advisory cross-process lock for the one-time caches ``run_tracking`` builds.

Lives beside ``mainOpenSimAD.py`` because that is its only caller. Keeping it
here (rather than in a repo-root orchestration module) means the solver package
does not import upward into the scripts that drive it.

No solver imports, so this module stays importable in milliseconds and can be
unit-tested without loading CasADi/OpenSim.
"""
import os
import time

# Fitting muscle-tendon polynomials takes minutes; a waiter must outlast it.
DEFAULT_LOCK_TIMEOUT = 3600.0      # give up waiting after 1 h and proceed anyway
DEFAULT_LOCK_STALE_AFTER = 1800.0  # treat a 30-min-old lock as crashed and break it

# One name for the whole shared-prep region. It must NOT be parameterized by
# anything that varies per window: the caches below are keyed by trial, but
# which of them a window needs depends on that window's own range of motion, so
# a per-window lock name would let two windows into the region at once.
SHARED_PREP_LOCK_NAME = "shared_prep"


class SharedPrepLock:
    """Serialize one-time writes into a folder shared by concurrent solves.

    ``run_tracking`` builds several caches into the session's Model folder the
    first time a trial is solved: ``*_mtParameters_{l,r}.npy``,
    ``data4PolynomialFitting_*.npy``, ``*_polynomial_{l,r}_*.npy`` and the
    adjusted ``dummy_motion_<trial>.mot``. Every window recomputes them if they
    are absent, so under a parallel pool all workers would build them at once:
    they overwrite each other's files, each spawns its own ``joblib`` pool
    (oversubscribing the machine by workers x cores), and the fitting routine
    deletes *every* ``motion4MA_*`` scratch file in the folder when it finishes
    (``muscleDataOpenSimAD.getPolynomialData``), including the ones its
    neighbours are still reading.

    Serialising that region means the first worker builds the caches while the
    others wait, then they all load from disk. The lock is a plain O_EXCL file,
    so it works across processes on every platform and costs one create+unlink
    when nothing is contended (the serial case).

    A lock older than ``stale_after`` is assumed to belong to a crashed worker
    and is broken; waiting longer than ``timeout`` gives up and proceeds
    unlocked, which is no worse than not locking at all. ``held`` records which
    of the two happened, so a caller that cares can log it.
    """

    def __init__(self, lock_dir, name=SHARED_PREP_LOCK_NAME,
                 timeout=DEFAULT_LOCK_TIMEOUT,
                 stale_after=DEFAULT_LOCK_STALE_AFTER,
                 poll_interval=0.5):
        self.lock_path = os.path.join(lock_dir, f".{name}.lock")
        self.timeout = timeout
        self.stale_after = stale_after
        self.poll_interval = poll_interval
        self.held = False
        self._token = None

    def acquire(self):
        """Take the lock. Returns True if held, False if it gave up waiting."""
        token = f"{os.getpid()} {time.time()}".encode()
        deadline = time.time() + self.timeout
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        while True:
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, token)
                finally:
                    os.close(fd)
                self._token = token
                self.held = True
                return True
            except FileExistsError:
                self._break_if_stale()

            if time.time() >= deadline:
                print(f"WARNING: timed out waiting for {self.lock_path}; "
                      f"proceeding without it.")
                self.held = False
                return False
            time.sleep(self.poll_interval)

    def _break_if_stale(self):
        """Remove a lock left behind by a worker that died mid-build.

        The break is done by renaming to a private name first: only one of
        several waiters can win that rename, so two of them can never both
        conclude the lock is free and then both create it.
        """
        try:
            if time.time() - os.path.getmtime(self.lock_path) <= self.stale_after:
                return
        except OSError:
            return  # holder released it between our check and the stat

        victim = f"{self.lock_path}.stale.{os.getpid()}"
        try:
            os.rename(self.lock_path, victim)
        except OSError:
            return  # another waiter won the race, or the holder released it
        print(f"Breaking stale prep lock: {self.lock_path}")
        try:
            os.unlink(victim)
        except OSError:
            pass

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
            self.held = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False
