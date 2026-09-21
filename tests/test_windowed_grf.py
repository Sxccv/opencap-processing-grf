"""Tests for the pure logic of the windowed-GRF pipeline.

Everything here is importable without CasADi, OpenSim or the OpenCap API, so
the suite runs in milliseconds. The solver itself is not covered - a single
window takes 5-15 minutes - but the concurrency arithmetic, the file layout,
the .mot reader and the coverage maths are, and those are what a refactor is
most likely to break silently.

Run with:  python -m pytest tests -q
"""
import importlib.util
import json
import multiprocessing
import os
import sys
import time

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "UtilsDynamicSimulations", "OpenSimAD"))

import pipeline_io
from parallel_config import (DEFAULT_PER_WORKER_BYTES, DEFAULT_RESERVE_BYTES,
                             compute_worker_count, merge_optimaltrajectories)
from sharedPrepLockOpenSimAD import SharedPrepLock

GiB = 1024 ** 3


def _load_numbered_script(filename, module_name):
    """Import a script whose name starts with a digit (not a valid identifier)."""
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(REPO_ROOT, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_sim = _load_numbered_script("02_run_grf_simulation.py", "run_grf_simulation")
build_csv = _load_numbered_script("03_build_grf_csv.py", "build_grf_csv")


# %% compute_worker_count

class TestComputeWorkerCount:
    def test_ram_is_the_binding_constraint(self):
        # 16 GiB - 1 GiB reserve = 15 GiB, at 2 GiB each -> 7 workers.
        assert compute_worker_count(16 * GiB, cpu_count=32, num_windows=32) == 7

    def test_cpu_count_caps_the_pool(self):
        assert compute_worker_count(256 * GiB, cpu_count=4, num_windows=32) == 4

    def test_never_more_workers_than_windows(self):
        assert compute_worker_count(256 * GiB, cpu_count=64, num_windows=3) == 3

    @pytest.mark.parametrize("available", [0, DEFAULT_RESERVE_BYTES,
                                           DEFAULT_RESERVE_BYTES + 1])
    def test_always_at_least_one_worker(self, available):
        """Below the reserve there is no RAM budget at all, but refusing to run
        would be worse than running one window at a time."""
        assert compute_worker_count(available, cpu_count=8, num_windows=8) == 1

    def test_reserve_is_actually_withheld(self):
        exactly_two = DEFAULT_RESERVE_BYTES + 2 * DEFAULT_PER_WORKER_BYTES
        assert compute_worker_count(exactly_two, 64, 64) == 2
        assert compute_worker_count(exactly_two - 1, 64, 64) == 1


# %% merge_optimaltrajectories

class TestMergeOptimalTrajectories:
    def _write_case(self, dyn_dir, case, payload):
        np.save(pipeline_io.trajectories_path(dyn_dir, case), payload)

    def test_returns_none_when_there_is_nothing_to_merge(self, tmp_path):
        assert merge_optimaltrajectories(str(tmp_path), "Trial") is None

    def test_collects_every_window_into_one_aggregate(self, tmp_path):
        dyn = str(tmp_path)
        for i in range(3):
            case = pipeline_io.case_name("Trial", i)
            self._write_case(dyn, case, {case: {"iter": i}})

        out = merge_optimaltrajectories(dyn, "Trial")
        merged = np.load(out, allow_pickle=True).item()
        assert set(merged) == {f"Trial_window_{i}" for i in range(3)}
        assert merged["Trial_window_2"]["iter"] == 2

    def test_a_stale_copy_of_a_neighbour_never_wins(self, tmp_path):
        """Older per-case files can carry a snapshot of their neighbours. Only
        the file named for a case is authoritative for that case."""
        dyn = str(tmp_path)
        self._write_case(dyn, "Trial_window_0", {
            "Trial_window_0": {"iter": 0},
            "Trial_window_1": {"iter": -1},  # stale snapshot
        })
        self._write_case(dyn, "Trial_window_1", {"Trial_window_1": {"iter": 1}})

        merged = np.load(merge_optimaltrajectories(dyn, "Trial"),
                         allow_pickle=True).item()
        assert merged["Trial_window_1"]["iter"] == 1

    def test_one_corrupt_file_does_not_lose_the_others(self, tmp_path):
        """Hours of solving already happened; a bad file must not discard it."""
        dyn = str(tmp_path)
        self._write_case(dyn, "Trial_window_0", {"Trial_window_0": {"iter": 0}})
        with open(pipeline_io.trajectories_path(dyn, "Trial_window_1"), "wb") as f:
            f.write(b"not a npy file at all")

        merged = np.load(merge_optimaltrajectories(dyn, "Trial"),
                         allow_pickle=True).item()
        assert set(merged) == {"Trial_window_0"}

    def test_other_trials_in_the_same_folder_are_ignored(self, tmp_path):
        dyn = str(tmp_path)
        self._write_case(dyn, "Trial_window_0", {"Trial_window_0": {"iter": 0}})
        self._write_case(dyn, "Other_window_0", {"Other_window_0": {"iter": 9}})

        merged = np.load(merge_optimaltrajectories(dyn, "Trial"),
                         allow_pickle=True).item()
        assert set(merged) == {"Trial_window_0"}


# %% SharedPrepLock

def _grab_lock(lock_dir, hold_seconds, queue):
    """Child-process body: take the lock, report success, hold it briefly."""
    lock = SharedPrepLock(lock_dir, timeout=5.0, poll_interval=0.01)
    acquired = lock.acquire()
    queue.put((os.getpid(), acquired, time.time()))
    time.sleep(hold_seconds)
    lock.release()


class TestSharedPrepLock:
    def test_uncontended_acquire_and_release(self, tmp_path):
        lock = SharedPrepLock(str(tmp_path))
        assert lock.acquire() is True
        assert lock.held
        assert os.path.exists(lock.lock_path)
        lock.release()
        assert not lock.held
        assert not os.path.exists(lock.lock_path)

    def test_context_manager_releases_even_on_error(self, tmp_path):
        lock_path = None
        with pytest.raises(RuntimeError):
            with SharedPrepLock(str(tmp_path)) as lock:
                lock_path = lock.lock_path
                raise RuntimeError("boom")
        assert not os.path.exists(lock_path)

    def test_a_second_holder_is_excluded(self, tmp_path):
        with SharedPrepLock(str(tmp_path)):
            other = SharedPrepLock(str(tmp_path), timeout=0.05,
                                   poll_interval=0.01)
            assert other.acquire() is False
            assert other.held is False

    def test_release_does_not_remove_a_lock_we_no_longer_own(self, tmp_path):
        """After a timeout broke ours and someone else retook it, releasing
        must not delete the new owner's lock."""
        mine = SharedPrepLock(str(tmp_path))
        mine.acquire()
        with open(mine.lock_path, "wb") as f:
            f.write(b"9999 0.0")  # somebody else's token
        mine.release()
        assert os.path.exists(mine.lock_path)

    def test_a_stale_lock_is_broken_not_waited_on(self, tmp_path):
        lock_path = os.path.join(str(tmp_path), ".shared_prep.lock")
        with open(lock_path, "wb") as f:
            f.write(b"1 0.0")
        os.utime(lock_path, (time.time() - 10_000, time.time() - 10_000))

        lock = SharedPrepLock(str(tmp_path), stale_after=60.0, timeout=1.0,
                              poll_interval=0.01)
        assert lock.acquire() is True
        lock.release()

    def test_an_unreadable_lock_still_honours_the_timeout(self, tmp_path, monkeypatch):
        """Regression: a getmtime that keeps failing used to `continue` past the
        deadline check, spinning forever at 100% CPU instead of giving up."""
        lock_path = os.path.join(str(tmp_path), ".shared_prep.lock")
        with open(lock_path, "wb") as f:
            f.write(b"1 0.0")

        def always_fails(_path):
            raise PermissionError("mtime unavailable")

        monkeypatch.setattr(os.path, "getmtime", always_fails)

        lock = SharedPrepLock(str(tmp_path), timeout=0.2, poll_interval=0.01)
        started = time.time()
        assert lock.acquire() is False
        assert time.time() - started < 5.0

    def test_only_one_of_many_processes_holds_it_at_a_time(self, tmp_path):
        """The property the whole class exists for, across real processes."""
        queue = multiprocessing.Queue()
        procs = [multiprocessing.Process(target=_grab_lock,
                                         args=(str(tmp_path), 0.15, queue))
                 for _ in range(4)]
        for p in procs:
            p.start()
        acquisitions = [queue.get(timeout=30) for _ in procs]
        for p in procs:
            p.join(timeout=30)

        assert all(acquired for _, acquired, _ in acquisitions)
        # Serialised: each waiter got in only after the previous one let go.
        times = sorted(t for _, _, t in acquisitions)
        assert times[-1] - times[0] >= 0.15 * (len(procs) - 1) * 0.5


# %% build_windows (02)

class TestBuildWindows:
    def test_whole_seconds_tile_the_range(self):
        assert run_sim.build_windows(0.0, 3.0) == [[0, 1], [1, 2], [2, 3]]

    def test_the_last_window_is_clipped_to_the_end(self):
        assert run_sim.build_windows(0.0, 2.6)[-1] == [2.0, 2.6]

    def test_a_short_tail_is_merged_into_its_neighbour(self):
        """A 0.1 s window is too short to solve; it joins the one before it."""
        windows = run_sim.build_windows(0.0, 2.1)
        assert windows == [[0.0, 1.0], [1.0, 2.1]]

    def test_a_single_short_window_is_kept(self):
        """Nothing to merge into - the trial is shorter than one step."""
        assert run_sim.build_windows(0.0, 0.3) == [[0.0, 0.3]]

    def test_windows_are_contiguous_and_cover_the_range(self):
        windows = run_sim.build_windows(0.0, 7.3)
        assert windows[0][0] == 0.0
        assert windows[-1][1] == 7.3
        for earlier, later in zip(windows, windows[1:]):
            assert earlier[1] == later[0]


# %% manifest_row (02)

class TestManifestRow:
    def _result(self, **kw):
        from pipeline_io import WindowResult
        defaults = dict(index=0, time_start=0.0, time_end=1.0, converged=True,
                        return_status="Solve_Succeeded")
        defaults.update(kw)
        return WindowResult(**defaults)

    def test_grf_path_is_stored_relative_to_the_session(self, tmp_path):
        session_root = str(tmp_path / "OpenCapData_x")
        grf = os.path.join(session_root, "OpenSimData", "Dynamics", "T", "g.mot")
        row = run_sim.manifest_row(self._result(grf_path=grf), session_root)
        assert row["grf_resultant_path"] == "OpenSimData/Dynamics/T/g.mot"

    def test_a_path_outside_the_session_stays_absolute(self, tmp_path):
        session_root = str(tmp_path / "OpenCapData_x")
        outside = str(tmp_path / "elsewhere" / "g.mot")
        row = run_sim.manifest_row(self._result(grf_path=outside), session_root)
        assert row["grf_resultant_path"] == outside

    def test_a_failed_window_records_why(self, tmp_path):
        row = run_sim.manifest_row(
            self._result(converged=False, return_status="Maximum_Iterations",
                         failure_reason="IPOPT did not converge"),
            str(tmp_path))
        assert row["converged"] is False
        assert row["failure_reason"] == "IPOPT did not converge"
        assert row["grf_resultant_path"] is None

    def test_forwards_all_seven_qc_fields_when_present(self, tmp_path):
        """The manifest is the only thing 03 ever reads - every QC field the
        worker measured must survive the trip through manifest_row."""
        row = run_sim.manifest_row(
            self._result(mean_vertical_grf=564.5, peak_vertical_grf=732.0,
                         min_clearance_m=-0.015, frac_frames_within_5mm=1.0,
                         dedrift_method="time",
                         dynamics_consistency_residual_N=12.3, flip_count=2),
            str(tmp_path))
        assert row["mean_vertical_grf"] == 564.5
        assert row["peak_vertical_grf"] == 732.0
        assert row["min_clearance_m"] == -0.015
        assert row["frac_frames_within_5mm"] == 1.0
        assert row["dedrift_method"] == "time"
        assert row["dynamics_consistency_residual_N"] == 12.3
        assert row["flip_count"] == 2

    def test_an_old_style_result_with_no_qc_fields_still_produces_a_row(
            self, tmp_path):
        """A WindowResult that never set any of the seven new fields (the old
        shape) must not crash manifest_row - it should just read back None."""
        row = run_sim.manifest_row(self._result(), str(tmp_path))
        for key in ("mean_vertical_grf", "peak_vertical_grf",
                    "min_clearance_m", "frac_frames_within_5mm",
                    "dedrift_method", "dynamics_consistency_residual_N",
                    "flip_count"):
            assert row[key] is None


# %% Coverage maths (03)

class TestCoverageMaths:
    def _w(self, start, end, index=0):
        return {"index": index, "time_start": start, "time_end": end}

    def test_adjacent_windows_merge_into_one_range(self):
        windows = [self._w(0, 1), self._w(1, 2), self._w(2, 3)]
        assert build_csv.merge_time_ranges(windows) == [[0, 3]]

    def test_a_missing_window_splits_the_range(self):
        windows = [self._w(0, 1), self._w(2, 3)]
        assert build_csv.merge_time_ranges(windows) == [[0, 1], [2, 3]]

    def test_windows_are_sorted_before_merging(self):
        windows = [self._w(2, 3), self._w(0, 1), self._w(1, 2)]
        assert build_csv.merge_time_ranges(windows) == [[0, 3]]

    def test_gaps_are_the_complement_of_the_coverage(self):
        assert build_csv.compute_gaps([[1, 2], [3, 4]], 0, 5) == \
            [[0, 1], [2, 3], [4, 5]]

    def test_full_coverage_has_no_gaps(self):
        assert build_csv.compute_gaps([[0, 5]], 0, 5) == []

    def test_no_coverage_is_one_whole_gap(self):
        assert build_csv.compute_gaps([], 0, 5) == [[0, 5]]

    def test_a_flagged_but_converged_window_becomes_a_gap_not_coverage(self):
        """A window can be IPOPT-converged (it is in `loaded` at all) and
        still be physiologically flagged; its span must fall out of
        valid_ranges into gap_ranges/flagged_ranges, mirroring what 03's
        main() does with `loaded` after splitting on window_physio_flags."""
        import pandas as pd

        w0 = self._w(0.0, 1.0, index=0)
        w0.update(peak_vertical_grf=700.0, mean_vertical_grf=600.0,
                  frac_frames_within_5mm=1.0, min_clearance_m=-0.01)
        w1 = self._w(1.0, 2.0, index=1)
        w1.update(peak_vertical_grf=2400.0, mean_vertical_grf=300.0,
                  frac_frames_within_5mm=1.0, min_clearance_m=-0.01)
        df0 = pd.DataFrame({"time": [0.0, 0.5]})
        df1 = pd.DataFrame({"time": [1.0, 1.5, 2.0]})
        loaded = [(w0, df0), (w1, df1)]

        mass_kg = 58.0
        flags = {w["index"]: build_csv.window_physio_flags(w, mass_kg, {})
                 for w, _ in loaded}
        assert flags[0] == []
        assert flags[1] != []  # implausible peak (2400 N > 2x BW)

        passing = [(w, df) for w, df in loaded if not flags[w["index"]]]
        flagged = [(w, df) for w, df in loaded if flags[w["index"]]]

        valid_ranges = build_csv.merge_time_ranges(
            build_csv.exported_spans(passing))
        gaps = build_csv.compute_gaps(valid_ranges, 0.0, 2.0)
        flagged_ranges = [
            {"index": w["index"], "time_start": float(df["time"].min()),
             "time_end": float(df["time"].max()), "reasons": flags[w["index"]]}
            for w, df in flagged if not df.empty
        ]

        assert valid_ranges == [[0.0, 0.5]]
        assert gaps == [[0.5, 2.0]]
        assert len(flagged_ranges) == 1
        assert flagged_ranges[0]["index"] == 1
        assert flagged_ranges[0]["time_start"] == 1.0
        assert flagged_ranges[0]["time_end"] == 2.0
        # And, critically, window 1's span is nowhere inside valid_ranges.
        assert not any(vs <= 1.5 <= ve for vs, ve in valid_ranges)


# %% pipeline_io

class TestPipelineIO:
    def test_every_path_hangs_off_the_session_folder(self):
        dyn = pipeline_io.dynamics_dir("/data", "OpenCapData_x", "T")
        assert dyn == os.path.join("/data", "OpenCapData_x", "OpenSimData",
                                   "Dynamics", "T")
        assert pipeline_io.manifest_path("/data", "OpenCapData_x", "T") == \
            os.path.join(dyn, "window_manifest_T.json")

    def test_case_naming_matches_the_files_run_tracking_writes(self):
        case = pipeline_io.case_name("T", 3)
        assert case == "T_window_3"
        assert pipeline_io.grf_resultant_path("/d", "T", case).endswith(
            "GRF_resultant_T_T_window_3.mot")
        assert pipeline_io.stats_path("/d", case).endswith("stats_T_window_3.npy")
        assert pipeline_io.trajectories_path("/d", case).endswith(
            "optimaltrajectories_T_window_3.npy")

    def test_reads_a_mot_file_past_its_header(self, tmp_path):
        mot = tmp_path / "t.mot"
        mot.write_text("Coordinates\nversion=1\nnRows=2\nendheader\n"
                       "time\tfx\tfy\n0.0\t1.0\t2.0\n0.5\t3.0\t4.0\n")
        df = pipeline_io.read_mot(str(mot))
        assert list(df.columns) == ["time", "fx", "fy"]
        assert df["fy"].tolist() == [2.0, 4.0]
        assert pipeline_io.mot_time_range(str(mot)) == (0.0, 0.5)

    def test_an_unusable_file_reports_no_time_range(self, tmp_path):
        assert pipeline_io.mot_time_range(str(tmp_path / "absent.mot")) == \
            (None, None)
        empty = tmp_path / "empty.mot"
        empty.write_text("endheader\ntime\n")
        assert pipeline_io.mot_time_range(str(empty)) == (None, None)


# %% load_converged_windows (03)

class TestLoadConvergedWindows:
    def _session(self, tmp_path, windows, files):
        session_id = "OpenCapData_x"
        dyn = pipeline_io.dynamics_dir(str(tmp_path), session_id, "T")
        os.makedirs(dyn, exist_ok=True)
        for name, body in files.items():
            with open(os.path.join(dyn, name), "w") as f:
                f.write(body)
        return session_id, {"trial_name": "T", "windows": windows}

    _GOOD_MOT = ("endheader\ntime\tground_force_right_vy\n"
                 "0.0\t100.0\n0.5\t200.0\n")

    def test_unconverged_windows_are_skipped(self, tmp_path):
        session_id, manifest = self._session(
            tmp_path,
            [{"index": 0, "time_start": 0, "time_end": 1, "converged": False,
              "grf_resultant_path": "OpenSimData/Dynamics/T/a.mot"}],
            {"a.mot": self._GOOD_MOT})
        assert build_csv.load_converged_windows(
            manifest, str(tmp_path), session_id) == []

    def test_a_window_whose_file_vanished_is_dropped(self, tmp_path, capsys):
        session_id, manifest = self._session(
            tmp_path,
            [{"index": 7, "time_start": 0, "time_end": 1, "converged": True,
              "grf_resultant_path": "OpenSimData/Dynamics/T/gone.mot"}],
            {})
        assert build_csv.load_converged_windows(
            manifest, str(tmp_path), session_id) == []
        assert "window 7 GRF file missing" in capsys.readouterr().out

    def test_each_mot_is_parsed_exactly_once(self, tmp_path, monkeypatch):
        """Reading was duplicated between the all-zero check and the concat."""
        session_id, manifest = self._session(
            tmp_path,
            [{"index": 0, "time_start": 0, "time_end": 1, "converged": True,
              "grf_resultant_path": "OpenSimData/Dynamics/T/a.mot"}],
            {"a.mot": self._GOOD_MOT})

        reads = []
        real_read = pipeline_io.read_mot
        monkeypatch.setattr(build_csv.pipeline_io, "read_mot",
                            lambda p: (reads.append(p), real_read(p))[1])

        loaded = build_csv.load_converged_windows(
            manifest, str(tmp_path), session_id)
        build_csv.concatenate_grf_frames([df for _, df in loaded])
        assert len(reads) == 1


# %% window_physio_flags (03) - the physiological gates replacing warn_all_zero
#
# Real measured values from the 2026-09-15 run, mass_kg = 58.0
# (BW = 58.0 * 9.80665 = 568.79 N).

class TestPhysioGates:
    MASS_KG = 58.0

    def test_a_plausible_window_is_not_flagged(self):
        """window 0: mean 564.5, peak 732.0, min_clearance_m -0.015, frac 1.0."""
        w = {"peak_vertical_grf": 732.0, "mean_vertical_grf": 564.5,
             "frac_frames_within_5mm": 1.0, "min_clearance_m": -0.015}
        assert build_csv.window_physio_flags(w, self.MASS_KG, {}) == []

    def test_implausibly_high_peak_is_flagged(self):
        """window 3: mean 321.9, peak 2367.7 (4.16x BW) -> flagged."""
        w = {"peak_vertical_grf": 2367.7, "mean_vertical_grf": 321.9,
             "frac_frames_within_5mm": None, "min_clearance_m": None}
        reasons = build_csv.window_physio_flags(w, self.MASS_KG, {})
        assert len(reasons) == 1
        assert "2367.7" in reasons[0]
        assert "implausible" in reasons[0]

    def test_near_zero_peak_produced_nothing_is_flagged(self):
        """window 5: mean 0.0, peak 0.1, clearance 0.089 -> flagged."""
        w = {"peak_vertical_grf": 0.1, "mean_vertical_grf": 0.0,
             "frac_frames_within_5mm": None, "min_clearance_m": 0.089}
        reasons = build_csv.window_physio_flags(w, self.MASS_KG, {})
        assert any("0.1" in r and "produced nothing" in r for r in reasons)

    def test_not_groundable_is_caught_only_by_the_clearance_gate(self):
        """window 4: mean 14.5, peak 151.7, min_clearance_m 0.072,
        frac_frames_within_5mm 0.0.

        The mean (14.5 N) is below 20% BW, but frac is 0 (never within 5mm),
        so the mean/stance contradiction gate must NOT fire - a low mean
        force while genuinely airborne is not a contradiction. The peak
        (151.7 N) is 27% BW, well under the 2x BW ceiling, so the
        implausible-peak gate must NOT fire either. Only the clearance gate
        (0.072 m > 0.005 m) should catch this window. If this test fails,
        the gate set has a hole: a window that is not physically groundable
        but whose peak/mean happen to look plausible would slip through.
        """
        w = {"peak_vertical_grf": 151.7, "mean_vertical_grf": 14.5,
             "frac_frames_within_5mm": 0.0, "min_clearance_m": 0.072}
        reasons = build_csv.window_physio_flags(w, self.MASS_KG, {})
        assert len(reasons) == 1
        assert "72.0 mm" in reasons[0]
        assert "not physically groundable" in reasons[0]

    def test_mean_stance_contradiction_gate_fires_alone(self):
        """Synthesised: mean 50.0, peak 300.0, frac 0.9, clearance -0.010.

        Grounded (frac 0.9 >= 0.20) but the mean force is far below 20% BW -
        a contradiction. Peak and clearance are both comfortably inside their
        own gates, so only the contradiction gate should fire.
        """
        w = {"peak_vertical_grf": 300.0, "mean_vertical_grf": 50.0,
             "frac_frames_within_5mm": 0.9, "min_clearance_m": -0.010}
        reasons = build_csv.window_physio_flags(w, self.MASS_KG, {})
        assert len(reasons) == 1
        assert "50.0" in reasons[0]
        assert "contradiction" in reasons[0]

    def test_all_none_inputs_are_safe_and_unflagged(self):
        """A window with every metric missing (e.g. an old-style manifest
        row) must not be flagged and must not raise."""
        w = {"peak_vertical_grf": None, "mean_vertical_grf": None,
             "frac_frames_within_5mm": None, "min_clearance_m": None}
        assert build_csv.window_physio_flags(w, self.MASS_KG, {}) == []


# %% concatenate_grf_frames (03)

class TestConcatenateGrfFrames:
    def test_boundary_duplicates_are_dropped_and_rows_sorted(self):
        import pandas as pd
        a = pd.DataFrame({"time": [1.0, 0.0], "fy": [10.0, 0.0]})
        b = pd.DataFrame({"time": [1.0, 2.0], "fy": [10.0, 20.0]})
        out = build_csv.concatenate_grf_frames([a, b])
        assert out["time"].tolist() == [0.0, 1.0, 2.0]

    def test_no_frames_yields_an_empty_frame(self):
        assert build_csv.concatenate_grf_frames([]).empty

    def test_valid_and_flag_reason_columns_survive_concatenation(self):
        """The merged CSV carries `valid`/`flag_reason` per-row; they must
        make it through concat, the time sort and the boundary de-dup."""
        import pandas as pd
        a = pd.DataFrame({"time": [0.0, 1.0], "fy": [1.0, 2.0],
                          "valid": [True, True], "flag_reason": ["", ""]})
        # t=1.0 is the shared boundary sample: identical on both sides, so
        # which copy de-dup keeps does not change the expected result.
        b = pd.DataFrame({"time": [1.0, 2.0], "fy": [2.0, 3.0],
                          "valid": [True, False],
                          "flag_reason": ["", "peak too high"]})
        out = build_csv.concatenate_grf_frames([a, b])
        assert out["time"].tolist() == [0.0, 1.0, 2.0]
        assert out["valid"].tolist() == [True, True, False]
        assert out["flag_reason"].tolist() == ["", "", "peak too high"]


# %% The manifest contract between 02 and 03

def test_02_writes_the_manifest_shape_03_reads(tmp_path):
    """The two scripts only meet through this file; keep them agreeing."""
    from pipeline_io import WindowResult

    session_id = "OpenCapData_x"
    dyn = pipeline_io.dynamics_dir(str(tmp_path), session_id, "T")
    os.makedirs(dyn)
    with open(os.path.join(dyn, "GRF_resultant_T_T_window_0.mot"), "w") as f:
        f.write("endheader\ntime\tground_force_right_vy\n0.0\t100.0\n")

    session_root = pipeline_io.session_dir(str(tmp_path), session_id)
    row = run_sim.manifest_row(
        WindowResult(0, 0.0, 1.0, converged=True,
                     return_status="Solve_Succeeded",
                     grf_path=pipeline_io.grf_resultant_path(
                         dyn, "T", pipeline_io.case_name("T", 0))),
        session_root)

    manifest_file = pipeline_io.manifest_path(str(tmp_path), session_id, "T")
    qc_marker = {"qc_version": 1, "kinematics_sha256": "abc123"}
    thresholds = {"peak_vertical_grf_bw_zero_max": 0.02,
                 "mean_vertical_grf_bw_min": 0.20,
                 "frac_frames_within_5mm_min": 0.20,
                 "peak_vertical_grf_bw_max": 2.0,
                 "clearance_gate_m": 0.005}
    with open(manifest_file, "w") as f:
        json.dump({"trial_name": "T", "session_id": session_id,
                   "windows": [row], "mass_kg": 58.0,
                   "thresholds": thresholds, "qc_marker": qc_marker}, f)

    with open(manifest_file) as f:
        manifest_json = json.load(f)
    loaded = build_csv.load_converged_windows(
        manifest_json, str(tmp_path), session_id)
    assert len(loaded) == 1
    assert loaded[0][1]["ground_force_right_vy"].tolist() == [100.0]

    # 03's consumer reads mass_kg/thresholds/qc_marker straight off the
    # manifest dict written by 02 - pin that contract too.
    assert manifest_json["mass_kg"] == 58.0
    assert manifest_json["thresholds"] == thresholds
    assert manifest_json["qc_marker"] == qc_marker
    assert build_csv.window_physio_flags(
        manifest_json["windows"][0], manifest_json["mass_kg"],
        manifest_json["thresholds"]) == []
