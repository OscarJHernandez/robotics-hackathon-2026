"""DAgger v2 guards: target placement, probe pinning, verified copies, per-machine job lock."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class TargetPlacementTest(unittest.TestCase):
    def test_defaults_reproduce_old_layouts(self):
        from nav.dagger.env import scene_layout
        from scene import HALF, TARGET_CLEARANCE, Arena
        arena = Arena(simple=True)
        try:
            for seed in range(5):
                a = arena.randomize(np.random.default_rng(seed))
                b = arena.randomize(np.random.default_rng(seed), TARGET_CLEARANCE, 0.3)
                self.assertEqual(a, b)
                # the old code: limit HALF - 0.3 and is_free(..., TARGET_CLEARANCE)
                tx, ty = a["target"]
                self.assertLessEqual(max(abs(tx), abs(ty)), HALF - 0.3)
                self.assertTrue(arena.is_free(tx, ty, TARGET_CLEARANCE))
            self.assertEqual(scene_layout(arena, 3), scene_layout(arena, 3, None, None))
        finally:
            arena.close()

    def test_wider_margins_hold(self):
        from nav.dagger.env import scene_layout
        from scene import HALF, Arena
        arena = Arena(simple=True)
        try:
            for scene in range(8):
                layout = scene_layout(arena, scene, 0.35, 0.45)
                tx, ty = layout["target"]
                self.assertLessEqual(max(abs(tx), abs(ty)), HALF - 0.45 + 1e-9)
                for o in layout["obstacles"]:
                    self.assertGreater(np.hypot(tx - o["x"], ty - o["y"]), o["radius"] + 0.35)
        finally:
            arena.close()

    def test_config_has_margins(self):
        from nav.dagger.config import load_config
        env = load_config()["env"]
        self.assertEqual((env["target_clearance"], env["target_wall"]), (0.35, 0.45))


class ProbePinTest(unittest.TestCase):
    def test_hash_mismatch_refuses_and_match_passes(self):
        from nav.dagger.featurize import Featurizer
        from nav.dagger.transfer import sha256
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "p.pt").write_bytes(b"not a probe")
            cfg = {"perception": "probe", "probe": "p.pt", "probe_sha256": "0" * 64, "patch_pool": 1,
                   "depth_cap": 10.0, "use_prev_command": False}
            with self.assertRaises(SystemExit) as ctx:
                Featurizer.from_config(cfg, Path(tmp))
            self.assertIn("SHA-256", str(ctx.exception))
            cfg["probe_sha256"] = sha256(Path(tmp) / "p.pt")
            try:  # the hash check passes; loading the fake file may fail afterwards, which is fine
                Featurizer.from_config(cfg, Path(tmp))
            except SystemExit as e:
                self.fail("hash check should pass: {}".format(e))
            except Exception:
                pass

    def test_pin_in_config(self):
        from nav.dagger.config import load_config
        self.assertEqual(len(load_config()["features"]["probe_sha256"]), 64)


class TransferTest(unittest.TestCase):
    def test_sha256(self):
        import hashlib
        from nav.dagger.transfer import sha256
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"hello")
            self.assertEqual(sha256(path), hashlib.sha256(b"hello").hexdigest())

    def test_commands(self):
        from nav.dagger import transfer
        cmd = transfer.scp_command("a", "h:b")
        self.assertEqual(cmd[0], "scp")
        self.assertIn("BatchMode=yes", cmd)
        hc = transfer.remote_hash_command("metacon", "data/x.pt")
        self.assertEqual(hc[0], "ssh")
        self.assertIn("BatchMode=yes", hc)
        self.assertIn("shasum -a 256", hc[-1])
        self.assertIn("sha256sum", hc[-1])
        self.assertIn("$HOME/Projects/robotics-hackathon-2026/data/x.pt", hc[-1])

    def test_push_verifies_and_raises_on_mismatch(self):
        from unittest import mock
        from nav.dagger import transfer
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f"
            path.write_bytes(b"hello")
            good = (transfer.sha256(path), 5)
            calls = []
            with mock.patch.object(transfer, "_run", side_effect=lambda c, allow_fail=False: calls.append(c) or ""):
                with mock.patch.object(transfer, "remote_info", return_value=good):
                    transfer.push("metacon", path, "data/f")
                with mock.patch.object(transfer, "remote_info", return_value=("0" * 64, 5)):
                    with self.assertRaises(RuntimeError):
                        transfer.push("metacon", path, "data/f")
                with mock.patch.object(transfer, "remote_info", return_value=(good[0], 4)):
                    with self.assertRaises(RuntimeError):
                        transfer.push("metacon", path, "data/f")
            self.assertTrue(any(c[0] == "scp" for c in calls))


class TransferSafetyTest(unittest.TestCase):
    def setUp(self):
        from unittest import mock
        from nav.dagger import transfer
        self.transfer, self.mock = transfer, mock
        self.calls = []
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def fake_run(self, cmd, allow_fail=False):
        self.calls.append(cmd)
        return ""

    def test_commands_are_lists_with_literal_remote(self):
        path = self.dir / "f"
        path.write_bytes(b"hello")
        with self.mock.patch.object(self.transfer, "_run", side_effect=self.fake_run), \
                self.mock.patch.object(self.transfer, "remote_info", return_value=(self.transfer.sha256(path), 5)):
            self.transfer.push("metacon", path, "data/f")
        scp = [c for c in self.calls if c[0] == "scp"][0]
        self.assertIsInstance(scp, list)
        self.assertEqual(scp[-1], "metacon:~/Projects/robotics-hackathon-2026/data/f.part")
        self.assertTrue(all(isinstance(c, list) and c[0] in ("scp", "ssh") for c in self.calls))
        self.assertEqual(self.transfer.remote_spec("h", "a/b"), "h:~/Projects/robotics-hackathon-2026/a/b")

    def test_push_uses_part_then_mv_and_cleans_up_on_mismatch(self):
        path = self.dir / "f"
        path.write_bytes(b"hello")
        good = (self.transfer.sha256(path), 5)
        with self.mock.patch.object(self.transfer, "_run", side_effect=self.fake_run), \
                self.mock.patch.object(self.transfer, "remote_info", return_value=good) as info:
            self.transfer.push("metacon", path, "data/f")
            info.assert_called_with("metacon", "data/f.part")
        self.assertIn("mv -f", self.calls[-1][-1])
        self.assertTrue(self.calls[-1][-1].rstrip().endswith('data/f"'))
        self.calls.clear()
        with self.mock.patch.object(self.transfer, "_run", side_effect=self.fake_run), \
                self.mock.patch.object(self.transfer, "remote_info", return_value=("0" * 64, 5)):
            with self.assertRaises(RuntimeError):
                self.transfer.push("metacon", path, "data/f")
        self.assertIn("rm -f", self.calls[-1][-1])
        self.assertFalse(any("mv -f" in c[-1] for c in self.calls if c[0] == "ssh"))

    def test_pull_verifies_part_then_replaces(self):
        target = self.dir / "sub" / "out.bin"

        def run(cmd, allow_fail=False):
            if cmd[0] == "scp":
                self.assertEqual(cmd[-1], str(target) + ".part")
                self.assertEqual(cmd[-2], "victory:~/Projects/robotics-hackathon-2026/data/out.bin")
                Path(cmd[-1]).write_bytes(b"hello")
            return ""

        import hashlib
        good = (hashlib.sha256(b"hello").hexdigest(), 5)
        with self.mock.patch.object(self.transfer, "_run", side_effect=run), \
                self.mock.patch.object(self.transfer, "list_remote", return_value=[]), \
                self.mock.patch.object(self.transfer, "remote_info", return_value=good):
            self.transfer.pull("victory", "data/out.bin", target)
        self.assertEqual(target.read_bytes(), b"hello")
        self.assertFalse(Path(str(target) + ".part").exists())
        target.unlink()
        with self.mock.patch.object(self.transfer, "_run", side_effect=run), \
                self.mock.patch.object(self.transfer, "list_remote", return_value=[]), \
                self.mock.patch.object(self.transfer, "remote_info", return_value=("0" * 64, 5)):
            with self.assertRaises(RuntimeError):
                self.transfer.pull("victory", "data/out.bin", target)
        self.assertFalse(target.exists())
        self.assertFalse(Path(str(target) + ".part").exists())

    def test_push_dir_skips_identical_and_copies_the_rest(self):
        (self.dir / "a.parquet").write_bytes(b"aaa")
        (self.dir / "b.parquet").write_bytes(b"bbbb")
        (self.dir / "c.txt").write_bytes(b"x")
        a = (self.transfer.sha256(self.dir / "a.parquet"), 3)

        def info(host, rel):
            if rel == "d/a.parquet":
                return a
            raise RuntimeError("missing")

        pushed = []
        with self.mock.patch.object(self.transfer, "remote_info", side_effect=info), \
                self.mock.patch.object(self.transfer, "push", side_effect=lambda h, p, r: pushed.append(r)):
            result = self.transfer.push_dir("h", self.dir, "d", patterns=("*.parquet",))
        self.assertEqual(result, [("a.parquet", 3, "skipped"), ("b.parquet", 4, "copied")])
        self.assertEqual(pushed, ["d/b.parquet"])

    def test_pull_dir(self):
        (self.dir / "a.bin").write_bytes(b"aaa")
        a = (self.transfer.sha256(self.dir / "a.bin"), 3)
        infos = {"d/a.bin": a, "d/b.bin": ("1" * 64, 9)}
        pulled = []
        with self.mock.patch.object(self.transfer, "list_remote", return_value=["a.bin", "b.bin", "sub/c.bin", "x.part"]), \
                self.mock.patch.object(self.transfer, "remote_info", side_effect=lambda h, r: infos[r]), \
                self.mock.patch.object(self.transfer, "pull", side_effect=lambda h, r, t: pulled.append(r)):
            result = self.transfer.pull_dir("h", "d", self.dir)
        self.assertEqual(result, [("a.bin", 3, "skipped"), ("b.bin", 9, "copied")])
        self.assertEqual(pulled, ["d/b.bin"])


class QueueRaceTest(unittest.TestCase):
    def test_threads_exactly_one_wins(self):
        import threading
        from nav.dagger.joblock import Busy, JobLock
        with tempfile.TemporaryDirectory() as tmp:
            barrier, results = threading.Barrier(8), []

            def work(i):
                lock = JobLock("hosty", tmp, log=lambda *_: None)
                barrier.wait()
                try:
                    lock.acquire("e{}".format(i))
                    results.append("won")
                except Busy:
                    results.append("busy")

            threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.assertEqual((results.count("won"), results.count("busy")), (1, 7))

    def test_dead_pid_same_host_taken_over_other_host_never(self):
        from nav.dagger.joblock import Busy, JobLock
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hostz.lock"
            dead = 2 ** 22 + 54321
            path.write_text(json.dumps({"pid": dead, "host": "hostz", "experiment": "old", "started": "t"}))
            lock = JobLock("hostz", tmp, log=lambda *_: None)
            lock.acquire("new")
            self.assertEqual(json.loads(path.read_text())["experiment"], "new")
            lock.release()
            self.assertFalse(path.exists())
            path.write_text(json.dumps({"pid": dead, "host": "elsewhere", "experiment": "far", "started": "t"}))
            with self.assertRaises(Busy):
                JobLock("hostz", tmp, log=lambda *_: None).acquire("new")
            self.assertEqual(json.loads(path.read_text())["experiment"], "far")

    def test_release_only_removes_own_lock(self):
        from nav.dagger.joblock import JobLock
        with tempfile.TemporaryDirectory() as tmp:
            lock = JobLock("hostw", tmp, log=lambda *_: None)
            lock.acquire("e")
            (Path(tmp) / "hostw.lock").write_text(json.dumps({"pid": 1, "host": "hostw", "experiment": "other"}))
            lock.release()
            self.assertTrue((Path(tmp) / "hostw.lock").exists())


class QueueTest(unittest.TestCase):
    def test_two_acquires_and_stale_takeover(self):
        from nav.dagger.joblock import Busy, JobLock
        with tempfile.TemporaryDirectory() as tmp:
            logs = []
            a = JobLock("hostx", tmp, log=logs.append)
            b = JobLock("hostx", tmp, log=logs.append)
            a.acquire("exp1")
            info = json.loads((Path(tmp) / "hostx.lock").read_text())
            self.assertEqual((info["pid"], info["experiment"]), (os.getpid(), "exp1"))
            with self.assertRaises(Busy):
                b.acquire("exp2")
            a.release()
            self.assertFalse((Path(tmp) / "hostx.lock").exists())
            with b:  # free again
                pass
            # stale: a lock written by a dead pid
            dead = 2 ** 22 + 12345
            (Path(tmp) / "hostx.lock").write_text(json.dumps({"pid": dead, "experiment": "old", "started": "t"}))
            b.acquire("exp3")
            self.assertTrue(any("stale" in line for line in logs))
            b.release()

    def test_context_manager(self):
        from nav.dagger.joblock import lock
        with tempfile.TemporaryDirectory() as tmp:
            with lock("e", "h", tmp):
                self.assertTrue((Path(tmp) / "h.lock").exists())
            self.assertFalse((Path(tmp) / "h.lock").exists())


if __name__ == "__main__":
    unittest.main()
