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
