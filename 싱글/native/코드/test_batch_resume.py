"""Control-flow tests with a small deterministic objective, not optical validation."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import batch_runner as batch
import native_cli as cli


class Objective:
    count = 0
    adj = 0

    def fg(self, p, y, alpha):
        self.count += 1
        self.adj += 1
        target = np.linspace(.1, .8, p.size).reshape(p.shape)
        delta = p - target
        value = float(np.mean(delta**2 + .2*delta**4))
        grad = (2*delta + .8*delta**3)/p.size
        return value, grad, value, 0., 1.


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive.zip'
        self.archive.write_bytes(b'fixture')
        self.sensor = np.ones((2, 2), dtype=np.float32)
        self.man = dict(scene='fixture', wavelength_nm=660., geometry=[5713,2857,512,3000],
                        sensor_sha256=cli.sha(self.sensor))

    def tearDown(self):
        self.temp.cleanup()

    def run_solver(self, folder, command, count):
        folder.mkdir(exist_ok=True)
        args = SimpleNamespace(run=folder, archive=self.archive, steps=count, until=count, alpha=.0001)
        with patch.object(cli, 'inputs', return_value=(folder,self.man,self.sensor)), \
             patch.object(cli, 'model', return_value=Objective()), contextlib.redirect_stdout(io.StringIO()):
            getattr(cli, command)(args)

    def test_gd_resume_repairs_torn_views_and_matches_continuous(self):
        full, split = self.root/'full', self.root/'split'
        self.run_solver(full, 'gd', 2)
        self.run_solver(split, 'gd', 1)
        for name in ('history.json','progress.json','latest_phase.npy'):
            (split/name).unlink()
        self.run_solver(split, 'gd', 1)
        np.testing.assert_array_equal(np.load(full/'latest_phase.npy'),np.load(split/'latest_phase.npy'))
        self.assertEqual(batch.read_json(split/'progress.json')['gd_accepted'],2)

    def prepare_lbfgs(self, path):
        path.mkdir()
        p = np.zeros((512,512), dtype=np.float64)
        np.save(path/'gd_150.npy', p)
        batch.write_json(path/'progress.json',dict(gd_accepted=150,phase_sha256=cli.sha(p)))

    def test_lbfgs_full_memory_resume_matches_continuous(self):
        full, split = self.root/'full', self.root/'split'
        for path in (full,split): self.prepare_lbfgs(path)
        self.run_solver(full,'lbfgs',2)
        self.run_solver(split,'lbfgs',1)
        out = split/'long_alpha_0.0001'
        for name in ('history.json','progress.json','phase_001.npy'): (out/name).unlink()
        self.run_solver(split,'lbfgs',2)
        np.testing.assert_array_equal(np.load(full/'long_alpha_0.0001/phase_002.npy'),np.load(out/'phase_002.npy'))
        self.assertEqual(batch.read_json(out/'progress.json')['accepted'],2)

    def test_solver_change_blocks_resume(self):
        path = self.root/'run'
        self.run_solver(path,'gd',1)
        config = batch.read_json(path/'native_gd_config.json')
        config['solver_sha256'] = 'changed'
        batch.write_json(path/'native_gd_config.json',config)
        with self.assertRaisesRegex(ValueError,'configuration mismatch'): self.run_solver(path,'gd',1)

    def test_lock_excludes_duplicate_runner_and_releases(self):
        path = self.root/'lock'
        with batch.batch_lock(path):
            with self.assertRaises(OSError):
                with batch.batch_lock(path): pass
        with batch.batch_lock(path): pass

    def test_manifest_rejects_path_escape(self):
        with self.assertRaises(ValueError): batch.case_entries([dict(id='../escape')])

    def test_child_crash_is_detected(self):
        import sys
        with self.assertRaisesRegex(RuntimeError,'exit code 7'):
            batch.invoke([sys.executable,'-c','raise SystemExit(7)'],self.root/'child.log')


if __name__ == '__main__': unittest.main()
