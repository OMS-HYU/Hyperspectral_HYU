import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import native_cli as cli
from tank_precision_recovery import solve


class PrecisionRecoveryTests(unittest.TestCase):
    @staticmethod
    def objective(x):
        d=x-np.linspace(.1,.8,x.size).reshape(x.shape)
        f=float(np.mean(d*d+.2*d**4))
        return f,(2*d+.8*d**3)/x.size,f,0.,1.

    def test_resume_matches_uninterrupted_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td)
            p=np.zeros((8,8))
            solve(root/'full',p,self.objective,{'test':1},4)
            solve(root/'split',p,self.objective,{'test':1},2)
            solve(root/'split',p,self.objective,{'test':1},4)
            np.testing.assert_array_equal(np.load(root/'full/phase_004.npy'),np.load(root/'split/phase_004.npy'))
            np.testing.assert_array_equal(p,np.zeros((8,8)))
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                solve(root/'split',p,self.objective,{'test':2},5)

    def test_convergence_does_not_invent_iterations(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            out=Path(td)
            count=solve(out,np.zeros((8,8)),lambda p:(0.,np.zeros_like(p),0.,0.,1.),{},60)
            self.assertEqual(count,0)
            self.assertIn('optimizer terminated',cli.loadj(out/'progress.json')['termination'])
            self.assertFalse((out/'phase_060.npy').exists())


if __name__=='__main__':
    unittest.main()
