import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
from rbc_native_solver import fit,spectral_operator


def objective(p,alpha):
    target=np.linspace(.05,.3,p.size).reshape(p.shape);d=p-target
    f=float(np.mean(d*d+.2*d**4))
    return f,(2*d+.8*d**3)/p.size,f,0.,1.


class RBCTests(unittest.TestCase):
    def test_full_resolution_memory_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            r=Path(td);weights=np.ones((12,12))
            fit(r/'full',objective,weights,'fixture',gd_budget=3,lbfgs_budget=4)
            fit(r/'split',objective,weights,'fixture',gd_budget=3,lbfgs_budget=4,until_updates=5)
            fit(r/'split',objective,weights,'fixture',gd_budget=3,lbfgs_budget=4)
            np.testing.assert_array_equal(np.load(r/'full/latest_phase.npy'),np.load(r/'split/latest_phase.npy'))
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                fit(r/'split',objective,weights,'changed',gd_budget=3,lbfgs_budget=4)

    def test_invalid_preconditioner_rejected(self):
        with self.assertRaises(ValueError):spectral_operator(-np.ones((8,8)))
        w=np.ones((8,8));w[1,0]=2
        with self.assertRaisesRegex(ValueError,'even'):spectral_operator(w)

    def test_phase_stays_periodic(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            fit(Path(td),objective,np.ones((12,12)),'fixture',gd_budget=2,lbfgs_budget=2)
            p=np.load(Path(td)/'latest_phase.npy')
            self.assertTrue(np.all((p>=0)&(p<2*np.pi)))


if __name__=='__main__':unittest.main()
