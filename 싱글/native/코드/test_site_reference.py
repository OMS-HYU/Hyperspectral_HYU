import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import site_intensity_run as runner
from site_reference_figure import render_arrays


class Objective:
    def fg(self,p,sensor,alpha=0):
        target=np.linspace(.1,.7,p.size).reshape(p.shape)
        d=p-target;value=float(np.mean(d*d+.2*d**4))
        return value,(2*d+.8*d**3)/p.size,value,0.,1.


class ReferenceTests(unittest.TestCase):
    def test_multiresolution_resume_matches_continuous(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(runner,'PLAN',[(64,1),(128,1),(256,1),(512,1)]):
            root=Path(td);full=root/'full';split=root/'split';full.mkdir();split.mkdir()
            runner.gd(full,Objective(),np.ones((2,2)),'fixture',4)
            runner.gd(split,Objective(),np.ones((2,2)),'fixture',2)
            runner.gd(split,Objective(),np.ones((2,2)),'fixture',4)
            np.testing.assert_array_equal(np.load(full/'latest_phase.npy'),np.load(split/'latest_phase.npy'))
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                runner.gd(split,Objective(),np.ones((2,2)),'changed',4)

    def test_amplitude_cannot_be_relabelled_as_intensity(self):
        a=np.zeros((2,2))
        with self.assertRaisesRegex(ValueError,'do not square amplitude'):
            render_arrays(a,a,a,a,Path('unused.png'),{'geometry':[5713,2857,512,3000],'observation':'amplitude'})


if __name__=='__main__':unittest.main()
