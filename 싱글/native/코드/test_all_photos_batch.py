import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import all_photos_batch as batch


class BatchTests(unittest.TestCase):
    def test_completed_skip_and_rbc_last(self):
        entries=[{'id':'normal','name':'RBC Normal'},{'id':'moon','name':'Moon'},
                 {'id':'tank','name':'Tank'},{'id':'boat','name':'Boat'}]
        summary={'ordinary':[dict(id='boat',known=True,wavelength_nm=660,gd_accepted=150,
                    lbfgs_accepted=60,final_circular_rmse_rad=.1,sensor_l2=.01)],
                 'precision_recovery_experiments':[dict(scene='tank',quality_pass=True,
                    wavelength_nm=660,recovery_float64_lbfgs_accepted=60)]}
        cases,skipped=batch.plan(entries,summary)
        self.assertEqual([x['id'] for x in cases],['moon','normal'])
        self.assertEqual(skipped,['boat','tank'])

    def test_child_failure_is_recorded_and_next_case_runs(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);archive=root/'input.zip';archive.write_bytes(b'fixture')
            np.save(root/'a.npy',np.zeros((512,512)))
            entries=[dict(id=x,name=x,phase='a.npy') for x in ['fixture_one','fixture_two']]
            args=['batch','--input-archive',str(archive),'--optics-archive',str(archive),'--output',str(root/'out')]
            with patch.object(sys,'argv',args), \
                 patch.object(batch.b,'EXPECTED_OPTICS',batch.b.sha256_file(archive)), \
                 patch.object(batch.b,'inspect_input',return_value=(root,root/'manifest.json',entries)), \
                 patch.object(batch.b,'invoke',side_effect=RuntimeError('simulated child failure')) as invoke:
                batch.main()
            self.assertEqual(invoke.call_count,2)
            queue=batch.b.read_json(root/'out/queue.json')
            self.assertEqual(queue['status'],'completed_with_failures')
            self.assertEqual(len(queue['results']),2)
            self.assertTrue(all(x['status']=='failed' for x in queue['results']))


if __name__=='__main__':unittest.main()
