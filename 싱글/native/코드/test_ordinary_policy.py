import unittest
import numpy as np
from ordinary_case_policy import convergence
from ordinary_policy_batch import route


class PolicyTests(unittest.TestCase):
    def test_low_sensor_residual_alone_does_not_stop(self):
        h=[dict(objective=1/(i+1),sensor_l2=.01) for i in range(11)]
        self.assertFalse(convergence(h,np.zeros((4,4)),np.zeros((4,4)))[0])

    def test_stable_loss_requires_stable_phase_and_good_sensor(self):
        h=[dict(objective=.0001,sensor_l2=.01) for _ in range(11)]
        self.assertTrue(convergence(h,np.zeros((4,4)),np.zeros((4,4)))[0])
        self.assertFalse(convergence(h,np.ones((4,4))*.1,np.zeros((4,4)))[0])
        h[-1]['sensor_l2']=.1
        self.assertFalse(convergence(h,np.zeros((4,4)),np.zeros((4,4)))[0])

    def test_rbc_never_dispatches_to_ordinary_pyramid(self):
        ordinary,rbc=route([dict(id='moon',name='Moon'),dict(id='triple',name='RBC Triple'),
                            dict(id='brightfield',name='RBC brightfield')])
        self.assertEqual([x['id'] for i,x in ordinary],['moon'])
        self.assertEqual(len(rbc),2)
        self.assertTrue(all(not x['ordinary_pyramid_dispatch'] for x in rbc))
        self.assertIn('Spectral Periodic',rbc[0]['method_family'])


if __name__=='__main__':unittest.main()
