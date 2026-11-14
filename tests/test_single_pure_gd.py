"""Numerical unit checks only; no claim of full B17 reconstruction."""
import unittest
import numpy as np
from 싱글.코드.pure_gd import descent
from 공통.object_models import TAU

class DescentTests(unittest.TestCase):
    def test_periodic_armijo(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(0,TAU,(16,16))
        target = rng.uniform(0,TAU,x.shape)
        def fg(p,grad):
            return float(np.sum(2*np.sin((p-target)/2)**2)), np.sin(p-target) if grad else None
        initial = fg(x,False)[0]
        for _ in range(20):
            f,g = fg(x,True)
            nx,step,back = descent(fg,x,f,g,-g,10.,cap=np.pi/2)
            self.assertTrue(np.all((nx>=0)&(nx<TAU)))
            self.assertLessEqual(fg(nx,False)[0],f+1e-12)
            x = nx
        self.assertLess(fg(x,False)[0],initial*1e-5)
        f,g = fg(x,True)
        with self.assertRaises(RuntimeError):
            descent(fg,x,f,g,g,1.)

if __name__ == "__main__":
    unittest.main()
