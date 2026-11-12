"""Contract and numerical tests; optional real full-native B17 forward smoke.

Run: python -m tests.test_contract
     python hyperspectral_phase/test_unified_simulator.py --full-forward
No reduced optical grids. These are not recovery accuracy experiments.
"""
import sys,unittest
import numpy as np
from scipy import fft
from common.object_models import N,TAU,wrap,image_phase,rbc_phase
from common.optics import GEOMETRY,GD_STEPS,LBFGS_STEPS,ALPHA,FullB17,phase_check,wavelength_check
from single.solver import prior,preconditioner
from common.spectrum_forward import multi_plan,continuum_plan,summed_forward

class Contract(unittest.TestCase):
    def test_fixed_contract(self):
        self.assertEqual(GEOMETRY,(3800,1429,512,3000))
        self.assertEqual((GD_STEPS,LBFGS_STEPS,ALPHA),(100,100,.015))
        for name in ("camera","camera_rot90","moon","brick","chelsea","coins"):
            p=image_phase(name)
            self.assertEqual(p.shape,(N,N))
            phase_check(wrap(p))
        for w in (440,550,660): self.assertEqual(wavelength_check(w),w)
        for w in (439,661,float("nan")):
            with self.assertRaises(ValueError): wavelength_check(w)
        with self.assertRaises(ValueError):phase_check(np.zeros((128,128)))
        with self.assertRaises(ValueError):phase_check(np.full((N,N),-.01))
    def test_circular_prior_and_preconditioner(self):
        rng=np.random.default_rng(19)
        p=rng.uniform(0,TAU,(N,N));v=rng.normal(size=p.shape)
        f,g=prior(p);eps=1e-5
        fd=(prior(p+eps*v,gradient=False)[0]-prior(p-eps*v,gradient=False)[0])/(2*eps)
        self.assertAlmostEqual(fd,float(g@v.ravel()),places=9)
        np.testing.assert_allclose(prior(p+TAU)[0],f,rtol=1e-12)
        for method in ("periodic","spectral"):
            a=v.ravel();b=preconditioner(method)(a)
            self.assertEqual(b.shape,a.shape);self.assertTrue(np.isfinite(b).all())
            self.assertGreater(float(a@b),0)
    def test_rbc_native_models(self):
        for kind,groups,layers in (("normal",100,1),("variable",100,1),("tilt",100,1),("double",75,2),("triple",50,3)):
            a=rbc_phase(kind,660)
            phase_check(a["phase"])
            self.assertEqual(a["labels"].shape,(N,N))
            self.assertEqual(len(np.unique(a["groupmap"][a["groupmap"]>0])),groups)
            self.assertEqual(int(a["labels"].max()),layers)
            self.assertEqual(len(a["diameter_um"]),100)
            np.testing.assert_allclose(a["phase"],wrap(TAU*a["opd_um"]/.660))
            self.assertTrue(np.any(a["labels"]==0))
    def test_spectral_plans(self):
        for k in (2,3):
            p=multi_plan([440,550,660][:k],[1]*k)
            self.assertEqual(p["K"],k)
            self.assertEqual(p["geometry"],list(GEOMETRY))
        for w,a in (([440],[1]),([440,440],[1,1]),([440,660],[1,-1]),([440,661],[1,1])):
            with self.assertRaises(ValueError):multi_plan(w,a)
        p=continuum_plan(440,660,11)
        self.assertIsNone(p["K"])
        self.assertAlmostEqual(sum(p["weights"]),220.)
        self.assertEqual(p["wavelengths_nm"][0],440.)
        self.assertEqual(p["wavelengths_nm"][-1],660.)
        # A synthetic bookkeeping stub checks intensity addition, not optics.
        class Stub:
            def predict(self,w,p):return np.full((3000,3000),w/1000)
        q=multi_plan([500,660],[1.,2.])
        y=summed_forward(Stub(),q,[np.zeros((N,N)),np.zeros((N,N))])
        self.assertEqual(y.shape,(3000,3000));self.assertAlmostEqual(float(y[0,0]),1.82)

def full_forward():
    """Compare the new wrapper with the frozen main at 660 nm on a full grid."""
    with fft.set_workers(2):
        engine=FullB17()
        from common.propagation import optics,measure
        p=wrap(image_phase("camera"))
        a=engine.predict(660,p)
        m,c=optics(660);b=measure(m,c,p)
        assert a.shape==b.shape==(3000,3000)
        assert np.isfinite(a).all() and a.min()>=0 and a.max()>0
        rel=float(np.linalg.norm(a-b)/np.linalg.norm(b))
        assert rel<1e-6,rel
        print("Native B17 forward smoke passed; relative difference:",rel,flush=True)

if __name__=="__main__":
    full="--full-forward" in sys.argv
    if full:sys.argv.remove("--full-forward")
    program=unittest.main(exit=False,verbosity=2)
    if not program.result.wasSuccessful():raise SystemExit(1)
    if full:full_forward()
