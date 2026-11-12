"""Fast K=1 routing/adapter contract tests, no B17 ZIP or optical FFT required."""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from 싱글.코드 import solver as u, pyramid as pmod
from 공통.object_models import RBC_MODELS, IMAGE_MODELS, image_phase, TAU

class RoutingTests(unittest.TestCase):
    def test_auto_by_scene(self):
        for scene in RBC_MODELS:
            self.assertEqual(u.resolve_method("auto",scene=scene),"spectral")
        for scene in IMAGE_MODELS:
            self.assertEqual(u.resolve_method("auto",scene=scene),"pyramid")
        self.assertEqual(u.resolve_method("pyramid",scene="triple"),"pyramid")
        self.assertEqual(u.resolve_method("compare",scene="camera"),"compare")
        self.assertEqual(u.resolve_method("periodic",scene="triple"),"periodic")

    def test_auto_measured(self):
        self.assertEqual(u.resolve_method("auto",object_type="rbc"),"spectral")
        self.assertEqual(u.resolve_method("auto",object_type="general"),"pyramid")
        with self.assertRaises(ValueError):
            u.resolve_method("auto")

    def test_invalid_cli_rejected_before_optics(self):
        cases=[
            ["recover","--mode","known","--sensor","s.npz","--output","out"],
            ["recover","--mode","blind","--wavelength","660","--sensor","s.npz","--output","out"],
            ["recover","--mode","known","--wavelength","660","--method","auto","--sensor","s.npz","--output","out"],
            ["single","--scene","custom","--true-wavelength","660","--mode","known","--wavelength","660","--output","out"],
            ["single","--scene","camera","--true-wavelength","700","--mode","known","--wavelength","660","--output","out"],
        ]
        for args in cases:
            with self.subTest(args=args),patch.object(u,"FullB17") as engine,contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as err:
                    u.main(args)
                self.assertEqual(err.exception.code,2)
                engine.assert_not_called()

    def test_default_single_routes_before_recovery(self):
        for scene,expected in [("triple","spectral"),("camera","pyramid")]:
            with self.subTest(scene=scene),tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp)/"run"
                with patch.object(u,"FullB17") as engine,patch.object(u,"generate"),patch.object(u,"recover") as recover,patch.object(u,"evaluate_selected"):
                    u.main(["single","--scene",scene,"--true-wavelength","660","--mode","known","--wavelength","660","--output",str(out)])
                    self.assertEqual(recover.call_args.args[-1],expected)

    def test_compare_same_observation_same_blind_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);obs=root/"sensor.npz";out=root/"inverse"
            y=np.ones((4,4))
            np.savez(obs,sensor=y)
            seen=[]
            def fake_fit(engine,observed,w,dest,method):
                seen.append((observed,w,method))
                self.assertTrue(dest.is_dir())
                return {"method":method,"sensor_sha256":u.digest(observed)}
            with patch.object(u,"sensor_check",side_effect=lambda y:y),patch.object(u,"scan_wavelength",return_value=(553,[{"wavelength_nm":553,"relative_l2":.1}])) as scan,patch.object(u,"fit_selected",side_effect=fake_fit):
                result=u.recover(object(),obs,out,"blind",None,"compare")
            scan.assert_called_once()
            self.assertIs(seen[0][0],seen[1][0])
            self.assertEqual([a[1:] for a in seen],[(553,"spectral"),(553,"pyramid")])
            self.assertEqual(result["spectral"]["sensor_sha256"],result["pyramid"]["sensor_sha256"])
            self.assertTrue((out/"comparison_recovery.json").exists())
            self.assertTrue((out/"blind_scan.json").exists())
            self.assertEqual(json.loads((out/"pyramid/recovery.json").read_text())["mode"],"blind")
            with self.assertRaises(FileExistsError):
                u.recover(object(),obs,out,"blind",None,"compare")

    def test_compare_evaluation_is_posthoc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"comparison_recovery.json").write_text("{}")
            with patch.object(u,"evaluate",side_effect=lambda t,s,d:{"method":d.name}) as e:
                result=u.evaluate_selected(Path("truth"),Path("sensor"),root)
            self.assertEqual(e.call_count,2)
            self.assertEqual(result["pyramid"]["method"],"pyramid")
            self.assertTrue((root/"comparison_evaluation.json").exists())

    def test_pyramid_adapter_wraps_optics_and_checkpoints(self):
        # Fake optical engine and legacy optimizer: this tests only the adapter.
        p=np.full((512,512),-0.4)
        y=np.ones((4,4))
        class Engine:
            archive_sha="test";source_fingerprint="test";inner_shas={}
            def optics(self,w):return None,None
            def value_grad(self,m,c,x,obs,alpha,grad):
                assert np.all(x>=0) and np.all(x<TAU)
                assert obs is y
                return (.04,np.zeros(x.size)) if grad else (.04,1.,y)
        legacy=types.ModuleType("cloud_k1_astronaut_multires")
        legacy.SCHEDULES={"pyramid64":((64,35),(128,35),(256,40),(512,40))}
        def gd(m,c,obs,vg,dest,schedule):
            vg(m,c,p,obs,0.,True)
            np.savez_compressed(dest/"gd_150_phase.npz",phase=p,gd_steps=150)
            return p,[{"gd_step":150,"sensor_loss":.04,"total_obj":.05,"variables_n":512}]
        legacy.gd_pyramid=gd
        reference=types.ModuleType("cloud_k1_endpoint_robust")
        def lb(m,c,obs,vg,dest,method,start):
            vg(m,c,start,obs,0.,True)
            np.savez_compressed(dest/"lbfgs_circular_checkpoint.npz",phase=p,step=150)
            return p,{"iterations":150,"termination":"test","history":[{"step":150,"objective":.05}]}
        reference.lbfgs=lb
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with patch.object(pmod,"gd_pyramid",side_effect=gd),patch.object(pmod,"lbfgs",side_effect=lb),patch.object(u,"sensor_check",return_value=y):
                result=pmod.fit_pyramid(Engine(),y,660,out)
            self.assertEqual(result["gd_iterations"],150)
            self.assertTrue(result["completed"])
            self.assertEqual(result["method"],"pyramid")
            for name in ("gd_150_phase.npz","gd_sealed.npz","recovered.npz","lbfgs_circular_checkpoint.npz"):
                with np.load(out/name,allow_pickle=False) as z:
                    self.assertTrue(np.allclose(z["phase"],TAU-.4))
            self.assertEqual(result["sensor_sha256"],u.digest(y))
            self.assertFalse(result["truth_in_inverse"])

    def test_new_builtin_images(self):
        for scene in ("checkerboard","astronaut"):
            p=image_phase(scene)
            self.assertEqual(p.shape,(512,512))
            self.assertTrue(np.isfinite(p).all())
            self.assertGreater(float(p.std()),0)
            self.assertGreaterEqual(float(p.min()),0)
            self.assertLessEqual(float(p.max()),TAU)

if __name__=="__main__":
    unittest.main(verbosity=2)
