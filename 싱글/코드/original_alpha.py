"""Independent equal-budget alpha continuation on the original B17 BL=.5 profile.
Regenerate exact sealed sensor and verify hashes. Phase truth only generates
simulated data and evaluates after inverse. No hardware or numerical changes.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import phase_check,digest,write_json
from 공통.object_models import wrap,image_phase
from 싱글.코드 import solver,wavelength_jacobian as jac
from 싱글.코드.original_geometry import OriginalProfile
from 싱글.코드.alpha_continuation import EXPECTED_TRUTH
from 싱글.코드.grid_convergence import spectrum_error

BASELINE_RUN=43609194176
BASELINE_RMSE=.09272863035423533
ALPHAS=(.005,.003,.001,.0003,.0001,0.)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--alpha",type=float,choices=ALPHAS,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    obs=root/"observation";obs.mkdir()
    base=a.baseline/"alpha005"
    rec=json.loads((base/"recovery.json").read_text())
    ev=json.loads((base/"evaluation.json").read_text())
    gen=json.loads((a.baseline/"observation/generation.json").read_text())
    cfg=json.loads((a.baseline/"experiment.json").read_text())
    assert rec["completed"] and rec["lbfgs_iterations"]==100 and rec["previous_lbfgs_iterations"]==100
    assert rec["alpha"]==.005 and rec["wavelength_nm"]==660.
    assert abs(ev["full_rmse_rad"]-BASELINE_RMSE)<1e-12
    assert cfg["sasm_blfactor"]==.5
    with np.load(base/"recovered.npz",allow_pickle=False) as z:
        start=phase_check(z["phase"]);assert float(z["wavelength_nm"])==660.
    assert digest(start)==rec["phase_sha256"]
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05");w=engine.w
        assert engine.config()==cfg
        truth=phase_check(wrap(image_phase("camera")))
        assert digest(truth)==EXPECTED_TRUTH==gen["truth_sha256"]
        clean=engine.predict(w,truth)
        sigma=.01*float(clean.mean())
        y=np.maximum(0.,clean+sigma*np.random.default_rng(672906).standard_normal(clean.shape))
        assert digest(clean)==gen["clean_sha256"]
        assert digest(y)==gen["sensor_sha256"]==rec["sensor_sha256"]
        np.savez_compressed(obs/"sensor_only.npz",sensor=y)
        np.savez_compressed(obs/"private_truth.npz",phase=truth,actual_wavelength_nm=w)
        write_json(obs/"generation.json",gen)
        del truth,clean;gc.collect()
        write_json(root/"experiment.json",dict(physical_profile=cfg,alpha=a.alpha,baseline_run=BASELINE_RUN,
            baseline_rmse_rad=BASELINE_RMSE,initial_phase_sha256=digest(start),
            sensor_sha256=digest(y),additional_budget=100,prior_gd_iterations=150,
            prior_lbfgs_iterations=200,truth_in_inverse=False))
        out=root/"inverse";P=solver.preconditioner("spectral")
        final=jac.refine(engine,y,w,start,out,a.alpha,"sealed_original_BL05_alpha005",P,
                         "original_BL05_alpha_continuation","fixed historical 660nm spectral P")
        rr=json.loads((out/"recovery.json").read_text())
        assert rr["sensor_sha256"]==rec["sensor_sha256"] and rr["initial_phase_sha256"]==rec["phase_sha256"]
        rr.update(physical_profile=cfg,previous_lbfgs_iterations=200,
                  total_lbfgs_iterations=200+rr["lbfgs_iterations"],baseline_run=BASELINE_RUN)
        write_json(out/"recovery.json",rr)
        score=solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
        with np.load(obs/"private_truth.npz",allow_pickle=False) as z:truth=phase_check(z["phase"])
        m,cam=engine.optics(w)
        def diagnostics(x):
            data,q,_=engine.value_grad(m,cam,x,y,0.,False)
            prior=solver.prior(x,1.,False)[0]
            off=float(np.angle(np.mean(np.exp(1j*(x-truth)))))
            e=np.angle(np.exp(1j*(x-truth-off)))
            return dict(rmse_rad=float(np.sqrt(np.mean(e**2))),sensor_l2=float(np.sqrt(data)),
                unweighted_prior=float(prior),objective=float(data+a.alpha*prior),scale=float(q),
                error_spectrum=spectrum_error(e))
        d=dict(alpha=a.alpha,baseline=diagnostics(start),final=diagnostics(final),truth=diagnostics(truth),
               score=score,accepted_additional_iterations=rr["lbfgs_iterations"],
               termination=rr["termination"],same_sensor_sha256=digest(y),
               same_initial_phase_sha256=digest(start),physical_profile=cfg)
        write_json(root/"diagnostics.json",d)
        print("ORIGINAL_ALPHA_FINAL",json.dumps(d),flush=True)
if __name__=="__main__":main()
