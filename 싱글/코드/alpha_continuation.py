"""Controlled regularization continuation on one sealed B17 Cameraman 660 nm case.

All arms regenerate and hash-check the original sensor; load the same sealed
fixed660 recovery; use the identical historical spectral preconditioner and
100 additional accepted L-BFGS steps. Truth is opened only after recovery.
No change to physical system or production defaults.
"""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
from scipy import fft
from 공통.optics import FullB17,ALPHA,phase_check,digest,write_json
from 싱글.코드 import solver
from 싱글.코드.wavelength_jacobian import refine
from 싱글.코드.grid_convergence import spectrum_error

W=660.
GRID=3800
SEED=672906
EXPECTED_SENSOR="5ddb08ec940c977996f4552a62a5aee2f6b154c73e3eff407f794a05a4152d86"
EXPECTED_TRUTH="5459e5287d0a86f96904c655832e28d3d710c09cbd6cba26708472da0228a388"
ALPHAS=(.015,.005,.001,.0003,0.)
BASELINE_RUN=43595625804

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha",type=float,required=True,choices=ALPHAS)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    obs=root/"observation"
    ref=a.baseline/"fixed660"
    prior_recovery=json.loads((ref/"recovery.json").read_text())
    prior_eval=json.loads((ref/"evaluation.json").read_text())
    assert prior_recovery["sensor_sha256"]==EXPECTED_SENSOR
    assert prior_recovery["geometry"]==[GRID,1429,512,3000]
    assert prior_recovery["wavelength_nm"]==W and prior_recovery["alpha"]==.015
    assert prior_recovery["completed"] and prior_recovery["lbfgs_iterations"]==100
    assert prior_recovery["gd_iterations"]==150
    with np.load(ref/"recovered.npz",allow_pickle=False) as z:
        start=phase_check(z["phase"]);assert float(z["wavelength_nm"])==W
    assert digest(start)==prior_recovery["phase_sha256"]
    assert abs(prior_eval["full_rmse_rad"]-.14390353332408887)<1e-12
    with fft.set_workers(4):
        engine=FullB17(propagation_n=GRID)
        solver.generate(engine,obs,"camera",W,SEED,20261127)
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:
            y=solver.sensor_check(z["sensor"])
        assert digest(y)==EXPECTED_SENSOR,("Sensor regeneration differs",digest(y))
        write_json(root/"experiment.json",dict(scene="camera",wavelength_nm=W,
            geometry=list(engine.geometry),seed=SEED,noise_pct_mean=1,
            sensor_sha256=digest(y),initial_phase_sha256=digest(start),
            baseline_run=BASELINE_RUN,baseline_rmse_rad=prior_eval["full_rmse_rad"],
            baseline_sensor_l2=prior_eval["sensor_l2"],alpha=a.alpha,
            previous_gd_iterations=150,previous_lbfgs_iterations=100,
            additional_lbfgs_budget=100,preconditioner="fixed historical 660 nm; alpha=0.015 in P for all arms",
            truth_in_inverse=False,comparison="one common baseline, independent equal-budget continuation arms"))
        P=solver.preconditioner("spectral")
        out=root/"inverse"
        final=refine(engine,y,W,start,out,a.alpha,"sealed_fixed660_GD150_LBFGS100",P,
                     "alpha_continuation","660 nm historical Jacobian; fixed P across alpha arms")
        rec=json.loads((out/"recovery.json").read_text())
        assert rec["initial_phase_sha256"]==digest(start) and rec["sensor_sha256"]==EXPECTED_SENSOR
        rec.update(previous_lbfgs_iterations=100,additional_lbfgs_iterations=rec["lbfgs_iterations"],
                   total_lbfgs_iterations=100+rec["lbfgs_iterations"],baseline_run=BASELINE_RUN,
                   alpha=a.alpha)
        write_json(out/"recovery.json",rec)
        # Post-hoc evaluation only. Truth is never passed to refine.
        scores=solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
        with np.load(obs/"private_truth.npz",allow_pickle=False) as z:
            truth=phase_check(z["phase"])
        assert digest(truth)==EXPECTED_TRUTH
        m,cam=engine.optics(W)
        def diagnostics(x):
            data,q,_=engine.value_grad(m,cam,x,y,0.,False)
            smooth=solver.prior(x,1.,False)[0]
            off=float(np.angle(np.mean(np.exp(1j*(x-truth)))))
            err=np.angle(np.exp(1j*(x-truth-off)))
            return dict(data_loss=float(data),unweighted_prior=float(smooth),
                objective_at_arm_alpha=float(data+a.alpha*smooth),scale=float(q),
                rmse_rad=float(np.sqrt(np.mean(err**2))),sensor_l2=float(np.sqrt(data)),
                error_spectrum=spectrum_error(err))
        d=dict(alpha=a.alpha,baseline=diagnostics(start),final=diagnostics(final),
               truth=diagnostics(truth),scores=scores,
               same_sensor_sha256=digest(y),same_initial_phase_sha256=digest(start),
               accepted_additional_iterations=rec["lbfgs_iterations"],
               total_lbfgs_iterations=rec["total_lbfgs_iterations"])
        write_json(root/"diagnostics.json",d)
        print("ALPHA_FINAL",json.dumps(d),flush=True)

if __name__=="__main__":main()
