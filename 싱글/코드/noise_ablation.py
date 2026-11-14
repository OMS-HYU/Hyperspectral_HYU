"""B17 Cameraman 660 nm noise ablation; physical geometry locked.

Five warm-start independent noise/alpha arms and one cold-start diagnostic.
Noise is sigma as a fraction of mean clean intensity, same deterministic draw.
Only the synthetic observation changes. No true phase enters the inverse.
"""
from pathlib import Path
import argparse,gc,json
import numpy as np
from scipy import fft
from 공통.optics import FullB17,phase_check,digest,write_json
from 공통.object_models import image_phase,wrap
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.grid_convergence import spectrum_error
from 싱글.코드.alpha_continuation import EXPECTED_SENSOR,EXPECTED_TRUTH

W=660.
GRID=3800
SEED=672906
BASELINE_RUN=43595625804
EXPECTED_CLEAN="a7b19650fe74850f62a518afe063927c3ccbec46bf5d777545fe87aa80c7abd7"
CASES={"A":(0.,0.,"warm"),"B":(0.,.001,"warm"),"C":(0.,.005,"warm"),
       "D":(.1,0.,"warm"),"E":(.1,.005,"warm"),"F":(0.,0.,"cold")}

def generate_observation(engine,obs,noise_pct):
    obs.mkdir()
    truth=phase_check(wrap(image_phase("camera")))
    assert digest(truth)==EXPECTED_TRUTH
    clean=engine.predict(W,truth)
    assert digest(clean)==EXPECTED_CLEAN,("Clean forward differs",digest(clean))
    draw=np.random.default_rng(SEED).standard_normal(clean.shape)
    reference=np.maximum(0.,clean+.01*float(clean.mean())*draw)
    assert digest(reference)==EXPECTED_SENSOR,("Original sensor replay differs",digest(reference))
    del reference
    sigma=noise_pct/100*float(clean.mean())
    y=np.maximum(0.,clean+sigma*draw)
    np.savez_compressed(obs/"sensor_only.npz",sensor=y)
    np.savez_compressed(obs/"private_truth.npz",phase=truth,actual_wavelength_nm=W)
    generation=dict(scene="camera",wavelength_nm=W,geometry=list(engine.geometry),
        noise_pct_mean=noise_pct,noise_sigma=sigma,seed=SEED,sensor_sha256=digest(y),
        clean_sha256=digest(clean),truth_sha256=digest(truth),
        original_noisy_sensor_sha256=EXPECTED_SENSOR,
        measured_noise_l2=float(np.linalg.norm(y-clean)/np.linalg.norm(clean)),
        B17_inner_shas=engine.inner_shas,truth_in_inverse=False)
    write_json(obs/"generation.json",generation)
    return generation

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();noise,alpha,initial=CASES[a.case]
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    obs=root/"observation"
    with fft.set_workers(4):
        engine=FullB17(propagation_n=GRID)
        generation=generate_observation(engine,obs,noise)
        gc.collect()
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:
            y=solver.sensor_check(z["sensor"])
        ref=a.baseline/"fixed660"
        refaudit=json.loads((ref/"recovery.json").read_text())
        assert refaudit["sensor_sha256"]==EXPECTED_SENSOR and refaudit["geometry"]==list(engine.geometry)
        assert refaudit["completed"] and refaudit["gd_iterations"]==150 and refaudit["lbfgs_iterations"]==100
        if initial=="warm":
            with np.load(ref/"recovered.npz",allow_pickle=False) as z:
                start=phase_check(z["phase"])
            assert digest(start)==refaudit["phase_sha256"]
            prior_gd=150;prior_lbfgs=100;budget=100
            init_label="sealed_1pct_noise_GD150_LBFGS100"
        else:
            # Explicitly disable the default GD smoothing after step 100.
            # This branch starts from zero and uses ONLY the clean observation.
            pyramid.REG=0.
            shared=root/"cold_gd";shared.mkdir()
            m,cam=engine.optics(W)
            def vg(model,camera,x,observed,reg,grad):
                return engine.value_grad(model,camera,wrap(x),observed,reg,grad)
            x,history=pyramid.gd_pyramid(m,cam,y,vg,shared,pyramid.SCHEDULES["pyramid64"])
            start=phase_check(wrap(x))
            np.savez_compressed(shared/"sealed.npz",phase=start)
            write_json(shared/"summary.json",dict(history=history,gd_iterations=150,alpha=0.,sensor_sha256=digest(y),initial_phase_sha256=digest(start)))
            del m,cam,x;gc.collect()
            prior_gd=150;prior_lbfgs=0;budget=200
            init_label="zero_start_clean_PyramidGD150_no_prior"
        jac.BUDGET=budget
        write_json(root/"experiment.json",dict(case=a.case,wavelength_nm=W,
            geometry=list(engine.geometry),noise_pct_mean=noise,alpha=alpha,
            initialization=init_label,baseline_run=BASELINE_RUN,
            sensor_sha256=digest(y),initial_phase_sha256=digest(start),
            prior_gd_iterations=prior_gd,prior_lbfgs_iterations=prior_lbfgs,
            additional_lbfgs_budget=budget,preconditioner="same fixed historical 660 nm spectral P (alpha=.015 in P only)",
            truth_in_inverse=False))
        out=root/"inverse"
        final=jac.refine(engine,y,W,start,out,alpha,init_label,solver.preconditioner("spectral"),
                         "noise_ablation","fixed historical 660 nm; unchanged across all arms")
        rec=json.loads((out/"recovery.json").read_text())
        assert rec["sensor_sha256"]==generation["sensor_sha256"]
        assert rec["initial_phase_sha256"]==digest(start)
        rec.update(previous_lbfgs_iterations=prior_lbfgs,additional_lbfgs_iterations=rec["lbfgs_iterations"],
            total_lbfgs_iterations=prior_lbfgs+rec["lbfgs_iterations"],noise_pct_mean=noise,case=a.case)
        write_json(out/"recovery.json",rec)
        # Truth only for sealed post-hoc evaluation.
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
                objective=float(data+alpha*smooth),scale=float(q),
                rmse_rad=float(np.sqrt(np.mean(err**2))),sensor_l2=float(np.sqrt(data)),
                error_spectrum=spectrum_error(err))
        result=dict(case=a.case,noise_pct_mean=noise,alpha=alpha,initialization=init_label,
            generation=generation,initial=diagnostics(start),final=diagnostics(final),
            truth=diagnostics(truth),scores=scores,accepted_additional_iterations=rec["lbfgs_iterations"],
            total_lbfgs_iterations=rec["total_lbfgs_iterations"],initial_phase_sha256=digest(start))
        write_json(root/"diagnostics.json",result)
        print("NOISE_FINAL",json.dumps(result),flush=True)

if __name__=="__main__":main()
