"""Experimental 660-nm Cameraman padding convergence and cross-grid inverse test.
Fixed physical geometry, B17 MAT checks, camera ROI and noise realization.
The 5700 grid is a numerical reference, not a certified converged physical truth.
All inverse candidates use exactly the previous fixed660 GD150 + spectral LBFGS100.
"""
from pathlib import Path
import argparse,gc,json,time
import numpy as np
from scipy import fft
import 공통.optics as optics
import 싱글.코드.solver as solver
import 싱글.코드.wavelength_jacobian as jac
from 공통.object_models import N,wrap
from 싱글.코드.pyramid import gd_pyramid,SCHEDULES
from 공통.optics import FullB17,phase_check,digest,write_json,ALPHA

W=660.
SEED=672906
GRIDS=(3800,4800,5700)
CASES={(3800,3800),(4800,4800),(5700,5700),(5700,3800),(5700,4800)}

def set_grid(n):
    assert n in GRIDS
    geometry=(n,1429,512,3000)
    # Only experiment-process configuration; original production files untouched.
    optics.GEOMETRY=geometry
    solver.GEOMETRY=geometry
    jac.GEOMETRY=geometry

def grid_spec(n):
    dx=350e-9;z=.005;pitch=W*1e-9*z/(n*dx)
    return dict(n=n,input_pitch_m=dx,output_pitch_m=pitch,
        padded_input_width_m=n*dx,output_width_m=n*pitch,
        meta_n=1429,meta_width_m=1429*dx,object_n=512,object_width_m=512*dx,
        object_meta_m=400e-6,meta_sensor_m=z,camera_n=3000,camera_pitch_m=2e-6,
        camera_width_m=.006,camera_model="bilinear point sampling; no pixel area integration",
        camera_resampling="upsampling" if pitch>2e-6 else "downsampling")

def profile_l2(a,b):
    # Relative mismatch after fitting a single nonnegative intensity gain.
    q=solver.profile_scale(b,a)
    return float(np.linalg.norm(q*a-b)/np.linalg.norm(b)),float(q)

def spectrum_error(err):
    power=np.abs(fft.fft2(err,norm="ortho"))**2
    f=fft.fftfreq(N)*N; r=np.hypot(f[:,None],f[None,:])
    bands=[]
    for lo,hi in ((0,16),(16,64),(64,128),(128,192),(192,256),(256,363)):
        q=(r>=lo)&(r<hi)
        bands.append(dict(lo=lo,hi=hi,rmse_contribution_rad=float(np.sqrt(power[q].sum()/err.size)),
                          error_energy_fraction=float(power[q].sum()/max(power.sum(),1e-30))))
    return bands

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-grid",type=int,required=True,choices=GRIDS)
    p.add_argument("--inverse-grid",type=int,required=True,choices=GRIDS)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if (a.source_grid,a.inverse_grid) not in CASES:p.error("Unsupported case")
    a.output.mkdir(parents=True,exist_ok=False)
    obs=a.output/"observation";shared=a.output/"shared_gd";shared.mkdir()
    t=time.monotonic()
    with fft.set_workers(4):
        set_grid(a.source_grid)
        engine=FullB17()
        solver.generate(engine,obs,"camera",W,SEED,20261127)
        with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:
            y=solver.sensor_check(z["sensor"])
        sensor_sha=digest(y)
        # Source grid is explicitly recorded before changing the inverse geometry.
        generation=json.loads((obs/"generation.json").read_text())
        assert generation["geometry"][0]==a.source_grid
        set_grid(a.inverse_grid)
        m,cam=engine.optics(W)
        assert m.c.propagation_n==a.inverse_grid and m.c.sensor_n==a.inverse_grid
        assert m.c.meta_n==1429 and m.c.object_n==512
        assert m.c.input_pitch_m==350e-9 and m.c.object_meta_m==400e-6
        assert m.c.meta_sensor_m==.005 and m.c.sensor_pitch_m==2e-6
        assert cam.w.shape==(3000,a.inverse_grid)
        assert np.allclose(np.asarray(cam.w.sum(axis=1)).ravel(),1)
        spec=grid_spec(a.inverse_grid)
        spec.update(asm_pass_fraction=float(np.count_nonzero(m.h1)/m.h1.size),
                    sasm_correction_pass_fraction=float(np.count_nonzero(m.hcorr)/m.hcorr.size))
        write_json(a.output/"experiment.json",dict(scene="camera",wavelength_nm=W,
            source_grid=grid_spec(a.source_grid),inverse_grid=spec,seed=SEED,
            sensor_sha256=sensor_sha,noise_pct_mean=1,alpha=ALPHA,
            gd_iterations=150,lbfgs_iterations=100,truth_in_inverse=False,
            B17_inner_shas=engine.inner_shas,source_grid_is_reference_not_converged=True,
            same_physical_geometry=True,phase_rmse_gate_rad=.15,sensor_l2_gate_fraction=.025,
            comparison="matched and common 5700 reference; fixed historical 660 preconditioner"))
        counts={"forward":0,"adjoint":0}
        def vg(model,camera,x,observed,alpha,grad):
            counts["forward"]+=1;counts["adjoint"]+=int(grad)
            return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
        phase,history=gd_pyramid(m,cam,y,vg,shared,SCHEDULES["pyramid64"])
        start=phase_check(wrap(phase))
        del m,cam,phase;gc.collect()
        np.savez_compressed(shared/"sealed.npz",phase=start)
        write_json(shared/"summary.json",dict(gd_iterations=150,sensor_sha256=sensor_sha,
            initial_phase_sha256=digest(start),history=history,**counts))
        out=a.output/"inverse"
        jac.refine(engine,y,W,start,out,ALPHA,"shared_Pyramid64_GD150",
            solver.preconditioner("spectral"),"fixed660_grid_test","660 nm historical Jacobian")
        gc.collect()
        # The inverse is sealed. Only now open truth for evaluation/diagnostics.
        metrics=solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out)
        with np.load(obs/"private_truth.npz",allow_pickle=False) as z:truth=phase_check(z["phase"])
        with np.load(out/"recovered.npz",allow_pickle=False) as z:rec=phase_check(z["phase"])
        err=np.angle(np.exp(1j*(rec-truth-metrics["offset_rad"])))
        clean_fit=engine.predict(W,truth)
        fit_clean_sha=digest(clean_fit)
        fit_vs_obs,q=profile_l2(clean_fit,y)
        set_grid(a.source_grid)
        if a.source_grid==a.inverse_grid:
            clean_src=clean_fit
        else:
            clean_src=engine.predict(W,truth)
        discrepancy,gain=profile_l2(clean_fit,clean_src)
        source_noise_l2=float(np.linalg.norm(clean_src-y)/np.linalg.norm(clean_src))
        diag=dict(source_grid=a.source_grid,inverse_grid=a.inverse_grid,
            source_clean_sha256=digest(clean_src),inverse_clean_sha256=fit_clean_sha,
            true_phase_sha256=digest(truth),same_observation_sha256=sensor_sha,
            clean_forward_mismatch_l2=discrepancy,clean_forward_gain=gain,
            true_phase_inverse_model_residual_to_noisy_observation=fit_vs_obs,
            true_phase_gain_fit_to_observation=q,source_noise_relative_l2=source_noise_l2,
            phase_error_frequency_bands=spectrum_error(err),elapsed_s=time.monotonic()-t)
        write_json(a.output/"diagnostics.json",diag)
        print("GRID_FINAL",json.dumps(dict(metrics=metrics,diagnostics=diag)),flush=True)

if __name__=="__main__":main()
