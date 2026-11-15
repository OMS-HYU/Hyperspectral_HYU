"""RBC-specific blind recovery comparison on the six wavelength-only successes.
Each arm regenerates identical observation, estimates wavelength from sensor only,
then executes its original initialization and common staged refinement.
No truth-based wavelength injection or phase selection.
"""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import phase_check,digest,write_json,ALPHA
from 공통.object_models import wrap
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.original_geometry import OriginalProfile
from 싱글.코드.original_blind import generate
from 싱글.코드.rbc_wavelength_only import CASES,estimate

def recover(engine,sensor,root,arm):
    with np.load(sensor,allow_pickle=False) as z:y=solver.sensor_check(z["sensor"])
    t0=time.monotonic()
    w,rank=estimate(engine,sensor,root)
    print("RBC_INVERSE_WAVELENGTH",w,flush=True)
    paths=[];total=0;P=solver.preconditioner("spectral")
    if arm=="spectral":
        # Exact historical RBC solver: full grid GD100, then spectral L-BFGS100.
        first=root/"alpha015";first.mkdir()
        info=solver.fit(engine,y,w,first,"spectral")
        with np.load(first/"recovered.npz",allow_pickle=False) as z:phase=phase_check(z["phase"])
        if abs(float(info["alpha"])-.015)>1e-12:raise ValueError("Unexpected historical alpha")
        gd_steps=100;total=info["lbfgs_iterations"]
        info.update(mode="blind",blind_experimental=True,physical_profile=engine.config(),
                    arm=arm,selected_wavelength_nm=w,previous_lbfgs_iterations=0,
                    total_lbfgs_iterations=total)
        write_json(first/"recovery.json",info);paths.append(first);gc.collect()
    else:
        shared=root/"shared_gd";shared.mkdir()
        m,cam=engine.optics(w)
        def vg(model,camera,x,observed,alpha,grad):
            return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
        x,h=pyramid.gd_pyramid(m,cam,y,vg,shared,pyramid.SCHEDULES["pyramid64"])
        phase=phase_check(wrap(x));del m,cam,x;gc.collect()
        gd_steps=150
        np.savez_compressed(shared/"sealed.npz",phase=phase)
        write_json(shared/"summary.json",dict(gd_iterations=gd_steps,sensor_sha256=digest(y),
            initial_phase_sha256=digest(phase),history=h))
    schedule=(("alpha005",.005),("alpha0003",.0003)) if arm=="spectral" else (("alpha015",.015),("alpha005",.005),("alpha0003",.0003))
    for name,alpha in schedule:
        out=root/name
        phase=jac.refine(engine,y,w,phase,out,alpha,
            paths[-1].name+"_sealed" if paths else "Pyramid64_GD150",
            P,"RBC_"+arm+"_staged","fixed historical 660nm spectral P")
        info=json.loads((out/"recovery.json").read_text())
        info.update(mode="blind",blind_experimental=True,physical_profile=engine.config(),
            arm=arm,gd_iterations=gd_steps,selected_wavelength_nm=w,
            previous_lbfgs_iterations=total,total_lbfgs_iterations=total+info["lbfgs_iterations"])
        total=info["total_lbfgs_iterations"];write_json(out/"recovery.json",info)
        paths.append(out);gc.collect()
    return paths,w,time.monotonic()-t0

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--arm",choices=("spectral","pyramid"),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    scene,true_w=CASES[a.case];obs=root/"observation"
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05")
        write_json(root/"experiment.json",dict(case=a.case,arm=a.arm,
            physical_profile=engine.config(),truth_in_inverse=False,
            schedule=[.015,.005,.0003],noise_pct_mean=1.,seed=672906,
            geometry_seed=20261127,comparison="same sensor and wavelength estimator; GD100 spectral vs GD150 pyramid; not equal GD budget"))
        generate(engine,obs,scene,true_w)
        paths,w,seconds=recover(engine,obs/"sensor_only.npz",root,a.arm)
        scores={out.name:solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out) for out in paths}
        rec=json.loads((paths[-1]/"recovery.json").read_text())
        result=dict(case=a.case,arm=a.arm,actual_wavelength_nm=true_w,
            estimated_wavelength_nm=w,wavelength_error_nm=abs(w-true_w),scores=scores,
            sensor_sha256=rec["sensor_sha256"],total_lbfgs_iterations=rec["total_lbfgs_iterations"],
            gd_iterations=rec["gd_iterations"],inverse_seconds=seconds)
        write_json(root/"comparison.json",result)
        print("RBC_INVERSE_FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
