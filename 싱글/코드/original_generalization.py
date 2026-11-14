"""Validate the original BL=.5 B17 profile beyond Cameraman.
Fixed phase-only sensor model, 660 nm, same 1% mean noise and fixed
GD150 -> LBFGS(.015,.005,.0003) schedule, no scene-specific tuning.
RBC truth uses existing projected-thickness model; no experimental cell data.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import phase_check,digest,write_json
from 공통.object_models import wrap,image_phase,rbc_phase,RBC_MODELS
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.original_geometry import OriginalProfile

SCENES=("boat","pepper","tank","normal","triple","mandrill")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",choices=SCENES,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    obs=root/"observation";obs.mkdir();shared=root/"shared_gd";shared.mkdir()
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05");w=engine.w;cfg=engine.config()
        write_json(root/"experiment.json",dict(physical_profile=cfg,scene=a.scene,
            schedule=[.015,.005,.0003],seed=672906,geometry_seed=20261127,truth_in_inverse=False))
        truth=rbc_phase(a.scene,w,seed=20261127) if a.scene in RBC_MODELS else {"phase":image_phase(a.scene)}
        truth["phase"]=phase_check(wrap(truth["phase"]))
        clean=engine.predict(w,truth["phase"])
        sigma=.01*float(clean.mean())
        y=solver.sensor_check(np.maximum(0.,clean+sigma*np.random.default_rng(672906).standard_normal(clean.shape)))
        np.savez_compressed(obs/"sensor_only.npz",sensor=y)
        np.savez_compressed(obs/"private_truth.npz",actual_wavelength_nm=w,**truth)
        write_json(obs/"generation.json",dict(physical_profile=cfg,scene=a.scene,seed=672906,
            geometry_seed=20261127,noise_sigma=sigma,noise_pct_mean=1.,
            truth_sha256=digest(truth["phase"]),sensor_sha256=digest(y),clean_sha256=digest(clean),
            model_assumption="projected RBC optical thickness, not measured cell phase" if a.scene in RBC_MODELS else "grayscale phase",
            truth_in_inverse=False))
        del truth,clean;gc.collect()
        m,cam=engine.optics(w)
        def vg(model,camera,x,observed,alpha,grad):
            return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
        x,h=pyramid.gd_pyramid(m,cam,y,vg,shared,pyramid.SCHEDULES["pyramid64"])
        start=phase_check(wrap(x));del m,cam,x;gc.collect()
        np.savez_compressed(shared/"sealed.npz",phase=start)
        write_json(shared/"summary.json",dict(gd_iterations=150,sensor_sha256=digest(y),
            initial_phase_sha256=digest(start),history=h))
        P=solver.preconditioner("spectral")
        paths=[];total=0
        for name,alpha in (("alpha015",.015),("alpha005",.005),("alpha0003",.0003)):
            out=root/name
            start=jac.refine(engine,y,w,start,out,alpha,"Pyramid64_GD150" if not paths else paths[-1].name+"_sealed",
                             P,"original_BL05_generalization","fixed historical 660nm spectral P")
            rec=json.loads((out/"recovery.json").read_text())
            rec.update(physical_profile=cfg,scene=a.scene,previous_lbfgs_iterations=total,
                       total_lbfgs_iterations=total+rec["lbfgs_iterations"])
            total=rec["total_lbfgs_iterations"]
            write_json(out/"recovery.json",rec)
            paths.append(out);gc.collect()
        # Only post-hoc evaluation can see the truth. Never select a phase using truth in the inverse.
        scores={out.name:solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out) for out in paths}
        write_json(root/"comparison.json",scores)
        print("GENERALIZATION_FINAL",json.dumps(dict(scene=a.scene,config=cfg,scores=scores)),flush=True)
if __name__=="__main__":main()
