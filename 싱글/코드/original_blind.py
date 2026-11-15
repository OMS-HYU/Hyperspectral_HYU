"""Blind K=1 validation on fixed original B17 BL=.5 physical profile.
Simulation generation and evaluation are isolated from inverse.
Coarse/fine flat library estimates wavelength, then fixed phase schedule.
Off-grid fine scan is 0.1 nm, no truth-dependent candidate selection.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import phase_check,digest,write_json
from 공통.object_models import wrap,image_phase,rbc_phase,RBC_MODELS,N
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.original_geometry import OriginalProfile

CASES={"camera440":("camera",440.),"camera550":("camera",550.),
       "camera660":("camera",660.),"camera587p6":("camera",587.6),
       "pepper660":("pepper",660.),"triple660":("triple",660.),
       "normal473p4":("normal",473.4)}

def generate(engine,obs,scene,w):
    obs.mkdir()
    truth=rbc_phase(scene,w,seed=20261127) if scene in RBC_MODELS else {"phase":image_phase(scene)}
    truth["phase"]=phase_check(wrap(truth["phase"]))
    clean=engine.predict(w,truth["phase"])
    sigma=.01*float(clean.mean())
    y=solver.sensor_check(np.maximum(0.,clean+sigma*np.random.default_rng(672906).standard_normal(clean.shape)))
    np.savez_compressed(obs/"sensor_only.npz",sensor=y)
    np.savez_compressed(obs/"private_truth.npz",actual_wavelength_nm=w,**truth)
    write_json(obs/"generation.json",dict(scene=scene,actual_wavelength_nm=w,seed=672906,
        geometry_seed=20261127,noise_sigma=sigma,noise_pct_mean=1.,
        truth_sha256=digest(truth["phase"]),sensor_sha256=digest(y),clean_sha256=digest(clean),
        model_assumption="projected RBC optical thickness; nondispersive index approximation outside 660nm" if scene in RBC_MODELS else "grayscale phase"))
def recover(engine,sensor,root):
    # No scene, truth, or actual wavelength argument.
    with np.load(sensor,allow_pickle=False) as z:y=solver.sensor_check(z["sensor"])
    w,rank=solver.scan_wavelength(engine,y)
    flat=np.zeros((N,N));norm=np.linalg.norm(y)
    tested={float(r["wavelength_nm"]) for r in rank}
    for k in range(max(4400,int(round(w*10))-10),min(6600,int(round(w*10))+10)+1):
        v=k/10.
        if v in tested:continue
        p=engine.predict(v,flat);q=solver.profile_scale(y,p)
        loss=float(np.linalg.norm(q*p-y)/norm)
        rank.append(dict(wavelength_nm=v,relative_l2=loss))
        print("BLIND_FINE",v,loss,flush=True)
        del p;gc.collect()
    rank.sort(key=lambda r:r["relative_l2"])
    w=float(rank[0]["wavelength_nm"])
    write_json(root/"blind_scan.json",dict(selected_wavelength_nm=w,ranking=rank,
        method="flat full sensor library coarse10nm integer fine then 0.1nm local fine",
        sensor_sha256=digest(y),truth_in_inverse=False))
    print("BLIND_SELECTED",w,flush=True)
    shared=root/"shared_gd";shared.mkdir()
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
                         P,"original_BL05_blind","fixed historical 660nm spectral P")
        rec=json.loads((out/"recovery.json").read_text())
        rec.update(mode="blind",blind_experimental=True,physical_profile=engine.config(),
                   selected_wavelength_nm=w,previous_lbfgs_iterations=total,
                   total_lbfgs_iterations=total+rec["lbfgs_iterations"])
        total=rec["total_lbfgs_iterations"];write_json(out/"recovery.json",rec)
        paths.append(out);gc.collect()
    return paths,w,rank
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    scene,true_w=CASES[a.case];obs=root/"observation"
    with fft.set_workers(4):
        # Engine profile label is fixed across every case; optics(w) uses
        # candidate w, not engine.w. No true wavelength enters recover.
        engine=OriginalProfile("original660_BL05")
        write_json(root/"experiment.json",dict(physical_profile=engine.config(),case=a.case,
            schedule=[.015,.005,.0003],truth_in_inverse=False,
            profile_label_wavelength_is_not_candidate=True))
        generate(engine,obs,scene,true_w)
        paths,w,rank=recover(engine,obs/"sensor_only.npz",root)
        scores={out.name:solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",out) for out in paths}
        result=dict(case=a.case,scene=scene,actual_wavelength_nm=true_w,selected_wavelength_nm=w,
            wavelength_error_nm=abs(w-true_w),scores=scores,top_flat_candidates=rank[:10])
        write_json(root/"comparison.json",result)
        print("ORIGINAL_BLIND_FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
