"""RBC-specific wavelength-only validation. Original B17 BL=.5 geometry.
Use existing flat phase scan, then 0.1 nm refinement. No phase inversion.
Evaluate wavelength only, independently from the general-image search.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.object_models import N
from 공통.optics import digest,write_json
from 싱글.코드 import solver
from 싱글.코드.original_geometry import OriginalProfile
from 싱글.코드.original_blind import generate

CASES={"normal440":("normal",440.),"normal550":("normal",550.),
       "normal587p6":("normal",587.6),"triple440":("triple",440.),
       "triple550":("triple",550.),"triple587p6":("triple",587.6)}

def estimate(engine,sensor,root):
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
        print("RBC_FINE",v,loss,flush=True)
        del p;gc.collect()
    rank.sort(key=lambda r:r["relative_l2"])
    w=float(rank[0]["wavelength_nm"])
    write_json(root/"wavelength_search.json",dict(selected_wavelength_nm=w,ranking=rank,
        sensor_sha256=digest(y),truth_in_inverse=False,phase_reconstruction=False))
    return w,rank

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    scene,true_w=CASES[a.case];obs=root/"observation"
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05")
        write_json(root/"experiment.json",dict(case=a.case,physical_profile=engine.config(),
            method="RBC flat-phase wavelength scan 10nm/1nm/0.1nm",truth_in_inverse=False,
            phase_reconstruction=False))
        generate(engine,obs,scene,true_w)
        w,rank=estimate(engine,obs/"sensor_only.npz",root)
        error=abs(w-true_w)
        result=dict(case=a.case,scene=scene,actual_wavelength_nm=true_w,
            estimated_wavelength_nm=w,wavelength_error_nm=error,
            pass_0p1_nm=error<=.100000001,pass_0p5_nm=error<=.500000001,
            top_candidates=rank[:10],phase_reconstruction=False,
            model_assumption="projected RBC optical thickness; nondispersive index approximation outside 660nm")
        write_json(root/"evaluation.json",result)
        print("RBC_WAVELENGTH_FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
