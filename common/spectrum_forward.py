#!/usr/bin/env python3
"""Foundation only: native B17 incoherent spectral forward model, no inverse.
Multi uses 2 or 3 independent phase maps. Continuum uses quadrature,
not a discrete known-K inverse. All optical evaluations are full size.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from common.optics import FullB17,GEOMETRY,phase_check,wavelength_check,digest

def multi_plan(wavelengths,weights):
    w=np.asarray(wavelengths,float);a=np.asarray(weights,float)
    if w.ndim!=1 or len(w) not in (2,3) or len(set(w.tolist()))!=len(w):
        raise ValueError("Multi foundation requires K=2 or 3 distinct wavelengths")
    if a.shape!=w.shape or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError("One positive finite spectral weight per wavelength")
    for v in w:wavelength_check(v)
    return {"mode":"multi","K":len(w),"wavelengths_nm":w.tolist(),"weights":a.tolist(),
            "geometry":list(GEOMETRY),"truth_parameterization":"independent-phase-v2",
            "inverse_status":"not implemented in unified foundation"}
def continuum_plan(start,stop,step):
    start=wavelength_check(start);stop=wavelength_check(stop);step=float(step)
    if not (start<stop and np.isfinite(step) and step>0):raise ValueError("Invalid continuum interval or step")
    n=int(np.ceil((stop-start)/step))
    if n>2200:raise ValueError("At most 2201 quadrature nodes")
    w=np.linspace(start,stop,n+1);h=(stop-start)/n
    a=np.full(n+1,h);a[[0,-1]]*=.5
    return {"mode":"continuum","K":None,"wavelengths_nm":w.tolist(),
            "weights":a.tolist(),"geometry":list(GEOMETRY),
            "quadrature":"trapezoid; weights in nm; uniform unit spectral density default",
            "inverse_status":"not implemented in unified foundation"}
def summed_forward(engine,plan,phases,density=None):
    """Sum intensities, never complex fields. No truth access inside any inverse."""
    w=plan["wavelengths_nm"];a=np.asarray(plan["weights"],float)
    if len(phases)!=len(w):raise ValueError("One native phase array per spectral node")
    if density is not None:
        if plan["mode"]!="continuum":raise ValueError("Density only for continuum")
        d=np.asarray(density,float)
        if d.shape!=a.shape or not np.isfinite(d).all() or np.any(d<0):raise ValueError("Invalid spectral density")
        a=a*d
    out=np.zeros((3000,3000),float)
    for wl,wt,p in zip(w,a,phases):
        out+=wt*engine.predict(wl,phase_check(p))
    return out
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest="mode",required=True)
    m=s.add_parser("multi");m.add_argument("--wavelengths",nargs="+",type=float,required=True);m.add_argument("--weights",nargs="+",type=float,required=True)
    c=s.add_parser("continuum");c.add_argument("--start",type=float,default=440);c.add_argument("--stop",type=float,default=660);c.add_argument("--step",type=float,default=10)
    for a in (m,c):
        a.add_argument("--output",type=Path,required=True)
        a.add_argument("--phases",type=Path,help="Optional NPZ phases[Nnodes,512,512], runs full forward only")
    a=p.parse_args(argv)
    plan=multi_plan(a.wavelengths,a.weights) if a.mode=="multi" else continuum_plan(a.start,a.stop,a.step)
    a.output.mkdir(parents=True,exist_ok=False)
    if a.phases:
        with np.load(a.phases,allow_pickle=False) as z:
            phases=z["phases"];density=z["density"] if "density" in z else None
            from scipy import fft
            with fft.set_workers(4):y=summed_forward(FullB17(),plan,phases,density)
        np.savez_compressed(a.output/"sensor_only.npz",sensor=y)
        plan["sensor_sha256"]=digest(y);plan["forward_completed"]=True
    else:plan["forward_completed"]=False
    (a.output/"plan.json").write_text(json.dumps(plan,indent=2),encoding="utf-8")
    print(json.dumps(plan,indent=2))
if __name__=="__main__":main()
