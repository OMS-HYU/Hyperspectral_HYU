#!/usr/bin/env python3
"""Shared native-grid B17 forward/adjoint and numerical contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scipy import fft
from common.object_models import wrap,TAU,N

GEOMETRY=(3800,1429,512,3000)
ALPHA=.015
GD_STEPS=100
LBFGS_STEPS=100
WMIN,WMAX=440.,660.
ROOT=Path(__file__).resolve().parent
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def write_json(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding="utf-8")
def phase_check(a):
    a=np.asarray(a)
    if a.shape!=(N,N) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>=TAU):
        raise ValueError("Object phase must be finite 512x512 in [0,2pi)")
    return a
def wavelength_check(w):
    w=float(w)
    if not WMIN<=w<=WMAX: raise ValueError("Wavelength outside original B17 LUT 440..660 nm")
    return w
class FullB17:
    """Reuse frozen MAIN original B17 operators, explicitly wrap LUT phase."""
    def __init__(self):
        from common.b17_loader import check_b17
        _,self.measure,self.value_grad,self.wrapped_rmse,self.archive_sha,self.inner_shas=check_b17()
        from common.propagation import load_physics
        from common.b17_core import Geometry,Model,Camera
        self.b,self.idx=load_physics()
        self.Geometry,self.Model,self.Camera=Geometry,Model,Camera
    def optics(self,w):
        w=wavelength_check(w)
        n,meta_n,obj,sensor=GEOMETRY
        c=self.Geometry(propagation_n=n,meta_n=meta_n,object_n=obj,sensor_n=n,
            wavelength_m=w*1e-9,input_pitch_m=350e-9,object_meta_m=400e-6,
            meta_sensor_m=.005,sensor_pitch_m=2e-6)
        lut=np.array([np.interp(w,self.b.lam,row) for row in self.b.lut])
        meta_phase=wrap(lut[self.idx])
        if meta_phase.shape!=(meta_n,meta_n) or not np.all((meta_phase>=0)&(meta_phase<TAU)):
            raise ValueError("Invalid original B17 metasurface phase")
        meta=np.exp(1j*meta_phase).astype(np.complex64)
        if not np.allclose(meta,np.exp(1j*lut[self.idx]).astype(np.complex64),rtol=0,atol=1e-7):
            raise RuntimeError("Wrapping changed complex metasurface transmission")
        m=self.Model(c,meta)
        for a in ("meta","h1","hcorr","qin","qout"):
            setattr(m,a,getattr(m,a).astype(np.complex64))
        cam=self.Camera(n,sensor,w*1e-9*c.meta_sensor_m/(n*c.input_pitch_m),c.sensor_pitch_m)
        return m,cam
    def predict(self,w,p):
        p=phase_check(p);m,c=self.optics(w)
        a=self.measure(m,c,p)
        if a.shape!=(3000,3000):raise RuntimeError("Not full sensor")
        return a
