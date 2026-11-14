#!/usr/bin/env python3
"""Shared native-grid B17 forward/adjoint and numerical contract."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.object_models import wrap,TAU,N

# Numerical padding is configurable; all physical dimensions are fixed.
DEFAULT_PROPAGATION_N=3800
META_N=1429
OBJECT_N=512
CAMERA_N=3000
INPUT_PITCH_M=350e-9
OBJECT_META_M=400e-6
META_SENSOR_M=.005
CAMERA_PITCH_M=2e-6
# Historical default retained for archived reports and compatibility.
GEOMETRY=(DEFAULT_PROPAGATION_N,META_N,OBJECT_N,CAMERA_N)

def propagation_grid(n=DEFAULT_PROPAGATION_N):
    """Even integer N, sufficient to embed object/metasurface; no physical changes."""
    if isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<META_N or n%2:
        raise ValueError("Propagation grid must be an even integer >= 1430")
    # Camera point coordinates must fit in native output grid at shortest wavelength.
    pitch=440e-9*META_SENSOR_M/(int(n)*INPUT_PITCH_M)
    x=(np.arange(CAMERA_N)-CAMERA_N//2)*CAMERA_PITCH_M/pitch+int(n)//2
    if x.min()<0 or x.max()>int(n)-1:
        raise ValueError("Propagation grid cannot cover fixed camera at 440 nm")
    return int(n)

def geometry_for(n=DEFAULT_PROPAGATION_N):
    return (propagation_grid(n),META_N,OBJECT_N,CAMERA_N)

def grid_details(n,w):
    n=propagation_grid(n);w=wavelength_check(w)
    pitch=w*1e-9*META_SENSOR_M/(n*INPUT_PITCH_M)
    return dict(propagation_n=n,meta_n=META_N,object_n=OBJECT_N,camera_n=CAMERA_N,
        input_pitch_m=INPUT_PITCH_M,object_meta_m=OBJECT_META_M,
        meta_sensor_m=META_SENSOR_M,camera_pitch_m=CAMERA_PITCH_M,
        output_pitch_m=pitch,input_window_m=n*INPUT_PITCH_M,output_window_m=n*pitch,
        resampling="upsampling" if pitch>CAMERA_PITCH_M else "downsampling" if pitch<CAMERA_PITCH_M else "identity")
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
    def __init__(self,propagation_n=DEFAULT_PROPAGATION_N):
        self.propagation_n=propagation_grid(propagation_n)
        self.geometry=geometry_for(self.propagation_n)
        from 공통.b17_loader import check_b17
        _,self.measure,self.value_grad,self.wrapped_rmse,self.archive_sha,self.inner_shas=check_b17()
        from 공통.propagation import load_physics
        from 공통.b17_core import Geometry,Model,Camera
        self.b,self.idx=load_physics()
        self.source_fingerprint=self.b.source_fingerprint
        self.Geometry,self.Model,self.Camera=Geometry,Model,Camera
    def optics(self,w):
        w=wavelength_check(w)
        n,meta_n,obj,sensor=self.geometry
        c=self.Geometry(propagation_n=n,meta_n=meta_n,object_n=obj,sensor_n=n,
            wavelength_m=w*1e-9,input_pitch_m=INPUT_PITCH_M,object_meta_m=OBJECT_META_M,
            meta_sensor_m=META_SENSOR_M,sensor_pitch_m=CAMERA_PITCH_M)
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
