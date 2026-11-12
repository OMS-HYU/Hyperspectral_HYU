"""Original B17 LUT loader and unchanged ASM/SASM propagation operators. Active geometry is full native grid."""
from __future__ import annotations
import hashlib, io, zipfile
from dataclasses import replace
from pathlib import Path
import h5py
import numpy as np
from scipy.io import loadmat
from scipy import fft
from scipy.sparse import csr_matrix
from dataclasses import dataclass

EXPECTED_B17_SHA256='6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716'

@dataclass(frozen=True)
class Geometry:
    wavelength_m:float=660e-9
    propagation_n:int=3800
    meta_n:int=1429
    object_n:int=512
    sensor_n:int=3800
    input_pitch_m:float=350e-9
    object_meta_m:float=400e-6
    meta_sensor_m:float=.005
    sensor_pitch_m:float=2e-6

def _centered(n,size):
    x=n//2-size//2
    return slice(x,x+size)

class Camera:
    def __init__(self,n,roi,pitch,target):
        x=(np.arange(roi)-roi//2)*target/pitch+n//2
        if x.min()<0 or x.max()>n-1:raise ValueError('camera ROI beyond coverage')
        x=np.clip(x,0,n-1);lo=np.minimum(np.floor(x).astype(int),n-2);a=x-lo
        self.w=csr_matrix((np.column_stack((1-a,a)).ravel(),(np.repeat(np.arange(roi),2),np.column_stack((lo,lo+1)).ravel())),shape=(roi,n))
    def forward(self,a):return (self.w@(self.w@a).T).T
    def adjoint(self,a):return (self.w.T@(self.w.T@a).T).T

class Model:
    def __init__(self,c,meta):
        n=c.propagation_n;self.c=c;self.obj=_centered(n,c.object_n);self.sensor=_centered(n,c.sensor_n)
        self.meta=np.zeros((n,n),dtype=np.complex128);m=_centered(n,c.meta_n);self.meta[m,m]=meta
        f=fft.fftfreq(n,c.input_pitch_m);fx,fy=f[None,:],f[:,None]
        spatial=(c.wavelength_m*fx)**2+(c.wavelength_m*fy)**2
        rt=np.sqrt(np.maximum(1-spatial,0));safe=np.maximum(rt,np.finfo(float).tiny)
        window=n*c.input_pitch_m;k=2*np.pi/c.wavelength_m
        b=((spatial<1)&(abs(c.object_meta_m*c.wavelength_m*fx/safe)<=window/2)&(abs(c.object_meta_m*c.wavelength_m*fy/safe)<=window/2))
        self.h1=b*np.exp(1j*k*c.object_meta_m*(-spatial/(1+rt)))
        correction=-spatial**2/(2*(1+rt)**2)
        b=((spatial<1)&(abs(c.meta_sensor_m*c.wavelength_m*fx*(1/safe-1))<=window/2)&(abs(c.meta_sensor_m*c.wavelength_m*fy*(1/safe-1))<=window/2))
        self.hcorr=b*np.exp(1j*k*c.meta_sensor_m*correction)
        pitch=c.wavelength_m*c.meta_sensor_m/(n*c.input_pitch_m)
        xin=(np.arange(n)-n//2)*c.input_pitch_m;xout=(np.arange(n)-n//2)*pitch
        self.qin=np.exp(1j*k*(xin[:,None]**2+xin[None,:]**2)/(2*c.meta_sensor_m))
        self.qout=np.exp(1j*k*(xout[:,None]**2+xout[None,:]**2)/(2*c.meta_sensor_m))
        self.scale=n*c.input_pitch_m**2/(1j*c.wavelength_m*c.meta_sensor_m)
    def asm(self,x,adjoint=False):return fft.ifft2(fft.fft2(x,norm='ortho')*(np.conj(self.h1) if adjoint else self.h1),norm='ortho')
    def sas(self,x):
        sp=fft.fft2(fft.ifftshift(x),norm='ortho')
        before=fft.fftshift(fft.ifft2(sp*self.hcorr,norm='ortho'))
        return fft.fftshift(fft.fft2(fft.ifftshift(before*self.qin),norm='ortho'))*self.scale*self.qout
    def sas_adjoint(self,y):
        r=y*np.conj(self.qout)*np.conj(self.scale)
        pre=fft.fftshift(fft.ifft2(fft.ifftshift(r),norm='ortho'))*np.conj(self.qin)
        sp=fft.fft2(fft.ifftshift(pre),norm='ortho')*np.conj(self.hcorr)
        return fft.fftshift(fft.ifft2(sp,norm='ortho'))
    def forward(self,x):
        full=np.zeros_like(self.meta);full[self.obj,self.obj]=x
        y=self.sas(self.asm(full)*self.meta)
        return y[self.sensor,self.sensor].copy()
    def adjoint(self,y):
        full=np.zeros_like(self.meta);full[self.sensor,self.sensor]=y
        x=self.asm(self.sas_adjoint(full)*np.conj(self.meta),adjoint=True)
        return x[self.obj,self.obj].copy()

class B17:
    def __init__(self,archive_path,*,verify_hash=True):
        p=Path(archive_path)
        if verify_hash and hashlib.sha256(p.read_bytes()).hexdigest()!=EXPECTED_B17_SHA256:raise ValueError('Original B17 archive hash mismatch')
        with zipfile.ZipFile(p) as a:
            l=next(s for s in a.namelist() if s.endswith('/bayesLUT_MSE_v6.3_nonoverlap.mat'))
            w=next(s for s in a.namelist() if s.endswith('/0.6NA_random_70_1_300_1mm_mapped_width.mat'))
            with h5py.File(io.BytesIO(a.read(l)),'r') as f:
                self.lam=np.asarray(f['wl_interp'],float).ravel()
                self.lut=np.asarray(f['lut_opt_interp'],float).T
            width=loadmat(io.BytesIO(a.read(w)))['mapped_width']
        if self.lut.shape[0]!=241 or self.lam[0]>440 or self.lam[-1]<660:raise ValueError('Unexpected LUT calibration')
        self.c=Geometry();start=width.shape[0]//2-self.c.meta_n//2
        self.indices=width[start:start+self.c.meta_n,start:start+self.c.meta_n].astype(int)-60
        if self.indices.min()<0 or self.indices.max()>=self.lut.shape[0]:raise ValueError('B17 pillar widths outside LUT')
