"""Original owner B17 full 3800-optical-grid single-wave model; no proprietary arrays distributed."""
from __future__ import annotations
import sys,os,io,zipfile,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat
SOURCE_DIRECTORY=Path(__file__).resolve().parent
from common.b17_core import B17,Geometry,Model,Camera,EXPECTED_B17_SHA256
ZIP=Path(os.environ.get('B17_ZIP',str(SOURCE_DIRECTORY.parent/'datasets'/'B17'/'05_NN_metacam-main.zip')))
N=3800; META=1429; OBJ=512; SENSOR=3000
WIDTH:np.ndarray|None=None
B17OBJ:B17|None=None

def load_physics():
 global WIDTH,B17OBJ
 if B17OBJ is None:
  B17OBJ=B17(ZIP,verify_hash=False)
  with zipfile.ZipFile(ZIP) as z:
   name=next(s for s in z.namelist() if s.endswith('0.6NA_random_70_1_300_1mm_mapped_width.mat'))
   widths=loadmat(io.BytesIO(z.read(name)))['mapped_width']
  s=widths.shape[0]//2-META//2
  WIDTH=widths[s:s+META,s:s+META].astype(np.int16)-60
  if WIDTH.min()<0 or WIDTH.max()>240:raise ValueError('B17 width out of LUT range')
 return B17OBJ,WIDTH

def optics(wavelength_nm):
 b,idx=load_physics()
 w=float(wavelength_nm)
 if w<440 or w>660:raise ValueError('Source out of original B17 LUT 440-660')
 c=Geometry(propagation_n=N,meta_n=META,object_n=OBJ,sensor_n=N,
    wavelength_m=w*1e-9,input_pitch_m=350e-9,object_meta_m=400e-6,
    meta_sensor_m=.005,sensor_pitch_m=2e-6)
 lut=np.array([np.interp(w,b.lam,row) for row in b.lut])
 meta=np.exp(1j*lut[idx]).astype(np.complex64)
 m=Model(c,meta)
 for attr in ('meta','h1','hcorr','qin','qout'):
  setattr(m,attr,getattr(m,attr).astype(np.complex64))
 native_pitch=w*1e-9*c.meta_sensor_m/(N*c.input_pitch_m)
 cam=Camera(N,SENSOR,native_pitch,c.sensor_pitch_m)
 return m,cam

def measure(m,cam,phase):
 f=m.forward(np.exp(1j*np.asarray(phase,dtype=np.float32)).astype(np.complex64))
 return cam.forward(np.asarray(f.real*f.real+f.imag*f.imag,dtype=np.float64))

def value_grad(m,cam,p,y,reg_strength=0.,calc_grad=True):
 import numpy as np
 x=np.asarray(p,dtype=np.float64).reshape(OBJ,OBJ)
 u=np.exp(1j*x.astype('float32')).astype('complex64')
 field=m.forward(u)
 pure=cam.forward(np.asarray(field.real*field.real+field.imag*field.imag,dtype=np.float64))
 pp=np.sum(pure*pure,dtype=np.float64)
 q=max(0.,float(np.sum(pure*y,dtype=np.float64)/max(pp,1e-40)))
 res=q*pure-y
 base=float(np.sum(res*res,dtype=np.float64)/np.sum(y*y,dtype=np.float64))
 if calc_grad:
  dp=(2*q/np.sum(y*y,dtype=np.float64)*res).astype('float32',copy=False)
  back=m.adjoint((cam.adjoint(dp)*field).astype(np.complex64))
  grad=(2*np.imag(np.conj(u)*back)).astype(np.float64)
 else:grad=None
 if reg_strength:
  dy=x[1:]-x[:-1];dx=x[:,1:]-x[:,:-1]
  penalty=float(np.mean(dy*dy)+np.mean(dx*dx))
  base+=reg_strength*penalty
  if calc_grad:
   d=2/(OBJ*(OBJ-1))
   grad[1:]+=reg_strength*d*dy;grad[:-1]-=reg_strength*d*dy
   grad[:,1:]+=reg_strength*d*dx;grad[:,:-1]-=reg_strength*d*dx
 return (base,grad.ravel()) if calc_grad else (base,q,pure)

def wrapped_rmse(p,truth):
 diff=np.asarray(p)-np.asarray(truth)
 off=np.angle(np.mean(np.exp(1j*diff)))
 dd=np.angle(np.exp(1j*(diff-off)))
 return float(np.sqrt(np.mean(dd*dd))),dd,float(off)
