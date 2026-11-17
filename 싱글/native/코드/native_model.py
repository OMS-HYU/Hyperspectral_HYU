"""Native B17 K=1 model from the 2026-11-16 5713 amplitude experiments.
Only archive-path and wavelength arguments differ from the original tested model.
"""
import json,hashlib,time,zipfile,io
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scipy import fft
from scipy.optimize import minimize
from scipy.io import loadmat
import h5py
R=Path(__file__).resolve().parent; B=R.parent; N=5713; M=2857; O=512; S=3000; DX=350e-9; Z1=.0004; Z2=.0063; PITCH=1.85e-6; W=660.; TAU=2*np.pi
sha=lambda a:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest();sl=lambda n,k:slice(n//2-k//2,n//2-k//2+k)
def save(n,obj):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def log(*x): print(*x,flush=True)
torch.set_num_threads(4);torch.set_num_interop_threads(1)
class Model:
 def __init__(self, archive, wavelength_nm=660.):
  p=Path(archive); W=float(wavelength_nm)
  if not 440 <= W <= 660: raise ValueError("B17 wavelength must be in 440–660 nm")
  self.archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
  with zipfile.ZipFile(p) as z:
   l=z.read(next(n for n in z.namelist() if n.endswith('/bayesLUT_MSE_v6.3_nonoverlap.mat')));b=z.read(next(n for n in z.namelist() if n.endswith('/0.6NA_random_70_1_300_1mm_mapped_width.mat')))
  assert hashlib.sha256(l).hexdigest()=='02c551f2e24fb01d91416ea21cd17d24ae1c5e6fab0d630bff8e1fb2b4feaeaa';assert hashlib.sha256(b).hexdigest()=='b7d799c1e3a5913e7e9034fd2efc31dc2dab98822058ef08e2e57470ac2d68f9'
  with h5py.File(io.BytesIO(l),'r') as f: lut=np.asarray(f['lut_opt_interp'],float).T;lam=np.asarray(f['wl_interp'],float).ravel()
  idx=loadmat(io.BytesIO(b))['mapped_width'].astype(np.int32)-60;v=np.array([np.interp(W,lam,row) for row in lut],np.float32);del l,b,lut
  self.meta=np.zeros((N,N),np.complex64);self.meta[sl(N,M),sl(N,M)]=1;self.meta[sl(N,2856),sl(N,2856)]=np.exp(1j*v[idx]);del idx
  self.h=np.empty((N,N),np.complex64);self.d=np.empty_like(self.h);self.q=np.empty_like(self.h)
  dk=2*np.pi/(N*DX);wl=torch.tensor(W*1e-9,dtype=torch.float32);k0=2*np.pi/wl;z1=torch.tensor(Z1,dtype=torch.float32);z2=torch.tensor(Z2,dtype=torch.float32);center=dk*torch.arange(-(N//2),N//2+1,dtype=torch.float32);f=torch.fft.ifftshift(center);x=center/dk*DX;klim=k0/torch.sqrt(1/np.pi**2*z1**2*dk**2+1);kx=f[None,:];ax=wl*kx/(2*np.pi)
  for a in range(0,N,96):
   e=min(a+96,N);ky=f[a:e,None];ay=wl*ky/(2*np.pi);kz=torch.sqrt(torch.clamp(k0**2-(kx**2+ky**2),min=0));mask=((ky**2/klim**2+kx**2/(dk*N/2)**2<=1)&(ky**2/(dk*N/2)**2+kx**2/klim**2<=1));self.h[a:e]=((kz>0)*torch.exp(1j*kz*z1)*(mask+~mask*1e-6)).numpy()
   rt=torch.sqrt(0j+(1-ax**2-ay**2));mask=(2*torch.abs(z2*(ax/rt-ax))<=.5*N*DX)&(2*torch.abs(z2*(ay/rt-ay))<=.5*N*DX);rt2=torch.sqrt(0j+1-(wl**2*(kx**2+ky**2))/(4*np.pi**2));self.d[a:e]=(mask*torch.exp(1j*k0*z2*(rt2-(1-wl**2*(kx**2+ky**2)/(8*np.pi**2))))).numpy();self.q[a:e]=torch.exp(1j*k0/(2*z2)*(x[a:e,None]**2+x[None,:]**2)).numpy()
  self.factor=float((wl*z2/DX).item()/(N*PITCH));self.count=0;self.adj=0;bg=np.ones((N,N),np.complex64);bg[sl(N,M),sl(N,M)]=np.exp(1j*np.float32(1));bg[sl(N,O),sl(N,O)]=0;self.bg=self.propagate(bg);log('MODEL READY',self.factor)
 def propagate(self,u):
  u=fft.fft2(u,norm='ortho',overwrite_x=True);u*=self.h;u=fft.ifft2(u,norm='ortho',overwrite_x=True);u*=self.meta;u=fft.fft2(u,norm='ortho',overwrite_x=True);u*=self.d;u=fft.ifft2(u,norm='ortho',overwrite_x=True);u*=self.q;return fft.fftshift(fft.fft2(fft.ifftshift(u),norm='ortho',overwrite_x=True))
 def field(self,p):
  self.count+=1;u=np.zeros((N,N),np.complex64);u[sl(N,O),sl(N,O)]=np.exp(1j*np.asarray(p,np.float32));u=self.propagate(u);u+=self.bg;return u
 def backward(self,u):
  self.adj+=1;u=fft.fftshift(fft.ifft2(fft.ifftshift(u),norm='ortho',overwrite_x=True))
  def mul(v,h):
   for a in range(0,N,96):v[a:a+96]*=np.conj(h[a:a+96])
  mul(u,self.q);u=fft.fft2(u,norm='ortho',overwrite_x=True);mul(u,self.d);u=fft.ifft2(u,norm='ortho',overwrite_x=True);mul(u,self.meta);u=fft.fft2(u,norm='ortho',overwrite_x=True);mul(u,self.h);u=fft.ifft2(u,norm='ortho',overwrite_x=True);return u[sl(N,O),sl(N,O)].copy()
 def cam(self,a):
  t=F.interpolate(a[None,None],scale_factor=self.factor,mode='bilinear',antialias=True,align_corners=False)[0,0];return t[sl(t.shape[0],S),sl(t.shape[1],S)]
 def measure(self,p):
  u=self.field(p);return self.cam(torch.from_numpy(np.abs(u))).numpy().copy()
 def fg(self,p,y,alpha):
  field=self.field(p);amp=torch.from_numpy(np.abs(field)).requires_grad_(True);pred=self.cam(amp);q=float(torch.sum(pred*y)/torch.sum(pred*pred));res=q*pred-y;loss=torch.sum(res*res)/torch.sum(y*y);loss.backward();gamp=amp.grad.numpy();av=amp.detach().numpy();dataloss=float(loss);del amp,pred,res,loss
  for a in range(0,N,96):field[a:a+96]*=gamp[a:a+96]/np.maximum(av[a:a+96],1e-20)
  del gamp,av;v=self.backward(field);g=np.imag(np.conj(np.exp(1j*p))*v).astype(np.float64);r,rg=prior(p);return dataloss+alpha*r,g+alpha*rg,dataloss,r,q

def prior(p):
 g=np.zeros_like(p);x=np.diff(p,axis=1);y=np.diff(p,axis=0);sx=np.sin(x)/x.size;sy=np.sin(y)/y.size;g[:,1:]+=sx;g[:,:-1]-=sx;g[1:]+=sy;g[:-1]-=sy;return float(np.mean(1-np.cos(x))+np.mean(1-np.cos(y))),g

