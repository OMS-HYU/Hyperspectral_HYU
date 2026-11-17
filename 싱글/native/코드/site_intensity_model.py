"""User's published full3800 B17 intensity geometry, zero exterior.

Original calibration/transfer coefficients; double precision propagation.
No background illumination and no amplitude-to-intensity relabeling.
"""
import sys
from pathlib import Path
import numpy as np

REPO=Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from 공통 import propagation as physical
from native_model import prior


class Model:
    def __init__(self,archive,wavelength=660):
        physical.SOURCE=Path(archive)
        self.optical,self.camera=physical.optics(wavelength)
        self.archive_sha256=physical.B17OBJ.archive_sha
        for name in ['meta','h1','hcorr','qin','qout']:
            setattr(self.optical,name,getattr(self.optical,name).astype(np.complex128))
        self.count=self.adj=0

    def field(self,phase):
        self.count+=1
        return self.optical.forward(np.exp(1j*np.asarray(phase,np.float64)))

    def measure(self,phase):
        field=self.field(phase)
        return self.camera.forward(np.abs(field)**2)

    def fg(self,phase,sensor,alpha=0):
        u=np.exp(1j*np.asarray(phase,np.float64))
        field=self.field(phase)
        predicted=self.camera.forward(np.abs(field)**2)
        denom=float(np.sum(sensor*sensor))
        gain=max(0.,float(np.sum(predicted*sensor)/max(float(np.sum(predicted**2)),1e-40)))
        residual=gain*predicted-sensor
        data=float(np.sum(residual**2)/denom)
        image_gradient=(2*gain/denom)*residual
        self.adj+=1
        back=self.optical.adjoint(self.camera.adjoint(image_gradient)*field)
        gradient=2*np.imag(np.conj(u)*back)
        penalty,pg=prior(phase)
        return data+alpha*penalty,gradient+alpha*pg,data,penalty,gain
