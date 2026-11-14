"""B17 original Real-scale physical geometry comparison for phase-only imaging.

The original folder's PhaseCamRealScaleConfig defines 1mm aperture, 0.4mm
object distance, 6.3mm sensor distance and 1.85um camera pitch.
Preserve legacy model, use a named experimental profile. All arms use the
same existing phase-only intensity forward/inverse and 5700 grid; a
separate arm tests the original SASM BLfactor=.5. Not a literal reproduction
of the original complex-object/amplitude/Adam implementation.
"""
import argparse,gc,io,json
from pathlib import Path
import numpy as np
from scipy import fft
from scipy.io import loadmat
from 공통.optics import FullB17,phase_check,wavelength_check,digest,write_json
from 공통.object_models import N,wrap,image_phase
from 공통.b17_core import read_b17_source,WIDTH_NAME
from 공통.propagation import SOURCE
from 싱글.코드 import solver,pyramid,wavelength_jacobian as jac
from 싱글.코드.alpha_continuation import EXPECTED_TRUTH

GRID=5700
CASES={"original440":(440.,2857,.0063,1.85e-6,1.),
       "original550":(550.,2857,.0063,1.85e-6,1.),
       "original660":(660.,2857,.0063,1.85e-6,1.),
       "aperture660":(660.,2857,.005,2e-6,1.),
       "original660_BL05":(660.,2857,.0063,1.85e-6,.5)}
class OriginalProfile(FullB17):
    def __init__(self,case):
        super().__init__(GRID)
        self.case=case
        self.w,self.meta_n,self.z2,self.pitch,self.bl=CASES[case]
        self.geometry=(GRID,self.meta_n,N,3000)
        data,_,hashes,_=read_b17_source(SOURCE)
        assert hashes==self.inner_shas
        width=loadmat(io.BytesIO(data[WIDTH_NAME]))["mapped_width"]
        self.width_shape=list(width.shape)
        # Original torch_pad_center pads the PHASE map with zero radians,
        # then applies a separate 1 mm aperture. Do not assume the MAT
        # dimensions equal the nominal aperture or synthesize pillar widths.
        if width.ndim!=2 or not np.isfinite(width).all():
            raise ValueError(f"Invalid B17 width map shape {width.shape}")
        if not np.all(width==np.rint(width)):
            raise ValueError("Noninteger B17 mapped widths")
        self.idx=width.astype(np.int32)-60
        if self.idx.min()<0 or self.idx.max()>=self.b.lut.shape[0]:
            raise ValueError(f"Original width map indices outside LUT: {self.idx.min()}..{self.idx.max()}")
        # Center convention matches torch_pad_center: floor(N/2).
        self.phase_source_slices=[];self.phase_dest_slices=[]
        for size in width.shape:
            left=self.meta_n//2-size//2
            lo=max(0,left);hi=min(self.meta_n,left+size)
            self.phase_source_slices.append(slice(lo-left,hi-left))
            self.phase_dest_slices.append(slice(lo,hi))
        self.mapped_pixels_in_aperture=int(np.prod([q.stop-q.start for q in self.phase_dest_slices]))
        self.zero_phase_padding_pixels=self.meta_n**2-self.mapped_pixels_in_aperture
        print("B17_MAP_AUDIT",json.dumps(dict(source_shape=self.width_shape,
            aperture_shape=[self.meta_n,self.meta_n],mapped_pixels=self.mapped_pixels_in_aperture,
            zero_phase_padding_pixels=self.zero_phase_padding_pixels,
            width_min_nm=int(width.min()),width_max_nm=int(width.max()))),flush=True)
    def optics(self,w):
        w=wavelength_check(w)
        n,meta_n,obj,sensor=self.geometry
        c=self.Geometry(propagation_n=n,meta_n=meta_n,object_n=obj,sensor_n=n,
          wavelength_m=w*1e-9,input_pitch_m=350e-9,object_meta_m=400e-6,
          meta_sensor_m=self.z2,sensor_pitch_m=self.pitch)
        lut=np.array([np.interp(w,self.b.lam,row) for row in self.b.lut])
        # Exactly the phase-padding convention in the original Real-scale code.
        phase=np.zeros((meta_n,meta_n),dtype=np.float64)
        phase[tuple(self.phase_dest_slices)]=wrap(lut[self.idx[tuple(self.phase_source_slices)]])
        meta=np.exp(1j*phase).astype(np.complex64)
        del phase
        m=self.Model(c,meta)
        if self.bl!=1.:
            f=fft.fftfreq(n,c.input_pitch_m)
            fx=c.wavelength_m*f[None,:];fy=c.wavelength_m*f[:,None]
            s=fx**2+fy**2;safe=np.sqrt(np.maximum(1-s,1e-30))
            mask=(s<1)&(2*np.abs(c.meta_sensor_m*fx*(1/safe-1))<=self.bl*n*c.input_pitch_m)&(2*np.abs(c.meta_sensor_m*fy*(1/safe-1))<=self.bl*n*c.input_pitch_m)
            m.hcorr*=mask
            del s,safe,mask
        for attr in ("meta","h1","hcorr","qin","qout"):
            setattr(m,attr,getattr(m,attr).astype(np.complex64))
        cam=self.Camera(n,sensor,w*1e-9*c.meta_sensor_m/(n*c.input_pitch_m),c.sensor_pitch_m)
        return m,cam
    def config(self):
        return dict(case=self.case,geometry=list(self.geometry),original_width_shape=self.width_shape,
          aperture_width_m=self.meta_n*350e-9,input_pitch_m=350e-9,
          mapped_pixels_in_aperture=self.mapped_pixels_in_aperture,
          zero_phase_padding_pixels=self.zero_phase_padding_pixels,
          phase_padding_policy="centered original width map; zero phase outside map within nominal aperture, as in original torch_pad_center",
          object_meta_m=400e-6,meta_sensor_m=self.z2,camera_pitch_m=self.pitch,
          camera_width_m=3000*self.pitch,sasm_blfactor=self.bl,
          wavelength_nm=self.w,propagation_n=GRID,
          original_reference="05_NN_metacam-main/src/metacam/nn/phasecam_realscale.py",
          model="phase-only intensity; current ASM/SASM and point-bilinear camera; not original amplitude/Adam code",
          B17_inner_shas=self.inner_shas)

def main():
    p=argparse.ArgumentParser();p.add_argument("--case",choices=tuple(CASES),required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    obs=root/"observation";obs.mkdir();shared=root/"shared_gd";shared.mkdir()
    with fft.set_workers(4):
        engine=OriginalProfile(a.case);w=engine.w;config=engine.config()
        write_json(root/"experiment.json",config)
        truth=phase_check(wrap(image_phase("camera")))
        assert digest(truth)==EXPECTED_TRUTH
        clean=engine.predict(w,truth)
        sigma=.01*float(clean.mean())
        y=np.maximum(0.,clean+sigma*np.random.default_rng(672906).standard_normal(clean.shape))
        solver.sensor_check(y)
        np.savez_compressed(obs/"sensor_only.npz",sensor=y)
        np.savez_compressed(obs/"private_truth.npz",phase=truth,actual_wavelength_nm=w)
        write_json(obs/"generation.json",dict(**config,scene="camera",seed=672906,
            truth_sha256=digest(truth),sensor_sha256=digest(y),clean_sha256=digest(clean),
            noise_sigma=sigma,noise_pct_mean=1.,truth_in_inverse=False))
        del truth,clean;gc.collect()
        m,cam=engine.optics(w)
        def vg(model,camera,x,observed,alpha,grad):
            return engine.value_grad(model,camera,wrap(x),observed,alpha,grad)
        x,h=pyramid.gd_pyramid(m,cam,y,vg,shared,pyramid.SCHEDULES["pyramid64"])
        start=phase_check(wrap(x));del m,cam,x;gc.collect()
        np.savez_compressed(shared/"sealed.npz",phase=start)
        write_json(shared/"summary.json",dict(gd_iterations=150,sensor_sha256=digest(y),initial_phase_sha256=digest(start),history=h))
        # Same existing optimization schedule as legacy baseline.
        P=solver.preconditioner("spectral")
        paths=[]
        first=root/"alpha015"
        final=jac.refine(engine,y,w,start,first,.015,"Pyramid64_GD150",P,
                        "original_geometry_alpha015","fixed historical 660 nm spectral P")
        paths.append(first)
        gc.collect()
        second=root/"alpha005"
        jac.refine(engine,y,w,final,second,.005,"alpha015_sealed",P,
                        "original_geometry_alpha005","fixed historical 660 nm spectral P")
        paths.append(second)
        # No truth reaches inverse. Physical profile metadata for each candidate.
        scores={}
        for path in paths:
            rec=json.loads((path/"recovery.json").read_text())
            rec.update(physical_profile=config,previous_lbfgs_iterations=100 if path==second else 0,
                total_lbfgs_iterations=rec["lbfgs_iterations"]+(100 if path==second else 0))
            write_json(path/"recovery.json",rec)
            scores[path.name]=solver.evaluate(obs/"private_truth.npz",obs/"sensor_only.npz",path)
        write_json(root/"comparison.json",scores)
        print("ORIGINAL_FINAL",json.dumps(dict(config=config,scores=scores)),flush=True)

if __name__=="__main__":main()
