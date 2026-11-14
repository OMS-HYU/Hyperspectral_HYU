"""Trace phase sensitivity through the unchanged B17 optical operator.
Diagnostic only: ground truth is allowed here, never for recovery.
Separate free-space propagating spectrum, numerical ASM truncation, finite
metasurface aperture, numerical SASM correction mask, and camera intensity.
"""
import argparse,json,gc
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.optics import FullB17,phase_check,digest,write_json
from 공통.object_models import image_phase,wrap,N
from 싱글.코드.wavelength_jacobian import probes
from 싱글.코드.alpha_continuation import EXPECTED_TRUTH
from 싱글.코드.grid_convergence import spectrum_error

RADII=(16,64,128,192,224,320)
GRID=3800

def power(a):
    return float(np.vdot(a,a).real)

def ratio(a,b):
    return float(a/max(b,1e-30))

def stage(m,cam,du,z,ref):
    n=m.c.propagation_n
    full=np.zeros_like(m.meta);full[m.obj,m.obj]=du
    sp=fft.fft2(full,norm="ortho")
    p0=power(full);ps=power(sp)
    del full
    f=fft.fftfreq(n,m.c.input_pitch_m)
    f2=(m.c.wavelength_m*f)**2
    propagating=(f2[:,None]+f2[None,:])<1
    pprop=power(sp[propagating]);del propagating
    sp*=m.h1
    p1=power(sp)
    a=fft.ifft2(sp,norm="ortho");del sp
    a*=m.meta
    p2=power(a)
    sp2=fft.fft2(fft.ifftshift(a),norm="ortho")
    p3=power(sp2);sp2*=m.hcorr
    p4=power(sp2);del sp2
    dz=m.sas(a);del a
    pf=power(dz)
    j=cam.forward(np.asarray(2*np.real(np.conj(z)*dz),dtype=np.float64))
    jj=power(j);rr=power(ref)
    j-=ref*(float(np.sum(j*ref))/rr)
    return dict(input_power=p0,parseval_relative_error=abs(ps-p0)/max(p0,1e-30),
      propagating_fraction=ratio(pprop,ps),
      asm_pass_fraction_of_input=ratio(p1,ps),
      asm_numerical_retention_of_propagating=ratio(p1,pprop),
      aperture_retention_after_asm=ratio(p2,p1),
      aperture_fraction_of_input=ratio(p2,p0),
      sasm_correction_retention_after_aperture=ratio(p4,p3),
      sasm_pass_fraction_of_input=ratio(p4,p0),
      sensor_field_power_unscaled=ratio(pf,p0),
      intensity_sensitivity_raw=float(np.sqrt(jj/rr)),
      intensity_sensitivity_gain_projected=float(np.sqrt(power(j)/rr)))

def load_phase(path,expected):
    r=json.loads((path/"recovery.json").read_text())
    e=json.loads((path/"evaluation.json").read_text())
    assert abs(e["full_rmse_rad"]-expected)<1e-12
    with np.load(path/"recovered.npz",allow_pickle=False) as z:p=phase_check(z["phase"])
    assert digest(p)==r["phase_sha256"]
    return p

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--wavelength",type=float,choices=(440.,550.,660.),required=True)
    p.add_argument("--best",type=Path,required=True)
    p.add_argument("--low-residual",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();w=a.wavelength;out=a.output;out.mkdir(parents=True,exist_ok=False)
    truth=phase_check(wrap(image_phase("camera")))
    assert digest(truth)==EXPECTED_TRUTH
    best=load_phase(a.best/"inverse",.12986880384061672)
    low=load_phase(a.low_residual/"inverse",.14452645528773578)
    with fft.set_workers(4):
        engine=FullB17(propagation_n=GRID);m,cam=engine.optics(w)
        results={}
        # Compare same phase perturbations at every wavelength. Flat and true
        # reference are both diagnostics, not candidate reconstructions.
        for name,phase in (("flat",np.zeros((N,N))),("truth",truth)):
            u=np.exp(1j*phase.astype(np.float32)).astype(np.complex64)
            z=m.forward(u)
            ref=cam.forward(np.asarray(z.real*z.real+z.imag*z.imag,dtype=np.float64))
            rows=[]
            for r in RADII:
                for kx,ky,d in probes(r):
                    row=stage(m,cam,(1j*u*d).astype(np.complex64),z,ref)
                    row.update(radius=r,kx=kx,ky=ky)
                    assert row["parseval_relative_error"]<.001
                    rows.append(row);print("STAGE",name,json.dumps(row),flush=True)
                gc.collect()
            keys=("propagating_fraction","asm_pass_fraction_of_input",
              "asm_numerical_retention_of_propagating","aperture_retention_after_asm",
              "aperture_fraction_of_input","sasm_correction_retention_after_aperture",
              "sasm_pass_fraction_of_input","intensity_sensitivity_gain_projected")
            radial=[dict(radius=r,**{k:float(np.median([row[k] for row in rows if row["radius"]==r])) for k in keys}) for r in RADII]
            results[name]=dict(probes=rows,radial=radial)
            del u,z,ref;gc.collect()
        # Compare the same two 660-nm solutions at all wavelengths.
        # At 440/550 these are counterfactual phase maps, not actual inversions.
        u0=np.exp(1j*truth.astype(np.float32)).astype(np.complex64)
        z0=m.forward(u0);y=cam.forward(np.asarray(z0.real*z0.real+z0.imag*z0.imag,dtype=np.float64))
        yy=power(y);rr=np.sqrt(yy)
        paths={}
        for name,phase in (("best_alpha005",best),("low_residual_clean_LBFGS",low)):
            off=float(np.angle(np.mean(np.exp(1j*(phase-truth)))))
            error=np.angle(np.exp(1j*(phase-truth-off)))
            aligned=wrap(truth+error)
            uc=np.exp(1j*aligned.astype(np.float32)).astype(np.complex64)
            delta=uc-u0
            dstat=stage(m,cam,delta,z0,y)
            # Actual complex field perturbation is not a linearized phase probe;
            # report its exact field transmission, then exact intensity residuals.
            curve=[]
            for t in (0.,.125,.25,.5,.75,.875,1.):
                x=wrap(truth+t*error)
                z=m.forward(np.exp(1j*x.astype(np.float32)).astype(np.complex64))
                pred=cam.forward(np.asarray(z.real*z.real+z.imag*z.imag,dtype=np.float64))
                q=float(np.sum(y*pred)/power(pred))
                curve.append(dict(t=t,phase_rmse_rad=float(t*np.sqrt(np.mean(error**2))),
                    fitted_gain=q,fixed_gain_sensor_l2=float(np.linalg.norm(pred-y)/rr),
                    profiled_sensor_l2=float(np.linalg.norm(q*pred-y)/rr),
                    complex_field_relative_l2=float(np.linalg.norm(z-z0)/np.linalg.norm(z0))))
                del z,pred
            paths[name]=dict(phase_sha256=digest(phase),error_spectrum=spectrum_error(error),
                exact_complex_field_delta_stages=dstat,interpolation=curve)
            print("PATH",name,json.dumps(paths[name]),flush=True)
        result=dict(wavelength_nm=w,geometry=list(engine.geometry),
          B17_inner_shas=engine.inner_shas,truth_sha256=digest(truth),
          common_phase_sources=dict(best_alpha005_run=43604536423,low_residual_run=43606880549),
          diagnostic_only=True,physical_system_changed=False,
          stage_normalization="power fractions; input complex tangent, unit RMS phase probes; intensity normalized by reference L2 after gain projection",
          radial_units="cycles across 512 x 350nm object; directions median",
          references=results,paths=paths)
        write_json(out/"diagnostics.json",result)
        print("TRANSFER_FINAL",json.dumps(result),flush=True)
if __name__=="__main__":main()
