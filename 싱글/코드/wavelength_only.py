"""Wavelength-only diagnostic: calibrated speckle power spectrum, original BL=.5.
No phase inverse. Compare object-insensitive statistical features with direct
flat-image correlation failures. All candidates scored with the same fixed
generic calibration ensemble, independent of case truth. The model is
simulation-matched; the wavelength estimator is experimental, not guaranteed.
"""
import argparse,gc,json
from pathlib import Path
import numpy as np
from scipy import fft,ndimage
from 공통.object_models import N,wrap
from 공통.optics import digest,phase_check,write_json
from 싱글.코드.original_geometry import OriginalProfile
from 싱글.코드.original_blind import CASES,generate
from 싱글.코드 import solver

PATCH=512
BINS=32
FREQ=np.linspace(.015,.5,BINS+1)
F=fft.fftfreq(PATCH)
R=np.hypot(F[:,None],F[None,:])
BIN=np.digitize(R,FREQ)-1
VALID=(BIN>=0)&(BIN<BINS)
LABEL=BIN[VALID]
COUNTS=np.bincount(LABEL,minlength=BINS)
WINDOW=np.hanning(PATCH);WINDOW=WINDOW[:,None]*WINDOW[None,:]
CENTERS=(600,1500,2400)

def feature(sensor):
    """Mean locally normalized 2D intensity PSD, radial log-band powers.
    Nine fixed sensor patches; no target-dependent ROI or scene labels.
    Remove local DC, taper, normalize spectral shape, discard absolute gain.
    """
    band=[]
    for cy in CENTERS:
        for cx in CENTERS:
            p=np.asarray(sensor[cy-PATCH//2:cy+PATCH//2,cx-PATCH//2:cx+PATCH//2],dtype=np.float64)
            p=(p-p.mean())/max(float(p.mean()),1e-30)
            power=np.abs(fft.fft2(p*WINDOW))**2
            b=np.bincount(LABEL,weights=power[VALID],minlength=BINS)/COUNTS
            band.append(b)
    b=np.mean(band,axis=0)
    b=b/max(float(np.sum(b)),1e-30)
    return np.log(np.maximum(b,1e-15))

def calibration_phases():
    out=[np.zeros((N,N),dtype=np.float64)]
    # Fixed generic phase patterns, not derived from any validation object.
    for seed,sigma,std in [(8120,8.,.8),(8121,2.,1.8)]:
        x=ndimage.gaussian_filter(np.random.default_rng(seed).standard_normal((N,N)),sigma)
        x=(x-x.mean())/max(float(x.std()),1e-30)
        out.append(phase_check(wrap(std*x)))
    return out

class Library:
    def __init__(self,engine,root):
        self.engine=engine;self.root=root;self.cache={};self.phases=calibration_phases()
    def get(self,w):
        w=round(float(w),6)
        if w in self.cache:return self.cache[w]
        m,cam=self.engine.optics(w);rows=[]
        for phase in self.phases:
            p=self.engine.measure(m,cam,phase)
            rows.append(feature(p));del p
        del m,cam;gc.collect()
        self.cache[w]=np.stack(rows)
        print("PSD_CALIBRATION",w,flush=True)
        return self.cache[w]
    def save(self):
        ws=sorted(self.cache)
        np.savez_compressed(self.root/"calibrated_features.npz",wavelengths_nm=np.asarray(ws),
            log_psd=np.stack([self.cache[w] for w in ws]),frequency_edges=FREQ)

def score(lib,obs,w):
    t=lib.get(w)
    d=np.sqrt(np.mean((t-obs[None,:])**2,axis=1))
    return dict(wavelength_nm=round(float(w),6),flat_psd=float(d[0]),
        ensemble_mean_psd=float(np.sqrt(np.mean((t.mean(axis=0)-obs)**2))),
        ensemble_min_psd=float(d.min()),template_distances=d.tolist())

def separated(rows,count,gap,metric):
    out=[]
    for r in sorted(rows,key=lambda r:(r[metric],r["wavelength_nm"])):
        if all(abs(r["wavelength_nm"]-v)>=gap for v in out):
            out.append(r["wavelength_nm"])
        if len(out)==count:break
    return out

def estimate(lib,obs):
    # Preregister the ensemble-mean metric for final wavelength selection.
    # Flat and ensemble-min rankings are secondary diagnostics.
    metric="ensemble_mean_psd";records={}
    def add(w):
        w=round(float(w),6)
        if w not in records:records[w]=score(lib,obs,w)
    for w in range(440,661,5):add(w)
    coarse=list(records.values())
    centers=separated(coarse,3,10.,metric)
    for c in centers:
        for w in range(max(440,int(c)-5),min(660,int(c)+5)+1):add(w)
    integer=list(records.values())
    fine_centers=separated(integer,2,2.,metric)
    for c in fine_centers:
        k=int(round(c*10))
        for i in range(max(4400,k-10),min(6600,k+10)+1):add(i/10.)
    ranking=sorted(records.values(),key=lambda r:(r[metric],r["wavelength_nm"]))
    return dict(selected_wavelength_nm=ranking[0]["wavelength_nm"],metric=metric,
        coarse_centers_nm=centers,fine_centers_nm=fine_centers,
        coarse_ranking=sorted(coarse,key=lambda r:r[metric]),ranking=ranking,
        secondary_flat_best=min(records.values(),key=lambda r:r["flat_psd"])["wavelength_nm"],
        secondary_ensemble_min_best=min(records.values(),key=lambda r:r["ensemble_min_psd"])["wavelength_nm"],
        truth_in_inverse=False)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",nargs="+",choices=tuple(CASES),default=list(CASES))
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    with fft.set_workers(4):
        engine=OriginalProfile("original660_BL05");lib=Library(engine,root)
        write_json(root/"experiment.json",dict(physical_profile=engine.config(),cases=a.cases,
            method="radial locally normalized speckle PSD; ensemble mean log-power distance",
            calibration="flat + two independent generic Gaussian-filtered random phase maps",
            patch_size=PATCH,patch_centers=list(CENTERS),radial_bins=BINS,
            calibration_phase_sha256=[digest(x) for x in lib.phases],truth_in_inverse=False,
            phase_reconstruction=False))
        # Generate each sensor using the exact prior blind case and retain only
        # measured features for inverse. Generation truth is not an argument to estimate.
        observed={}
        for case in a.cases:
            scene,w=CASES[case];dest=root/case;dest.mkdir();obs=dest/"observation"
            generate(engine,obs,scene,w)
            with np.load(obs/"sensor_only.npz",allow_pickle=False) as z:
                y=solver.sensor_check(z["sensor"]);f=feature(y)
                np.savez_compressed(dest/"observed_feature.npz",log_psd=f)
            del y;gc.collect()
            observed[case]=f
        results={}
        for case,f in observed.items():
            dest=root/case
            r=estimate(lib,f)
            write_json(dest/"wavelength_search.json",r)
            # Evaluation only: exact true wavelength score, even if omitted
            # from the candidate search. Never changes r or its ranking.
            scene,w=CASES[case]
            oracle=score(lib,f,w)
            true_rank=1+sum(v[r["metric"]]<oracle[r["metric"]] for v in r["ranking"])
            est=r["selected_wavelength_nm"];err=abs(est-w)
            ev=dict(case=case,scene=scene,actual_wavelength_nm=w,estimated_wavelength_nm=est,
                wavelength_error_nm=err,pass_0p1_nm=err<=.100000001,pass_0p5_nm=err<=.500000001,
                true_wavelength_score_posthoc=oracle,true_rank_posthoc=true_rank,
                true_wavelength_was_searched=any(abs(v["wavelength_nm"]-w)<1e-8 for v in r["ranking"]),
                flat_psd_estimate_nm=r["secondary_flat_best"],ensemble_min_estimate_nm=r["secondary_ensemble_min_best"],
                coarse_centers_nm=r["coarse_centers_nm"],fine_centers_nm=r["fine_centers_nm"],
                top_candidates=r["ranking"][:5],phase_reconstruction=False)
            write_json(dest/"evaluation.json",ev);results[case]=ev
            write_json(root/"comparison.json",results);lib.save()
            print("WAVELENGTH_ONLY_FINAL",json.dumps(ev),flush=True)
if __name__=="__main__":main()
