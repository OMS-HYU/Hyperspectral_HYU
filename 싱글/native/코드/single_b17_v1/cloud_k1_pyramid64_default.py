#!/usr/bin/env python3
"""K1 default: native-B17 Pyramid64 GD150 -> full-512 L-BFGS150.

Optics/phase/sensor geometry (3800,1429,512,3000) stays original.
This script runs only single-wavelength synthetic tests; all true-phase
data are sealed and opened only by post-hoc evaluation.
"""
from __future__ import annotations
import argparse,gc,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy import fft
from skimage import data,color,transform

import cloud_k1_astronaut_multires as pyramid
import cloud_k1_endpoint_robust as reference

SCENES=("moon","brick","chelsea","camera_rot90")
WAVELENGTHS=(440,550,660)
SEEDS={"moon":261201,"brick":261202,"chelsea":261203,"camera_rot90":261204}
NOISE_PCTS=(1,3)
GEOMETRY=(3800,1429,512,3000)
Pyramid64_SCHEDULE=((64,35),(128,35),(256,40),(512,40))
ROOT=Path.cwd()/"b17_k1_pyramid64_stress_outputs"

def arrhash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def case_seed(scene,noise_pct):
    return SEEDS[scene] + (10000 if noise_pct==3 else 0)

def true_phase(scene):
    if scene=="moon":image=data.moon()
    elif scene=="brick":image=data.brick()
    elif scene=="chelsea":image=color.rgb2gray(data.chelsea())*255.0
    elif scene=="camera_rot90":image=np.rot90(data.camera(),1)
    else:raise ValueError(scene)
    if image.shape!=(512,512):
        image=transform.resize(image,(512,512),order=3,anti_aliasing=True,preserve_range=True)
    u8=np.uint8(np.rint(np.clip(image,0,255)))
    return np.ascontiguousarray(u8,dtype=np.float64)*(2*np.pi/255.)

def self_test():
    assert pyramid.SCHEDULES["pyramid64"]==Pyramid64_SCHEDULE
    assert reference.REG_LAMBDA==.015
    assert sum(n for _,n in Pyramid64_SCHEDULE)==150
    for scene in SCENES:
        phase=true_phase(scene)
        assert phase.shape==(512,512) and np.isfinite(phase).all()
        assert 0<=float(phase.min())<=float(phase.max())<=2*np.pi
        assert np.array_equal(phase,true_phase(scene))
        print("K1_SCENE_VALID",scene,"phase_std",float(np.std(phase)),flush=True)
    pyramid.self_test()
    print("K1_PYRAMID64_DEFAULT_SELFTEST_OK",flush=True)

def generate(scene,wavelength,noise_pct,optics,measure,dest):
    obs=dest/"sensor_only.npz";private=dest/"private_truth.npz"
    if obs.exists() or private.exists():
        assert obs.exists() and private.exists(),"Corrupt cached observation/truth pair"
        with np.load(obs,allow_pickle=False) as z:
            assert int(z["seed"])==case_seed(scene,noise_pct)
            assert int(z["noise_pct"])==noise_pct
            assert float(z["wavelength_nm"])==wavelength
        return
    phase=true_phase(scene)
    metasurface,cam=optics(wavelength)
    clean=measure(metasurface,cam,phase)
    sigma=float(np.mean(clean))*noise_pct/100.
    rng=np.random.default_rng(case_seed(scene,noise_pct))
    y=np.maximum(0.,clean+sigma*rng.standard_normal(clean.shape))
    reference.atomic_npz(obs,sensor=y,noise_sigma=np.float64(sigma),
        seed=np.int32(case_seed(scene,noise_pct)),noise_pct=np.int32(noise_pct),
        wavelength_nm=np.float64(wavelength))
    reference.atomic_npz(private,phase=phase,wavelength_nm=np.float64(wavelength))
    print("K1_SENSOR_CREATED",json.dumps({"scene":scene,"nm":wavelength,
        "noise_pct":noise_pct,"seed":case_seed(scene,noise_pct),
        "sensor_sha256":arrhash(y)}),flush=True)
    del metasurface,cam,clean,y,phase;gc.collect()

def run(scene,wavelength,noise_pct):
    from k1_cloud_batch import check_b17
    if noise_pct==3 and wavelength!=660:
        raise ValueError("3 percent stress is restricted to 660 nm")
    optics,measure,value_grad,wrapped_rmse,b17_sha,inner_shas=check_b17()
    dest=ROOT/f"{scene}_{wavelength}nm_noise{noise_pct}"
    dest.mkdir(parents=True,exist_ok=True)
    generate(scene,wavelength,noise_pct,optics,measure,dest)
    obs=dest/"sensor_only.npz"
    sensor_zip_sha=reference.sha(obs)
    with np.load(obs,allow_pickle=False) as z:y=z["sensor"].copy()
    sensor_sha=arrhash(y)
    audit={
        "scene":scene,"wavelength_nm":wavelength,"noise_pct":noise_pct,
        "seed":case_seed(scene,noise_pct),"sensor_sha256":sensor_sha,
        "sensor_zip_sha256":sensor_zip_sha,
        "B17_original_zip_SHA256":b17_sha,"B17_LUT_width_map_shas":inner_shas,
        "geometry":GEOMETRY,"phase_encoding":"2*pi*uint8/255",
        "default_method":"Pyramid64 GD150 -> native L-BFGS150",
        "pyramid_schedule":Pyramid64_SCHEDULE,
        "GD_1_100":"no prior","GD_101_150":"circular alpha 0.015",
        "L_BFGS":"circular alpha 0.015 maxiter150",
        "true_phase_used_in_fitting":False,
        "original_metaphysics_and_sensor_unchanged":True}
    (dest/"manifest.json").write_text(json.dumps(audit,indent=2))
    t0=time.monotonic()
    with fft.set_workers(1):
        m,cam=optics(wavelength)
        gd150,hist=pyramid.gd_pyramid(m,cam,y,value_grad,dest,Pyramid64_SCHEDULE)
        reference.atomic_npz(dest/"gd150_phase.npz",phase=gd150,gd_steps=np.int32(150))
        x,lb_info=reference.lbfgs(m,cam,y,value_grad,dest,"circular",gd150)
        result=reference.evaluate(wavelength,scene,"circular",x,
            sensor_zip_sha,m,cam,y,value_grad,wrapped_rmse,dest,lb_info,hist)
    result.update(scene=scene,wavelength_nm=wavelength,noise_pct=noise_pct,
        noise_seed=case_seed(scene,noise_pct),sensor_array_sha256=sensor_sha,
        algorithm="Pyramid64 GD150 + L-BFGS150",
        pyramid_schedule=Pyramid64_SCHEDULE,geometry=GEOMETRY,
        B17_original=True,true_phase_only_posthoc=True,
        elapsed_seconds=time.monotonic()-t0)
    (dest/"evaluation_circular.json").write_text(json.dumps(result,indent=2))
    print("RESULT_K1_STRESS",json.dumps({k:result[k] for k in
        ("scene","wavelength_nm","noise_pct","noise_seed","phase_rmse_rad",
         "sensor_l2","passed","lbfgs_actual_steps","elapsed_seconds",
         "sensor_array_sha256")}),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scene",choices=SCENES)
    p.add_argument("--wavelength",type=int,choices=WAVELENGTHS)
    p.add_argument("--noise-pct",type=int,choices=NOISE_PCTS,default=1)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test()
    if args.self_test:return
    if args.scene is None or args.wavelength is None:
        p.error("scene and wavelength required")
    run(args.scene,args.wavelength,args.noise_pct)

if __name__=="__main__":main()
