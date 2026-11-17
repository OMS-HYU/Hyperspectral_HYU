#!/usr/bin/env python3
"""B17 K=1 Astronaut 660nm phase-pyramid optimization (original 3800^2 forward grid).

Changes ONLY optimization parameter resolution; no downsampling of the B17
LUT/meta mask, the optical FFTs, or camera observation. Each arm uses exactly
150 Armijo GD updates across pyramid stages, then L-BFGS-B maxiter=150 in native
512x512 phase variables. Objective and regularizer are computed on the 512^2
phase field. Ground truth never enters inverse or parameter selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import fft, sparse, ndimage

import cloud_k1_endpoint_robust as reference

SCENE = "astronaut"
WAVELENGTH_NM = 660
ROOT = Path.cwd() / "b17_k1_multires_outputs"
SCHEDULES = {
    "pyramid64": ((64,35),(128,35),(256,40),(512,40)),
    "pyramid128": ((128,50),(256,50),(512,50)),
}
REG = .015
EXPECTED_SENSOR_ARRAY_SHA = "4cde1bf19ef9a5dc10d46445c90880cb45f93ed42e3cebdd1df1ffc956975b7a"
ORIGINAL_ZERO_RMSE = .20491063836522266
ORIGINAL_ZERO_SENSOR_L2 = .04501716279730178


def arrhash(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def make_interp(n, target=512):
    """1D linear interpolation matrix; [target x n]."""
    if not 2 <= n <= target:
        raise ValueError(n)
    if n==target:
        return sparse.identity(target, dtype=np.float64, format="csr")
    t=np.linspace(0,n-1,target)
    lo=np.floor(t).astype(np.int64)
    hi=np.minimum(lo+1,n-1)
    f=t-lo
    rows=np.concatenate([np.arange(target),np.arange(target)])
    cols=np.concatenate([lo,hi])
    weights=np.concatenate([1.-f,f])
    return sparse.coo_matrix((weights,(rows,cols)),shape=(target,n),dtype=np.float64).tocsr()


def full_from_vars(vars_, B):
    tmp=B @ np.asarray(vars_,dtype=np.float64)
    return np.asarray((B @ tmp.T).T,dtype=np.float64)


def grad_to_vars(grad_full, B):
    tmp=B.T @ np.asarray(grad_full,dtype=np.float64)
    return np.asarray((B.T @ tmp.T).T,dtype=np.float64)


def sample_full_to_vars(x, n):
    if n==512:
        return np.array(x,dtype=np.float64,copy=True)
    coords=np.linspace(0,511,n)
    yy,xx=np.meshgrid(coords,coords,indexing="ij")
    return ndimage.map_coordinates(np.asarray(x,dtype=np.float64),
                                  [yy,xx],order=1,mode="nearest")


def self_test():
    reference.self_test()
    assert reference.REG_LAMBDA==REG
    for name,sch in SCHEDULES.items():
        assert sum(c for _,c in sch)==150, (name,sch)
        assert sch[-1][0]==512
        assert all(sch[i][0]<sch[i+1][0] for i in range(len(sch)-1))
    rng=np.random.default_rng(10111)
    for n in (8,16,32,128,512):
        B=make_interp(n,target=512)
        a=rng.normal(size=(n,n))
        y=rng.normal(size=(512,512))
        full=full_from_vars(a,B)
        adj=grad_to_vars(y,B)
        lhs=float(np.sum(full*y))
        rhs=float(np.sum(a*adj))
        rel=abs(lhs-rhs)/max(1.,abs(lhs),abs(rhs))
        assert rel<1e-11,(n,rel)
        assert full.shape==(512,512)
        assert np.all(np.isfinite(full))
        print("INTERP_ADJOINT_TEST_OK",n,rel,flush=True)
    ones=np.ones((64,64),dtype=np.float64)*1.234
    assert np.max(np.abs(full_from_vars(ones,make_interp(64))-1.234))<1e-12
    print("ALL_MULTIPYRAMID_SELFTESTS_PASSED",flush=True)


def coupled_value_grad(model,cam,y,value_grad,phase_vars,B,step,grad=True):
    p=full_from_vars(phase_vars,B)
    lam=REG if step>100 else 0.
    if grad:
        data_loss,g=value_grad(model,cam,p,y,0.,True)
        gf=np.asarray(g).reshape(512,512)
        if lam:
            e,pg=reference.penalty(p,"circular",lam,grad=True)
            total=float(data_loss+e)
            gf=gf+pg
        else:
            total=float(data_loss)
        return total,grad_to_vars(gf,B),float(data_loss)
    data_loss,_,_=value_grad(model,cam,p,y,0.,False)
    if lam:
        data_loss+=reference.penalty(p,"circular",lam,grad=False)[0]
    return float(data_loss)


def gd_pyramid(model,cam,y,value_grad,dest,schedule):
    vars_=None
    prev_B=None
    history=[]
    accepted=0
    start=time.monotonic()
    for n,steps in schedule:
        B=make_interp(n)
        if vars_ is None:
            vars_=np.zeros((n,n),dtype=np.float64)
        else:
            vars_=sample_full_to_vars(full_from_vars(vars_,prev_B),n)
        rate=0.
        print("PYRAMID_STAGE_START",n,steps,"global",accepted,flush=True)
        loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,accepted,True)
        for k in range(steps):
            current_step=accepted+1
            if current_step==101:
                rate=0.
                loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,current_step,True)
            gn=float(np.sum(g*g))
            if gn<=1e-42 or not np.isfinite(gn):
                raise RuntimeError(f"Gradient vanished at {n=} step={current_step}")
            gmax=float(np.max(np.abs(g)))
            max_rate=.25/max(gmax,1e-30)
            rate_try=max_rate if rate==0. else min(rate*1.5,max_rate)
            for back in range(32):
                candidate=vars_-rate_try*g
                new_loss=coupled_value_grad(model,cam,y,value_grad,
                                            candidate,B,current_step,False)
                if np.isfinite(new_loss) and new_loss <= loss - 1e-4*rate_try*gn:
                    break
                rate_try*=.5
            else:
                raise RuntimeError(f"Armijo failed at stage n={n}, GD={current_step}, loss={loss}")
            vars_=candidate
            loss,g,data=coupled_value_grad(model,cam,y,value_grad,vars_,B,current_step,True)
            rate=rate_try
            accepted=current_step
            if accepted%10==0 or k+1==steps:
                rec={"gd_step":accepted,"variables_n":n,"total_obj":float(loss),
                     "sensor_loss":float(data),"backtracks":back,
                     "elapsed_s":time.monotonic()-start}
                history.append(rec)
                print("MULTIRES_GD",json.dumps(rec),flush=True)
            if accepted%25==0 or k+1==steps:
                reference.atomic_npz(dest/f"gd_{accepted:03d}_phase.npz",
                                     phase=full_from_vars(vars_,B),
                                     gd_steps=np.int32(accepted),parameter_n=np.int32(n))
        prev_B=B
    assert accepted==150
    assert schedule[-1][0]==512
    phase=full_from_vars(vars_,prev_B)
    return phase,history


def run(scheme):
    from k1_cloud_batch import check_b17
    optics,measure,value_grad,wrapped_rmse,b17_hash,inner_hashes=check_b17()
    dest=ROOT/scheme
    dest.mkdir(parents=True,exist_ok=True)
    reference.generate(WAVELENGTH_NM,optics,measure,dest,SCENE)
    obs=dest/"sensor_only.npz"
    with np.load(obs,allow_pickle=False) as z:
        y=z["sensor"].copy()
        assert float(z["wavelength_nm"])==WAVELENGTH_NM
    observed_hash=arrhash(y)
    assert observed_hash==EXPECTED_SENSOR_ARRAY_SHA,(observed_hash,EXPECTED_SENSOR_ARRAY_SHA)
    zipped_observation_hash=reference.sha(obs)
    manifest={
        "scene":SCENE,"wavelength_nm":WAVELENGTH_NM,"schedule":SCHEDULES[scheme],
        "source_B17_zip_sha":b17_hash,"B17_inner_sha":inner_hashes,
        "geometry":[3800,1429,512,3000],"same_sensor_array_sha256":observed_hash,
        "GD_total":150,"GD_1_100_no_prior":True,"GD_101_150_circular_alpha":REG,
        "L_BFGS_maxiter":150,"L_BFGS_circular_alpha":REG,
        "true_phase_used_for_fitting":False,
        "baseline_run":globals().get("BASELINE_RUN_ID",42810998498),"baseline_phase_rmse":ORIGINAL_ZERO_RMSE,
        "baseline_sensor_l2":ORIGINAL_ZERO_SENSOR_L2
    }
    (dest/"manifest.json").write_text(json.dumps(manifest,indent=2))
    start=time.monotonic()
    with fft.set_workers(1):
        model,cam=optics(WAVELENGTH_NM)
        gd150,hist=gd_pyramid(model,cam,y,value_grad,dest,SCHEDULES[scheme])
        reference.atomic_npz(dest/"gd150_phase.npz",phase=gd150,gd_steps=np.int32(150))
        final,lb_info=reference.lbfgs(model,cam,y,value_grad,dest,"circular",gd150)
        evaluation=reference.evaluate(WAVELENGTH_NM,SCENE,"circular",final,
            zipped_observation_hash,model,cam,y,value_grad,wrapped_rmse,
            dest,lb_info,hist)
    evaluation.update(
        scheme=scheme,
        schedule=SCHEDULES[scheme],
        sensor_array_sha256=observed_hash,
        original_zero_phase_rmse_rad=ORIGINAL_ZERO_RMSE,
        original_zero_sensor_l2=ORIGINAL_ZERO_SENSOR_L2,
        relative_change_rmse_percent=100*(evaluation["phase_rmse_rad"]/
                                          ORIGINAL_ZERO_RMSE-1),
        total_elapsed_seconds=time.monotonic()-start,
        true_phase_only_posthoc=True)
    (dest/"evaluation_circular.json").write_text(json.dumps(evaluation,indent=2))
    print("RESULT_MULTIPYRAMID",json.dumps({
        k:evaluation[k] for k in ("scheme","phase_rmse_rad","sensor_l2",
             "passed","lbfgs_actual_steps","sensor_array_sha256",
             "relative_change_rmse_percent","total_elapsed_seconds")
    }),flush=True)
    return evaluation


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--scheme",choices=SCHEDULES)
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    self_test()
    if args.self_test:
        return
    if args.scheme is None:
        p.error("--scheme is required")
    run(args.scheme)


if __name__ == "__main__":
    main()
