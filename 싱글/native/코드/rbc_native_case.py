"""Preserved physical-RBC phase inputs, native full-resolution separate solver."""
import argparse
import gc
import hashlib
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import fft
import native_cli as io
import batch_runner as b
from tank_precision_recovery import DoubleModel
from rbc_native_solver import fit
from render_five_panel import render

HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser()
    for k in ['run','archive','truth','calibration']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--scene',choices=['normal','variable','tilt','double','triple'],required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--metadata',type=Path)
    a=ap.parse_args();r=a.run;r.mkdir(parents=True,exist_ok=True)
    validation=io.loadj(a.calibration/'validation.json')
    if not validation['passed']:raise ValueError('Native RBC numerical validation required')
    def stage(name):io.write(r/'stage.json',dict(stage=name));print('RBC_STAGE',name,flush=True)
    with b.batch_lock(r/'case.lock'):
        if not (r/'manifest.json').exists():
            stage('generate')
            b.invoke([sys.executable,str(HERE/'native_cli.py'),'generate','--run',str(r),'--archive',str(a.archive),
                      '--truth',str(a.truth),'--scene',a.scene,'--seed',str(a.seed),'--wavelength','660'],r/'generation.log')
        man=io.loadj(r/'manifest.json')
        if man['scene']!=a.scene or man['noise_seed']!=a.seed:raise ValueError('RBC run identity differs')
        sensor=np.load(r/'sensor.npy',allow_pickle=False)
        if io.sha(sensor)!=man['sensor_sha256']:raise ValueError('Sensor hash mismatch')
        weights=np.load(a.calibration/'spectral_weights.npy',allow_pickle=False)
        if io.sha(weights)!=validation['weights_sha256']:raise ValueError('Native spectral calibration hash mismatch')
        identity=dict(scene=a.scene,sensor_sha256=man['sensor_sha256'],archive_sha256=man['archive_sha256'],
            solver_sha256=hashlib.sha256((HERE/'rbc_native_solver.py').read_bytes()).hexdigest(),
            native_model_sha256=hashlib.sha256((HERE/'native_model.py').read_bytes()).hexdigest(),
            model_precision_sha256=hashlib.sha256((HERE/'tank_precision_recovery.py').read_bytes()).hexdigest(),
            calibration_weights_sha256=io.sha(weights),phase_resolution=512,
            method='Full BB/Armijo GD100 + native spectral-preconditioned periodic L-BFGS100',alpha=.015)
        stage('full_GD100_spectral_periodic_L100')
        with fft.set_workers(4):
            model=DoubleModel(a.archive,660)
            if model.archive_sha256!=man['archive_sha256']:raise ValueError('Archive mismatch')
            target=torch.from_numpy(sensor)
            result=fit(r/'full_solver',lambda p,alpha:model.fg(p,target,alpha),weights,identity)
        phase=np.load(r/'full_solver/latest_phase.npy',allow_pickle=False)
        io.save_phase(r/'final_phase.npy',phase)
        stage('posthoc_evaluate')
        # Truth/masks are opened only after the estimate has been sealed.
        truth=np.load(r/'private_truth.npy',allow_pickle=False)
        if io.sha(truth)!=man['truth_sha256']:raise ValueError('RBC truth hash mismatch')
        error=np.angle(np.exp(1j*(phase-truth)));rms=lambda x:float(np.sqrt(np.mean(x*x)))
        roi=truth>1e-8
        roi_source='phase-support mask for posthoc ROI only; cell/group metadata unavailable'
        groups=None;labels=None
        if a.metadata:
            with np.load(a.metadata,allow_pickle=False) as z:
                if io.sha(z['phase'])!=man['truth_sha256']:raise ValueError('ROI metadata phase does not match this exact truth')
                labels=z['labels'];groups=z['groupmap'];roi=labels>0;roi_source='original matched phase/labels/groupmap metadata'
        with fft.set_workers(4):
            predicted=model.measure(phase);gain=float(np.sum(predicted*sensor)/np.sum(predicted*predicted))
        l2=float(np.linalg.norm((gain*predicted-sensor).ravel())/np.linalg.norm(sensor.ravel()))
        global_pass=rms(error)<=.15 and l2<=.025
        roi_rmse=rms(error[roi]) if np.any(roi) else None
        group_scores=[];layers={}
        if labels is not None:
            for k in np.unique(labels[labels>0]):layers[str(int(k))]=rms(error[labels==k])
            for k in np.unique(groups[groups>0]):group_scores.append(dict(group=int(k),rmse_rad=rms(error[groups==k])))
        cell_validated=bool(group_scores) and roi_rmse<=.15 and all(x['rmse_rad']<=.15 for x in group_scores) and all(x<=.15 for x in layers.values())
        evaluation=dict(scene=a.scene,wavelength_nm=660,variant=identity['method'],geometry=man['geometry'],observation='amplitude',
             gd_accepted=result['gd_accepted'],lbfgs_accepted=result['lbfgs_accepted'],termination=result['termination'],
             circular_rmse_rad=rms(error),linear_rmse_rad=rms(phase-truth),roi_rmse_rad=roi_rmse,
             background_rmse_rad=rms(error[~roi]) if np.any(~roi) else None,
             reference_float32_sensor_l2=l2,recovery_float64_sensor_l2=l2,
             sensor_metric_model='native double precision; reference float32 not separately evaluated',
             global_phase_sensor_pass=global_pass,quality_pass=global_pass and cell_validated,
             cell_group_validation_available=bool(group_scores),cell_group_pass=cell_validated,
             roi_source=roi_source,group_scores=group_scores,layer_rmse_rad=layers,
             sensor_sha256=man['sensor_sha256'],truth_sha256=man['truth_sha256'],phase_sha256=io.sha(phase),
             original_phase_source='preserved 39-input archive; not regenerated or replaced',
             truth_roi_used_for_optimization=False,preview='comparison.png')
        io.write(r/'completion_policy.json',dict(gd_accepted=result['gd_accepted'],lbfgs_accepted=result['lbfgs_accepted']))
        render(r,r/'comparison.png',phase_path=r/'final_phase.npy')
        io.write(r/'evaluation.json',evaluation)
        stage('completed');print(evaluation,flush=True)


if __name__=='__main__':main()
