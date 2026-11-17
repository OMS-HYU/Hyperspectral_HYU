"""Native ordinary-photo GD<=150/L-BFGS<=150 with truth-free convergence stops."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import scipy
import torch
from scipy import fft
import native_cli as cli
import native_model as nm
import batch_runner as b
from tank_precision_recovery import DoubleModel,solve

HERE=Path(__file__).resolve().parent
THRESHOLDS=dict(sensor_l2=.025,relative_objective_change=1e-5,phase_rms_rad=1e-3,window=10)


def convergence(history,current_phase,previous_phase):
    """No truth/ROI/group map enters this decision."""
    window=THRESHOLDS['window']
    if len(history)<window+1 or previous_phase is None:return False,{'reason':'insufficient history'}
    values=np.array([x['objective'] for x in history[-window-1:]],float)
    l2=float(history[-1]['sensor_l2'])
    relative=float(np.ptp(values)/max(abs(values[0]),1e-30))
    phase_rms=float(np.sqrt(np.mean(np.angle(np.exp(1j*(current_phase-previous_phase)))**2)))
    evidence=dict(sensor_l2=l2,relative_objective_change=relative,phase_rms_rad=phase_rms,truth_used=False)
    good=np.isfinite([relative,phase_rms,l2]).all() and l2<=THRESHOLDS['sensor_l2'] and relative<=THRESHOLDS['relative_objective_change'] and phase_rms<=THRESHOLDS['phase_rms_rad']
    return bool(good),evidence


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ['run','archive','truth']:ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--scene',required=True);ap.add_argument('--seed',type=int,required=True)
    a=ap.parse_args();r=a.run;r.mkdir(parents=True,exist_ok=True)
    if scipy.__version__!='1.17.0':raise RuntimeError('SciPy 1.17.0 required')
    def stage(name):b.write_json(r/'policy_stage.json',dict(stage=name));print('POLICY_STAGE',name,flush=True)
    common=['--run',str(r),'--archive',str(a.archive)]
    base=[sys.executable,str(HERE/'native_cli.py')]
    with b.batch_lock(r/'ordinary_case.lock'):
        if not (r/'manifest.json').exists():
            stage('generate')
            b.invoke(base+['generate',*common,'--truth',str(a.truth),'--scene',a.scene,'--seed',str(a.seed),'--wavelength','660'],r/'policy_children.log')
        man=cli.loadj(r/'manifest.json')
        if man['scene']!=a.scene or man['noise_seed']!=a.seed:raise ValueError('Run identity mismatch')
        stage('gd_up_to_150')
        previous=None;gd_reason='budget';gd_evidence={}
        stop_path=r/'gd_policy_stop.json'
        if stop_path.exists():
            stopped=cli.loadj(stop_path);gd_reason=stopped['reason'];gd_evidence=stopped.get('evidence',{})
            if b.array_sha(r/'latest_phase.npy')!=stopped['phase_sha256']:raise ValueError('Stopped GD phase changed')
        else:
            while True:
                total=0
                if (r/'gd_state.npz').exists():
                    with np.load(r/'gd_state.npz',allow_pickle=False) as z:total=int(z['total'])
                if total>=150:break
                b.invoke(base+['gd',*common,'--steps',str(min(10,150-total))],r/'policy_children.log')
                p=np.load(r/'latest_phase.npy',allow_pickle=False);gp=cli.loadj(r/'progress.json')
                good,evidence=convergence(cli.loadj(r/'history.json'),p,previous)
                # Only stop GD after the full 512-square parameter stage has started.
                if good and gp['grid']==512 and gp['gd_accepted']>=120:
                    gd_reason='sensor_and_phase_converged';gd_evidence=evidence;break
                previous=p.copy()
            cli.write(stop_path,dict(reason=gd_reason,evidence=gd_evidence,phase_sha256=b.array_sha(r/'latest_phase.npy')))
        gd_count=int(cli.loadj(r/'progress.json')['gd_accepted'])
        initial=np.load(r/'latest_phase.npy',allow_pickle=False)
        stage('refine_up_to_150')
        with fft.set_workers(4):
            _,_,sensor=cli.inputs(a);target=torch.from_numpy(sensor);model=DoubleModel(a.archive,man['wavelength_nm'])
            if model.archive_sha256!=man['archive_sha256']:raise ValueError('Archive identity mismatch')
            out=r/'recovery_float64';out.mkdir(exist_ok=True)
            config=dict(alpha=.0001,init_sha256=cli.sha(initial),sensor_sha256=cli.sha(sensor),gd_accepted=gd_count,
                        scipy=scipy.__version__,model_sha256=hashlib.sha256((HERE/'native_model.py').read_bytes()).hexdigest(),
                        policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        solver_sha256=hashlib.sha256((HERE/'tank_precision_recovery.py').read_bytes()).hexdigest(),
                        maxcor=10,maxls=20,ftol=1e-15,gtol=1e-10,objective_scale=1e5,truth_used=False)
            if (out/'optimizer_state.npz').exists():
                # Existing exact full-memory state can be reused only with identical physical inputs.
                saved=cli.loadj(out/'config.json')
                for key in ['alpha','init_sha256','sensor_sha256','gd_accepted','scipy','model_sha256']:
                    if saved[key]!=config[key]:raise ValueError('Refinement identity mismatch: '+key)
                config=saved
            validation=out/'precision_validation.json'
            if not validation.exists():
                _,g,*_=model.fg(initial,target,.0001);d=g/max(float(np.max(np.abs(g))),1e-30);h=1e-3
                numeric=(model.fg(initial+h*d,target,.0001)[0]-model.fg(initial-h*d,target,.0001)[0])/(2*h)
                analytic=float(np.sum(g*d));err=abs(numeric-analytic)/max(abs(analytic),1e-30)
                cli.write(validation,dict(relative_gradient_error=err,truth_used=False))
            if cli.loadj(validation)['relative_gradient_error']>.01:raise RuntimeError('Gradient validation failed')
            count=0;previous=None;reason='budget';evidence={}
            if (out/'optimizer_state.npz').exists():
                with np.load(out/'optimizer_state.npz',allow_pickle=False) as z:count=int(z['accepted'])
                previous=np.load(out/f'phase_{count:03}.npy',allow_pickle=False)
            while count<150:
                requested=min(150,count+10)
                new_count=solve(out,initial,lambda p:model.fg(p,target,.0001),config,requested)
                if new_count==0:
                    cli.save_phase(out/'phase_000.npy',initial);count=0;reason='optimizer_stationary';break
                phase=np.load(out/f'phase_{new_count:03}.npy',allow_pickle=False)
                good,evidence=convergence(cli.loadj(out/'history.json'),phase,previous)
                count=new_count
                if good:reason='sensor_and_phase_converged';break
                if count<requested:reason=cli.loadj(out/'progress.json')['termination'];break
                previous=phase.copy()
            cli.write(r/'completion_policy.json',dict(gd_accepted=gd_count,lbfgs_accepted=count,gd_max=150,lbfgs_max=150,
                      gd_stop_reason=gd_reason,refine_stop_reason=reason,gd_stop_evidence=gd_evidence,
                      refine_stop_evidence=evidence,thresholds=THRESHOLDS,truth_used_for_stopping=False))
        del model,target
        import gc
        gc.collect()
        stage('posthoc_evaluate')
        b.invoke([sys.executable,str(HERE/'evaluate_precision_case.py'),*common],r/'policy_children.log')
        stage('completed')


if __name__=='__main__':main()
