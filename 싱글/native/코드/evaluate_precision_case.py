"""Post-hoc evaluation of a generic native float64 L60 case."""
import argparse
import gc
from pathlib import Path
import numpy as np
from scipy import fft
import native_cli as cli
from native_model import Model
from tank_precision_recovery import DoubleModel


def evaluate(run, archive):
    args=argparse.Namespace(run=run,archive=archive)
    _,man,sensor=cli.inputs(args)
    out=run/'recovery_float64'
    status=cli.loadj(out/'progress.json')
    policy=cli.loadj(run/'completion_policy.json') if (run/'completion_policy.json').exists() else None
    count=int(status['accepted'])
    if policy:
        if count!=policy['lbfgs_accepted'] or not 0<=count<=150:raise ValueError('Policy/checkpoint count mismatch')
    elif count!=60:raise ValueError('60 accepted updates required for legacy protocol')
    truth=np.load(run/'private_truth.npy',allow_pickle=False)
    phase_path=out/f'phase_{count:03}.npy'
    phase=np.load(phase_path,allow_pickle=False)
    if cli.sha(truth)!=man['truth_sha256'] or cli.sha(phase)!=status['phase_sha256']:
        raise ValueError('Evaluation hashes mismatch')
    delta=phase-truth
    circular=float(np.sqrt(np.mean(np.angle(np.exp(1j*delta))**2)))
    linear=float(np.sqrt(np.mean(delta**2)))
    metrics={}
    with fft.set_workers(4):
        for name,cls in [('reference_float32',Model),('recovery_float64',DoubleModel)]:
            model=cls(archive,man['wavelength_nm'])
            pred=model.measure(phase).astype(np.float64)
            gain=float(np.sum(pred*sensor)/np.sum(pred*pred))
            metrics[name+'_sensor_l2']=float(np.linalg.norm((gain*pred-sensor).ravel())/np.linalg.norm(sensor.astype(np.float64).ravel()))
            del model,pred
            gc.collect()
    from render_five_panel import render
    figure_metadata=render(run,out/'comparison.png',phase_path=phase_path)
    result=dict(scene=man['scene'],wavelength_nm=man['wavelength_nm'],noise_seed=man['noise_seed'],
                sensor_sha256=man['sensor_sha256'],truth_sha256=man['truth_sha256'],phase_sha256=cli.sha(phase),
                gd_accepted=int(cli.loadj(run/'progress.json')['gd_accepted']),lbfgs_accepted=count,circular_rmse_rad=circular,linear_rmse_rad=linear,
                **metrics,quality_pass=circular<=.15 and metrics['reference_float32_sensor_l2']<=.025,
                variant='float32 zero-start GD up to 150 + float64 refinement up to 150' if policy else 'float32 zero-start GD150 + float64 propagation L60, new curvature',
                stopping_policy=policy,
                truth_used_for_evaluation_only=True,common_reference_L60_complete=False,
                preview='comparison.png',figure=figure_metadata)
    cli.write(out/'evaluation.json',result)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--archive',type=Path,required=True)
    a=ap.parse_args();print(evaluate(a.run,a.archive),flush=True)
