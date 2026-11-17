"""Post-hoc evaluation only; truth is never imported by the recovery optimizer."""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
from scipy import fft

import native_cli as cli
from tank_precision_recovery import DoubleModel
from native_model import Model


def render_comparison(r, phases, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4.4))
    for ax,(label,path) in zip(axes,[('Truth',r/'private_truth.npy'),*phases.items()]):
        ax.imshow(np.load(path,allow_pickle=False),cmap='twilight',vmin=0,vmax=cli.TAU)
        ax.set_title(label)
        ax.axis('off')
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.88,wspace=.04)
    fig.savefig(output,dpi=140)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    a=ap.parse_args()
    r,man,sensor=cli.inputs(a)
    truth=np.load(r/'private_truth.npy',allow_pickle=False)
    if cli.sha(truth)!=man['truth_sha256']:
        raise ValueError('Truth hash mismatch')
    out=r/'recovery_float64'
    progress=cli.loadj(out/'progress.json')
    count=progress['accepted']
    phases={'GD150':r/'gd_150.npy','reference_L37':r/'long_alpha_0.0001/phase_037.npy',
            f'float64_L{count}':out/f'phase_{count:03}.npy'}
    results={}
    for label,path in phases.items():
        p=np.load(path,allow_pickle=False)
        d=p-truth
        results[label]=dict(phase_sha256=cli.sha(p),circular_rmse_rad=float(np.sqrt(np.mean(np.angle(np.exp(1j*d))**2))),
                           linear_rmse_rad=float(np.sqrt(np.mean(d*d))))
    with fft.set_workers(4):
        for name,cls in [('reference_float32',Model),('recovery_float64',DoubleModel)]:
            optical=cls(a.archive,man['wavelength_nm'])
            for label,path in phases.items():
                predicted=optical.measure(np.load(path,allow_pickle=False))
                gain=float(np.sum(predicted.astype(np.float64)*sensor)/np.sum(predicted.astype(np.float64)**2))
                residual=gain*predicted.astype(np.float64)-sensor
                l2=float(np.linalg.norm(residual.ravel())/np.linalg.norm(sensor.astype(np.float64).ravel()))
                results[label][name+'_sensor_l2']=l2
            del optical,predicted,residual
            gc.collect()
    render_comparison(r,phases,out/'comparison.png')
    final=results[f'float64_L{count}']
    report=dict(scene='tank',wavelength_nm=man['wavelength_nm'],noise_seed=man['noise_seed'],
                sensor_sha256=man['sensor_sha256'],truth_sha256=man['truth_sha256'],
                reference_gd_accepted=150,reference_lbfgs_accepted=37,recovery_lbfgs_accepted=count,
                variant='float64 propagation recovery from GD150; curvature initialized separately',
                common_reference_L60_complete=False,truth_used_for_evaluation_only=True,
                quality_pass=final['circular_rmse_rad']<=.15 and final['reference_float32_sensor_l2']<=.025,
                results=results,preview='comparison.png')
    cli.write(out/'evaluation.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
