"""Native model numerical gate before dispatching any physical RBC case."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from scipy import fft
import native_cli as io
from tank_precision_recovery import DoubleModel
from rbc_native_solver import calibrate,fit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    report=a.output/'validation.json'
    if report.exists():
        if not io.loadj(report)['passed']:raise RuntimeError('Stored RBC validation failed')
        print('VALIDATION_ALREADY_PASSED',flush=True);return
    with fft.set_workers(4):
        model=DoubleModel(a.archive,660)
        p=.2*np.sin(np.arange(512)[None,:]/17)*np.ones((512,1))
        observed=torch.from_numpy(model.measure(np.zeros((512,512))))
        _,g,*_=model.fg(p,observed,.015);d=g/max(float(np.max(np.abs(g))),1e-30);h=1e-3
        numerical=(model.fg(p+h*d,observed,.015)[0]-model.fg(p-h*d,observed,.015)[0])/(2*h)
        analytic=float(np.sum(g*d));err=abs(numerical-analytic)/max(abs(analytic),1e-30)
        print('RBC_NATIVE_GRADIENT_ERROR',err,flush=True)
        if err>1e-3:raise RuntimeError('RBC native gradient failed')
        weights,meta=calibrate(model)
        io.save_phase(a.output/'spectral_weights.npy',weights)
        io.write(a.output/'spectral_calibration.json',meta)
        # Four true full-grid updates test the solver's native numerical path.
        target=torch.from_numpy(model.measure(p))
        identity=dict(archive_sha256=model.archive_sha256,gradient_error=err,kind='native numerical smoke, no cell truth')
        smoke=fit(a.output/'numerical_smoke',lambda phase,alpha:model.fg(phase,target,alpha),weights,
                  identity,gd_budget=2,lbfgs_budget=2)
        history=io.loadj(a.output/'numerical_smoke/history.json')
        descent=all(nextrow['objective']<=row['objective']*(1+1e-12) for row,nextrow in zip(history,history[1:]) if row['stage']==nextrow['stage'])
        passed=err<1e-3 and smoke['gd_accepted']==2 and smoke['lbfgs_accepted']==2 and descent
        io.write(report,dict(passed=passed,gradient_relative_error=err,spectral_operator=meta,
                 native_smoke=smoke,within_stage_descent=descent,geometry=[5713,2857,512,3000],
                 observation='amplitude',archive_sha256=model.archive_sha256,alpha=.015,
                 weights_sha256=io.sha(weights),truth_or_roi_used_for_optimization=False,
                 scope='Numerical dispatch gate only; does not establish RBC recovery performance'))
        print('RBC_NATIVE_VALIDATION',passed,flush=True)
        if not passed:raise RuntimeError('Native RBC numerical smoke failed')


if __name__=='__main__':main()
