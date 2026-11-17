"""Separate float64 recovery experiment; never overwrites the reference solver.

Keeps the calibrated transfer arrays and geometry, promotes propagation,
object phase, camera interpolation and loss to double precision. Starts from
the verified GD150 phase, not truth. Original float32 L37 is preserved.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy import fft
from scipy.optimize import _lbfgsb

import native_cli as cli
import native_model as nm
from batch_runner import batch_lock


class DoubleModel(nm.Model):
    def propagate(self, u):
        return super().propagate(np.asarray(u, dtype=np.complex128))

    def field(self, p):
        self.count += 1
        u = np.zeros((nm.N, nm.N), dtype=np.complex128)
        u[nm.sl(nm.N, nm.O), nm.sl(nm.N, nm.O)] = np.exp(1j*np.asarray(p, np.float64))
        return self.propagate(u) + self.bg


def solve(out, initial, fun, config, until):
    """Atomic full-memory L-BFGS state; terminal convergence is never counted as an update."""
    out.mkdir(parents=True, exist_ok=True)
    state = out/'optimizer_state.npz'
    n = initial.size
    m = 10
    data = dict(x=initial.ravel().copy(), f=np.array(0.), g=np.zeros(n),
                wa=np.zeros(2*m*n+5*n+11*m*m+8*m), iwa=np.zeros(3*n, np.int32),
                task=np.zeros(2, np.int32), ln_task=np.zeros(2, np.int32),
                lsave=np.zeros(4, np.int32), isave=np.zeros(44, np.int32), dsave=np.zeros(29))
    accepted = nfev = 0
    elapsed = 0.
    history = []
    last = np.zeros(4)
    if state.exists():
        if cli.loadj(out/'config.json') != config:
            raise ValueError('Recovery configuration changed; use a separate output directory')
        with np.load(state, allow_pickle=False) as z:
            data = {k:z[k] for k in data}
            accepted, nfev = int(z['accepted']), int(z['nfev'])
            elapsed, last = float(z['elapsed']), z['last']
            history = json.loads(str(z['history_json']))
        if data['task'][0] != 1 or accepted != len(history):
            raise ValueError('Not an accepted-iteration resume state')
    else:
        cli.write(out/'config.json', config)
    start = time.monotonic()
    termination = 'paused at requested accepted count'
    def progress():
        return dict(accepted=accepted, requested=until, termination=termination,
                    sensor_l2=float(np.sqrt(last[1])), objective=float(last[0]), nfev=nfev,
                    phase_sha256=cli.sha(np.mod(data['x'].reshape(initial.shape), cli.TAU)),
                    elapsed_seconds=elapsed+time.monotonic()-start,
                    protocol='float64 propagation recovery; separate from reference float32 L60')
    while accepted < until:
        _lbfgsb.setulb(m, data['x'], np.zeros(n), np.zeros(n), np.zeros(n,np.int32),
                      data['f'], data['g'], 1e-15/np.finfo(float).eps, 1e-10,
                      data['wa'], data['iwa'], data['task'], data['lsave'],
                      data['isave'], data['dsave'], 20, data['ln_task'])
        if data['task'][0] == 3:
            j,g,d,reg,q = fun(data['x'].reshape(initial.shape))
            if not np.isfinite(j) or not np.isfinite(g).all():
                raise FloatingPointError('Nonfinite recovery objective/gradient')
            data['f'] = np.array(j*1e5)
            data['g'] = np.asarray(g.ravel()*1e5, np.float64)
            last = np.array([j,d,reg,q])
            nfev += 1
        elif data['task'][0] == 1:
            accepted += 1
            history.append(dict(accepted=accepted,objective=float(last[0]),sensor_l2=float(np.sqrt(last[1])),nfev=nfev))
            cli.save_state(state, **data, accepted=accepted, nfev=nfev, last=last,
                           elapsed=elapsed+time.monotonic()-start, history_json=json.dumps(history))
            cli.save_phase(out/f'phase_{accepted:03}.npy', np.mod(data['x'].reshape(initial.shape),cli.TAU))
            cli.write(out/'history.json',history)
            cli.write(out/'progress.json',progress())
            print('ACCEPT',json.dumps(progress()),flush=True)
        else:
            termination = f'optimizer terminated: task={data["task"].tolist()}'
            break
    cli.write(out/'progress.json',progress())
    print('END',json.dumps(progress()),flush=True)
    return accepted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--until',type=int,default=60)
    a = ap.parse_args()
    if scipy.__version__ != '1.17.0' or not 1 <= a.until <= 60:
        raise ValueError('Requires SciPy 1.17.0 and until in 1..60')
    r,man,y = cli.inputs(a)
    p = np.load(r/'gd_150.npy',allow_pickle=False)
    gd = cli.loadj(r/'progress.json')
    if gd['gd_accepted'] != 150 or cli.sha(p) != gd['phase_sha256']:
        raise ValueError('GD150 checkpoint mismatch')
    out = r/'recovery_float64'
    out.mkdir(exist_ok=True)
    config = dict(variant='float64 propagation, original calibrated complex64 transfer coefficients',
                  initialization='same zero-start GD150; new curvature; no truth',
                  init_sha256=cli.sha(p), sensor_sha256=man['sensor_sha256'],
                  archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(Path(nm.__file__).read_bytes()).hexdigest(),
                  solver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scipy=scipy.__version__, alpha=.0001, maxcor=10,maxls=20,ftol=1e-15,gtol=1e-10)
    if config['archive_sha256'] != man['archive_sha256']:
        raise ValueError('Archive mismatch')
    with batch_lock(out/'run.lock'), fft.set_workers(4):
        model = DoubleModel(a.archive,man['wavelength_nm'])
        target = torch.from_numpy(y)
        validation = out/'precision_validation.json'
        if not validation.exists():
            # A directional derivative check does not use the private truth.
            f,g,*_ = model.fg(p,target,.0001)
            direction = g/max(np.max(np.abs(g)),1e-30)
            h = 1e-3
            fp = model.fg(p+h*direction,target,.0001)[0]
            fm = model.fg(p-h*direction,target,.0001)[0]
            analytic = float(np.sum(g*direction))
            numeric = (fp-fm)/(2*h)
            error = abs(numeric-analytic)/max(abs(analytic),1e-30)
            cli.write(validation,dict(objective=f,analytic=analytic,numeric=numeric,
                                     relative_gradient_error=error,truth_used=False))
            print('VALIDATION',error,flush=True)
            if error > .01:
                raise RuntimeError('Double precision gradient validation failed')
        elif cli.loadj(validation)['relative_gradient_error'] > .01:
            raise RuntimeError('Stored gradient validation failed')
        count = solve(out,p,lambda phase:model.fg(phase,target,.0001),config,a.until)
        if count < a.until:
            raise RuntimeError('Recovery stopped before requested accepted count; inspect progress')


if __name__ == '__main__':
    main()
