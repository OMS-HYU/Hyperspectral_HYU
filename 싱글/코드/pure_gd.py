"""K=1 GD-only controlled benchmark; existing B17 optics and evaluator unchanged."""
from __future__ import annotations
import argparse, json, tempfile, time
from pathlib import Path
import numpy as np
from scipy import fft
from 공통.object_models import N, wrap, TAU
from 공통.optics import FullB17, GEOMETRY, ALPHA, phase_check, wavelength_check, digest, write_json
from 싱글.코드.solver import sensor_check, prior, preconditioner, scan_wavelength, evaluate
from 싱글.코드.pyramid import gd_pyramid, SCHEDULES

METHODS = ("full100", "full150", "pyramid150", "spectral150")

def descent(fg, x, f, g, direction, step, cap=None, max_backtracks=32):
    """Armijo on wrapped phase; objective must be periodic."""
    slope = float(np.dot(g.ravel(), direction.ravel()))
    if not np.isfinite(slope) or slope >= 0:
        raise RuntimeError("Non-descent or stationary direction")
    if cap is not None:
        step = min(step, cap / max(float(np.max(np.abs(direction))), 1e-30))
    for back in range(max_backtracks):
        trial = wrap(x + step * direction)
        nf = fg(trial, False)[0]
        if np.isfinite(nf) and nf <= f + 1e-4 * step * slope:
            return trial, step, back
        step *= .5
    raise RuntimeError("Armijo exhausted")

def full_gd(value_grad, m, cam, y, method, dest):
    x = np.zeros(N*N, dtype=np.float64)
    history = []
    count = 100 if method == "full100" else 150
    P = preconditioner("spectral") if method == "spectral150" else None
    step = 1e5
    def fg(p, grad, alpha=0.):
        f = value_grad(m, cam, p, y, 0., grad)
        e, pg = prior(p, alpha, grad)
        return float(f[0] + e), (f[1].ravel() + pg if grad else None), float(f[0])
    f, g, data = fg(x, True)
    history.append(dict(stage="GD", iteration=0, objective=f, sensor_l2=float(np.sqrt(data))))
    for k in range(1, count+1):
        alpha = ALPHA if k > 100 else 0.
        if k == 101:
            f, g, data = fg(x, True, alpha)
            step = 1e5
        d = -P(g) if P is not None and k > 100 else -g
        x, used, back = descent(lambda p, grad: fg(p, grad, alpha), x, f, g, d, step,
                                cap=np.pi/2 if P is not None and k > 100 else None,
                                max_backtracks=25)
        f, g, data = fg(x, True, alpha)
        row = dict(stage="GD", iteration=k, objective=f, sensor_l2=float(np.sqrt(data)),
                   step=used, backtracks=back, alpha=alpha,
                   preconditioned=bool(P is not None and k > 100))
        history.append(row)
        print(json.dumps(row), flush=True)
        write_json(dest/"history.json", history)
        if k in (25, 50, 75, 100, 125, 150):
            np.savez_compressed(dest/f"gd_{k:03d}.npz", phase=x.reshape(N,N), iteration=k)
        step = min(used*1.3, 1e7)
    return x.reshape(N,N), history, count

def fit_gd_only(engine, y, w, dest, method, mode):
    if method not in METHODS:
        raise ValueError(method)
    dest.mkdir(parents=True, exist_ok=False)
    y = sensor_check(y)
    w = wavelength_check(w)
    m, cam = engine.optics(w)
    calls = dict(forward=0, adjoint=0)
    t0 = time.monotonic()
    def wrapped_value_grad(model, camera, x, observed, alpha, grad):
        physical = wrap(x)
        phase_check(physical.reshape(N,N))
        calls["forward"] += 1
        calls["adjoint"] += int(grad)
        return engine.value_grad(model, camera, physical, observed, alpha, grad)
    if method == "pyramid150":
        with tempfile.TemporaryDirectory(prefix="pure_gd_pyramid_") as tmp:
            p, rows = gd_pyramid(m, cam, y, wrapped_value_grad, Path(tmp), SCHEDULES["pyramid64"])
            for source in sorted(Path(tmp).glob("*.npz")):
                with np.load(source, allow_pickle=False) as z:
                    payload = {k:z[k] for k in z.files}
                payload["phase"] = phase_check(wrap(payload["phase"]))
                np.savez_compressed(dest/source.name, **payload)
        history = [dict(stage="GD", iteration=r["gd_step"], objective=r["total_obj"],
                        sensor_l2=float(np.sqrt(r["sensor_loss"])), variables_n=r["variables_n"])
                   for r in rows]
        steps = 150
    else:
        p, history, steps = full_gd(wrapped_value_grad, m, cam, y, method, dest)
    p = phase_check(wrap(p))
    loss, scale, pure = wrapped_value_grad(m, cam, p, y, 0., False)
    np.savez_compressed(dest/"gd_sealed.npz", phase=p, iteration=steps)
    np.savez_compressed(dest/"recovered.npz", phase=p, wavelength_nm=w, scale=scale)
    np.savez_compressed(dest/"predicted.npz", sensor=scale*pure)
    write_json(dest/"history.json", history)
    audit = dict(method=method, mode=mode, wavelength_nm=w, gd_iterations=steps,
                 lbfgs_iterations=0, completed=True, termination="iteration_budget",
                 sensor_l2=float(np.sqrt(loss)), sensor_sha256=digest(y), phase_sha256=digest(p),
                 geometry=list(GEOMETRY), B17_archive_sha256=engine.archive_sha,
                 B17_source_fingerprint=engine.source_fingerprint, B17_inner_shas=engine.inner_shas,
                 phase_range=[0,float(TAU)], alpha=ALPHA if steps>100 else 0.,
                 truth_in_inverse=False, seconds=time.monotonic()-t0,
                 preconditioner="historical 660 nm spectral, GD101-150" if method=="spectral150" else None,
                 pyramid_schedule=SCHEDULES["pyramid64"] if method=="pyramid150" else None, **calls)
    write_json(dest/"recovery.json", audit)
    return audit

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sensor", type=Path, required=True)
    p.add_argument("--truth", type=Path, help="Optional post-hoc evaluation only")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("known","blind"), default="known")
    p.add_argument("--wavelength", type=float)
    p.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    a = p.parse_args(argv)
    if (a.mode=="known") != (a.wavelength is not None):
        p.error("Known requires --wavelength; blind forbids it")
    if len(set(a.methods)) != len(a.methods):
        p.error("Duplicate methods")
    if a.wavelength is not None:
        wavelength_check(a.wavelength)
    with np.load(a.sensor, allow_pickle=False) as z:
        y = sensor_check(z["sensor"])
    a.output.mkdir(parents=True, exist_ok=False)
    results = {}
    with fft.set_workers(4):
        engine = FullB17()
        w = a.wavelength
        if a.mode == "blind":
            w, ranking = scan_wavelength(engine,y)
            write_json(a.output/"blind_scan.json",ranking)
        for method in a.methods:
            child = a.output/method
            try:
                audit = fit_gd_only(engine,y,w,child,method,a.mode)
                results[method] = {"status":"completed","recovery":audit}
                if a.truth is not None:
                    results[method]["evaluation"] = evaluate(a.truth,a.sensor,child)
            except Exception as exc:
                results[method] = {"status":"error","error":repr(exc)}
                print(method,repr(exc),flush=True)
            write_json(a.output/"comparison.json",results)
    if any(v["status"]=="error" for v in results.values()):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
