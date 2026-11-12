"""Generate the EXACT historical synthetic RBC observation from its sealed archive.

The raw optical phase is used ONLY to generate the historical sensor (identical
to the original 2026-11-12 experiment). Inversion receives sensor_only.npz;
truth is canonical wrapped and stored separately. No checkpoint is used to
initialize the new 100+100 run. Exact sensor hash mismatch is a hard failure.
"""
from pathlib import Path
import hashlib,io,json,zipfile
import numpy as np
from 공통.optics import digest,phase_check,GEOMETRY,write_json
from 공통.object_models import wrap

ARCHIVE_SHA="1e2e71047ac9ab0329ab88972d3a7e4170413073b2863d4aed7aaa43399eac6d"
SENSOR_SHA="4bb694a4aa5cba938c4101ed9e932eb8864db33863a5a7074b42933009f474fc"
GD_SHA="bf037b13cdbf10e52a60ab8adfd42b8d16434d6f70667c48603eb4314b94c8dc"
FINAL_SHA="c3d82c82c5709a422c4b613e5313774c49d0cba04e9d8d0276fd438221156c9a"

def generate_reference(engine,archive,dest):
    archive=Path(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=ARCHIVE_SHA:
        raise ValueError("Wrong historical spectral reference archive")
    with zipfile.ZipFile(archive) as z:
        with np.load(io.BytesIO(z.read("reference/results/independent100_signed_truth.npz")),allow_pickle=False) as a:
            raw=a["phase"].copy()
        with np.load(io.BytesIO(z.read("reference/fixed_results/truth_wrapped.npz")),allow_pickle=False) as a:
            truth={k:a[k].copy() for k in ("phase","labels","groupmap")}
        manifest=json.loads(z.read("reference/fixed_results/manifest.json"))
    if not np.array_equal(wrap(raw),truth["phase"]):
        raise ValueError("Reference truth mismatch")
    phase_check(truth["phase"])
    if manifest["sensor_sha256"]!=SENSOR_SHA:
        raise ValueError("Reference manifest sensor mismatch")
    # Reproduce the original physical forward; raw phase is not an inverse input.
    m,cam=engine.optics(660.)
    clean=engine.measure(m,cam,raw)
    sigma=.01*float(clean.mean())
    y=np.maximum(0.,clean+sigma*np.random.default_rng(672906).standard_normal(clean.shape))
    if digest(y)!=SENSOR_SHA:
        raise RuntimeError(f"Historical sensor is not byte-identical: {digest(y)}")
    dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/"sensor_only.npz",sensor=y)
    np.savez_compressed(dest/"private_truth.npz",actual_wavelength_nm=660.,**truth)
    write_json(dest/"generation.json",dict(scene="historical_triple",seed=672906,noise_sigma=sigma,geometry=list(GEOMETRY),sensor_sha256=SENSOR_SHA,reference_archive_sha256=ARCHIVE_SHA,B17_inner_shas=engine.inner_shas,B17_source_fingerprint=engine.source_fingerprint,truth_in_inverse=False))
    print("Historical sensor SHA-256 verified:",SENSOR_SHA,flush=True)

def compare_reference(out):
    audit=json.loads((out/"recovery.json").read_text())
    evaluation=json.loads((out/"evaluation.json").read_text())
    with np.load(out/"gd_sealed.npz",allow_pickle=False) as a: gd_sha=digest(a["phase"].ravel())
    with np.load(out/"recovered.npz",allow_pickle=False) as a: phase_sha=digest(a["phase"].ravel())
    result=dict(reference_roi_rmse_rad=.077085576096947,reference_groups_passed=49,reference_group31_rmse_rad=.3298432165686897,new_roi_rmse_rad=evaluation["roi_rmse_rad"],new_groups_passed=evaluation["groups_passed"],new_group31_rmse_rad=next(g["rmse_rad"] for g in evaluation["groups"] if g["group"]==31),same_sensor=audit["sensor_sha256"]==SENSOR_SHA,gd_exact=gd_sha==GD_SHA,final_exact=phase_sha==FINAL_SHA,gd_sha256=gd_sha,phase_sha256=phase_sha,completed=audit["completed"])
    write_json(out/"historical_comparison.json",result)
    print("REFERENCE_COMPARISON",json.dumps(result),flush=True)
