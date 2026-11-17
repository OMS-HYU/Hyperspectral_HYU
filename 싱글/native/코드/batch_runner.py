"""Resumable single-track native experiment queue.

The runner refuses to synthesize missing input manifests or phase arrays. It
only starts the existing native CLI after validating the extracted case
metadata, and records every blocked or failed case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import contextlib
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
import numpy as np


HERE = Path(__file__).resolve().parent
TRACK = HERE.parent
DEFAULT_ARCHIVE = TRACK / "데이터" / "05_NN_metacam-main.zip"
TAU = 2 * 3.141592653589793
EXPECTED_OPTICS = '6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716'

def array_sha(path: Path) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.load(path, allow_pickle=False)).tobytes()).hexdigest()


@contextlib.contextmanager
def batch_lock(path: Path):
    """OS lock survives neither a crash nor reboot; no stale PID deletion needed."""
    with path.open('a+b') as stream:
        stream.seek(0); stream.write(b'0'); stream.flush(); stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def system_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def find_manifest(root: Path) -> Path:
    candidates = sorted(root.rglob("*.json"))
    for path in candidates:
        try:
            data = read_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and any(key in data for key in ("cases", "scenes", "items", "experiments")):
            return path
        if isinstance(data, list) and data and all(isinstance(item, dict) for item in data):
            return path
    raise FileNotFoundError("No case manifest with cases/scenes/items/experiments was found")


def case_entries(manifest: Any) -> list[dict[str, Any]]:
    if isinstance(manifest, list):
        entries = manifest
    else:
        entries = next((manifest[key] for key in ("cases", "scenes", "items", "experiments") if key in manifest), None)
    if not isinstance(entries, list) or not entries or not all(isinstance(item, dict) for item in entries):
        raise ValueError("Manifest must contain a non-empty list of case objects")
    result = []
    for index, item in enumerate(entries):
        name = item.get("id") or item.get("name") or item.get("scene")
        phase = item.get("phase") or item.get("phase_path") or item.get("phase_file")
        if not name:
            raise ValueError(f"Manifest entry {index} must explicitly contain id/name")
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(name)):
            raise ValueError(f'Unsafe scene id: {name}')
        result.append({"id": str(name), "phase": str(phase) if phase else None, **item})
    return result


def resolve_phase(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or root.resolve() not in path.parents:
        raise FileNotFoundError(f"Phase array is missing or outside extracted archive: {relative}")
    return path


def inspect_input(input_archive: Path, work_root: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    if not input_archive.is_file():
        raise FileNotFoundError(f"Input archive not found: {input_archive}")
    extraction = work_root / "input"
    extraction.mkdir(parents=True, exist_ok=True)
    marker = extraction / ".archive_sha256"
    digest = sha256_file(input_archive)
    if marker.exists() and marker.read_text(encoding="ascii").strip() != digest:
        raise RuntimeError("Input extraction belongs to a different archive; refusing to replace it")
    if not marker.exists():
        with zipfile.ZipFile(input_archive) as archive:
            for member in archive.infolist():
                target = (extraction / member.filename).resolve()
                if not target.is_relative_to(extraction.resolve()):
                    raise ValueError('Archive path traversal')
            archive.extractall(extraction)
        marker.write_text(digest, encoding="ascii")
    manifest_path = find_manifest(extraction)
    entries = case_entries(read_json(manifest_path))
    phase_archive = manifest_path.parent / "phase_inputs.npz"
    if phase_archive.is_file():
        phase_dir = extraction / "phases"
        phase_dir.mkdir(exist_ok=True)
        with np.load(phase_archive, allow_pickle=False) as arrays:
            for entry in entries:
                if entry["phase"] is None:
                    if entry["id"] not in arrays.files:
                        raise KeyError(f"Manifest case has no matching phase array: {entry['id']}")
                    phase = np.asarray(arrays[entry["id"]])
                    digest = hashlib.sha256(np.ascontiguousarray(phase).tobytes()).hexdigest()
                    if digest != entry.get('sha256'):
                        raise ValueError(f"Manifest phase hash mismatch: {entry['id']}")
                    if phase.shape != (512, 512) or not np.isfinite(phase).all() or phase.min() < 0 or phase.max() >= 2 * np.pi + 1e-6:
                        raise ValueError(f"Invalid phase array for case: {entry['id']}")
                    output = phase_dir / f"{entry['id']}.npy"
                    if not output.exists():
                        np.save(output, phase)
                    elif array_sha(output) != digest:
                        raise ValueError(f'Extracted phase was modified: {output}')
                    entry["phase"] = str(output.relative_to(extraction))
    if any(entry["phase"] is None for entry in entries):
        raise ValueError("Manifest must provide phase paths or phase_inputs.npz must provide matching arrays")
    return extraction, manifest_path, entries


def phase_path_for(entry: dict[str, Any], extracted: Path) -> Path:
    return resolve_phase(extracted, entry["phase"])


def invoke(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        with subprocess.Popen(command, cwd=TRACK.parent, stdout=log, stderr=subprocess.STDOUT, text=True) as child:
            write_json(log_path.parent / 'active_child.json', dict(pid=child.pid, command=command))
            returncode = child.wait()
    if returncode:
        raise RuntimeError(f"Command failed with exit code {returncode}: {command[0]}")


def run_case(entry: dict[str, Any], extracted: Path, args: argparse.Namespace) -> dict[str, Any]:
    scene = entry["id"]
    run = args.output / scene
    truth = phase_path_for(entry, extracted)
    if run.exists() and not run.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {run}")
    status_path = run / "batch_status.json"
    status = read_json(status_path) if status_path.exists() else {
        "scene": scene,
        "input_phase": str(truth),
        "input_phase_sha256": sha256_file(truth),
        "status": "started",
        "events": [],
    }
    log_path = run / "batch.log"
    base = [sys.executable, str(TRACK / "run.py")]
    archive = args.optics_archive
    try:
        status['status'] = 'running'
        status.pop('error', None)
        write_json(status_path, status)
        expected = dict(scene=scene, noise_seed=args.seed, wavelength_nm=args.wavelength,
                        truth_sha256=array_sha(truth), archive_sha256=EXPECTED_OPTICS)
        if (run / 'manifest.json').exists():
            manifest = read_json(run / 'manifest.json')
            if any(manifest.get(k) != v for k, v in expected.items()):
                raise ValueError('Run identity differs; refusing to regenerate or resume')
            if array_sha(run / 'sensor.npy') != manifest['sensor_sha256']:
                raise ValueError('Sensor changed')
        if not (run / "manifest.json").exists():
            # A killed generation is deterministic, but never overwrite its evidence.
            partial = [run / n for n in ('sensor.npy', 'private_truth.npy') if (run / n).exists()]
            if partial:
                recovery = run / ('partial_generate_' + str(time.time_ns()))
                recovery.mkdir()
                for path in partial:
                    path.replace(recovery / path.name)
            seed = args.seed
            wavelength = args.wavelength
            invoke(base + ["generate", "--run", str(run), "--archive", str(archive), "--truth", str(truth),
                           "--scene", scene, "--seed", str(seed), "--wavelength", str(wavelength)], log_path)
            status["events"].append("generated")
        # --steps adds accepted updates; the CLI repairs projections from its atomic state.
        total = 0
        if (run / 'gd_state.npz').exists():
            with np.load(run / 'gd_state.npz', allow_pickle=False) as state:
                total = int(state['total'])
        invoke(base + ["gd", "--run", str(run), "--archive", str(archive), "--steps", str(max(1,150-total))], log_path)
        gd_progress = read_json(run / "progress.json")
        if gd_progress.get("gd_accepted") != 150 or not gd_progress.get("complete"):
            raise RuntimeError(f"GD did not complete 150 accepted updates: {gd_progress.get('gd_accepted')}")
        status["events"].append("gd150")
        terminal = run / 'long_alpha_0.0001' / 'progress.json'
        if terminal.exists() and 'optimizer terminated' in read_json(terminal).get('termination', ''):
            raise RuntimeError('Numerical optimizer termination; no automatic curvature restart')
        invoke(base + ["lbfgs", "--run", str(run), "--archive", str(archive), "--alpha", "0.0001", "--until", "60"], log_path)
        lbfgs_progress = read_json(run / "long_alpha_0.0001" / "progress.json")
        phase = run / "long_alpha_0.0001" / "phase_060.npy"
        if lbfgs_progress.get("accepted") != 60 or not phase.is_file():
            raise RuntimeError(f"L-BFGS did not complete 60 accepted updates: {lbfgs_progress.get('accepted')}")
        status["events"].append("lbfgs60")
        evaluation = run / "evaluation_L60.json"
        for label, checkpoint in [('GD150', run / 'gd_150.npy'), ('L60', phase)]:
            target = run / f'evaluation_{label}.json'
            if not target.exists():
                invoke(base + ['evaluate', '--run', str(run), '--archive', str(archive), '--truth', str(truth),
                               '--phase', str(checkpoint), '--output', str(target)], log_path)
            ev = read_json(target)
            manifest = read_json(run / 'manifest.json')
            if any(ev[k] != v for k,v in dict(phase_sha256=array_sha(checkpoint), truth_sha256=array_sha(truth),
                                            sensor_sha256=manifest['sensor_sha256']).items()):
                raise ValueError('Evaluation hashes do not match checkpoint')
        ev = read_json(evaluation)
        if not np.isfinite([ev['circular_rmse'], ev['sensor_l2']]).all():
            raise ValueError('Non-finite evaluation')
        status["events"].append("evaluated")
        status["status"] = 'passed' if ev['circular_rmse'] <= .15 and ev['sensor_l2'] <= .025 else 'failed_gate'
        status['evaluation'] = ev
        status.pop('error', None)
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = str(exc)
    status["updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(status_path, status)
    return status


def run_batch(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-archive", type=Path, required=True,
                        help="Google Drive archive containing the explicit case manifest and phase arrays")
    parser.add_argument("--optics-archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wavelength", type=float, default=660.0)
    parser.add_argument("--seed", type=int, default=672912)
    parser.add_argument('--restart-limit', type=int, default=3, help='Crash retries per scene; numerical termination stops immediately')
    parser.add_argument('--retry-delay', type=float, default=10)
    parser.add_argument("--only", action="append", default=[],
                        help="Run only this case id; repeat for multiple cases")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    for key in ('input_archive','optics_archive','output'):
        setattr(args,key,getattr(args,key).resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'resume_command.json', dict(python=sys.executable,
               args=[str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]))
    identity = dict(input_archive_sha256=sha256_file(args.input_archive),
                    optics_sha256=sha256_file(args.optics_archive), wavelength=args.wavelength, seed=args.seed,
                    solver_sha256=sha256_file(HERE / 'native_cli.py'), model_sha256=sha256_file(HERE / 'native_model.py'),
                    only=args.only)
    identity_path = args.output / 'batch_identity.json'
    if identity_path.exists() and read_json(identity_path) != identity:
        raise ValueError('Batch identity/code changed; use a separate output directory')
    write_json(identity_path, identity)
    queue_path = args.output / "queue.json"
    batch = {"system": system_snapshot(), "input_archive": str(args.input_archive),
             "input_archive_sha256": sha256_file(args.input_archive) if args.input_archive.is_file() else None}
    try:
        if args.wavelength != 660:
            raise ValueError('440/550, RBC and Blind remain blocked pending separate validation')
        if sha256_file(args.optics_archive) != EXPECTED_OPTICS:
            raise ValueError('Original B17 archive hash differs from the active protocol')
        import scipy
        if scipy.__version__ != '1.17.0':
            raise ValueError('Exact resume requires scipy==1.17.0')
        extracted, manifest_path, entries = inspect_input(args.input_archive, args.output / "_source")
        completed = {x['id'] for x in read_json(TRACK / '결과/최신_20261116/native_results.json')['ordinary']
                     if x.get('known') and x['wavelength_nm']==args.wavelength and x['gd_accepted']==150
                     and x['lbfgs_accepted']==60 and x['final_circular_rmse_rad']<=.15 and x['sensor_l2']<=.025}
        blocked = [x['id'] for x in entries if 'RBC' in x.get('name','')]
        entries = [x for x in entries if x['id'] not in completed and x['id'] not in blocked]
        entries.sort(key=lambda x: x['id'] != 'tank')
        batch.update(skipped_completed=sorted(completed), blocked_rbc=blocked)
        if args.only:
            entries = [entry for entry in entries if entry["id"] in args.only]
            if not entries:
                raise ValueError(f"None of --only cases exist in the manifest: {args.only}")
        batch.update({"manifest": str(manifest_path), "count": len(entries),
                      "cases": [{"id": item["id"], "phase": item["phase"]} for item in entries]})
        write_json(queue_path, batch)
        print(json.dumps(batch, ensure_ascii=False, indent=2))
        if args.inspect_only:
            return 0
    except Exception as exc:
        batch.update({"status": "blocked", "error": str(exc)})
        write_json(queue_path, batch)
        print(json.dumps(batch, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    results = []
    with contextlib.nullcontext():  # main holds the OS lock across inspection and execution.
        for entry in entries:
            # Persist each attempt. Restart the CLI after a crash using the same atomic state.
            for attempt in range(args.restart_limit + 1):
                result = run_case(entry, extracted, args)
                result['attempt'] = attempt
                write_json(args.output / entry['id'] / 'batch_status.json', result)
                if result['status'] != 'failed': break
                terminal = args.output / entry['id'] / 'long_alpha_0.0001/progress.json'
                if terminal.exists() and 'optimizer terminated' in read_json(terminal).get('termination',''): break
                if attempt < args.restart_limit: time.sleep(args.retry_delay)
            results.append(result)
            batch['results'] = results
            batch['status'] = 'running'
            write_json(queue_path, batch)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result['status'] != 'passed': break  # Tank gate before extending the queue.
    batch["results"] = results
    batch["status"] = "completed" if len(results)==len(entries) and all(item["status"] == "passed" for item in results) else "stopped_with_failures"
    write_json(queue_path, batch)
    return 0 if batch["status"] == "completed" else 1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if '--help' in argv or '-h' in argv:
        return run_batch(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', type=Path, required=True)
    known, _ = parser.parse_known_args(argv)
    known.output.mkdir(parents=True, exist_ok=True)
    lock = batch_lock(known.output / '.case.lock')
    try:
        lock.__enter__()
    except OSError:
        print('A runner already owns this batch; nothing started.', flush=True)
        return 0
    try:
        return run_batch(argv)
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
