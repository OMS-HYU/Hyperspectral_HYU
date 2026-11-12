"""Replay the exact historical Spectral Periodic solver with sealed GD100.

Only paths are adapted; original ZIP is unchanged. No new GD run is performed.
Historical reference and newly generated output are kept separate.
"""
from pathlib import Path
import argparse, hashlib, os, subprocess, sys, zipfile

ROOT=Path(__file__).resolve().parents[2]
ARCHIVE=Path(__file__).resolve().parent/"B17_K1_RBC_0_2pi_Spectral_Periodic_LBFGS100_FULL_20261112.zip"
B17=ROOT/"데이터"/"B17"/"05_NN_metacam-main.zip"
ARCHIVE_SHA="1e2e71047ac9ab0329ab88972d3a7e4170413073b2863d4aed7aaa43399eac6d"
B17_SHA="6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716"

def verify(p, expected):
    if not p.is_file():
        raise FileNotFoundError(p)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        raise ValueError(f"Original file SHA-256 mismatch: {p}")

def patch(p, replacements):
    s=p.read_text(encoding="utf-8")
    for old,new in replacements:
        if old not in s: raise ValueError(f"Expected source path not found: {p}: {old}")
        s=s.replace(old,new)
    compile(s,str(p),"exec")
    p.write_text(s,encoding="utf-8")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=ARCHIVE)
    parser.add_argument("--b17",type=Path,default=B17)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare-only",action="store_true")
    a=parser.parse_args()
    archive=a.archive.resolve(); b17=a.b17.resolve(); out=a.output.resolve()
    verify(archive,ARCHIVE_SHA);verify(b17,B17_SHA)
    out.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        # Exact verified archive; only source/reference are copied, never original results.
        for n in z.namelist():
            if n.startswith("reference/") or n in ("run_spectral.py","finalize.py"):
                z.extract(n,out)
    patch(out/"reference"/"b17_optics.py",[
        ("ORIG=Path('/mnt/data/single_k1_input/05_NN_metacam-main(4).zip')",
         "ORIG=Path(os.environ['B17_ZIP'])")])
    baseline=("/mnt/data/rbc_fixed_0_2pi","reference")
    # Baseline is injected through the existing environment variable, so no solver math changes.
    for name in ("run_spectral.py","finalize.py"):
        p=out/name
        pairs=[("/mnt/data/rbc_periodic_improved/results/sealed.npz", "reference/periodic_results/sealed.npz")]
        if name=="finalize.py":
            pairs.append(("/mnt/data/rbc_periodic_improved/results/history.json","reference/periodic_results/history.json"))
        patch(p,pairs)
    if a.prepare_only:
        print("Prepared and compiled historical sources; no optical calculation executed:",out)
        return
    env=os.environ.copy()
    env.update(B17_ZIP=str(b17), B17_BASELINE_DIR=str(out/"reference"), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for name in ("run_spectral.py","finalize.py"):
        subprocess.run([sys.executable,str(out/name)],cwd=out,env=env,check=True)
    print("Replay completed; compare results/comparison.json with historical record.")

if __name__=="__main__":
    main()
