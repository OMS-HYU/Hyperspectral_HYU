"""Validate the original B17 physical arrays before optical use."""
from pathlib import Path
import os, hashlib, zipfile
from common import propagation

LUT_SHA="02c551f2e24fb01d91416ea21cd17d24ae1c5e6fab0d630bff8e1fb2b4feaeaa"
WIDTH_SHA="b7d799c1e3a5913e7e9034fd2efc31dc2dab98822058ef08e2e57470ac2d68f9"
ORIGINAL_ZIP_SHA="6264a21cb0433c691cd4511f1f9f428f8f92b160cc4388845b6c9b0586fad716"
def check_b17():
    path=Path(os.environ.get("B17_ZIP",str(Path(__file__).resolve().parents[1]/"datasets"/"B17"/"05_NN_metacam-main.zip")))
    if not path.is_file():
        raise FileNotFoundError(f"Original B17 ZIP not found: {path}. See datasets/B17/README.md or set B17_ZIP.")
    outer=hashlib.sha256(path.read_bytes()).hexdigest()
    expected={"bayesLUT_MSE_v6.3_nonoverlap.mat":LUT_SHA,"0.6NA_random_70_1_300_1mm_mapped_width.mat":WIDTH_SHA}
    with zipfile.ZipFile(path) as z:
        actual={name:hashlib.sha256(z.read(next(f for f in z.namelist() if f.endswith(name)))).hexdigest() for name in expected}
    if actual!=expected: raise ValueError(f"B17 internal MAT hashes mismatch: {actual}")
    if propagation.ZIP!=path:
        propagation.ZIP=path
        propagation.B17OBJ=propagation.WIDTH=None
    return propagation.optics,propagation.measure,propagation.value_grad,propagation.wrapped_rmse,outer,actual
