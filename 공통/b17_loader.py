"""Resolve a shared B17 source and verify original MAT bytes, never substitute data."""
from 공통 import propagation

def check_b17():
    source=propagation.default_source()
    if propagation.SOURCE!=source:
        propagation.SOURCE=source
        propagation.B17OBJ=propagation.WIDTH=None
    b,_=propagation.load_physics()
    return propagation.optics,propagation.measure,propagation.value_grad,propagation.wrapped_rmse,b.archive_sha,b.inner_shas
