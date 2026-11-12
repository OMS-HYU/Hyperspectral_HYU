"""Continuum quadrature/forward; inverse not implemented."""
import sys
from 공통.spectrum_forward import main as shared_main, continuum_plan, summed_forward
def main(argv=None):
    shared_main(["continuum", *(sys.argv[1:] if argv is None else argv)])
