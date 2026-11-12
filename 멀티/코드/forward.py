"""K=2–3 forward; inverse not implemented."""
import sys
from 공통.spectrum_forward import main as shared_main, multi_plan, summed_forward
def main(argv=None):
    shared_main(["multi", *(sys.argv[1:] if argv is None else argv)])
