#!/usr/bin/env python3
"""Active single-spectrum K=1 native launcher; legacy explicitly opt-in."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    legacy=bool(args and args[0]=='--legacy')
    target=ROOT/'코드'/('unified_simulator.py' if legacy else 'native_cli.py')
    if legacy: args=args[1:]
    return subprocess.run([sys.executable,str(target),*(args or ['--help'])],check=False).returncode
if __name__=='__main__':raise SystemExit(main())
