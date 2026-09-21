"""Install the pinned SAM 2 source after requirements.txt (no CUDA compiler needed)."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, SAM2_BUILD_CUDA='0')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-build-isolation',
                    '-r', str(root/'requirements-sam2.txt')], env=env, check=True)


if __name__ == '__main__':
    main()
