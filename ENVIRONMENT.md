# Execution environment and installation status

The final fixed-setting validation artifacts were checked against the following local environment.

| Component | Version |
|---|---|
| Python | 3.11.3 |
| OS | Windows NT 10.0, build 26200 |
| GPU | NVIDIA GeForce RTX 3070 Ti |
| CUDA | 12.1 |
| cuDNN | 9.1.0 |
| PyTorch | 2.5.1+cu121 |
| Torchvision | 0.20.1+cu121 |
| SAM 2 package | 1.0 |
| OpenAI CLIP | 1.0.1 |
| NumPy | 1.26.4 |
| Pandas | 3.0.0 |
| SciPy | 1.13.1 |
| OpenCV contrib | 4.10.0.84 |
| scikit-learn | 1.6.1 |
| scikit-image | 0.20.0 |
| BM3D | 4.0.3 |
| Pillow | 10.3.0 |
| Matplotlib | 3.9.4 |
| Shapely | 2.0.4 |
| Streamlit | 1.33.0 |
| OpenPyXL | 3.1.5 |

The checkpoint used for the final run was `sam2.1_hiera_large.pt`. Model weights are not included in this repository.


## Release dependency changes (2026-09-15)

The table above records the pre-existing local analysis environment, not a clean installation from requirements.txt. The release pins pandas 2.2.3 because Streamlit 1.33 requires pandas <3. The invalid PyPI requirement sam-2==1.0 is replaced by an official source archive pinned to commit 393ae336a752d26e68fb9a586e3d4ac14ff1e3c5; install it using scripts/install_sam2.py after requirements.txt.

A fresh virtual environment was created, but its network dependency installation was blocked by the execution service's approval/usage limit. Therefore a clean install and pip check have not yet passed. The GitHub Actions workflow performs these checks on a new runner after upload. Local verification used the environment in the table, including pandas 3.0.0, and does not prove the new dependency combination has installed successfully.

Do not use the historical package table as an alternative requirements file. CUDA hardware/software availability and model weights must be prepared separately.
