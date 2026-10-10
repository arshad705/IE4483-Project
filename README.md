# IE4483 Mini Project 2: Dogs vs. Cats (+ CIFAR-10)

Binary image classifier (dog vs. cat) in PyTorch, extended to CIFAR-10 (multi-class and class imbalance).
Course: IE4483, NTU.

The detailed plan, roles and workflow are in [`project-plan.md`](project-plan.md).

## Setup

Requirements:
- Python 3.12.x
- NVIDIA GPU with CUDA (recommended). CPU and Apple Silicon work for subset tests, but full training is slow.
- VS Code with the Jupyter extension

```bash
# 1. Create and activate a virtual environment
py -3.12 -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install PyTorch (NOT in requirements.txt). Same version for everyone; pick the build for your machine below.
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

# 3. Install the other packages
pip install -r requirements.txt

# 4. Check
python -c "import torch, torchvision, numpy, pandas, matplotlib, sklearn, PIL, tqdm; print(torch.__version__, torch.cuda.is_available())"
```

Which PyTorch build to install (check your driver with `nvidia-smi`):

| Your machine | Index URL / command |
|---|---|
| NVIDIA GPU, driver CUDA ≥ 12.6 | `--index-url https://download.pytorch.org/whl/cu126` |
| Older NVIDIA card (GTX 10-series or older) | Use `cu126`; newer CUDA builds drop some older GPUs |
| No NVIDIA GPU (Windows / Linux) | `--index-url https://download.pytorch.org/whl/cpu` |
| Mac (Apple Silicon) | `pip install torch==2.14.1 torchvision==0.29.1` (uses MPS) |

No conda needed: pip wheels include the CUDA runtime; you only need the NVIDIA driver.

In VS Code, open a notebook and choose **Select Kernel → Python Environments → venv**.

## Dataset

Download the dataset from the course link and place it in `datasets/` at the repo root. It is not committed.

```
datasets/
├── train/cat/    (10,000 images)
├── train/dog/    (10,000 images)
├── val/cat/      (2,500 images)
├── val/dog/      (2,500 images)
└── test/         (500 images: 1.jpg to 500.jpg, no labels)
```

If the zip contains a `__MACOSX` folder, delete it, and remove any `._*` or `.DS_Store` files inside `datasets/`.

CIFAR-10 downloads itself into `data/` on the first run (via torchvision). It is not committed either.

## Project structure

```
common.py                  shared code: seed, data loading, models, train/evaluate, submission
01_configA_cnn.ipynb       Config A: custom CNN from scratch
02_resnet_B_Bp_C.ipynb     Configs B, B′, C, C-noaug (ResNet-18) + submission.csv
03_comparison.ipynb        comparison, error analysis (reads saved results, no GPU needed)
04_cifar10.ipynb           CIFAR-10 and class imbalance
outputs/                   results CSVs, prediction CSVs, figures (committed)
checkpoints/               model weights (not committed; shared Drive)
```

Every notebook has `FULL_RUN = False` at the top. Keep it `False` (2,000 train / 500 val subset) while developing. Only the trainer sets it to `True`.

## Reproducibility

- `common.set_seed(42)` runs first in every notebook (Python, NumPy, PyTorch seeds; deterministic cuDNN).
- All final training runs on one machine, so results are directly comparable. Each results file records the GPU and git commit.
- Use the pinned versions above. Different GPUs or CUDA versions can still change results slightly.

## Team workflow

1. **Build:** each member writes their notebook on a branch and tests it with `FULL_RUN = False`.
2. **Freeze:** everything merged to `master`; `common.py` locked.
3. **Train:** the trainer runs every notebook from `master` with `FULL_RUN = True`.
4. **Analyse:** everyone works from the saved files in `outputs/`.

Rules:
- Never push straight to `master`. Create a branch, push it, and open a pull request.
- Clear notebook outputs before committing (VS Code: *Clear All Outputs*).
- Do not commit `venv/`, `datasets/`, `data/`, `checkpoints/`, `*.pth` or `*.zip`.
- Checkpoints go in the shared Drive folder (link in group chat), named like `resnet18_C_seed42.pth`.
- Results CSVs, prediction CSVs and figures go in `outputs/` and are committed.
- Add your work to `CONTRIBUTIONS.md` as you go (needed for the report).

## Troubleshooting

**`DLL load failed ... An Application Control policy has blocked this file`** (Windows): Smart App Control blocked a package binary. Find the file with:

```
Get-WinEvent -LogName "Microsoft-Windows-CodeIntegrity/Operational" -MaxEvents 30 | ForEach-Object { if ($_.Message -match "attempted to load\s+(\S+)") { $Matches[1] } } | Sort-Object -Unique
```

Try a restart first. pandas is pinned to 2.3.3 because 3.0.6 was blocked. Do not turn off Smart App Control.

## References

- Kaggle Dogs vs. Cats: https://www.kaggle.com/c/dogs-vs-cats
- Simonyan & Zisserman (2014), *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
- He et al. (2016), *Deep Residual Learning for Image Recognition*.
- Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10).
