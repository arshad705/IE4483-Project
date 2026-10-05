# IE4483 Mini Project 2: Dogs vs. Cats (+ CIFAR-10)

Binary image classifier (dog vs. cat) in PyTorch, extended to CIFAR-10 (multi-class and class imbalance).
Course: IE4483, NTU.

The detailed plan is in [`project-plan.md`](project-plan.md).

## Setup

Requirements:
- Python 3.12.x
- NVIDIA GPU with CUDA (recommended). CPU also works, but training is slow.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install PyTorch (NOT in requirements.txt)
#    Check your driver's CUDA version with: nvidia-smi
#    Then copy the install command from https://pytorch.org/get-started/locally/
#    Use a CUDA build at or below the version nvidia-smi shows. torchaudio is not needed.
#    Tested with: torch 2.14.1+cu126
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. Install the other packages
pip install -r requirements.txt

# 4. Check that PyTorch sees the GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Dataset

Download the dataset from the course Drive link and place it in `datasets/`. It is not committed to git.

```
datasets/
├── train/cat/    (10,000 images)
├── train/dog/    (10,000 images)
├── val/cat/      (2,500 images)
├── val/dog/      (2,500 images)
└── test/         (500 images: 1.jpg to 500.jpg, no labels)
```

CIFAR-10 downloads itself into `data/` on the first run (via torchvision). It is not committed either.

## Planned structure and commands

The code lives in one notebook, as set out in `project-plan.md`:

​```
IE4483_Project2.ipynb   # all code, run top to bottom
datasets/               # local only, not committed
data/                   # CIFAR-10, auto-downloaded, not committed
checkpoints/            # saved weights, not committed
outputs/                # figures and result tables for the report
submission.csv          # final output
​```

## Reproducibility

- Seeds (Python, NumPy, PyTorch) and deterministic cuDNN are set in Cell 1 of the notebook.
- Use the pinned package versions above. Different GPUs or CUDA versions can still change results slightly.
- Run the notebook cells top to bottom.

## Team workflow

- Never push straight to `master`. Create a branch, push it, and open a pull request for review.
- Do not commit `venv/`, `datasets/`, `data/`, `checkpoints/` or `*.pth` files.
- Trained checkpoints are shared through the team Drive folder. Name files like resnet18_configC_seed42.pth
- Per-image prediction CSVs and result tables go in `outputs/` and are committed.
- Each member adds their work to `CONTRIBUTIONS.md` as they go (needed for the report).

## References

- Kaggle Dogs vs. Cats: https://www.kaggle.com/c/dogs-vs-cats
- Simonyan & Zisserman (2014), *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
- He et al. (2016), *Deep Residual Learning for Image Recognition*.
- Krizhevsky (2009), *Learning Multiple Layers of Features from Tiny Images* (CIFAR-10).