# IE4483 Mini Project 2 — Dogs vs. Cats (+ CIFAR-10) Plan

## Overview

Binary image classifier (dog vs. cat) built with PyTorch. Four model configurations are trained and compared, with **ResNet-18 + Linear FC (fine-tuned, Config C)** selected as the final model for submission. The same model is then adapted for CIFAR-10 multi-class classification and class imbalance.

**Execution:** Shared code in `common.py` + one notebook per part (see [Code Structure](#code-structure)). All members build code in parallel. **All final training runs on one machine, by one member (the trainer)**, so every result is directly comparable.

**Dataset location:** `datasets/train/`, `datasets/val/`, `datasets/test/`
**Output:** `submission.csv` with columns `id` (1–500) and `label` (1=dog, 0=cat)

---

## File Splits

| Files |
|------|
| `common.py` (shared code)
| `01_configA_cnn.ipynb` 
| `02_resnet_B_Bp_C.ipynb` (+ submission) 
| `03_comparison.ipynb` 
| `04_cifar10.ipynb` 
| 

### Phases

| Phase | Who | What | Done when |
|-------|-----|------|-----------|
| 1. Build | All, in parallel | Write own notebook; test with `FULL_RUN = False` (subset) on own machine | Notebook runs top-to-bottom after kernel restart |
| 2. Freeze | All | Merge everything to `master`; lock `common.py` | Code freeze date: **TBD (proposed 19 Oct)** |
| 3. Train | Trainer only | Run each notebook from `master` with `FULL_RUN = True` | All results CSVs committed; checkpoints uploaded to shared Drive |
| 4. Analyse | All, in parallel | `03_comparison.ipynb`, error analysis, report — from saved files only | Report parts a–h drafted |

### Handoff contract (every notebook, before Phase 2)

1. **Runs top-to-bottom on a subset** on the author's machine. One switch at the top: `FULL_RUN = False` (subset: 2,000 train / 500 val) or `True` (full data). The trainer only flips this switch.
2. **Imports all shared code from `common.py`.** No copy-pasted data loading, transforms or training loops — every config must use the identical pipeline.
3. **Saves to the agreed paths** (see [Output conventions](#output-conventions)).
4. **Merged to `master` before training.** The trainer runs `master`, never a personal branch.
5. **Outputs cleared before every commit** (VS Code: *Clear All Outputs*), to avoid notebook merge conflicts.

Any change to `common.py` after the freeze means retraining the affected configs. The `git_commit` column in every results file shows which code produced which number.

### Output conventions

**Config IDs:** `A`, `B`, `Bp` (B′), `C`, `C-noaug`, `cifar-bal`, `cifar-imb`, `cifar-imb-wloss`, `cifar-imb-sampler`

| File | Path | Committed? |
|------|------|-----------|
| Checkpoint | `checkpoints/<model>_<config>_seed<N>.pth` (e.g. `resnet18_C_seed42.pth`) | No — shared Drive (link in group chat) |
| Per-epoch results | `outputs/results_<config>_seed<N>.csv` | Yes |
| Per-image val predictions | `outputs/preds_<config>_seed<N>.csv` (`path, label, pred, prob_dog`) | Yes |
| Per-class accuracy (CIFAR) | `outputs/perclass_<config>_seed<N>.csv` | Yes |
| Figures | `outputs/*.png` | Yes |

**Results CSV columns (one row per epoch):**

```
config, seed, epoch, train_loss, train_acc, eval_loss, eval_acc, eval_split, lr, seconds, gpu, git_commit
```

- `eval_split` = `val` for Dogs vs. Cats, `test` for CIFAR-10 (no separate CIFAR val set; no model selection on it).
- Config B (SVM) writes a single row with `epoch = 1` and empty train-loss columns.

---

## Project Requirements Compliance

| PDF Requirement | How Fulfilled |
|----------------|--------------|
| **Step 1** — Load train/val/test from provided dataset | `common.py` — `ImageFolder` + custom `TestDataset` |
| **Step 2** — Preprocess + augment (scaling, rotation, flipping) | `common.py` — `RandomResizedCrop` + `RandomRotation` + `RandomHorizontalFlip` + `ColorJitter` |
| **Step 3** — Design classification model (CNN or pretrained backbone) | `01` (Config A), `02` (Configs B, B′, C); `get_resnet18()` in `common.py` |
| **Step 4** — Try different parameters; train on train set, validate on val set | `01`, `02` — val set never used for weight updates |
| **Step 5** — Generate submission.csv (id + label, 1=dog, 0=cat) | `02` — `common.write_submission()` with format checks |
| **Part a** — State image counts + describe preprocessing | `02` prints counts; transforms in `common.py` |
| **Part b** — Model, architecture, dims, loss, training, code, reproducibility | `common.py`, `01`, `02` — seed fixed, one trainer, GPU + commit logged |
| **Part c** — Discuss parameter choices + reasons | Hyperparameter table below; justified in report |
| **Part d** — Val accuracy + submission.csv | `02` (training + submission), `03` (final table) |
| **Part e** — Analyse correct/incorrect samples | `03` — from `preds_C_seed42.csv` (val set; test set unlabelled) |
| **Part f** — Compare different models and data processing | `03` — reads all results CSVs (A, B, B′, C, C-noaug) |
| **Part g** — CIFAR-10 adaptation, describe changes, report test results | `04` |
| **Part h** — Handle class imbalance with ≥2 approaches | `04` — Weighted loss + WeightedRandomSampler |

---

## Environment Setup

### Requirements

- Python 3.12.x
- NVIDIA GPU with CUDA (recommended; CPU and Apple Silicon work for subset tests)
- VS Code with the Jupyter extension (kernel via `ipykernel`, included in `requirements.txt`)

### Setup Steps (Windows PowerShell)

```bash
# 1. Create and activate the virtual environment
py -3.12 -m venv venv
venv\Scripts\activate

# 2. Install PyTorch — same version for everyone; CUDA build depends on your driver (see README)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126

# 3. Install everything else (pinned versions)
pip install -r requirements.txt

# 4. Check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- **torch is not in `requirements.txt`** on purpose: each machine needs the CUDA build matching its driver. The *version* (2.14.1) is the same for everyone.
- **pandas is pinned to 2.3.3**, not 3.x: Windows Smart App Control blocks an unsigned pandas 3.0.6 binary on at least one team laptop.
- To regenerate `requirements.txt`: `pip freeze | findstr /v /b /i "torch" | Out-File requirements.txt -Encoding ascii`

### Project File Structure

```
IE4483-Project/
├── venv/                        ← virtual environment (not committed)
├── datasets/                    ← local only, not committed
│   ├── train/cat/   (10,000)    train/dog/ (10,000)
│   ├── val/cat/     (2,500)     val/dog/   (2,500)
│   └── test/        (500: 1.jpg – 500.jpg)
├── data/                        ← CIFAR-10, auto-downloaded (not committed)
├── checkpoints/                 ← model weights (not committed; shared Drive)
├── outputs/                     ← results CSVs, preds CSVs, figures (committed)
├── common.py                    ← shared code: seed, data, models, train/eval, submission
├── 01_configA_cnn.ipynb         ← Config A
├── 02_resnet_B_Bp_C.ipynb       ← Configs B, B′, C, C-noaug + submission.csv
├── 03_comparison.ipynb          ← Parts d, e, f — reads saved files only, no GPU needed
├── 04_cifar10.ipynb             ← Parts g, h
├── submission.csv               ← final only (from trainer's Config C run)
├── sampleSubmission.csv
├── project-plan.md
├── README.md
├── CONTRIBUTIONS.md
└── requirements.txt
```

> **GitHub sharing policy:** Commit code (`common.py`, notebooks with outputs cleared), `outputs/`, the **final** `submission.csv`, and docs. Never commit `venv/`, `datasets/`, `data/`, `checkpoints/` or `*.pth`. Checkpoints go to the shared Drive folder (link in group chat); the final `resnet18_C_seed42.pth` may also be attached to a GitHub Release.

---

## Model Configurations (for Part f Comparison)

Four configurations are trained and compared. Only **Config C** is used for final submission. Config B′ isolates the effect of fine-tuning from classifier type. **C-noaug** (Config C without augmentation) shows the effect of data augmentation.

| Config | Model | Classifier | Training | Purpose |
|--------|-------|-----------|---------|---------|
| **A**  | Custom CNN (4 conv blocks, scratch) | Linear FC | Train from scratch | Baseline — Week 9 CNN |
| **B**  | ResNet-18 (frozen backbone) | SVM (sklearn) | Extract features → fit SVM | Week 6 SVM + Week 8 features |
| **B′** | ResNet-18 (frozen backbone) | Linear FC | Frozen backbone, train FC only | Isolates classifier type (SVM vs FC) |
| **C**  | ResNet-18 (unfreeze layer4 + fc) | Linear FC (fine-tuned) | Fine-tune layer4 + fc | **Final model** — Week 8 transfer learning |
| **C-noaug** | Same as C | Same as C | No train augmentation | Data-processing comparison (Part f) |

**Why B′ is needed (Part f):**
B and C change *two things at once* (frozen→fine-tuned backbone AND SVM→FC classifier). Adding B′ allows clean isolation:
- B → B′: same frozen backbone, SVM replaced by FC → isolates **classifier type**
- B′ → C: same FC classifier, backbone unfrozen → isolates **fine-tuning effect**

**Expected val accuracy:**
```
Config A  (Custom CNN)          → ~75–82%
Config B  (Frozen ResNet + SVM) → ~88–91%
Config B' (Frozen ResNet + FC)  → ~89–92%
Config C  (Fine-tuned ResNet)   → ~91–94%   ← SELECTED
```

---

## Code Structure

Build order: **`common.py` first** (everything imports it) → `01`, `02`, `04` in parallel → `03`.

### `common.py` — shared code

- `SEED = 42`, `set_seed(seed)`: `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, `cudnn.benchmark = False`
- `DEVICE`: `cuda` → `mps` → `cpu`
- `DATA_DIR`, `FULL_RUN` handling, `SUBSET_SIZE = {"train": 2000, "val": 500}`
- Transforms:
  - Train: `RandomRotation(15)`, `RandomHorizontalFlip(0.5)`, `RandomResizedCrop(224, scale=(0.8, 1.0))`, `ColorJitter(0.2, 0.2, 0.2)`, `ToTensor()`, `Normalize(ImageNet mean/std)`
  - Train without augmentation (for C-noaug) and val/test: `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize`
- `get_dataloaders(full_run, augment=True)` — `ImageFolder` for train/val, seeded subset, seeded `DataLoader` generator; asserts `cat=0`, `dog=1`
- `TestDataset` — preserves numeric file IDs, sorted 1..500
- `get_resnet18(num_classes, mode)` — `mode` ∈ `'features'` (B), `'frozen_fc'` (B′), `'finetune'` (C); loads `ResNet18_Weights.IMAGENET1K_V1`; prints frozen/unfrozen status
- `set_frozen_bn_eval(model, keep_train)` — keeps frozen BatchNorm layers in eval mode so running stats stay at ImageNet values:
  ```python
  model.train()
  for name, m in model.named_modules():
      if isinstance(m, nn.BatchNorm2d) and not name.startswith(tuple(keep_train)):
          m.eval()   # keep_train=() for B′, ("layer4",) for C
  ```
- `train_one_epoch(...)`, `evaluate(...)` → loss, accuracy
- `fit(model, config, ...)` — epoch loop; writes `outputs/results_<config>_seed<N>.csv`; saves best checkpoint
- `predict_val(...)` → `outputs/preds_<config>_seed<N>.csv`
- `write_submission(model, path)` — predicts the 500 test images, writes `submission.csv`, checks: 500 rows, columns `id,label`, ids 1..500, labels ∈ {0, 1}, matches `sampleSubmission.csv` ids
- `run_metadata()` → GPU name and current git commit, stored in every results row

**Status:** [ ] pending

---

### `01_configA_cnn.ipynb` — Config A

**Report section:** Parts b, c, f | **Owner:** TBD

- Defines `CustomCNN(nn.Module)`:
  ```
  Input: (B, 3, 224, 224)
  → Conv2d(3, 32, 3, pad=1) + BatchNorm + ReLU + MaxPool2d(2)   → (B, 32, 112, 112)
  → Conv2d(32, 64, 3, pad=1) + BatchNorm + ReLU + MaxPool2d(2)  → (B, 64, 56, 56)
  → Conv2d(64, 128, 3, pad=1) + BatchNorm + ReLU + MaxPool2d(2) → (B, 128, 28, 28)
  → Conv2d(128, 256, 3, pad=1) + BatchNorm + ReLU + MaxPool2d(2)→ (B, 256, 14, 14)
  → AdaptiveAvgPool2d(4,4)                                      → (B, 256, 4, 4)
  → Flatten                                                     → (B, 4096)
  → Linear(4096, 512) + ReLU + Dropout(0.5)
  → Linear(512, 2)                                              → logits
  ```
- Verifies output shape with a dummy tensor
- Trains Config A with `common.fit` (5 epochs, Adam lr=1e-3, StepLR ×0.5/5ep)
  - ⚠️ Config A is a baseline only. Unequal epochs vs. C are intentional and stated in the report.

**Lecture grounding:** Conv+pooling (Week 9), BatchNorm+Dropout (Week 11), backprop (Week 7)

**Status:** [ ] pending

---

### `02_resnet_B_Bp_C.ipynb` — Configs B, B′, C, C-noaug + submission

**Report section:** Parts a, b, c, d, f | **Owner:** TBD

- Prints dataset image counts (Part a)
- **Config B:** extract 512-dim frozen ResNet-18 features for train/val → `sklearn.svm.SVC(kernel='rbf', C=1.0)` → val accuracy (one results row)
- **Config B′:** frozen backbone + FC, 5 epochs, Adam lr=1e-3, StepLR ×0.5/5ep
- **Config C:** unfreeze layer4 + fc, 15 epochs, Adam lr=1e-4, StepLR ×0.5/3ep; best checkpoint by val accuracy
- **Config C-noaug:** identical to C, train transform without augmentation
- Saves val predictions for C (`preds_C_seed42.csv`) for Part e
- Writes `submission.csv` from the best Config C checkpoint via `common.write_submission`

**Lecture grounding:** Transfer learning (Week 8), SVM (Week 6), fine-tuning (Week 11)

**Status:** [ ] pending

---

### `03_comparison.ipynb` — analysis (no GPU needed)

**Report section:** Parts d, e, f | **Owner:** TBD

- Reads `outputs/results_*.csv` → comparison table: A / B / B′ / C / C-noaug val accuracy
- Conclusions from the comparison:
  - A vs C → pretrained weights matter (Week 8)
  - B vs B′ → SVM vs FC on same frozen features (classifier type)
  - B′ vs C → frozen vs fine-tuned backbone, same FC head (fine-tuning effect)
  - C vs C-noaug → effect of data augmentation
- Training curves (loss + accuracy vs epoch) for A, B′, C
- From `preds_C_seed42.csv`: confusion matrix, final val accuracy, 4 correct + 4 incorrect val images with predicted vs actual label
- Notes that the test set is unlabelled, so the val set is used for Part e
- Saves all figures to `outputs/`

**Status:** [ ] pending

---

### `04_cifar10.ipynb` — Parts g, h

**Report section:** Parts g, h | **Owner:** TBD

**Part g — CIFAR-10:**
- Loads CIFAR-10 via `torchvision.datasets.CIFAR10(root='./data', download=True)`
- Resizes 32×32 → 224×224 for ResNet compatibility
- `get_resnet18(num_classes=10, mode='finetune')` from `common.py`
- Trains 10 epochs (Adam lr=1e-4, as Config C); reports test accuracy (10,000 labelled images)
- Documents changes vs Dogs vs. Cats: `Linear(512, 10)` instead of 2; `CIFAR10` instead of `ImageFolder`; no custom test loader needed

**Part h — Class imbalance:**
- Simulates imbalance: classes 0, 1, 2 (airplane, automobile, bird) reduced to 20% (~1,000 each); classes 3–9 stay at 5,000

| Config | Setup |
|--------|-------|
| `cifar-bal` | Balanced CIFAR-10 (the Part g run) |
| `cifar-imb` | Imbalanced, no fix |
| `cifar-imb-wloss` | Imbalanced + Weighted CrossEntropyLoss (inverse class frequency) |
| `cifar-imb-sampler` | Imbalanced + WeightedRandomSampler |

- Reports overall accuracy, **per-class recall and macro-F1** (accuracy alone hides minority-class failure) → `perclass_<config>_seed42.csv`
- All four runs on the same machine (the trainer's), so they are comparable

**CIFAR-10 facts to cite in report:**
- 60,000 colour images, 32×32, 10 classes
- 50,000 train (5,000/class, balanced) + 10,000 test
- Source: Krizhevsky, Nair, Hinton — https://www.cs.toronto.edu/~kriz/cifar.html

**Status:** [ ] pending

---

## Hyperparameters

| Parameter | Config A | Config B′ | Config C / C-noaug | CIFAR (g, h) | Reason |
|-----------|---------|----------|---------|---------|--------|
| Optimizer | Adam | Adam | Adam | Adam | Adaptive LR (Week 11) |
| Learning Rate | 1e-3 | 1e-3 | 1e-4 | 1e-4 | Lower for fine-tuning (Week 8) |
| Scheduler | StepLR ×0.5/5ep | StepLR ×0.5/5ep | StepLR ×0.5/3ep | StepLR ×0.5/3ep | LR decay (Week 11) |
| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss | CE (weighted for `-wloss`) | Standard multi-class |
| Batch Size | 32 | 32 | 32 | 32 | Fits 4 GB GPU (ResNet-18 train step peak ≈ 0.8 GB on GTX 1650) |
| Epochs | 5 | 5 | 15 | 10 | A & B′ for comparison; C is final |
| Seed | 42 | 42 | 42 | 42 | Reproducibility |
| BN frozen layers eval | N/A | ✅ all | ✅ all except layer4 | ✅ all except layer4 | Prevent BN stat drift (Week 11) |

Batch size stays fixed at 32 for every run — changing it changes results more than the hardware does.

---

## Execution Order

**Build (Phase 1):** `common.py` → `01`, `02`, `04` in parallel → `03`

**Train (Phase 3, trainer only, from `master` with `FULL_RUN = True`):**

```
1. 02_resnet_B_Bp_C.ipynb   → secures the final model + submission.csv first
2. 04_cifar10.ipynb         → longest runs; start early
3. 01_configA_cnn.ipynb     → short baseline
4. Commit outputs/, upload checkpoints → everyone runs 03_comparison.ipynb
```

Time the first full epoch of each notebook and extrapolate before committing to the full schedule. Backup if the trainer's machine is unavailable: Kaggle (change only `DATA_DIR`).

---

## Report Mapping

| Part | Marks | Notebooks | Lecture Weeks |
|------|-------|-----------|--------------|
| a | 10% | `common.py`, `02` | 9 |
| b | 20% | `common.py`, `01`, `02` | 7, 8, 9, 11 |
| c | 10% | `01`, `02` | 7, 11 |
| d | 20% | `02`, `03` | — |
| e | 10% | `03` | 9 |
| f | 10% | `01`, `02`, `03` | 5, 6, 8, 9, 11 |
| g | 10% | `04` | 8, 9 |
| h | 10% | `04` | 11 |
| **Total** | **100%** | | |

---

## Lecture Grounding (IE4483 — Weeks 5 to 11)

### Week 5 — Decision Tree (ID3/C4.5) — Lectures 13–15
- **Use in part (f):** Conventional classifiers like Decision Trees split on individual pixel features — they lose all spatial structure. This justifies moving to CNNs.

### Week 6 — KNN, SVM — Lectures 16–18
- **Use in part (f):** Config B uses SVM on ResNet features. Discuss why SVM on raw pixels would fail at 224×224 resolution (150K+ features). With ResNet features (512-dim) it becomes feasible but still underperforms end-to-end fine-tuning.

### Week 7 — Neural Networks, Backpropagation — Lectures 19–21
- **Use in parts (b) and (c):** Explain how all CNN models learn via backpropagation and gradient descent. Config C starts from ImageNet weights (better initialisation) vs. Config A which starts from random weights.

### Week 8 — Deep Learning, Image Processing, Computer Vision — Lectures 22–24
- **Use in parts (b) and (f):** Deep networks learn feature hierarchies (edges → shapes → objects). Transfer learning: ResNet-18 pretrained on ImageNet already encodes general visual features applicable to Dogs vs. Cats and CIFAR-10.

### Week 9 — Convolutional Neural Networks — Lectures 25–27
- **Use in parts (a), (b), (e):** Core week. Describe conv layers, pooling, receptive fields, feature maps. For part (e): analyse why model fails — background clutter, unusual pose, occlusion confuse spatial features.

### Week 11 — Regularisation and Optimisation — Lectures 31–33
- **Use in parts (b), (c), (f), (h):**
  - **Dropout (0.5):** prevents co-adaptation of neurons, reduces overfitting
  - **BatchNorm:** stabilises training, allows higher learning rates
  - **Adam:** adaptive learning rate per parameter, better than plain SGD for fine-tuning
  - **StepLR:** decays learning rate as training progresses
  - **Weighted loss** (part h): cost-sensitive learning — penalises misclassification of minority classes more heavily


---

## Reproducibility

- `common.set_seed(42)` is called first in every notebook: `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic = True`, `cudnn.benchmark = False`
- Subsets and `DataLoader` shuffling use seeded `torch.Generator`s
- **All final runs on one machine (the trainer's)**, so comparisons are not affected by GPU/CUDA differences. Every results row records `gpu` and `git_commit`.
- Package versions pinned in `requirements.txt`; torch pinned to 2.14.1 (CUDA build per machine)
- Notebooks must run top-to-bottom after a kernel restart
- Same seed on different hardware can still give slightly different numbers — never compare results across machines

---

## Notes

- `venv/`, `datasets/`, `data/`, `checkpoints/`, `*.pth`, `*.zip`, `__MACOSX/` are in `.gitignore`
- CIFAR-10 in `data/` is auto-downloaded on first run of `04` — do not manually place files there
- CIFAR-10 is perfectly balanced (5,000/class) — Part h requires artificial imbalance
- Test set (Dogs vs. Cats) is unlabelled — Part e uses the **val set**; state this in the report
- Only **Config C** is the final selected model — A, B, B′ and C-noaug exist for comparison only
- `NUM_WORKERS = 0` is the safe default in notebooks on Windows; raise it only if tested
- Windows **Smart App Control** can block unsigned package DLLs (`An Application Control policy has blocked this file`). Check `Microsoft-Windows-CodeIntegrity/Operational` in Event Viewer for the file name; a restart fixed one torch case. Do not disable Smart App Control (cannot easily be turned back on).
- Reference: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*
- PDF references: VGG [Simonyan & Zisserman, 2014] and ResNet [He et al., 2016] — cite both even if only ResNet is used

---

## Changelog

- **2026-10-10** — Switched from a single notebook to `common.py` + 4 notebooks; one trainer runs all full training; added handoff contract, output conventions and C-noaug config; environment updated (Python 3.12, torch 2.14.1 + cu126, no torchaudio, pandas 2.3.3, ipykernel); fixed `numpy.seed` → `np.random.seed`; Part h reports per-class recall and macro-F1.
