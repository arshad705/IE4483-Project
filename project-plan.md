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

**Config IDs:** `A`, `B`, `Bp` (B′), `C`, `C-noaug`, `sweep-lr1e-3`, `sweep-lr1e-4`, `sweep-lr1e-5`, `cifar-bal`, `cifar-imb`, `cifar-imb-wloss`, `cifar-imb-sampler`

| File | Path | Committed? |
|------|------|-----------|
| Checkpoint | `checkpoints/<model>_<config>_seed<N>.pth` (e.g. `resnet18_C_seed42.pth`, `linear_Bp_seed42.pth`) | No — shared Drive (link in group chat) |
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
| **Step 4** — Try different parameters; train on train set, validate on val set | `02` — learning-rate sweep for Config C (1e-3 / 1e-4 / 1e-5); all configs validated on val, never trained on it |
| **Step 5** — Generate submission.csv (id + label, 1=dog, 0=cat) | `02` — `common.write_submission()` with format checks |
| **Part a** — State image counts + describe preprocessing | `02` prints counts; transforms in `common.py` |
| **Part b** — Model, architecture, dims, loss, training, code, reproducibility | `common.py`, `01`, `02` — seed fixed, one trainer, GPU + commit logged |
| **Part c** — Discuss parameter choices + reasons | Hyperparameter table below + LR sweep results (`results_sweep-*.csv`) as evidence |
| **Part d** — Val accuracy + submission.csv | `02` (training + submission), `03` (final table) |
| **Part e** — Analyse correct/incorrect samples | `03` — from `preds_C_seed42.csv` (val set; test set unlabelled) |
| **Part f** — Compare different models and data processing | `03` — comparison chain B → B′ → C-noaug → C (one change per step) + A vs C |
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

Five configurations are trained and compared. Only **Config C** is used for final submission.

| Config | Model | Classifier | Train augmentation | What learns | Purpose |
|--------|-------|-----------|-----|---------|---------|
| **A**  | Custom CNN (4 conv blocks) | Linear FC | Yes | Everything, from scratch | Baseline — Week 9 CNN |
| **B**  | ResNet-18 (frozen) | SVM (sklearn) | No | SVM only (on fixed features) | Week 6 SVM + Week 8 features |
| **B′** | ResNet-18 (frozen) | Linear FC | No | FC only (on the same fixed features as B) | Isolates classifier type |
| **C-noaug** | ResNet-18 (layer4 unfrozen) | Linear FC | No | layer4 + FC | Isolates fine-tuning |
| **C**  | ResNet-18 (layer4 unfrozen) | Linear FC | Yes | layer4 + FC | **Final model** — isolates augmentation |

### Comparison chain (Part f)

Each step changes **exactly one thing**, so any accuracy difference can be attributed to that change:

```
B ──────────► B′ ──────────────► C-noaug ──────────► C
   SVM → FC       frozen → fine-tuned     no aug → aug
 (classifier)       (fine-tuning)         (augmentation)
```

- **B → B′:** same features, same inputs; only the classifier changes (SVM vs linear layer).
- **B′ → C-noaug:** same head, same inputs (no augmentation); only whether layer4 learns. The LR also differs (1e-3 vs 1e-4), justified because a new layer trained from scratch needs a larger step than pretrained weights (Week 8).
- **C-noaug → C:** identical model and schedule; only augmentation changes (Week 11). Compare val accuracy **and the train–val gap** (overfitting).
- **A vs C:** scratch CNN vs pretrained ResNet at the same training budget. **Limitation (state in report):** architecture and pretraining both differ, so this shows the combined effect, not pretraining alone. A ResNet-18-from-scratch run would separate them but is not planned (compute).

**Why B and B′ use fixed features:** the backbone is frozen and its BatchNorm is fixed, so without augmentation every image gives the same 512 numbers every epoch. Features are extracted **once** (eval transforms, `augment=False`) and reused by both B and B′ — mathematically identical to running the frozen network each epoch, but B′'s 15 epochs take seconds instead of 15 full passes.

**Expected val accuracy** (estimates only — replace with measured results):
```
Config A       (Custom CNN, 15 ep)     → ~80–88%
Config B       (Frozen ResNet + SVM)   → ~88–92%
Config B'      (Frozen ResNet + FC)    → ~88–92%
Config C-noaug (Fine-tuned, no aug)    → ~92–96%
Config C       (Fine-tuned ResNet)     → ~93–96%   ← SELECTED
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

**Status:** [x] done (full run on L40S, commit fea3316)

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
- Trains Config A with `common.fit` (up to 15 epochs, Adam lr=1e-3, StepLR ×0.5/3ep, train augmentation on; best-val checkpoint) — same budget and schedule as C
- `keep_bn_train=None` in `fit()`: Config A trains all its BatchNorm layers (nothing is frozen)

**Lecture grounding:** Conv+pooling (Week 9), BatchNorm+Dropout (Week 11), backprop (Week 7)

**Status:** [x] done (full run on L40S, commit fea3316)

---

### `02_resnet_B_Bp_C.ipynb` — Configs B, B′, C, C-noaug + submission

**Report section:** Parts a, b, c, d, f | **Owner:** TBD

Run in this order:

1. **Dataset counts** — `describe_dataset()` (Part a)
2. **LR sweep for C (always on the subset)** — `get_dataloaders(full_run=False)` regardless of `FULL_RUN`; Config C setup for 3 epochs at lr = 1e-3, 1e-4, 1e-5 → `results_sweep-lr<x>_seed42.csv`. All three on one machine in one go. Purpose: evidence for the LR choice in Part c. Takes minutes.
3. **Shared fixed features (B and B′)** — `get_dataloaders(FULL_RUN, augment=False)` (⚠️ **must** be `augment=False`, otherwise features come from randomly augmented images) → `get_resnet18(mode="features")` → `extract_features()` once for train and val; keep in memory
4. **Config B:** `sklearn.svm.SVC(kernel='rbf', C=1.0)` on the features → `save_result_row("B", acc)`
5. **Config B′:** `nn.Linear(512, 2)` trained with `fit(..., model_name="linear")` on `TensorDataset` loaders of the same features; up to 15 epochs, Adam lr=1e-3, StepLR ×0.5/3ep. Runs in seconds.
6. **Config C-noaug:** `get_dataloaders(FULL_RUN, augment=False)`; layer4 + fc; up to 15 epochs, Adam lr=1e-4, StepLR ×0.5/3ep; `keep_bn_train=("layer4",)`
7. **Config C:** identical to C-noaug except `augment=True` (unless the sweep clearly favours another LR — then use it for both C and C-noaug and record why)
8. `predict_val()` for C → `preds_C_seed42.csv` (Part e)
9. `write_submission()` from the best Config C checkpoint

**Lecture grounding:** Transfer learning (Week 8), SVM (Week 6), fine-tuning (Week 11)

**Status:** [x] done (full run on L40S, commit fea3316)

---

### `03_comparison.ipynb` — analysis (no GPU needed)

**Report section:** Parts d, e, f | **Owner:** TBD

- Reads `outputs/results_*.csv` → comparison table: A / B / B′ / C-noaug / C best val accuracy (+ epoch reached)
- Conclusions follow the comparison chain (see Model Configurations):
  - B → B′ → classifier type (SVM vs linear layer, same features) — Week 6
  - B′ → C-noaug → fine-tuning effect (frozen vs layer4 trained) — Week 8
  - C-noaug → C → augmentation effect: val accuracy **and train–val gap** — Week 11
  - A vs C → scratch CNN vs pretrained ResNet at equal budget (architecture + pretraining combined; state the limitation) — Weeks 8, 9
- LR sweep table/plot from `results_sweep-*.csv` (Part c)
- Training curves (loss + accuracy vs epoch) for A, B′, C-noaug, C
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
| Learning Rate | 1e-3 | 1e-3 | 1e-4 (confirmed by sweep) | 1e-4 | Random/new weights need larger steps; pretrained weights need small steps or they are destroyed (Week 8) |
| Scheduler | StepLR ×0.5/3ep | StepLR ×0.5/3ep | StepLR ×0.5/3ep | StepLR ×0.5/3ep | Same decay shape for all; halves LR 4× in 15 epochs (Week 11) |
| Max epochs | 15 | 15 | 15 | 10 | **Same budget** for all Dogs vs. Cats configs; CIFAR runs are compared only with each other |
| Checkpoint kept | Best val | Best val | Best val | Last epoch | Best-val = "up to 15 epochs" for everyone; never select on the CIFAR test set |
| Train augmentation | Yes | No (fixed features) | C-noaug: No / C: Yes | Yes | One change per step in the comparison chain |
| Loss | CrossEntropyLoss | CrossEntropyLoss | CrossEntropyLoss | CE (weighted for `-wloss`) | Standard multi-class |
| Batch Size | 32 | 32 | 32 | 32 | Fits 4 GB GPU (ResNet-18 train step peak ≈ 0.8 GB on GTX 1650) |
| Seed | 42 | 42 | 42 | 42 | Reproducibility |
| BN frozen layers eval | N/A (all train) | N/A (fixed features) | ✅ all except layer4 | ✅ all except layer4 | Prevent BN stat drift (Week 11) |

**Rules behind these choices:**
- **Same for all:** max epochs, LR schedule, batch size, data, seed, validation set. Otherwise a difference could just mean "trained longer".
- **Different by design:** starting LR (suits each model type — forcing one LR would handicap someone) and augmentation (deliberately varied in the chain).
- Reported val accuracy is the best epoch's; the same (small, optimistic) selection effect applies to every config, so comparisons stay fair. State this in the report.

Batch size stays fixed at 32 for every run — changing it changes results more than the hardware does.

---

## Execution Order

**Build (Phase 1):** `common.py` → `01`, `02`, `04` in parallel → `03`

**Train (Phase 3, trainer only, from `master` with `FULL_RUN = True`):**

```
1. 02_resnet_B_Bp_C.ipynb   → LR sweep (minutes) → B, B′ (fast) → C-noaug → C + submission.csv
2. 04_cifar10.ipynb         → longest runs; start early
3. 01_configA_cnn.ipynb     → 15-epoch baseline
4. Commit outputs/, upload checkpoints → everyone runs 03_comparison.ipynb
```

Time the first full epoch of each notebook and extrapolate before committing to the full schedule. Backup if the trainer's machine is unavailable: Kaggle (change only `DATA_DIR`).

---

## Report Mapping

| Part | Marks | Notebooks | Lecture Weeks |
|------|-------|-----------|--------------|
| a | 10% | `common.py`, `02` | 9 |
| b | 20% | `common.py`, `01`, `02` | 7, 8, 9, 11 |
| c | 10% | `01`, `02` (LR sweep), `03` | 7, 8, 11 |
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
- `NUM_WORKERS = 4` in `common.py` (used for the full run, faster epochs); set it to 0 locally if a Windows notebook hangs. Workers are seeded, so results stay repeatable
- Windows **Smart App Control** can block unsigned package DLLs (`An Application Control policy has blocked this file`). Check `Microsoft-Windows-CodeIntegrity/Operational` in Event Viewer for the file name; a restart fixed one torch case. Do not disable Smart App Control (cannot easily be turned back on).
- Reference: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*
- PDF references: VGG [Simonyan & Zisserman, 2014] and ResNet [He et al., 2016] — cite both even if only ResNet is used

---

## Changelog

- **2026-10-10** — Switched from a single notebook to `common.py` + 4 notebooks; one trainer runs all full training; added handoff contract, output conventions and C-noaug config; environment updated (Python 3.12, torch 2.14.1 + cu126, no torchaudio, pandas 2.3.3, ipykernel); fixed `numpy.seed` → `np.random.seed`; Part h reports per-class recall and macro-F1.
- **2026-10-11** — Training setup revised for fair comparison: StepLR ×0.5/5ep never triggered in 5-epoch runs (A, B′), so all Dogs vs. Cats configs now share **max 15 epochs + StepLR ×0.5/3ep**, keeping the best-val checkpoint; starting LR still differs by model type (justified). Comparison chain **B → B′ → C-noaug → C** (one change per step); B and B′ share fixed features extracted once with `augment=False` (B′ now runs in seconds); A vs C limitation stated (architecture + pretraining). Added LR sweep for C on the subset (Step 4 / Part c). Expected accuracies revised (estimates).
