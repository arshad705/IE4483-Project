# IE4483 Mini Project 2 — Dogs vs. Cats (+ CIFAR-10) Plan

## Overview

Binary image classifier (dog vs. cat) built with PyTorch. Three model configurations are trained and compared, with **ResNet-18 + Linear FC (fine-tuned)** selected as the final model for submission. The same model is then adapted for CIFAR-10 multi-class classification.

**Lecture briefing:** Try different models, select only 1, give reasons. Use lecture content (weeks 5–11) to analyse and evaluate.

**Dataset location:** `datasets/train/`, `datasets/val/`, `datasets/test/`
**Output:** `submission.csv` with columns `id` (1–500) and `label` (1=dog, 0=cat)

---

## Project Requirements Compliance

| PDF Requirement | How Fulfilled |
|----------------|--------------|
| **Step 1** — Load train/val/test from provided dataset | `utils/dataset.py` with `ImageFolder` + custom `TestDataset` |
| **Step 2** — Preprocess + augment (scaling, rotation, flipping) | `RandomResizedCrop` + `RandomRotation` + `RandomHorizontalFlip` + `ColorJitter` |
| **Step 3** — Design classification model (CNN or pretrained backbone) | 3 configs tried; ResNet-18 + FC selected as final |
| **Step 4** — Try different parameters; train on train set, validate on val set | Hyperparameter experiments; val set never used for weight updates |
| **Step 5** — Generate submission.csv (id + label, 1=dog, 0=cat) | `predict.py` outputs 500-row CSV |
| **Part a** — State image counts + describe preprocessing | Subset size documented; full augmentation pipeline described |
| **Part b** — At least 1 model, architecture, dims, loss, training, code, reproducibility | All 3 configs documented; ResNet-18 selected; seed=42 fixed |
| **Part c** — Discuss parameter choices + reasons | Hyperparameter table with lecture-grounded justification |
| **Part d** — Val accuracy + submission.csv | Val accuracy logged per epoch; submission.csv generated |
| **Part e** — Analyse correct/incorrect samples | Val set used (test set unlabelled); 1–2 cases discussed |
| **Part f** — Compare different models and data processing | 3-way comparison: Custom CNN vs. ResNet+SVM vs. ResNet+FC |
| **Part g** — CIFAR-10 adaptation, describe changes, report test results | ResNet-18 adapted to 10-class; changes documented |
| **Part h** — Handle class imbalance with ≥2 approaches | Weighted loss + WeightedRandomSampler |

---

## Environment Setup

### Requirements

- Python 3.12 (recommended — stable, fully supported by PyTorch 2.x and all dependencies)
- NVIDIA GPU with CUDA (local machine)

### Virtual Environment Setup (Required)

All development must be done inside a virtual environment to keep dependencies isolated.

```bash
# 1. Create the virtual environment inside the project folder
python -m venv venv

# 2. Activate it (Windows)
venv\Scripts\activate

# 3. Install CUDA-enabled PyTorch (match your NVIDIA driver)
#    Check your CUDA version with: nvidia-smi
#    Then pick the right wheel at: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install remaining dependencies
pip install numpy pandas matplotlib scikit-learn tqdm Pillow

# 5. Freeze versions for reproducibility
pip freeze > requirements.txt
```

> To reactivate the environment in a new terminal: `venv\Scripts\activate`
> To deactivate: `deactivate`

### Python Packages

```
torch
torchvision
torchaudio
numpy
pandas
matplotlib
scikit-learn
tqdm
Pillow
```

### Project File Structure (after setup)

```
ie4483 prj/
├── venv/                    ← virtual environment (do not commit)
├── datasets/
│   ├── train/cat/   (10,000 images)
│   ├── train/dog/   (10,000 images)
│   ├── val/cat/     (2,500 images)
│   ├── val/dog/     (2,500 images)
│   └── test/        (500 images: 1.jpg – 500.jpg)
├── data/                    ← CIFAR-10 auto-downloaded here by torchvision
├── sampleSubmission.csv
├── IE4483-Project2.pdf
├── project-plan.md          ← this file
├── requirements.txt         ← frozen package versions
├── train.py                 ← main training script (Dogs vs. Cats)
├── predict.py               ← generates submission.csv
├── models/
│   ├── custom_cnn.py        ← Config A: Custom CNN definition
│   └── resnet_finetune.py   ← Config B & C: ResNet-18 wrapper
├── utils/
│   ├── dataset.py           ← DataLoader setup + augmentation
│   └── evaluate.py          ← accuracy, confusion matrix, sample plots
├── cifar10/
│   ├── train_cifar10.py     ← CIFAR-10 training (parts g & h)
│   └── imbalance.py         ← class imbalance utilities
├── checkpoints/             ← saved model weights (.pth files)
├── outputs/                 ← saved plots and figures for report
└── submission.csv           ← final output
```

---

## Model Configurations (for Part f Comparison)

Three configurations are trained and compared. Only **Config C** is used for final submission.

| Config | Model | Classifier | Training | Purpose |
|--------|-------|-----------|---------|---------|
| **A** | Custom CNN (4 conv blocks, scratch) | Linear FC | Train from scratch | Baseline — Week 9 CNN |
| **B** | ResNet-18 (frozen backbone) | SVM (sklearn) | Extract features → fit SVM | Week 6 SVM + Week 8 features |
| **C** | ResNet-18 (unfreeze layer4 + fc) | Linear FC (fine-tuned) | Fine-tune end-to-end | **Final model** — Week 8 transfer learning |

**Expected val accuracy progression:**
```
Config A (Custom CNN)    → ~75–82%   ← no pretrained weights, limited data
Config B (ResNet + SVM)  → ~88–91%   ← pretrained features, but fixed classifier
Config C (ResNet + FC)   → ~91–94%   ← pretrained features + end-to-end fine-tuning  ← SELECTED
```

**Part (f) conclusion:** Transfer learning beats training from scratch. End-to-end fine-tuning beats fixed feature extraction with SVM. Augmentation further improves Config C accuracy.

---

## Sub-Tasks

---

### Sub-Task 1 — Data Loading & Augmentation

**Status:** [ ] pending

**Intent:**
Set up the PyTorch `Dataset` and `DataLoader` for train, val, and test splits with appropriate image transforms. All three model configs consume this same pipeline.

**Expected Outcomes:**
- `utils/dataset.py` provides `get_dataloaders(data_dir, batch_size, img_size, subset_size)` returning train/val/test loaders
- Images resized to 224×224 (standard for ResNet-18; also works for Custom CNN)
- Train split applies augmentation; val/test apply only resize + normalize
- Supports subset of images (start: 2,000 train + 500 val) via `subset_size` parameter

**Augmentation Pipeline (train only) — covers PDF Step 2:**
- `RandomRotation(degrees=15)` ← rotation (explicitly named in PDF)
- `RandomHorizontalFlip(p=0.5)` ← flipping (explicitly named in PDF)
- `RandomResizedCrop(224, scale=(0.8, 1.0))` ← scaling (explicitly named in PDF)
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`
- `ToTensor()`
- `Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])` ← ImageNet stats for ResNet compatibility

**Val/Test Pipeline:**
- `Resize(256)` → `CenterCrop(224)`
- `ToTensor()`
- `Normalize(same as above)`

**Todo List:**
- [ ] Create `utils/dataset.py` with `ImageFolder` for train/val
- [ ] Create custom `TestDataset` for unlabelled test folder (reads filenames to preserve IDs)
- [ ] Implement `get_dataloaders()` with configurable `subset_size`
- [ ] Verify label mapping: `ImageFolder` assigns alphabetically — confirm `cat=0`, `dog=1`
- [ ] Sanity check: print class-to-index mapping and one sample batch shape

**Relevant Context:**
- Train images: `datasets/train/cat/`, `datasets/train/dog/`
- Test images: `datasets/test/1.jpg` → `500.jpg` (IDs must be preserved for submission)
- Label convention: `1=dog`, `0=cat` (matches `sampleSubmission.csv`)

---

### Sub-Task 2 — Config A: Custom CNN Model

**Status:** [ ] pending

**Intent:**
Build a CNN from scratch as the baseline for comparison in part (f). Trained briefly (~5 epochs) to get a rough val accuracy — not used for final submission. Demonstrates understanding of CNN design using Week 9 lecture content.

**Architecture:**
```
Input: (B, 3, 224, 224)
→ Conv2d(3, 32, 3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)     → (B, 32, 112, 112)
→ Conv2d(32, 64, 3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)    → (B, 64, 56, 56)
→ Conv2d(64, 128, 3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)   → (B, 128, 28, 28)
→ Conv2d(128, 256, 3, padding=1) + BatchNorm + ReLU + MaxPool2d(2)  → (B, 256, 14, 14)
→ AdaptiveAvgPool2d(4, 4)                                           → (B, 256, 4, 4)
→ Flatten                                                           → (B, 4096)
→ Linear(4096, 512) + ReLU + Dropout(0.5)
→ Linear(512, 2)                                                    → logits (B, 2)
```

**Lecture grounding:**
- Conv + pooling layers → Week 9 (CNN architecture)
- BatchNorm + Dropout → Week 11 (regularisation)
- Backpropagation trains all weights from random init → Week 7

**Todo List:**
- [ ] Create `models/custom_cnn.py` with `CustomCNN(nn.Module)`
- [ ] Use `BatchNorm2d` after each conv for training stability (Week 11)
- [ ] Use `Dropout(0.5)` before final linear layer to reduce overfitting (Week 11)
- [ ] Verify output shape with a dummy tensor
- [ ] Train for ~5 epochs via `train.py --model cnn --epochs 5` and record val accuracy

---

### Sub-Task 3 — Config B & C: ResNet-18 Model

**Status:** [ ] pending

**Intent:**
Load pretrained ResNet-18, configure it for two uses:
- **Config B** — frozen backbone, features fed to SVM (sklearn)
- **Config C** — unfreeze `layer4` + `fc`, fine-tune end-to-end ← final model

**Strategy:**
- Load `resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)` — pretrained on ImageNet (Week 8: transfer learning)
- For Config B: freeze all layers, remove `fc`, extract 512-dim feature vectors → fit `sklearn.svm.SVC`
- For Config C: freeze early layers, unfreeze `layer4` + `fc`, replace `fc` with `Linear(512, 2)`

**Lecture grounding:**
- Pretrained backbone → Week 8 (deep learning, feature hierarchies, transfer learning)
- SVM classifier → Week 6 (SVM)
- Fine-tuning with lower LR → Week 11 (optimisation for deep models)

**Todo List:**
- [ ] Create `models/resnet_finetune.py` with `get_resnet18(num_classes, freeze_backbone)` function
- [ ] Load pretrained weights using `ResNet18_Weights.IMAGENET1K_V1`
- [ ] For Config B: add `get_resnet18_features()` function that returns 512-dim vectors (removes FC layer)
- [ ] For Config C: unfreeze `layer4` + `fc`; replace `fc` with `nn.Linear(512, num_classes)`
- [ ] Verify frozen/unfrozen layers by printing `requires_grad` status

---

### Sub-Task 4 — Training Loop

**Status:** [ ] pending

**Intent:**
Implement the main training script for Configs A and C. Config B (SVM) is fitted separately using sklearn after feature extraction.

**Training Configuration:**

| Hyperparameter | Config A (Custom CNN) | Config C (ResNet-18 FC) | Reason |
|----------------|----------------------|------------------------|--------|
| Optimizer | Adam | Adam | Adaptive LR, better convergence (Week 11) |
| Learning Rate | 1e-3 | 1e-4 | Lower for pretrained — avoid destroying learned weights (Week 8) |
| Scheduler | StepLR (×0.5 every 5 ep) | StepLR (×0.5 every 3 ep) | LR decay as model converges (Week 11) |
| Loss Function | CrossEntropyLoss | CrossEntropyLoss | Standard for multi-class classification |
| Batch Size | 32 | 32 | Balance between GPU memory and gradient stability |
| Epochs | 5 (comparison only) | 15 (full training) | Config A only needs rough accuracy for part (f) |
| Random Seed | 42 | 42 | Reproducibility (Part b requirement) |

**Config B (SVM) Training:**
- Extract 512-dim features from frozen ResNet-18 for all train images
- Fit `sklearn.svm.SVC(kernel='rbf', C=1.0)` on extracted features
- Evaluate on val set features → record accuracy

**Expected Outcomes:**
- `train.py` accepts `--model [cnn|resnet]`, `--epochs`, `--lr`, `--batch_size`, `--subset`
- Trains on GPU if available, falls back to CPU
- Prints per-epoch train loss, train accuracy, val accuracy
- Saves best model weights to `checkpoints/<model_name>_best.pth`

**Todo List:**
- [ ] Create `train.py` with argument parsing
- [ ] Implement `train_one_epoch()` and `evaluate()` functions
- [ ] Add `torch.manual_seed(42)`, `random.seed(42)`, `numpy.seed(42)` for reproducibility
- [ ] Add `torch.backends.cudnn.deterministic = True`, `benchmark = False`
- [ ] Add SVM feature extraction + fitting in a `train_svm()` function
- [ ] Use `tqdm` for progress bars
- [ ] Save best checkpoint when val accuracy improves
- [ ] Log train/val loss and accuracy per epoch to console and CSV

---

### Sub-Task 5 — Evaluation & Analysis

**Status:** [ ] pending

**Intent:**
Compute final validation accuracy for all 3 configs, plot training curves, and produce correct/incorrect prediction visualisations for the report (parts d, e, f).

**Expected Outcomes:**
- Comparison table of all 3 configs for part (f)
- Training curves (loss + accuracy vs. epoch) for Config A and C saved as PNG
- 3–5 sample **val** images with predicted vs. actual label (correct and incorrect cases) for part (e)
- Confusion matrix for Config C on val set

**Todo List:**
- [ ] Implement `plot_training_curves(train_losses, val_accs)` in `utils/evaluate.py`
- [ ] Implement `show_sample_predictions(model, dataloader, n=8)` — correct and incorrect cases from val set
- [ ] Compute confusion matrix using `sklearn.metrics.confusion_matrix`
- [ ] Build 3-config comparison table (Config A / B / C val accuracy)
- [ ] Save all figures to `outputs/` for report inclusion
- [ ] For part (e): select 1–2 misclassified val images and discuss using Week 9 terminology (texture, spatial features, pose, background clutter)

**Note — Part (e):** Test set is unlabelled so ground truth is unknown. Val set is used for correct/incorrect analysis. State this explicitly in the report.

---

### Sub-Task 6 — Test Prediction & submission.csv

**Status:** [ ] pending

**Intent:**
Run the best Config C checkpoint on the 500 test images and generate `submission.csv`.

**Expected Outcomes:**
- `predict.py` loads Config C checkpoint and runs inference on `datasets/test/`
- Output `submission.csv` has exactly 500 rows, columns `id` and `label`
- `id` matches image filename integer (`5.jpg` → `id=5`)
- `label` is `1` for dog and `0` for cat

**Todo List:**
- [ ] Create `predict.py` that loads model from `--checkpoint` argument
- [ ] Sort test images by numeric ID before inference
- [ ] Apply val/test transform pipeline (no augmentation)
- [ ] Write results to `submission.csv` using pandas
- [ ] Verify: row count = 500, label values are only 0 or 1

---

### Sub-Task 7 — CIFAR-10 Extension (Parts g & h)

**Status:** [ ] pending

**Intent:**
Adapt Config C (ResNet-18 + FC) for CIFAR-10 10-class classification. Then simulate class imbalance and apply two fixes.

**CIFAR-10 Dataset Facts (cite in report):**
- Source: Krizhevsky, Nair, and Hinton — https://www.cs.toronto.edu/~kriz/cifar.html
- 60,000 colour images at 32×32 pixels across 10 mutually exclusive classes
- Training: 50,000 images (5,000 per class — perfectly balanced)
- Test: 10,000 images (1,000 per class)
- Loaded automatically via `torchvision.datasets.CIFAR10(root='./data', download=True)` — no manual download needed
- Downloaded to `data/` folder (~163 MB, first run only)

**CIFAR-10 Class Labels:**

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

**Changes vs. Dogs vs. Cats (Part g):**
- Output layer: `Linear(512, 10)` instead of 2 — via `get_resnet18(num_classes=10)`
- Input: CIFAR-10 images are 32×32 — resize to 224×224 for ResNet compatibility
- Loss: `CrossEntropyLoss` unchanged — now 10-class
- Dataset loaded via `torchvision.datasets.CIFAR10` instead of `ImageFolder`
- No custom test loader needed — CIFAR-10 test set has ground truth labels

**Class Imbalance Simulation (Part h):**
- CIFAR-10 is perfectly balanced by default — imbalance must be artificially created
- Reduce classes 0, 1, 2 (airplane, automobile, bird) to 20% → 1,000 samples each
- Classes 3–9 remain at full 5,000 samples each

**Two Imbalance Fixes:**
1. **Weighted CrossEntropyLoss** — compute inverse-frequency class weights, pass to `nn.CrossEntropyLoss(weight=...)` — cost-sensitive learning (Week 11)
2. **WeightedRandomSampler** — assign per-sample weights inversely proportional to class frequency, oversample minority classes during training

**Results Table (4 configs to train and report):**

| Config | Setup | Expected outcome |
|--------|-------|-----------------|
| 1 | Balanced CIFAR-10 | Best accuracy baseline |
| 2 | Imbalanced, no fix | Drops on minority classes |
| 3 | Imbalanced + weighted loss | Improves minority class accuracy |
| 4 | Imbalanced + weighted sampler | Further improves balance |

**Todo List:**
- [ ] Create `cifar10/train_cifar10.py` adapting the Dogs vs. Cats training loop
- [ ] Use `get_resnet18(num_classes=10)` — reuse existing model wrapper
- [ ] Create `cifar10/imbalance.py` with `get_class_weights()` and `get_weighted_sampler()` utilities
- [ ] Simulate imbalance by subsampling classes 0, 1, 2 to 20% in training set
- [ ] Train all 4 configurations and log test accuracy for each
- [ ] Build results table comparing all 4 configs for part (h)

---

## Key Steps Summary (Execution Order)

```
Step 1   Setup venv and install packages
Step 2   Implement data pipeline (Sub-Task 1)
Step 3   Build Custom CNN — Config A (Sub-Task 2)
Step 4   Build ResNet-18 wrapper — Config B & C (Sub-Task 3)
Step 5   Implement training loop (Sub-Task 4)
Step 6   Train Config A for ~5 epochs → record rough val accuracy
Step 7   Extract ResNet features → fit SVM (Config B) → record val accuracy
Step 8   Train Config C fully (15 epochs) → record best val accuracy
Step 9   Select Config C as final model, justify using lecture concepts
Step 10  Evaluation + plots — training curves, confusion matrix, sample predictions (Sub-Task 5)
Step 11  Generate submission.csv using Config C (Sub-Task 6)
Step 12  CIFAR-10 extension — parts g and h (Sub-Task 7)
```

---

## Report Mapping

Each row maps the exact document question to the sub-task that produces the answer.

### Part a — 10%
> *"State the amount of image data that you used to form your training set and testing set. Describe the data pre-processing procedures and image augmentations (if any)."*

| What to write | Where it comes from |
|--------------|-------------------|
| How many train/val images used | Sub-Task 1 — `subset_size` parameter (start: 2,000 train + 500 val) |
| Preprocessing steps | Sub-Task 1 — resize to 224×224, normalize with ImageNet stats |
| Augmentations applied | Sub-Task 1 — `RandomRotation`, `RandomHorizontalFlip`, `RandomResizedCrop`, `ColorJitter` |
| Lecture week to cite | Week 9 (CNN input preprocessing) |

---

### Part b — 20%
> *"Select or build at least one machine learning model to construct your classifier. Clearly describe the model you use, including a figure of model architecture, the input and output dimensions, structure of the model, loss function(s), training strategy, etc. Include your code and instructions on how to run the code. Ensure reproducibility."*

| What to write | Where it comes from |
|--------------|-------------------|
| Model selected | Sub-Task 3 — Config C: ResNet-18 + Linear FC, fine-tuned |
| Why this model was chosen | Sub-Tasks 2, 3 — compared with Config A and B first |
| Architecture figure + dims | Sub-Task 2 (Config A) and Sub-Task 3 (Config C) — layer-by-layer diagrams in plan |
| Loss function | Sub-Task 4 — `CrossEntropyLoss` for both |
| Training strategy | Sub-Task 4 — Adam, StepLR, batch size 32, 15 epochs for Config C |
| Code + run instructions | `train.py --model resnet --epochs 15 --lr 1e-4 --subset 2000` |
| Reproducibility | Sub-Task 4 — `seed=42`, `cudnn.deterministic=True` |
| Lecture weeks to cite | Week 7 (backprop), Week 8 (deep learning, transfer learning), Week 9 (CNN), Week 11 (regularisation) |

---

### Part c — 10%
> *"Discuss how you consider and determine the parameters (e.g., learning rate, etc.) / settings of your model as well as your reasons of doing so."*

| What to write | Where it comes from |
|--------------|-------------------|
| Learning rate choice (1e-4) | Sub-Task 4 — lower LR for pretrained model to avoid overwriting ImageNet weights (Week 8) |
| Optimiser choice (Adam) | Sub-Task 4 — adaptive LR, better convergence than SGD (Week 11) |
| Scheduler (StepLR) | Sub-Task 4 — LR decays as model converges, avoids overshooting (Week 11) |
| Batch size (32) | Sub-Task 4 — balance between GPU memory and stable gradient estimates |
| Epochs (15) | Sub-Task 4 — sufficient for fine-tuning; best checkpoint saved automatically |
| Dropout (0.5) | Sub-Task 2 — reduces overfitting on limited data (Week 11) |
| Lecture weeks to cite | Week 7 (gradient descent), Week 11 (optimisation, regularisation) |

---

### Part d — 20%
> *"Report the classification accuracy on validation set. Apply the classifier(s) built to the test set. Submit the submission.csv with the results you obtained."*

| What to write | Where it comes from |
|--------------|-------------------|
| Val accuracy (Config C) | Sub-Task 4 — best val accuracy logged during training |
| Val accuracy (Config A, B) | Sub-Tasks 2, 4 — recorded for comparison |
| submission.csv | Sub-Task 6 — `predict.py` generates 500-row CSV from Config C checkpoint |
| CSV format | `id` (1–500) + `label` (1=dog, 0=cat) — matches `sampleSubmission.csv` |

---

### Part e — 10%
> *"Analyse some correctly and incorrectly classified (if any) samples in the test set. Select 1–2 cases to discuss the strength and weakness of the model."*

| What to write | Where it comes from |
|--------------|-------------------|
| Sample images shown | Sub-Task 5 — `show_sample_predictions()` on **val set** (test set is unlabelled) |
| Correct cases | Images where model is confident and correct — clear frontal view, uncluttered background |
| Incorrect cases | Images where model fails — occluded animal, unusual pose, background clutter |
| Strength discussion | Sub-Task 5 — high accuracy on standard images; good spatial feature extraction (Week 9) |
| Weakness discussion | Sub-Task 5 — fails when spatial features are ambiguous (Week 9 — receptive field, texture) |
| ⚠️ Note | Test set has no ground truth labels — state in report that val set is used as proxy |

---

### Part f — 10%
> *"Discuss how different choice of models and data processing may affect the project in terms of accuracy on validation set."*

| What to write | Where it comes from |
|--------------|-------------------|
| Model comparison table | Sub-Task 5 — Config A (~75–82%) vs. Config B (~88–91%) vs. Config C (~91–94%) |
| Why Config A is weakest | No pretrained weights, trains from random init — Week 7, 9 |
| Why Config B is middle | Good features (ResNet) but SVM is a fixed classifier, no end-to-end learning — Week 6, 8 |
| Why Config C is best | Pretrained features + fine-tuned end-to-end — Week 8, 11 |
| Data processing effect | Run Config C with/without augmentation → show accuracy difference |
| Conventional vs. deep | Decision Tree / SVM (Weeks 5–6) lose spatial structure; CNNs (Week 9) preserve it |
| Lecture weeks to cite | Week 5 (Decision Tree), Week 6 (SVM), Week 8 (deep learning), Week 9 (CNN), Week 11 (regularisation) |

---

### Part g — 10%
> *"Apply and improve your classification algorithm to a multi-category image classification problem for CIFAR-10. Describe details about the dataset and classification problem. Explain how your algorithm can tackle this problem, and what changes you make comparing to solving Dogs vs. Cats problem. Report your results for the testing set of CIFAR-10."*

| What to write | Where it comes from |
|--------------|-------------------|
| CIFAR-10 description | Sub-Task 7 — 60,000 images, 32×32, 10 classes, 50K train / 10K test, perfectly balanced |
| Changes from Dogs vs. Cats | Sub-Task 7 — output `Linear(512, 10)`, dataset loader change, no custom test set needed |
| How ResNet-18 handles it | Week 8 — same pretrained features, generalise to 10-class output |
| Test accuracy | Sub-Task 7 — Config 1 (balanced) test accuracy reported |
| Lecture weeks to cite | Week 8 (transfer learning), Week 9 (CNN for multi-class) |

---

### Part h — 10%
> *"Train the classifier for (g), while some of the classes in CIFAR-10 training dataset contains much fewer labelled data. How can you improve your algorithm to tackle the data unbalancing issue? Describe and justify at least 2 approaches you use."*

| What to write | Where it comes from |
|--------------|-------------------|
| How imbalance was created | Sub-Task 7 — classes 0, 1, 2 reduced to 1,000 samples each; classes 3–9 stay at 5,000 |
| Effect of imbalance (no fix) | Sub-Task 7 — Config 2 accuracy drops on minority classes |
| Fix 1: Weighted CrossEntropyLoss | Sub-Task 7 — inverse-frequency weights; cost-sensitive learning — Week 11 |
| Fix 2: WeightedRandomSampler | Sub-Task 7 — oversample minority classes during training |
| Justification for both | Each approach attacks imbalance differently: loss weighting vs. sampling strategy |
| Results comparison | Sub-Task 7 — 4-row table: balanced / imbalanced / fix1 / fix2 |
| Lecture weeks to cite | Week 11 (optimisation, cost-sensitive learning) |

---

### Summary Table

| Part | Marks | Sub-Tasks | Lecture Weeks |
|------|-------|-----------|--------------|
| a | 10% | 1 | 9 |
| b | 20% | 2, 3, 4 | 7, 8, 9, 11 |
| c | 10% | 4 | 7, 11 |
| d | 20% | 4, 6 | — |
| e | 10% | 5 | 9 |
| f | 10% | 2, 3, 4, 5 | 5, 6, 8, 9, 11 |
| g | 10% | 7 | 8, 9 |
| h | 10% | 7 | 11 |
| **Total** | **100%** | | |

---

## Lecture Grounding (IE4483 — Weeks 5 to 11)

Use these specific lecture weeks to ground your report analysis and justify design decisions.

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

- Fix `torch.manual_seed(42)`, `random.seed(42)`, `numpy.seed(42)` at the top of all scripts
- Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`
- Record exact package versions via `pip freeze > requirements.txt` after environment setup

---

## Notes

- `venv/` must not be committed — add to `.gitignore`
- CIFAR-10 in `data/` is auto-downloaded on first run — do not manually place files there
- CIFAR-10 is perfectly balanced (5,000/class) — part (h) requires artificial imbalance simulation
- Test set (Dogs vs. Cats) is unlabelled — part (e) uses **val set** where ground truth is known; state this in the report
- Only **Config C** is the final selected model — Configs A and B exist for comparison and part (f) only
- Reference: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*
- PDF references: VGG [Simonyan & Zisserman, 2014] and ResNet [He et al., 2016] — cite both even if only ResNet is used
