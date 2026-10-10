"""Shared code for IE4483 Mini Project 2 (Dogs vs. Cats + CIFAR-10).

Every notebook imports from here, so all configs use the same seed handling,
data pipeline, models, training loop and output formats.

Typical use in a notebook:

    from common import *
    set_seed()
    FULL_RUN = False                      # only the trainer sets True
    train_loader, val_loader = get_dataloaders(FULL_RUN)
    model = get_resnet18(mode="finetune")
    ...
    history = fit(model, "C", train_loader, val_loader, optimizer, criterion,
                  epochs=15, scheduler=scheduler, keep_bn_train=("layer4",))

See project-plan.md (Code Structure, Output conventions) for the contract.

Note: a notebook does 5 things with this file:
    1. set_seed()             -> fix randomness
    2. get_dataloaders()      -> load the pictures
    3. get_resnet18()         -> build a model (or define your own, e.g. Config A)
    4. fit()                  -> train it; results CSV + best model saved automatically
    5. predict_val() / write_submission() -> save predictions and submission.csv
Everything else in this file supports those five steps.
"""

from __future__ import annotations

import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

__all__ = [
    "SEED", "DEVICE", "DATA_DIR", "OUTPUT_DIR", "CHECKPOINT_DIR", "SUBSET_SIZE",
    "BATCH_SIZE", "NUM_WORKERS", "CLASS_TO_IDX",
    "set_seed", "get_device", "get_transforms", "maybe_subset", "make_loader",
    "get_dataloaders", "describe_dataset", "TestDataset", "get_test_loader",
    "get_resnet18", "describe_trainable", "set_frozen_bn_eval",
    "train_one_epoch", "evaluate", "fit", "save_result_row", "extract_features",
    "predict_val", "write_submission", "load_checkpoint", "checkpoint_path",
    "run_metadata",
]

# ---------------------------------------------------------------------------
# SETTINGS: folders, batch size, image size. Change values here, not in notebooks.
# ---------------------------------------------------------------------------

# Same seed for every run, so results can be repeated.
SEED = 42
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IE4483_DATA_DIR", REPO_ROOT / "datasets"))  # override for Kaggle
OUTPUT_DIR = REPO_ROOT / "outputs"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

SUBSET_SIZE = {"train": 2000, "val": 500}  # used when FULL_RUN is False
BATCH_SIZE = 32       # fixed for every run (see plan: changing it changes results)
NUM_WORKERS = 0       # safe default in notebooks on Windows
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_TO_IDX = {"cat": 0, "dog": 1}

# The columns of every results CSV (one row per epoch). Same for all notebooks.
RESULT_COLUMNS = [
    "config", "seed", "epoch", "train_loss", "train_acc", "eval_loss", "eval_acc",
    "eval_split", "lr", "seconds", "gpu", "git_commit",
]

# ---------------------------------------------------------------------------
# STEP 1: GET READY (fix randomness, pick GPU/CPU, label results)
# ---------------------------------------------------------------------------


# Fixes all randomness (Python, NumPy, PyTorch), so the same run gives the same result.
# Call this first in every notebook.
def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch; make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Picks where to run: NVIDIA GPU if there is one, else Mac GPU, else CPU.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# The chosen device. Every model and batch is moved here.
DEVICE = get_device()


# Internal: keeps randomness fixed if pictures are loaded with several workers.
def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic seed (matters if NUM_WORKERS > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Records WHICH GPU and WHICH version of the code made a result.
# Added to every results row automatically.
def run_metadata() -> dict:
    """GPU name and current git commit; '-dirty' means uncommitted changes to tracked files."""
    if DEVICE.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = DEVICE.type
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if dirty:
            commit += "-dirty"
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"gpu": gpu, "git_commit": commit}


# ---------------------------------------------------------------------------
# STEP 2: GET THE PICTURES (load, resize, augment, batch)
# ---------------------------------------------------------------------------


# How each picture is prepared before the model sees it.
# Train: random rotate/flip/crop/colour changes (augmentation) -> more variety.
# Val/test: just resize + centre crop, no randomness (fair, repeatable scoring).
def get_transforms(augment: bool = True):
    """Return (train_tf, eval_tf). augment=False gives the C-noaug train transform."""
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if not augment:
        return eval_tf, eval_tf
    train_tf = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


# Takes a small random sample of pictures (same sample every time) for quick tests.
def maybe_subset(ds, n: int | None, seed: int = SEED):
    """Random but reproducible subset of size n (None or n >= len -> full dataset)."""
    if n is None or n >= len(ds):
        return ds
    g = torch.Generator().manual_seed(seed)
    return Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())


# Wraps a set of pictures so they are served to the model in batches of 32.
def make_loader(ds, shuffle: bool, batch_size: int = BATCH_SIZE, seed: int = SEED,
                sampler=None) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle if sampler is None else False,
        sampler=sampler, num_workers=NUM_WORKERS, generator=g,
        worker_init_fn=_seed_worker, pin_memory=DEVICE.type == "cuda",
    )


# MAIN DATA FUNCTION. Gives the train and val pictures, ready in batches.
# full_run=False -> small sample (testing); True -> all pictures (trainer only).
# augment=False -> no augmentation (for Config C-noaug).
# Also checks that cat = 0 and dog = 1.
def get_dataloaders(full_run: bool, augment: bool = True, batch_size: int = BATCH_SIZE,
                    seed: int = SEED):
    """Train and val loaders for Dogs vs. Cats. Val is never shuffled (predictions line up with files)."""
    train_tf, eval_tf = get_transforms(augment)
    train_full = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_full = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tf)
    if train_full.class_to_idx != CLASS_TO_IDX or val_full.class_to_idx != CLASS_TO_IDX:
        raise ValueError(f"Unexpected label mapping: {train_full.class_to_idx} / {val_full.class_to_idx}")

    if full_run:
        train_ds, val_ds = train_full, val_full
    else:
        train_ds = maybe_subset(train_full, SUBSET_SIZE["train"], seed)
        val_ds = maybe_subset(val_full, SUBSET_SIZE["val"], seed)
    print(f"{'FULL' if full_run else 'SUBSET'} run | train {len(train_ds)} | val {len(val_ds)} "
          f"| augment={augment} | device={DEVICE}")
    return make_loader(train_ds, True, batch_size, seed), make_loader(val_ds, False, batch_size, seed)


# Counts the pictures in each folder (for report Part a).
def describe_dataset() -> pd.DataFrame:
    """Image counts per split and class (report Part a)."""
    rows = []
    for split in ("train", "val"):
        for cls in CLASS_TO_IDX:
            rows.append({"split": split, "class": cls,
                         "images": sum(1 for _ in (DATA_DIR / split / cls).glob("*.jpg"))})
    rows.append({"split": "test", "class": "unlabelled",
                 "images": sum(1 for _ in (DATA_DIR / "test").glob("*.jpg"))})
    return pd.DataFrame(rows)


# Reads the 500 unlabelled test pictures and keeps each picture's id (1.jpg -> 1).
class TestDataset(Dataset):
    """Unlabelled test images named '<id>.jpg'. Returns (image, id), sorted by id."""

    def __init__(self, root, transform):
        self.files = sorted(Path(root).glob("*.jpg"), key=lambda p: int(p.stem))
        if not self.files:
            raise FileNotFoundError(f"No .jpg files in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        return self.transform(Image.open(path).convert("RGB")), int(path.stem)


# The test pictures in batches, in id order. Used by write_submission().
def get_test_loader(batch_size: int = BATCH_SIZE) -> DataLoader:
    _, eval_tf = get_transforms()
    return DataLoader(TestDataset(DATA_DIR / "test", eval_tf), batch_size=batch_size,
                      shuffle=False, num_workers=NUM_WORKERS)


# Internal: the file name of each val picture, so predictions can be matched to files.
def _sample_paths(ds) -> list[str]:
    """File paths of an ImageFolder or a Subset of one, in dataset order."""
    if isinstance(ds, Subset):
        base = _sample_paths(ds.dataset)
        return [base[i] for i in ds.indices]
    return [path for path, _ in ds.samples]


# ---------------------------------------------------------------------------
# STEP 3: GET A MODEL (build ResNet-18, freeze parts, save/load)
# ---------------------------------------------------------------------------


# MAIN MODEL FUNCTION. Builds ResNet-18 pre-trained on ImageNet, in one of 3 ways:
#   'features'  (Config B):  nothing learns; only turns each picture into 512 numbers for the SVM
#   'frozen_fc' (Config B'): only the new last layer learns
#   'finetune'  (Config C):  the last block (layer4) + the new last layer learn
def get_resnet18(num_classes: int = 2, mode: str = "finetune", pretrained: bool = True) -> nn.Module:
    """ResNet-18 for Configs B, B' and C.

    mode='features'  (B):  everything frozen, fc removed -> 512-dim feature vectors for the SVM
    mode='frozen_fc' (B'): everything frozen, new trainable Linear(512, num_classes)
    mode='finetune'  (C):  layer4 + new fc trainable, everything else frozen
    """
    if mode not in {"features", "frozen_fc", "finetune"}:
        raise ValueError(f"unknown mode: {mode!r}")
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    if mode == "features":
        model.fc = nn.Identity()
    else:
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # new layer: trainable
        if mode == "finetune":
            for p in model.layer4.parameters():
                p.requires_grad = True
    model = model.to(DEVICE)
    describe_trainable(model)
    return model


# Prints which parts of the model will learn and which are frozen (a quick check).
def describe_trainable(model: nn.Module) -> None:
    """Print which top-level blocks are trainable (sanity check for freezing)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    blocks = []
    for name, child in model.named_children():
        params = list(child.parameters())
        if params:
            blocks.append(f"{name}:{'train' if any(p.requires_grad for p in params) else 'frozen'}")
    print(f"trainable {trainable:,} / {total:,} params | " + " ".join(blocks))


# Safety fix: stops frozen BatchNorm layers from quietly changing during training.
def set_frozen_bn_eval(model: nn.Module, keep_train=()) -> None:
    """Put BatchNorm layers outside `keep_train` prefixes in eval mode.

    Frozen BN layers would otherwise update their running mean/var in train mode,
    drifting away from the ImageNet statistics. Use keep_train=() for B' and
    keep_train=("layer4",) for C.
    """
    prefixes = tuple(keep_train)
    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not (prefixes and name.startswith(prefixes)):
            m.eval()


# Builds the checkpoint file name, e.g. checkpoints/resnet18_C_seed42.pth
def checkpoint_path(model_name: str, config: str, seed: int = SEED) -> Path:
    return CHECKPOINT_DIR / f"{model_name}_{config}_seed{seed}.pth"


# Loads a saved (trained) model back from its .pth file.
def load_checkpoint(model: nn.Module, path) -> nn.Module:
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# STEP 4-6: TRAIN, MEASURE, SAVE RESULTS, PREDICT
# ---------------------------------------------------------------------------


# One round of learning: show every training picture once and adjust the model.
# Returns the average loss and accuracy for that round.
def train_one_epoch(model, loader, optimizer, criterion, keep_bn_train=None):
    """One pass over loader. keep_bn_train=None trains all BN layers (Config A);
    a tuple freezes BN outside those prefixes (B': (), C: ("layer4",))."""
    model.train()
    if keep_bn_train is not None:
        set_frozen_bn_eval(model, keep_bn_train)
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


# Measures the model on a set of pictures WITHOUT learning. Returns loss and accuracy.
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


# Internal: results file name, e.g. outputs/results_C_seed42.csv
def _results_path(config: str, seed: int) -> Path:
    return OUTPUT_DIR / f"results_{config}_seed{seed}.csv"


# Saves a single result in the standard CSV format.
# Used for Config B (SVM), which has no training rounds.
def save_result_row(config: str, eval_acc: float, seed: int = SEED, eval_split: str = "val",
                    **extra) -> Path:
    """Write a single-row results file, e.g. for Config B (SVM): no epochs, no train loss."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    row = {c: "" for c in RESULT_COLUMNS}
    row.update(config=config, seed=seed, epoch=1, eval_acc=eval_acc, eval_split=eval_split,
               **run_metadata())
    row.update({k: v for k, v in extra.items() if k in RESULT_COLUMNS})
    path = _results_path(config, seed)
    pd.DataFrame([row], columns=RESULT_COLUMNS).to_csv(path, index=False)
    print(f"saved {path.name}: eval_acc={eval_acc:.4f}")
    return path


# MAIN TRAINING FUNCTION. Repeats for each round (epoch):
#   learn (train_one_epoch) -> measure (evaluate) -> write a CSV row -> save the best model.
# Use this for every trained config (A, B', C, C-noaug, CIFAR).
def fit(model, config: str, train_loader, eval_loader, optimizer, criterion, epochs: int,
        scheduler=None, keep_bn_train=None, model_name: str = "resnet18", seed: int = SEED,
        eval_split: str = "val") -> pd.DataFrame:
    """Train for `epochs`, log every epoch to outputs/results_<config>_seed<N>.csv and save a checkpoint.

    eval_split='val'  -> keeps the checkpoint with the best val accuracy.
    eval_split='test' -> keeps the LAST epoch (never select a model on the test set; used for CIFAR-10).
    The CSV is rewritten after every epoch, so a crash keeps the epochs already finished.
    """
    if eval_split not in {"val", "test"}:
        raise ValueError("eval_split must be 'val' or 'test'")
    OUTPUT_DIR.mkdir(exist_ok=True)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    meta = run_metadata()
    if meta["git_commit"].endswith("-dirty"):
        print("WARNING: uncommitted changes — results will be tagged '-dirty'.")
    ckpt = checkpoint_path(model_name, config, seed)
    results = _results_path(config, seed)
    rows, best_acc = [], -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, keep_bn_train)
        eval_loss, eval_acc = evaluate(model, eval_loader, criterion)
        if scheduler is not None:
            scheduler.step()
        seconds = time.time() - t0

        rows.append({"config": config, "seed": seed, "epoch": epoch,
                     "train_loss": round(train_loss, 5), "train_acc": round(train_acc, 5),
                     "eval_loss": round(eval_loss, 5), "eval_acc": round(eval_acc, 5),
                     "eval_split": eval_split, "lr": lr, "seconds": round(seconds, 1), **meta})
        pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(results, index=False)

        improved = eval_split == "val" and eval_acc > best_acc
        if improved:
            best_acc = eval_acc
        if improved or eval_split == "test":
            torch.save(model.state_dict(), ckpt)
        print(f"[{config}] epoch {epoch}/{epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} "
              f"| {eval_split} loss {eval_loss:.4f} acc {eval_acc:.4f} | lr {lr:.1e} | {seconds:.0f}s"
              + (" *" if improved else ""))

    print(f"saved {results.name} and {ckpt.name}")
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


# Config B only: turns every picture into 512 numbers using the frozen ResNet,
# so the SVM can learn from them.
@torch.no_grad()
def extract_features(model, loader):
    """Run a 'features'-mode ResNet over loader -> (X [n, 512], y [n]) numpy arrays for the SVM."""
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(DEVICE)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# Saves the model's guess for EVERY val picture (right or wrong) to a CSV.
# Used later for the confusion matrix and right/wrong examples (Part e).
@torch.no_grad()
def predict_val(model, val_loader, config: str, seed: int = SEED) -> pd.DataFrame:
    """Per-image val predictions -> outputs/preds_<config>_seed<N>.csv (path, label, pred, prob_dog)."""
    model.eval()
    paths = _sample_paths(val_loader.dataset)
    labels, preds, probs = [], [], []
    for x, y in val_loader:
        p = torch.softmax(model(x.to(DEVICE)), dim=1)[:, CLASS_TO_IDX["dog"]].cpu()
        labels += y.tolist()
        probs += p.tolist()
        preds += (p >= 0.5).long().tolist()
    if len(paths) != len(preds):
        raise RuntimeError("val loader order does not match dataset (is it shuffled?)")
    rel = [os.path.relpath(p, DATA_DIR) for p in paths]
    df = pd.DataFrame({"path": rel, "label": labels, "pred": preds,
                       "prob_dog": np.round(probs, 5)})
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"preds_{config}_seed{seed}.csv"
    df.to_csv(out, index=False)
    print(f"saved {out.name}: accuracy {(df.label == df.pred).mean():.4f} on {len(df)} images")
    return df


# Guesses all 500 test pictures, writes submission.csv (id, label: 1 = dog, 0 = cat),
# then checks the format: 500 rows, ids 1..500, labels only 0/1.
@torch.no_grad()
def write_submission(model, path="submission.csv", sample_path=None, expected_n: int = 500) -> pd.DataFrame:
    """Predict the test set and write id,label (1 = dog, 0 = cat), then check the format."""
    model.eval()
    ids, preds = [], []
    for x, batch_ids in get_test_loader():
        preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
        ids += batch_ids.tolist()
    sub = pd.DataFrame({"id": ids, "label": preds}).sort_values("id").reset_index(drop=True)
    path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    sub.to_csv(path, index=False)

    chk = pd.read_csv(path)
    assert list(chk.columns) == ["id", "label"], chk.columns
    assert len(chk) == expected_n, f"{len(chk)} rows, expected {expected_n}"
    assert chk["id"].tolist() == list(range(1, expected_n + 1)), "ids are not 1..N"
    assert set(chk["label"].unique()) <= {0, 1}, "labels must be 0 or 1"
    sample = Path(sample_path) if sample_path else REPO_ROOT / "sampleSubmission.csv"
    if sample.exists():
        ref = pd.read_csv(sample)
        assert list(ref.columns) == list(chk.columns) and ref["id"].tolist() == chk["id"].tolist(), \
            "does not match sampleSubmission.csv format"
    print(f"{path.name} OK | {len(chk)} rows | predicted dogs: {int(chk['label'].sum())}")
    return chk
