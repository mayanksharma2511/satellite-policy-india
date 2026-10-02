"""Train a CNN to estimate village consumption from Landsat tiles.

Meant to run on a free Kaggle GPU (also runs on CPU, slowly). Two experiments:

  crossfit  Every village gets a prediction from a model that never saw it.
            Villages are split into 5 folds by district (neighbouring villages look
            alike, so splitting by village would leak), and each fold is predicted by
            a model trained on the other four. These predictions are what the roads
            study uses later.
  loso      Leave one state out: train on five states, predict the sixth. This
            measures how much accuracy drops in a state the model has never seen.

Each experiment also fits a simple baseline (ridge regression on per-band summary
statistics), so we can see whether the CNN is worth it.

Label: log of SECC 2012 consumption per person (secc_cons_pc_rural).

Outputs (in --out): predictions_crossfit.csv, predictions_loso.csv, metrics.json

Kaggle:  python train_cnn.py --mode all
Quick test on fake data: see tests at the bottom of the README.
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

SEED = 2027
FOLDS = 5


# ---------------------------------------------------------------- data

def find_file(name, roots):
    for root in roots:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits:
            return sorted(hits)
    raise FileNotFoundError(f"{name} not found under {roots}")


def load(roots):
    shards = find_file("shard_*.npz", roots)
    images, ids = [], []
    for path in shards:
        data = np.load(path, allow_pickle=True)
        images.append(data["images"])
        ids.extend(data["shrid2"].tolist())
    images = np.concatenate(images)

    villages = pd.read_csv(find_file("villages.csv.gz", roots)[0], dtype={"shrid2": str})
    table = pd.DataFrame({"shrid2": ids}).merge(villages, on="shrid2", how="left")
    table["y"] = np.log(table["secc_cons_pc_rural"])
    table["district"] = table["state_name"] + "/" + table["district_name"]
    print(f"Loaded {len(table):,} tiles from {len(shards)} shards; "
          f"{table['y'].notna().sum():,} have a consumption label")
    return images, table


def district_folds(table):
    """Assign whole districts to folds, so neighbouring villages stay together."""
    rng = np.random.default_rng(SEED)
    districts = table["district"].unique()
    fold_of = dict(zip(districts, rng.permutation(len(districts)) % FOLDS))
    return table["district"].map(fold_of).to_numpy()


class Tiles(torch.utils.data.Dataset):
    def __init__(self, images, idx, y, mean, std, augment):
        self.images, self.idx, self.y = images, idx, y
        self.mean = torch.tensor(mean).view(6, 1, 1)
        self.std = torch.tensor(std).view(6, 1, 1)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = (torch.from_numpy(self.images[j].astype(np.float32)) - self.mean) / self.std
        if self.augment:  # satellite images have no "up": flips and rotations are free data
            if torch.rand(1) < 0.5:
                x = x.flip(2)
            x = torch.rot90(x, int(torch.randint(0, 4, (1,))), dims=(1, 2))
        target = self.y[j] if self.y is not None else 0.0
        return x, torch.tensor(target, dtype=torch.float32)


# ---------------------------------------------------------------- model

def make_model():
    """ResNet-18 pretrained on ImageNet, adapted from 3 colour channels to 6 bands."""
    from torchvision.models import resnet18, ResNet18_Weights
    try:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pretrained = True
    except Exception as err:  # e.g. no internet on the Kaggle notebook
        print(f"Could not load pretrained weights ({err}); training from scratch")
        model = resnet18(weights=None)
        pretrained = False

    old = model.conv1.weight.data  # (64, 3, 7, 7), channels R, G, B
    conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        if pretrained:
            # our bands: 0 blue, 1 green, 2 red, 3 NIR, 4 SWIR1, 5 SWIR2
            conv.weight[:, 2] = old[:, 0]
            conv.weight[:, 1] = old[:, 1]
            conv.weight[:, 0] = old[:, 2]
            conv.weight[:, 3:] = old.mean(dim=1, keepdim=True)  # new bands start as "average colour"
            conv.weight *= 3 / 6  # keep the overall activation scale similar
    model.conv1 = conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def band_stats(images, idx):
    sample = images[idx[:5000]].astype(np.float32)
    return sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3)) + 1e-6


def train_and_predict(images, y, train_idx, predict_idx, args, device):
    """Train on train_idx (holding out 10% of districts for choosing the epoch), predict predict_idx."""
    rng = np.random.default_rng(SEED)
    groups = args.table["district"].to_numpy()[train_idx]
    holdout_groups = rng.choice(np.unique(groups), size=max(1, len(np.unique(groups)) // 10),
                                replace=False)
    is_val = np.isin(groups, holdout_groups)
    fit_idx, val_idx = train_idx[~is_val], train_idx[is_val]

    mean, std = band_stats(images, fit_idx)
    y_mean, y_std = y[fit_idx].mean(), y[fit_idx].std()
    y_scaled = (y - y_mean) / y_std  # NaN for unlabelled villages; never used in training

    def loader(idx, augment, shuffle, target=True):
        ds = Tiles(images, idx, y_scaled if target else None, mean, std, augment)
        return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=shuffle,
                                           num_workers=args.workers, pin_memory=device == "cuda")

    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(loader(fit_idx, True, True))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.MSELoss()

    def predict(idx):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(device_type=device, enabled=device == "cuda"):
            for x, _ in loader(idx, False, False, target=False):
                out.append(model(x.to(device)).float().squeeze(1).cpu())
        return torch.cat(out).numpy() * y_std + y_mean

    best, best_state = -np.inf, None
    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        for x, t in loader(fit_idx, True, True):
            x, t = x.to(device), t.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), t)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        val_r2 = r2_score(y[val_idx], predict(val_idx))
        print(f"    epoch {epoch + 1}/{args.epochs}: validation R2 {val_r2:.3f} "
              f"({time.time() - start:.0f}s)")
        if val_r2 > best:
            best = val_r2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(predict_idx)


# ---------------------------------------------------------------- baseline

def summary_features(images):
    """Per-band mean, std and 10th/50th/90th percentiles: a 'no deep learning' baseline."""
    x = images.reshape(len(images), 6, -1).astype(np.float32)
    feats = [x.mean(2), x.std(2), *np.percentile(x, [10, 50, 90], axis=2)]
    return np.concatenate(feats, axis=1)


def ridge_predict(features, y, train_idx, predict_idx):
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(features[train_idx], y[train_idx])
    return model.predict(features[predict_idx])


# ---------------------------------------------------------------- experiments

def report(name, table, pred, y, groups):
    ok = ~np.isnan(y) & ~np.isnan(pred)
    rows = {"all": r2_score(y[ok], pred[ok])}
    for g in sorted(set(groups[ok])):
        m = ok & (groups == g)
        rows[g] = r2_score(y[m], pred[m])
    print(f"  {name}: " + ", ".join(f"{k} {v:.3f}" for k, v in rows.items()))
    return rows


def run_crossfit(images, table, feats, args, device):
    y = table["y"].to_numpy()
    fold = district_folds(table)
    labelled = ~np.isnan(y)
    cnn, ridge = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k in range(FOLDS):
        train_idx = np.where(labelled & (fold != k))[0]
        predict_idx = np.where(fold == k)[0]
        print(f"Cross-fit fold {k + 1}/{FOLDS}: train {len(train_idx):,}, predict {len(predict_idx):,}")
        cnn[predict_idx] = train_and_predict(images, y, train_idx, predict_idx, args, device)
        ridge[predict_idx] = ridge_predict(feats, y, train_idx, predict_idx)
    states = table["state_name"].to_numpy()
    metrics = {"cnn": report("CNN R2", table, cnn, y, states),
               "ridge_baseline": report("Ridge R2", table, ridge, y, states)}
    pd.DataFrame({"shrid2": table["shrid2"], "fold": fold, "pred_cnn": cnn,
                  "pred_ridge": ridge}).to_csv(args.out / "predictions_crossfit.csv", index=False)
    return metrics


def run_loso(images, table, feats, args, device):
    y = table["y"].to_numpy()
    states = table["state_name"].to_numpy()
    labelled = ~np.isnan(y)
    cnn, ridge = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for state in sorted(set(states)):
        train_idx = np.where(labelled & (states != state))[0]
        predict_idx = np.where(states == state)[0]
        print(f"Leave out {state}: train {len(train_idx):,}, predict {len(predict_idx):,}")
        cnn[predict_idx] = train_and_predict(images, y, train_idx, predict_idx, args, device)
        ridge[predict_idx] = ridge_predict(feats, y, train_idx, predict_idx)
    metrics = {"cnn": report("CNN R2 (unseen state)", table, cnn, y, states),
               "ridge_baseline": report("Ridge R2 (unseen state)", table, ridge, y, states)}
    pd.DataFrame({"shrid2": table["shrid2"], "pred_cnn": cnn,
                  "pred_ridge": ridge}).to_csv(args.out / "predictions_loso.csv", index=False)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["crossfit", "loso", "all"], default="all")
    parser.add_argument("--data", nargs="*", default=["/kaggle/input", "data"])
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("OUT", "outputs")))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    args.out.mkdir(parents=True, exist_ok=True)

    images, table = load(args.data)
    args.table = table
    feats = summary_features(images)
    metrics = {"n_tiles": len(table), "n_labelled": int(table["y"].notna().sum()),
               "epochs": args.epochs, "device": device}
    if args.mode in ("crossfit", "all"):
        metrics["crossfit"] = run_crossfit(images, table, feats, args, device)
    if args.mode in ("loso", "all"):
        metrics["loso"] = run_loso(images, table, feats, args, device)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
