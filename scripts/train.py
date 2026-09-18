"""Train ChipStain.

    python scripts/train.py --config configs/ours.yaml --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from chipstain.data import PairDataset, make_splits
from chipstain.losses import LOSSES
from chipstain.metrics import image_metrics
from chipstain.model import ChipStainNet


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    ms = []
    for x, y in loader:
        mu, _ = model(x.to(device))
        for p, g in zip(mu.cpu().numpy()[:, 0], y.numpy()[:, 0]):
            ms.append(image_metrics(p, g))
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/raw/hela_kyoto")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs:
        cfg["epochs"] = a.epochs
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = get_device()
    run_dir = os.path.join(a.out, f"{cfg['name']}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)

    train, val, _ = make_splits(a.data)
    tl = DataLoader(PairDataset(train, crop=cfg["crop"], augment=True), batch_size=cfg["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(PairDataset(val, crop=None, augment=False), batch_size=4, shuffle=False, num_workers=0)

    model = ChipStainNet(cfg["encoder"], cfg["pretrained"], cfg["uncertainty"]).to(device)
    loss_fn = LOSSES[cfg["loss"]]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    steps = cfg["epochs"] * len(tl) * cfg["iters_per_epoch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)

    best, log = -1e9, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        n = 0
        for _ in range(cfg["iters_per_epoch"]):  # several random-crop passes per epoch
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                mu, logvar = model(x)
                loss = loss_fn(mu, logvar, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
                n += 1
        vm = validate(model, vl, device)
        rec = {"epoch": ep + 1, "train_loss": tot / n, **{f"val_{k}": v for k, v in vm.items()}, "time_min": (time.time() - t0) / 60}
        log.append(rec)
        print(json.dumps(rec))
        score = vm["pearson"] + vm["ssim"]
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": ep + 1, "val": vm}, os.path.join(run_dir, "best.pt"))
    json.dump(log, open(os.path.join(run_dir, "log.json"), "w"), indent=1)
    yaml.safe_dump(cfg, open(os.path.join(run_dir, "config.yaml"), "w"))
    print("best val:", best, "->", run_dir)


if __name__ == "__main__":
    main()
