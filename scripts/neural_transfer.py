"""Neural transfer test: human iPSC-derived motor-neuron cultures (Google in-silico-labeling
data, Condition A, Christiansen et al. Cell 2018, CC BY 4.0).

    python scripts/download_isl_neurons.py
    python scripts/neural_transfer.py --stage all --out report/results/neural

Stages
  prep       rescale every well so nuclei match the HeLa training scale (median nucleus diameter,
             measured on DAPI vs H2B) and cache bright-field planes + normalised DAPI as .npy
  zplane     pick the bright-field z-plane on the 2 VALIDATION wells (zero-shot Pearson)
  zeroshot   apply the HeLa-trained checkpoints unchanged to the 3 TEST wells
  finetune   fine-tune with k = 1, 2, 5, 20 training wells, initialised from ChipStain (HeLa)
             or from ImageNet only, then evaluate on the 3 test wells
Reference nuclei for the downstream metric come from the same watershed pipeline run on the
real DAPI image (no manual labels exist for this data).
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
import torch
from skimage import filters, io, measure, morphology, transform
from tqdm import tqdm

from chipstain.data import make_splits, load_pair, normalize_input, pad_to_multiple
from chipstain.losses import gaussian_nll, l1_loss
from chipstain.metrics import calibration_metrics, image_metrics, match_f1, segment_nuclei
from chipstain.model import ChipStainNet, predict_tta
from scripts.evaluate import get_device

VAL_WELLS = ["r05c04", "r05c07"]  # held out from the 22 training wells for z-plane / checkpoint selection


def nuc_diameter(img):
    sm = filters.gaussian(img.astype(np.float32), 1.5)
    m = sm > filters.threshold_otsu(sm)
    try:
        m = morphology.remove_small_objects(m, max_size=20)
    except TypeError:  # scikit-image < 0.26
        m = morphology.remove_small_objects(m, min_size=21)
    a = np.array([p.area for p in measure.regionprops(measure.label(m)) if p.area > 20])
    return float(np.sqrt(4 * np.median(a) / np.pi))


def nucleus_scale(root):
    """Rescale factor that matches the median nucleus diameter (neuron DAPI vs HeLa H2B)."""
    _, _, test = make_splits("data/raw/hela_kyoto")
    d_hela = float(np.median([nuc_diameter(load_pair(s)[1]) for s in test if s.timepoint == 1]))
    d_neu = float(np.median([nuc_diameter(io.imread(f)) for f in sorted(glob.glob(os.path.join(root, "*", "*_dapi.png")))]))
    return d_hela / d_neu, d_hela, d_neu


def dapi_range(root):
    train_dapi = [io.imread(f).astype(np.float32) for f in sorted(glob.glob(os.path.join(root, "train", "*_dapi.png")))]
    return np.percentile(np.concatenate([t[::4, ::4].ravel() for t in train_dapi]), [1, 99.8])


def select_scale_z(root, ckpt, device, scales, out):
    """Choose (rescale factor, z-plane) by zero-shot Pearson on the VALIDATION wells only."""
    model, _ = load_model(ckpt, device)
    lo, hi = dapi_range(root)
    res = []
    for w in VAL_WELLS:
        dapi0 = io.imread(os.path.join(root, "train", f"{w}_dapi.png")).astype(np.float32)
        for sc in scales:
            g = np.clip((transform.rescale(dapi0, sc, anti_aliasing=True, preserve_range=True) - lo) / (hi - lo), 0, 1).astype(np.float32)
            for bf in sorted(glob.glob(os.path.join(root, "train", f"{w}_bf_z*.png"))):
                z = int(bf.split("_z")[-1][:2])
                x = normalize_input(transform.rescale(io.imread(bf).astype(np.float32), sc, anti_aliasing=True, preserve_range=True))
                p, _ = infer(model, x, device, tta=False)
                res.append({"well": w, "scale": sc, "z": z, "pearson": image_metrics(p, g)["pearson"]})
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(out, "selection_val.csv"), index=False)
    best = df.groupby(["scale", "z"])["pearson"].mean().idxmax()
    return float(best[0]), int(best[1]), df


def prep(root, cache, scale):
    os.makedirs(cache, exist_ok=True)
    _, d_hela, d_neu = nucleus_scale(root)
    dapi_files = sorted(glob.glob(os.path.join(root, "*", "*_dapi.png")))
    lo, hi = dapi_range(root)
    meta = {"scale": scale, "d_hela_px": d_hela, "d_neuron_px": d_neu, "dapi_lo": float(lo), "dapi_hi": float(hi), "wells": {}}
    for f in tqdm(dapi_files, desc="prep"):
        split, well = f.split("/")[-2], os.path.basename(f).split("_")[0]
        dapi = transform.rescale(io.imread(f).astype(np.float32), scale, anti_aliasing=True, preserve_range=True)
        np.save(os.path.join(cache, f"{well}_dapi.npy"), np.clip((dapi - lo) / (hi - lo), 0, 1).astype(np.float32))
        for bf in sorted(glob.glob(os.path.join(root, split, f"{well}_bf_z*.png"))):
            z = int(bf.split("_z")[-1][:2])
            x = transform.rescale(io.imread(bf).astype(np.float32), scale, anti_aliasing=True, preserve_range=True)
            np.save(os.path.join(cache, f"{well}_bf_z{z:02d}.npy"), normalize_input(x))
        meta["wells"][well] = split
    json.dump(meta, open(os.path.join(cache, "meta.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "wells"}))
    return meta


def load_model(ckpt, device, uncertainty=None):
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ck["config"]
    m = ChipStainNet(cfg["encoder"], False, cfg["uncertainty"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), cfg


@torch.no_grad()
def infer(model, x, device, tta=True):
    h, w = x.shape
    xp, _ = pad_to_multiple(x)
    xt = torch.from_numpy(xp)[None, None].to(device)
    if tta:
        mu, ale, epi = predict_tta(model, xt)
        var = (ale if ale is not None else 0) + epi
    else:
        mu, lv = model(xt)
        var = lv.exp() if lv is not None else torch.zeros_like(mu)
    return mu[0, 0, :h, :w].cpu().numpy(), np.sqrt(var[0, 0, :h, :w].cpu().numpy())


def score(p, g, s):
    m = image_metrics(p, g)
    sp, sg = segment_nuclei(np.clip(p, 0, 1)), segment_nuclei(g)
    f = match_f1(sp, sg)
    m.update({"seg_f1_vs_real": f["f1"], "n_pred": f["n_pred"], "n_real": f["n_gt"], "mean_sigma": float(s.mean())})
    if s.max() > 0:
        m.update(calibration_metrics(p, g, s))
    return m


def wells_of(meta, split):
    return sorted(w for w, s in meta["wells"].items() if s == split and (split != "train" or w not in VAL_WELLS))


def evaluate_wells(model, cache, wells, z, device, tta=True, save_example=None):
    rows = []
    for j, w in enumerate(wells):
        x = np.load(os.path.join(cache, f"{w}_bf_z{z:02d}.npy"))
        g = np.load(os.path.join(cache, f"{w}_dapi.npy"))
        p, s = infer(model, x, device, tta)
        rows.append({"well": w, **score(p, g, s)})
        if save_example and j == 0:  # a 512 x 512 crop of the first test well for the figure
            r0, c0 = (x.shape[0] - 512) // 2, (x.shape[1] - 512) // 2
            crop = lambda a: a[r0:r0 + 512, c0:c0 + 512].astype(np.float16)  # noqa: E731
            np.savez_compressed(save_example, bf=crop(x), dapi=crop(g), pred=crop(p), sigma=crop(s))
    return rows


def finetune(init, cache, train_wells, val_wells, z, device, steps=1200, lr=2e-4, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if init == "imagenet":
        model = ChipStainNet("resnet34", True, True).to(device)
    else:
        model, _ = load_model(init, device)
    X = [np.load(os.path.join(cache, f"{w}_bf_z{z:02d}.npy")) for w in train_wells]
    Y = [np.load(os.path.join(cache, f"{w}_dapi.npy")) for w in train_wells]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state = -1e9, None
    for step in range(1, steps + 1):
        model.train()
        xb, yb = [], []
        for _ in range(8):
            i = rng.integers(len(X))
            h, w = X[i].shape
            r, c = rng.integers(0, h - 256), rng.integers(0, w - 256)
            k, fl = rng.integers(4), rng.random() < 0.5
            xa, ya = np.rot90(X[i][r:r + 256, c:c + 256], k), np.rot90(Y[i][r:r + 256, c:c + 256], k)
            if fl:
                xa, ya = np.fliplr(xa), np.fliplr(ya)
            xb.append(np.ascontiguousarray(xa)); yb.append(np.ascontiguousarray(ya))
        xb = torch.from_numpy(np.stack(xb))[:, None].to(device)
        yb = torch.from_numpy(np.stack(yb))[:, None].to(device)
        mu, lv = model(xb)
        loss = gaussian_nll(mu, lv, yb, beta=0.5)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 100 == 0 or step == steps:
            model.eval()
            v = evaluate_wells(model, cache, val_wells, z, device, tta=False)
            sc = float(np.mean([r["pearson"] for r in v]))
            if sc > best:
                best, best_state = sc, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval(), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["select", "prep", "zeroshot", "finetune", "all"])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.35, 0.45, 0.55, 0.7, 0.85, 1.0])
    ap.add_argument("--root", default="data/raw/isl_rubin")
    ap.add_argument("--cache", default="outputs/neural_cache")
    ap.add_argument("--out", default="report/results/neural")
    ap.add_argument("--chipstain", default="runs/chipstain_nll_s0/best.pt")
    ap.add_argument("--baseline", default="runs/baseline_unet_s0/best.pt")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 20])
    ap.add_argument("--steps", type=int, default=1200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = get_device()
    sel_file = os.path.join(a.out, "selection.json")
    if a.stage in ("select", "all") or not os.path.exists(sel_file):
        sc_nuc, d_hela, d_neu = nucleus_scale(a.root)
        scale, z, df = select_scale_z(a.root, a.chipstain, device, sorted(set(a.scales + [round(sc_nuc, 3)])), a.out)
        json.dump({"scale": scale, "z": z, "nucleus_matched_scale": sc_nuc, "d_hela_px": d_hela, "d_neuron_px": d_neu,
                   "val_pearson": df.groupby(["scale", "z"])["pearson"].mean().reset_index().to_dict("records")}, open(sel_file, "w"), indent=1)
        print(f"selected on validation wells: scale {scale}, z {z} (nucleus-matched scale {sc_nuc:.3f})")
    sel = json.load(open(sel_file))
    z = sel["z"]
    meta_file = os.path.join(a.cache, "meta.json")
    if a.stage in ("prep", "all") or not os.path.exists(meta_file) or abs(json.load(open(meta_file))["scale"] - sel["scale"]) > 1e-6:
        meta = prep(a.root, a.cache, sel["scale"])
    else:
        meta = json.load(open(meta_file))
    test_wells = wells_of(meta, "test")
    if a.stage in ("zeroshot", "all"):
        rows = []
        for name, ck in [("ChipStain (HeLa, zero-shot)", a.chipstain), ("U-Net baseline (HeLa, zero-shot)", a.baseline)]:
            model, _ = load_model(ck, device)
            ex = "outputs/neural_example_zeroshot.npz" if name.startswith("ChipStain") else None
            rows += [{"model": name, **r} for r in evaluate_wells(model, a.cache, test_wells, z, device, tta=True, save_example=ex)]
        pd.DataFrame(rows).to_csv(os.path.join(a.out, "zeroshot.csv"), index=False)
        print(pd.DataFrame(rows).groupby("model")[["pearson", "ssim", "seg_f1_vs_real", "mean_sigma"]].mean())
    if a.stage in ("finetune", "all"):
        train_pool = wells_of(meta, "train")
        rows = []
        for k in a.ks:
            tw = train_pool[:k]
            for init_name, init in [("from ChipStain (HeLa)", a.chipstain), ("from ImageNet", "imagenet")]:
                model, vbest = finetune(init, a.cache, tw, VAL_WELLS, z, device, steps=a.steps)
                ex = "outputs/neural_example_finetuned.npz" if (k == max(a.ks) and init_name.startswith("from ChipStain")) else None
                for r in evaluate_wells(model, a.cache, test_wells, z, device, tta=True, save_example=ex):
                    rows.append({"k_wells": k, "init": init_name, "val_pearson": vbest, **r})
                pd.DataFrame(rows).to_csv(os.path.join(a.out, "finetune.csv"), index=False)
                print(k, init_name, pd.DataFrame(rows).query("k_wells == @k and init == @init_name")[["pearson", "seg_f1_vs_real", "mean_sigma"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
