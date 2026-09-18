"""Image-quality, calibration and downstream-segmentation metrics."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import filters, measure, morphology, segmentation, feature
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------- image quality (inputs are normalised targets in [0,1]) ----------
def image_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.clip(pred, 0, 1).astype(np.float64)
    gt = np.clip(gt, 0, 1).astype(np.float64)
    return {
        "mae": float(np.abs(pred - gt).mean()),
        "psnr": float(peak_signal_noise_ratio(gt, pred, data_range=1.0)),
        "ssim": float(structural_similarity(gt, pred, data_range=1.0)),
        "pearson": float(np.corrcoef(pred.ravel(), gt.ravel())[0, 1]),
    }


# ---------- calibration ----------
def sparsification(err: np.ndarray, unc: np.ndarray, n_bins: int = 20):
    """Sparsification curve: remove the most-uncertain pixels first and track the
    MAE of the remainder. Returns fractions, curve_by_unc, curve_oracle.
    AUSE = area between the two curves (lower is better)."""
    err = err.ravel()
    unc = unc.ravel()
    order_u = np.argsort(-unc)
    order_o = np.argsort(-err)
    fr = np.linspace(0, 0.99, n_bins)
    cu, co = [], []
    n = len(err)
    for f in fr:
        k = int(n * f)
        cu.append(err[order_u[k:]].mean())
        co.append(err[order_o[k:]].mean())
    cu, co = np.array(cu), np.array(co)
    ause = float(np.trapezoid(cu - co, fr) / max(cu[0], 1e-9))
    return fr, cu, co, ause


def calibration_metrics(pred, gt, unc, n_pix: int = 200_000) -> dict:
    err = np.abs(pred - gt).ravel()
    u = unc.ravel()
    idx = np.random.default_rng(0).choice(len(err), size=min(n_pix, len(err)), replace=False)
    rho = spearmanr(u[idx], err[idx]).statistic
    _, _, _, ause = sparsification(err, u)
    return {"spearman_unc_err": float(rho), "ause": ause}


# ---------- downstream nuclei segmentation ----------
def segment_nuclei(fluo01: np.ndarray, min_size: int = 30) -> np.ndarray:
    """Simple label-free-agnostic pipeline: blur -> Otsu -> watershed on distance."""
    sm = filters.gaussian(fluo01, sigma=1.0)
    thr = filters.threshold_otsu(sm)
    mask = sm > thr
    mask = morphology.remove_small_objects(mask, min_size)
    mask = ndi.binary_fill_holes(mask)
    dist = ndi.distance_transform_edt(mask)
    peaks = feature.peak_local_max(dist, min_distance=6, labels=measure.label(mask))
    markers = np.zeros_like(mask, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)


def match_f1(pred_lab: np.ndarray, gt_lab: np.ndarray, iou_thr: float = 0.5) -> dict:
    """Object-level precision/recall/F1 by greedy IoU matching."""
    gt_ids = np.unique(gt_lab)[1:]
    pr_ids = np.unique(pred_lab)[1:]
    if len(gt_ids) == 0 or len(pr_ids) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_gt": len(gt_ids), "n_pred": len(pr_ids)}
    gt_area = np.array([(gt_lab == g).sum() for g in gt_ids])[:, None]
    pr_area = np.array([(pred_lab == p).sum() for p in pr_ids])[None, :]
    inter = np.zeros((len(gt_ids), len(pr_ids)))
    gi = {g: i for i, g in enumerate(gt_ids)}
    pi = {p: i for i, p in enumerate(pr_ids)}
    both = (gt_lab > 0) & (pred_lab > 0)
    for g, p in zip(gt_lab[both], pred_lab[both]):
        inter[gi[g], pi[p]] += 1
    iou = inter / (gt_area + pr_area - inter + 1e-9)
    tp = 0
    used = set()
    for i in np.argsort(-iou.max(1)):
        j = int(iou[i].argmax())
        if iou[i, j] >= iou_thr and j not in used:
            tp += 1
            used.add(j)
    prec = tp / len(pr_ids)
    rec = tp / len(gt_ids)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1, "n_gt": int(len(gt_ids)), "n_pred": int(len(pr_ids))}
