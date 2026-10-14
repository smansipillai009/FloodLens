"""
Train the water-segmentation U-Net.
    python -m app.ml.train --data data/processed/kurosiwo --epochs 30
Saves: checkpoint, norm_stats.json, metrics.json (val + held-out test events).
Loss: BCE + Dice, masked by `valid`.  Needs torch + segmentation-models-pytorch.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from app.config import get_config
from app.ml import metrics as M
from app.ml.dataset import PatchDataset, compute_norm_stats

logger = logging.getLogger("train")


def _loss(logits, y, v):
    import torch
    import torch.nn.functional as F
    bce = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits) * v
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + (y * v).sum() + 1)
    return bce + dice


def evaluate(model, ds, device, thr=0.5, bs=8) -> dict:
    import torch
    model.eval()
    tot = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    t0, n = time.time(), 0
    with torch.no_grad():
        for i in range(0, len(ds), bs):
            batch = [ds[j] for j in range(i, min(i + bs, len(ds)))]
            x = torch.tensor(np.stack([b[0] for b in batch])).to(device)
            pred = (torch.sigmoid(model(x)) >= thr).cpu().numpy()[:, 0]
            for k, b in enumerate(batch):
                tot = M.add(tot, M.confusion(pred[k], b[1][0] > 0.5, b[2][0] > 0))
                n += 1
    s = M.scores(tot)
    s["inference_s_per_patch"] = (time.time() - t0) / max(n, 1)
    return s


def main():
    import torch
    from torch.utils.data import DataLoader
    from app.ml.model import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train-patches", type=int, default=0, help="subset if compute-limited")
    a = ap.parse_args()
    cfg = get_config()["ml"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train = PatchDataset(a.data, "train", augment=True)
    if a.max_train_patches:
        train.files = train.files[:: max(1, len(train.files) // a.max_train_patches)]
    norm = compute_norm_stats(train)
    train.norm = norm
    val, test = PatchDataset(a.data, "val", norm), PatchDataset(a.data, "test", norm)

    model = build_model(cfg["encoder"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    best, ck = -1.0, Path(cfg["checkpoint"]); ck.parent.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=2)
    for ep in range(a.epochs):
        model.train(); tl = 0
        for x, y, v in dl:
            x, y, v = x.to(dev), y.to(dev), v.to(dev)
            opt.zero_grad(); l = _loss(model(x), y, v); l.backward(); opt.step(); tl += float(l.detach())
        vs = evaluate(model, val, dev, cfg["prob_threshold"])
        logger.info("ep %d loss %.4f val IoU %.3f F1 %.3f", ep, tl / len(dl), vs["iou"], vs["f1"])
        if vs["iou"] > best:
            best = vs["iou"]; torch.save({"state": model.state_dict(), "encoder": cfg["encoder"]}, ck)
    model.load_state_dict(torch.load(ck, map_location=dev)["state"])
    Path(cfg["norm_stats"]).write_text(json.dumps(norm), encoding="utf-8")
    out = {"val": evaluate(model, val, dev, cfg["prob_threshold"]),
           "test_heldout_events": evaluate(model, test, dev, cfg["prob_threshold"]),
           "ensemble_validated": False,
           "note": "Himalayan unseen-scene metrics are added by the separate check-only comparison script"}
    Path(cfg["metrics_file"]).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
