"""
Steps 2+3 in one script (reads the SAME patches as training; independent of tune_baseline_kurosiwo.py).

  python scripts/eval_unet_vs_baseline.py --data data/processed/kurosiwo --max-test 2000

1. U-Net: sweeps the probability threshold 0.05..0.95 on the VAL split and PICKS the threshold there
   (best IoU on val; ties -> higher threshold).  TEST is then reported at that threshold (and at 0.5) -
   test is never used to choose anything.
2. Baseline: the simple SAR rule (VV drop <= c dB AND post VV <= p dB) for a small grid + the live-pipeline defaults,
   evaluated on VAL and TEST with the same patches, valid mask and pixel counting.
3. Writes models/checkpoints/eval_table.json and prints a Markdown table for the report.
Val has few patches (one event) -> the chosen threshold is noisy; the script prints how flat the val curve is.
Needs torch + segmentation-models-pytorch (run on the Colab/GPU machine if not installed locally).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
NB = 100                      # histogram bins over probability 0..1


def load(npz) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(npz)
    sar = z["sar"].astype(np.float32)
    y = z["y"] > 0
    v = (z["valid"] > 0) & np.all(np.isfinite(sar), axis=0)
    return sar, y, v


def hist_update(h: dict, prob: np.ndarray, y: np.ndarray, v: np.ndarray) -> None:
    b = np.clip((prob * NB).astype(int), 0, NB - 1)
    h["pos"] += np.bincount(b[v & y], minlength=NB)
    h["neg"] += np.bincount(b[v & ~y], minlength=NB)


def sweep_from_hist(h: dict) -> list[dict]:
    """Confusion for every threshold t_k = k/NB (pred = prob >= t_k), from the two histograms."""
    out = []
    pos_total, neg_total = int(h["pos"].sum()), int(h["neg"].sum())
    for k in range(1, NB):
        tp = int(h["pos"][k:].sum()); fp = int(h["neg"][k:].sum())
        fn = pos_total - tp
        d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
        out.append({"threshold": k / NB, "iou": d(tp, tp + fp + fn), "f1": d(2 * tp, 2 * tp + fp + fn),
                    "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
                    "area_diff_pct": d(100.0 * (fp - fn), tp + fn)})
    return out


def baseline_counts(sar: np.ndarray, y: np.ndarray, v: np.ndarray, change_db: float, post_db: float) -> tuple[int, int, int]:
    pre_vv, post_vv = sar[0], sar[2]
    pred = ((post_vv - pre_vv) <= change_db) & (post_vv <= post_db)
    return int((pred & y & v).sum()), int((pred & ~y & v).sum()), int((~pred & y & v).sum())


def prf(tp: int, fp: int, fn: int) -> dict:
    d = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"iou": d(tp, tp + fp + fn), "f1": d(2 * tp, 2 * tp + fp + fn), "precision": d(tp, tp + fp),
            "recall": d(tp, tp + fn), "area_diff_pct": d(100.0 * (fp - fn), tp + fn)}


def pick(sweep: list[dict]) -> dict:
    ok = [r for r in sweep if r["iou"] == r["iou"]]
    best = max(r["iou"] for r in ok)
    near = [r for r in ok if r["iou"] >= best - 1e-9]
    return max(near, key=lambda r: r["threshold"])


def subset(files: list, n: int) -> list:
    if n <= 0 or len(files) <= n:
        return files
    idx = np.linspace(0, len(files) - 1, n).astype(int)
    return [files[i] for i in idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-val", type=int, default=0, help="0 = all val patches")
    ap.add_argument("--max-test", type=int, default=2000, help="evenly spaced subset of test patches (0 = all)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--out", default="models/checkpoints/eval_table.json")
    ap.add_argument("--skip-unet", action="store_true", help="baseline rules only (no torch needed)")
    a = ap.parse_args(argv)
    from app.config import get_config
    cfg = get_config()
    fd = cfg["flood_detection"]
    data = Path(a.data)
    files = {s: subset(sorted((data / s).rglob("*.npz")), a.max_val if s == "val" else a.max_test) for s in ("val", "test")}
    for s, f in files.items():
        print(f"{s}: {len(f)} patches")
        if not f:
            print(f"FATAL: no patches in {data / s}")
            return 2

    # ---- baseline rules -------------------------------------------------------------------------
    rules = [(fd["water_change_db_vv"], fd["water_post_max_db_vv"], "live-pipeline defaults")]
    rules += [(c, p, "") for c in (-2.0, -3.0, -4.0, -5.0) for p in (-12.0, -15.0, -18.0)
              if (c, p) != (fd["water_change_db_vv"], fd["water_post_max_db_vv"])]
    base = {s: {r[:2]: [0, 0, 0] for r in rules} for s in files}
    # ---- U-Net ----------------------------------------------------------------------------------
    hist = {s: {"pos": np.zeros(NB, np.int64), "neg": np.zeros(NB, np.int64)} for s in files}
    model = None
    if not a.skip_unet:
        import torch
        from app.ml.model import build_model, make_input
        ck_path, norm_path = ROOT / cfg["ml"]["checkpoint"], ROOT / cfg["ml"]["norm_stats"]
        ck = torch.load(ck_path, map_location="cpu")
        model = build_model(ck.get("encoder", cfg["ml"]["encoder"]))
        model.load_state_dict(ck["state"]); model.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"; model.to(dev)
        norm = json.loads(norm_path.read_text(encoding="utf-8"))
        mean = np.asarray(norm["mean"], np.float32).reshape(-1)[:, None, None]; std = np.asarray(norm["std"], np.float32).reshape(-1)[:, None, None]
        print("U-Net device:", dev)

    for s, fl in files.items():
        batch_x, batch_meta = [], []

        def flush():
            if not batch_x:
                return
            import torch
            try:
                xs = [np.stack(batch_x)]
                groups = [list(range(len(batch_x)))]
            except ValueError:                      # unequal shapes -> one by one
                xs = [x[None] for x in batch_x]; groups = [[i] for i in range(len(batch_x))]
            for x, g in zip(xs, groups):
                with torch.no_grad():
                    pr = torch.sigmoid(model(torch.tensor(x).to(dev)))[:, 0].cpu().numpy()
                for j, i in enumerate(g):
                    hist_update(hist[s], pr[j], *batch_meta[i])
            batch_x.clear(); batch_meta.clear()

        for n, f in enumerate(fl, 1):
            sar, y, v = load(f)
            for (c, p, _) in rules:
                t = baseline_counts(sar, y, v, c, p)
                base[s][(c, p)] = [u + w for u, w in zip(base[s][(c, p)], t)]
            if model is not None:
                batch_x.append((make_input(*sar) - mean) / std); batch_meta.append((y, v))
                if len(batch_x) >= a.bs:
                    flush()
            if n % 500 == 0:
                print(f"  {s}: {n}/{len(fl)}")
        if model is not None:
            flush()

    result = {"n_patches": {s: len(f) for s, f in files.items()}, "baseline": [], "unet": None}
    for (c, p, note) in rules:
        row = {"change_db": c, "post_db": p, "note": note}
        for s in files:
            row[s] = prf(*base[s][(c, p)])
        result["baseline"].append(row)

    if model is not None:
        sv = sweep_from_hist(hist["val"]); st = sweep_from_hist(hist["test"])
        best = pick(sv)
        k = int(round(best["threshold"] * NB)) - 1
        near = [r["threshold"] for r in sv if r["iou"] >= best["iou"] - 0.01]
        result["unet"] = {"chosen_threshold_from_val": best["threshold"], "val_at_chosen": best, "test_at_chosen": st[k],
                          "test_at_0.5": st[49], "val_at_0.5": sv[49],
                          "val_thresholds_within_0.01_iou": [min(near), max(near)],
                          "val_curve": sv[4::10], "test_curve": st[4::10]}

    # ---- print ----------------------------------------------------------------------------------
    f3 = lambda x: "nan" if x != x else f"{x:.3f}"  # noqa: E731
    print("\n| Method | split | IoU | F1 | Precision | Recall | Area diff % |\n|---|---|---|---|---|---|---|")
    for row in result["baseline"]:
        tag = f"Baseline dVV<={row['change_db']:g} dB & VV<={row['post_db']:g} dB" + (" (defaults)" if row["note"] else "")
        for s in ("val", "test"):
            m = row[s]
            print(f"| {tag} | {s} | {f3(m['iou'])} | {f3(m['f1'])} | {f3(m['precision'])} | {f3(m['recall'])} | {m['area_diff_pct']:+.0f} |")
    if result["unet"]:
        u = result["unet"]
        for tag, m, s in (("U-Net thr 0.50", u["val_at_0.5"], "val"), ("U-Net thr 0.50", u["test_at_0.5"], "test"),
                          (f"U-Net thr {u['chosen_threshold_from_val']:.2f} (chosen on val)", u["val_at_chosen"], "val"),
                          (f"U-Net thr {u['chosen_threshold_from_val']:.2f} (chosen on val)", u["test_at_chosen"], "test")):
            print(f"| {tag} | {s} | {f3(m['iou'])} | {f3(m['f1'])} | {f3(m['precision'])} | {f3(m['recall'])} | {m['area_diff_pct']:+.0f} |")
        lo, hi = u["val_thresholds_within_0.01_iou"]
        print(f"\nVal IoU is within 0.01 of its best for thresholds {lo:.2f}-{hi:.2f}"
              f"{' (flat curve: the choice is not well determined by 29 val patches; prefer the middle of that range)' if hi - lo >= 0.2 else ''}.")
        print(f"==> set ml.prob_threshold: {u['chosen_threshold_from_val']:.2f} in configs/config.yaml (only if test at that threshold is not worse than at 0.50).")
    Path(ROOT / a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
