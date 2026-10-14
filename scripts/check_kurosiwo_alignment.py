"""
Do the Kuro Siwo flood LABELS line up with the SAR images? (Day-1 sanity check before any tuning or training.)

  python scripts/check_kurosiwo_alignment.py --data data/processed/kurosiwo [--split train] [--patches 200]

Prints, over N patches:
  * how dark / how much darker the flood pixels are than non-flood land (post VV dB, change dB);
  * the same statistic with the label transposed / flipped / rotated. If one of those shows a much stronger 'flood is darker' contrast than the identity,
    the labels are stored in a different orientation than the images and prepare_kurosiwo.py must apply that transform.
Verdict lines: OK / WEAK CONTRAST (labels may simply be hard for SAR, e.g. flooded vegetation) / ORIENTATION MISMATCH.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

TRANSFORMS = {"identity": lambda a: a, "transpose": lambda a: a.T, "flipud": np.flipud, "fliplr": np.fliplr,
              "rot90": lambda a: np.rot90(a, 1), "rot180": lambda a: np.rot90(a, 2), "rot270": lambda a: np.rot90(a, 3)}
MIN_FLOOD_PIXELS = 2000
CONTRAST_OK_DB = 2.0          # flood must be at least this much darker (post VV) than land, on average
MISMATCH_MARGIN_DB = 1.0      # another orientation must beat the identity by this much to be called a mismatch


def contrast(files, transform) -> dict:
    fl_post, ld_post, fl_chg, ld_chg = [], [], [], []
    for f in files:
        z = np.load(f)
        sar = z["sar"].astype("float32")
        y = TRANSFORMS[transform](z["y"]).astype(bool)
        if y.shape != sar[0].shape:
            continue
        ok = z["valid"].astype(bool)
        post, chg = sar[2], sar[2] - sar[0]
        fl_post.append(post[ok & y]); ld_post.append(post[ok & ~y]); fl_chg.append(chg[ok & y]); ld_chg.append(chg[ok & ~y])
    cat = lambda l: np.concatenate(l) if l else np.array([])  # noqa: E731
    fp, lp, fc, lc = cat(fl_post), cat(ld_post), cat(fl_chg), cat(ld_chg)
    if fp.size == 0 or lp.size == 0:
        return {"flood_px": int(fp.size)}
    return {"flood_px": int(fp.size), "flood_post_med": float(np.median(fp)), "land_post_med": float(np.median(lp)),
            "darker_by_db": float(np.median(lp) - np.median(fp)), "flood_change_med": float(np.median(fc)), "land_change_med": float(np.median(lc)),
            "frac_flood_below_minus15": float((fp < -15).mean()), "frac_land_below_minus15": float((lp < -15).mean())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--patches", type=int, default=200)
    a = ap.parse_args(argv)
    files = sorted((Path(a.data) / a.split).rglob("*.npz"))
    if not files:
        print(f"No patches under {Path(a.data) / a.split}"); return 2
    sel = [files[i] for i in np.linspace(0, len(files) - 1, min(len(files), a.patches)).astype(int)]
    res = {t: contrast(sel, t) for t in TRANSFORMS}
    base = res["identity"]
    print(f"{len(sel)} patches from '{a.split}'. 'darker_by' = median post-VV of land minus median post-VV of flood pixels (dB, higher = labels sit on dark water)\\n")
    print(f"{'orientation':10s} {'flood px':>9s} {'flood post':>10s} {'land post':>10s} {'darker by':>10s} {'flood chg':>10s} {'<-15dB: flood/land':>20s}")
    for t, r in res.items():
        if "darker_by_db" not in r:
            print(f"{t:10s} {r['flood_px']:9d}  (no data)"); continue
        print(f"{t:10s} {r['flood_px']:9d} {r['flood_post_med']:10.2f} {r['land_post_med']:10.2f} {r['darker_by_db']:10.2f} {r['flood_change_med']:10.2f} "
              f"{100 * r['frac_flood_below_minus15']:9.1f}% /{100 * r['frac_land_below_minus15']:5.1f}%")
    if base.get("flood_px", 0) < MIN_FLOOD_PIXELS or "darker_by_db" not in base:
        print(f"\\nVERDICT: too few flood pixels ({base.get('flood_px', 0)}) to judge; use more --patches or another --split."); return 1
    best_t = max((t for t in res if "darker_by_db" in res[t]), key=lambda t: res[t]["darker_by_db"])
    if best_t != "identity" and res[best_t]["darker_by_db"] > base["darker_by_db"] + MISMATCH_MARGIN_DB:
        print(f"\\nVERDICT: ORIENTATION MISMATCH. The labels fit the images much better after '{best_t}' "
              f"({res[best_t]['darker_by_db']:.1f} dB vs {base['darker_by_db']:.1f} dB). prepare_kurosiwo.py must apply this transform to mask/valid_mask."); return 4
    if base["darker_by_db"] < CONTRAST_OK_DB:
        print(f"\\nVERDICT: WEAK CONTRAST ({base['darker_by_db']:.1f} dB). Orientation looks right, but flood pixels are barely darker than land: expected for flooded vegetation/urban "
              "areas. A fixed dB threshold will score poorly here; the U-Net (which sees pre/post together) is the right tool, and the threshold baseline will be low."); return 1
    print(f"\\nVERDICT: OK. Flood pixels are {base['darker_by_db']:.1f} dB darker than land and the identity orientation is the best fit."); return 0


if __name__ == "__main__":
    sys.exit(main())
