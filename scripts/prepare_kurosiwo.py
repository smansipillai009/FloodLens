"""
Convert the Kuro Siwo GRD WebDataset into the patch format expected by app/ml/dataset.py.

  python scripts/prepare_kurosiwo.py --src /data/Kuro-Siwo-Webdataset --inspect 20     # Day 1: look first
  python scripts/prepare_kurosiwo.py --src /data/Kuro-Siwo-Webdataset --dst data/processed/kurosiwo \
         [--units auto|db|linear] [--pre sec1|sec2|both] [--speckle lee|none] [--event-key KEY] \
         [--val-events E1 E2 | --val-fraction 0.1] [--test-events E3 ...] [--limit-shards N]

Input layout (Kuro Siwo GRD WebDataset, as documented on the HF dataset card; verify with --inspect):
  <src>/train_GRD/shard-*.tar, <src>/test_GRD/shard-*.tar
  each sample = <key>.{flood_vv,flood_vh,sec1_vv,sec1_vh,sec2_vv,sec2_vh,dem,mask,valid_mask}.npy + <key>.info.json
  mask: 0 no water, 1 permanent water, 2 flood.   flood_* = post-event, sec1/sec2 = the two pre-event images.

Output (what PatchDataset reads):
  <dst>/{train,val,test}/<event_id>/<key>[_sec1|_sec2].npz
        sar    float32[4,H,W] dB = pre_VV, pre_VH, post_VV, post_VH
        y      uint8[H,W]     1 = NEW flood water
        valid  uint8[H,W]
  <dst>/prepare_meta.json  (units, preprocessing, event lists, counts: keep it with the checkpoint)

Rules enforced here
  * Split BY EVENT, never by patch. Official test_GRD -> `test`. Events in both train and test shards => hard error.
  * Permanent water (mask == 1) is IGNORED (valid = 0) by default: we want NEW flooding, and detect_flood()
    removes pre-event permanent water anyway.
  * Units are never guessed silently: `auto` proposes a unit from the FIRST sample only if it is unambiguous;
    run --inspect first and pass --units explicitly for the real conversion.
  * Preprocessing parity with inference: the live pipeline feeds Lee-filtered linear sigma0 -> dB, so by default the
    same Lee filter (app.services.sentinel1.lee_filter, same window/ENL) is applied here. If Kuro Siwo turns out to be
    already speckle-filtered, compare both on a few patches and decide with --speckle none; record the choice.
  * DEM is not used (the live model input is SAR only; terrain is handled by masks).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "mask", "valid_mask")
OPTIONAL = ("sec2_vv", "sec2_vh", "dem")
EVENT_KEY_CANDIDATES = ("event_id", "event", "flood_event", "event_name", "actid", "activation", "emsr", "act_id")
NODATA_DB_FILL = -25.0          # same fill the inference code uses for masked pixels (app/ml/infer.py)
MIN_LINEAR_FLOOR = 1e-10        # same floor as infer.py
LINEAR_MEDIAN_MAX = 10.0        # heuristic bound used ONLY to *propose* a unit in `auto`
VAL_FRACTION_DEFAULT = 0.1


class EventIdError(KeyError):
    """info.json carries no usable event identifier."""


# ── reading shards (stdlib only; no webdataset dependency) ───────────────────

def shard_paths(folder: Path, limit: int = 0) -> list[Path]:
    shards = sorted(Path(folder).glob("*.tar"))
    return shards[:limit] if limit else shards


def iter_samples(shards: Iterable[Path]) -> Iterator[tuple[str, dict[str, bytes]]]:
    """Yield (sample_key, {extension: bytes}). WebDataset key = basename up to the FIRST dot."""
    for shard in shards:
        cur_key: Optional[str] = None
        cur: dict[str, bytes] = {}
        with tarfile.open(shard, "r") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                base = Path(m.name).name
                if "." not in base:
                    continue
                key, ext = base.split(".", 1)
                if key != cur_key:
                    if cur_key is not None and cur:
                        yield cur_key, cur
                    cur_key, cur = key, {}
                cur[ext] = tf.extractfile(m).read()
        if cur_key is not None and cur:
            yield cur_key, cur


def npy(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b), allow_pickle=False).astype(np.float32)


def read_arrays(files: dict[str, bytes]) -> tuple[dict[str, np.ndarray], dict]:
    missing = [n for n in REQUIRED if f"{n}.npy" not in files]
    if missing:
        raise ValueError(f"sample is missing {missing}; has {sorted(files)}")
    arrs = {n: npy(files[f"{n}.npy"]) for n in REQUIRED + OPTIONAL if f"{n}.npy" in files}
    info = json.loads(files["info.json"].decode("utf-8")) if "info.json" in files else {}
    return arrs, info


# ── units ────────────────────────────────────────────────────────────────────

def detect_units(sar_arrays: Iterable[np.ndarray]) -> str:
    """Return 'db', 'linear' or 'ambiguous' for the SAR arrays of ONE sample."""
    vals = np.concatenate([a[np.isfinite(a) & (a != 0)].ravel() for a in sar_arrays])
    if vals.size == 0:
        return "ambiguous"
    med = float(np.median(vals))
    if med < 0:
        return "db"
    if 0 < med < LINEAR_MEDIAN_MAX and float(vals.min()) >= 0:
        return "linear"
    return "ambiguous"


def to_linear(a: np.ndarray, units: str) -> np.ndarray:
    """Linear power with nodata = 0 (the convention of app/services/sentinel1.py)."""
    a = np.where(np.isfinite(a), a, 0.0).astype(np.float64)
    if units == "db":
        out = np.where(a != 0, np.power(10.0, a / 10.0), 0.0)
    elif units == "linear":
        out = np.where(a > 0, a, 0.0)
    else:
        raise ValueError(f"units must be 'db' or 'linear', got {units!r}")
    return out.astype(np.float32)


def linear_to_db(lin: np.ndarray) -> np.ndarray:
    """Same dB conversion as app/ml/infer.py; nodata (<=0) -> NaN."""
    return np.where(lin > 0, 10.0 * np.log10(np.maximum(lin, MIN_LINEAR_FLOOR)), np.nan).astype(np.float32)


# ── one sample -> training patch ─────────────────────────────────────────────

def build_patch(arrs: dict[str, np.ndarray], pre: str, units: str, speckle: str,
                permanent: str = "ignore", lee_window: int = 5, enl: float = 4.4) -> dict:
    """pre: 'sec1' | 'sec2'.  Returns {'sar': [4,H,W] dB, 'y': uint8, 'valid': uint8}."""
    names = (f"{pre}_vv", f"{pre}_vh", "flood_vv", "flood_vh")
    lin = {n: to_linear(arrs[n], units) for n in names}
    if speckle == "lee":
        from app.services.sentinel1 import lee_filter   # identical filter to inference
        lin = {n: lee_filter(a, lee_window, enl) for n, a in lin.items()}
    db = [linear_to_db(lin[n]) for n in names]
    finite = np.all([np.isfinite(d) for d in db], axis=0)

    mask = arrs["mask"]
    valid = (arrs["valid_mask"] > 0.5) & finite & np.isin(mask, (0, 1, 2))
    if permanent == "ignore":
        valid &= mask != 1
    y = (mask == 2).astype(np.uint8)
    sar = np.stack([np.where(np.isfinite(d), d, NODATA_DB_FILL) for d in db]).astype(np.float32)
    return {"sar": sar, "y": y, "valid": valid.astype(np.uint8)}


# ── events / splits ──────────────────────────────────────────────────────────

def _slug(v) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(v)).strip("_") or "unknown"


def event_id_of(info: dict, event_key: Optional[str]) -> str:
    if event_key:
        if event_key not in info:
            raise EventIdError(f"--event-key {event_key!r} not in info.json keys {sorted(info)}")
        return _slug(info[event_key])
    for k in EVENT_KEY_CANDIDATES:
        if k in info and info[k] not in (None, ""):
            return _slug(info[k])
    raise EventIdError("Could not find an event identifier in info.json. Keys present: "
                       f"{sorted(info)}. Re-run with --event-key <key> (run --inspect to see values).")


def val_by_hash(event: str, fraction: float) -> bool:
    return int(hashlib.sha1(event.encode()).hexdigest(), 16) % 1000 < int(fraction * 1000)


def assign_split(event: str, shard_group: str, val_events: set, test_events: set, val_fraction: float) -> str:
    if shard_group == "test" or event in test_events:
        return "test"
    if event in val_events or (not val_events and val_by_hash(event, val_fraction)):
        return "val"
    return "train"


def resolve_root(src: Path) -> Path:
    """Return the folder that directly contains train_GRD/ (searches up to 3 levels below `src`)."""
    if (src / "train_GRD").is_dir():
        return src
    if src.is_dir():
        for cand in sorted(src.rglob("train_GRD")):
            if cand.is_dir() and len(cand.relative_to(src).parts) <= 3:
                print(f"Note: found train_GRD under {cand.parent}; using that as --src.")
                return cand.parent
    return src


def explain_missing(src: Path) -> None:
    print(f"\nCould not find <src>/train_GRD/*.tar with --src {src}")
    if not src.exists():
        print("  That folder does not exist. The dataset has probably not been downloaded yet (or went to another path).")
        return
    kids = sorted(x.name for x in src.iterdir())[:15]
    tars = [str(x.relative_to(src)) for x in list(src.rglob("*.tar"))[:8]]
    print(f"  Contents of {src}: {kids or '(empty)'}")
    print(f"  .tar files found below it: {tars or 'none'}")
    print("  Expected: <src>/train_GRD/*.tar and <src>/test_GRD/*.tar (download both; a few shards each is enough to start).")


# ── commands ─────────────────────────────────────────────────────────────────

def stats_line(name: str, a: np.ndarray) -> str:
    f = a[np.isfinite(a)]
    if f.size == 0:
        return f"  {name:10s} no finite values"
    p = np.percentile(f, [1, 50, 99])
    return (f"  {name:10s} shape={a.shape} min={f.min():.4g} p01={p[0]:.4g} med={p[1]:.4g} "
            f"p99={p[2]:.4g} max={f.max():.4g} nan%={100 * (1 - f.size / a.size):.1f}")


def cmd_inspect(src: Path, n: int, event_key: Optional[str]) -> int:
    src = resolve_root(src)
    folder = src / "train_GRD"
    if not shard_paths(folder):
        explain_missing(src); return 2
    unit_votes, labels, events = Counter(), Counter(), Counter()
    for i, (key, files) in enumerate(iter_samples(shard_paths(folder, 1))):
        if i >= n:
            break
        arrs, info = read_arrays(files)
        if i < 2:
            print(f"\n== sample {key}\n  files: {sorted(files)}")
            for name in REQUIRED + OPTIONAL:
                if name in arrs:
                    print(stats_line(name, arrs[name]))
            print(f"  info.json: {json.dumps(info)[:600]}")
        unit_votes[detect_units([arrs[k] for k in ("sec1_vv", "sec1_vh", "flood_vv", "flood_vh")])] += 1
        labels.update(np.unique(arrs["mask"]).astype(int).tolist())
        try:
            events[event_id_of(info, event_key)] += 1
        except EventIdError as e:
            if i == 0:
                print(f"\n!! {e}")
    print(f"\nUnit votes: {dict(unit_votes)}")
    print(f"Label values seen (expect only 0,1,2): {sorted(labels)}")
    print(f"Event ids seen in these samples: {dict(events)}")
    only = next(iter(unit_votes)) if len(unit_votes) == 1 else None
    print("Proposed: --units " + (only if only and only != "ambiguous" else "<decide manually: votes disagree/ambiguous>"))
    return 0


def cmd_convert(a: argparse.Namespace) -> int:
    src, dst = resolve_root(Path(a.src)), Path(a.dst)
    groups = {"train": src / "train_GRD", "test": src / "test_GRD"}
    for name, p in groups.items():
        if not shard_paths(p):
            print(f"No .tar shards in {p} (the {name} split is required)."); explain_missing(src); return 2
    val_events, test_events = set(map(_slug, a.val_events or [])), set(map(_slug, a.test_events or []))
    pres = ["sec1", "sec2"] if a.pre == "both" else [a.pre]
    units = a.units
    counts, events, skipped = Counter(), defaultdict(set), Counter()

    for group, folder in groups.items():
        for key, files in iter_samples(shard_paths(folder, a.limit_shards)):
            try:
                arrs, info = read_arrays(files)
            except ValueError:
                skipped["incomplete_sample"] += 1; continue
            if any(f"{p}_vv" not in arrs or f"{p}_vh" not in arrs for p in pres):
                skipped["missing_pre_image"] += 1; continue
            try:
                event = event_id_of(info, a.event_key)
            except EventIdError as e:
                print(f"FATAL: {e}"); return 3
            if units == "auto":
                proposed = detect_units([arrs[k] for k in ("sec1_vv", "sec1_vh", "flood_vv", "flood_vh")])
                if proposed == "ambiguous":
                    print(f"FATAL: units ambiguous for sample {key}; run --inspect, then pass --units db|linear"); return 4
                units = proposed
                print(f"Units proposed from the first sample: {units} (fixed for the whole run; verify with --inspect)")
            split = assign_split(event, group, val_events, test_events, a.val_fraction)
            if a.max_per_split and counts[split] >= a.max_per_split:
                continue
            for p in pres:
                patch = build_patch(arrs, p, units, a.speckle, a.permanent)
                if patch["valid"].sum() == 0:
                    skipped["no_valid_pixels"] += 1; continue
                out = dst / split / event / (f"{key}_{p}.npz" if a.pre == "both" else f"{key}.npz")
                out.parent.mkdir(parents=True, exist_ok=True)
                patch["sar"] = patch["sar"].astype(np.float16 if a.store_dtype == "float16" else np.float32)
                np.savez_compressed(out, **patch)
                counts[split] += 1
                events[split].add(event)

    empty = [k for k in ("train", "val", "test") if counts[k] == 0]
    overlap = (events["train"] | events["val"]) & events["test"]
    meta = {"units": units, "pre": a.pre, "speckle": a.speckle, "permanent_water": a.permanent,
            "event_key": a.event_key or "auto", "store_dtype": a.store_dtype, "counts": dict(counts), "skipped": dict(skipped),
            "events": {k: sorted(v) for k, v in events.items()}, "leaky_events": sorted(overlap)}
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "prepare_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("units", "pre", "speckle", "counts", "skipped")}, indent=2))
    if not counts:
        print(f"FATAL: nothing was written. Skipped: {dict(skipped)}. "
              "If 'missing_pre_image' dominates, the requested --pre image (e.g. sec2) is absent: use --pre sec1.")
        return 6
    if empty:
        print(f"WARNING: split(s) with zero patches: {empty}. Training/evaluation will fail until each split has data "
              "(check --val-events / --val-fraction / --test-events).")
    if overlap:
        print(f"FATAL: events present in BOTH train/val and test: {sorted(overlap)}. This would leak. "
              "Hold them out with --test-events (and delete the output folder) and re-run.")
        return 5
    print("Wrote", dst, "| events per split:", {k: len(v) for k, v in events.items()})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder containing train_GRD/ and test_GRD/")
    ap.add_argument("--dst", default="data/processed/kurosiwo")
    ap.add_argument("--inspect", type=int, metavar="N", help="print stats for the first N samples and exit")
    ap.add_argument("--units", choices=["auto", "db", "linear"], default="auto")
    ap.add_argument("--pre", choices=["sec1", "sec2", "both"], default="sec1",
                    help="which pre-event image is the reference ('both' doubles the data)")
    ap.add_argument("--speckle", choices=["lee", "none"], default="lee")
    ap.add_argument("--permanent", choices=["ignore", "zero"], default="ignore")
    ap.add_argument("--store-dtype", choices=["float16", "float32"], default="float16",
                    help="dtype of stored dB values; float16 (resolution ~0.01-0.03 dB) roughly halves disk use")
    ap.add_argument("--event-key", default=None, help="info.json key holding the event id")
    ap.add_argument("--val-events", nargs="*", default=None)
    ap.add_argument("--test-events", nargs="*", default=None, help="extra train-shard events to hold out as test")
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION_DEFAULT)
    ap.add_argument("--limit-shards", type=int, default=0)
    ap.add_argument("--max-per-split", type=int, default=0, help="stop writing a split after this many patches (quick tests)")
    a = ap.parse_args(argv)
    return cmd_inspect(Path(a.src), a.inspect, a.event_key) if a.inspect else cmd_convert(a)


if __name__ == "__main__":
    sys.exit(main())
