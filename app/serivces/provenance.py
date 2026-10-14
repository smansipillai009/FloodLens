"""
FloodLens Nepal — Provenance & Manifest
=======================================
Creates and writes run manifests that capture every input ID, date,
parameter and version used in a run. Enables full auditability.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

FLOODLENS_VERSION = "0.2.0"


def create_run_directory(outputs_root: Path, run_id: Optional[str] = None) -> tuple[str, Path]:
    """
    Create a unique run directory under outputs_root/runs/<run_id>/.

    Returns
    -------
    (run_id, run_dir)
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]  # short UUID for readability
    run_dir = outputs_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run directory created: %s", run_dir)
    return run_id, run_dir


def write_manifest(
    run_dir: Path,
    run_id: str,
    request_dict: dict[str, Any],
    scene_selection: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> Path:
    """
    Write manifest.json to the run directory.
    This file is the audit trail for the run.

    Parameters
    ----------
    run_dir : Path
        The run-specific output directory.
    run_id : str
        Unique identifier for this run.
    request_dict : dict
        Serialized RunRequest dict.
    scene_selection : dict, optional
        Serialized SceneSelectionResult dict.
    extra : dict, optional
        Any additional metadata to include.
    """
    from app.config import config_hash, get_config, locked_hash
    import platform, importlib.metadata as md
    versions = {}
    for pkg in ("numpy", "rasterio", "geopandas", "shapely", "networkx", "torch", "segmentation-models-pytorch"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = None
    manifest = {
        "floodlens_version": FLOODLENS_VERSION,
        "python": platform.python_version(),
        "package_versions": versions,
        "config_hash": config_hash(),
        "config_locked_hash": locked_hash(),
        "config_snapshot": get_config(),
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "request": request_dict,
        "scene_selection": scene_selection,
        "extra": extra or {},
        "attributions": [
            "Contains modified Copernicus Sentinel data 2026.",
            (
                "Produced using Copernicus WorldDEM-30 "
                "© DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 "
                "provided under COPERNICUS by the European Union and ESA; all rights reserved."
            ),
            "© OpenStreetMap contributors.",
        ],
        "prototype_disclaimer": (
            "Educational prototype — not operational emergency guidance. "
            "All results require field verification."
        ),
    }

    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Manifest written: %s", manifest_path)
    return manifest_path


def hash_config_file(config_path: Path) -> str:
    """Return SHA-256 hex digest of a config file (for locking configs at eval time)."""
    sha256 = hashlib.sha256()
    sha256.update(config_path.read_bytes())
    return sha256.hexdigest()
