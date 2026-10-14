"""
FloodLens Nepal — Situation Report Generator
============================================
Reads ONLY from a validated AnalysisSummary (summary.json).
Governance rule: NO hardcoded numbers in this file or the template.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from app.api.schemas import AnalysisSummary
from app.config import get_config

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
TEMPLATE_FILE = "situation_report.md.j2"


def load_summary_from_json(summary_path: Path) -> AnalysisSummary:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return AnalysisSummary.model_validate(data)


def render_report(
    summary: AnalysisSummary,
    templates_dir: Optional[Path] = None,
    template_file: str = TEMPLATE_FILE,
) -> str:
    if templates_dir is None:
        templates_dir = TEMPLATES_DIR
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        undefined=StrictUndefined,
        autoescape=False,
    )
    order = {"potentially_cut_off": 0, "uncertain": 1, "connected": 2}
    n = get_config()["report"]["max_settlement_rows"]
    rows = sorted(summary.settlement_results, key=lambda r: (order[r.status.value], r.settlement_name or ""))
    rows = [r for r in rows if r.status.value != "connected"][:n]
    rendered = env.get_template(template_file).render(summary=summary, rows=rows)
    logger.info("Report rendered (%d chars).", len(rendered))
    return rendered


def write_report(
    summary: AnalysisSummary,
    out_path: Path,
    templates_dir: Optional[Path] = None,
) -> Path:
    out_path.write_text(render_report(summary, templates_dir), encoding="utf-8")
    logger.info("Situation report written: %s", out_path)
    return out_path


def generate_report_from_run(
    run_dir: Path,
    templates_dir: Optional[Path] = None,
) -> Optional[Path]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        logger.error("summary.json not found in %s.", run_dir)
        return None
    return write_report(
        load_summary_from_json(summary_path),
        run_dir / "situation_report_en.md",
        templates_dir,
    )
