"""Portable filesystem locations shared by the paper scripts.

Defaults are relative to the repository. Environment variables allow large
data and an ACORN checkout to live elsewhere without editing source files.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
DATA_DIR = PAPER_ROOT / "data"
CONFIG_DIR = PAPER_ROOT / "configs"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


WORK_DIR = _path_from_env("ACORN_WORK_DIR", REPO_ROOT / "work")
DOI_ROOT = _path_from_env("ACORN_DOI_DIR", REPO_ROOT / "external" / "doi")
ACORN_SOURCE_DIR = _path_from_env(
    "ACORN_SOURCE_DIR", REPO_ROOT / "external" / "acorn"
)
BUILD_DIR = _path_from_env("ACORN_BUILD_DIR", REPO_ROOT / "build")

MODEL_CHECKPOINTS_DIR = DOI_ROOT / "models"
MODEL_RUNS_DIR = WORK_DIR / "model_runs"
RAW_CRYO_DIR = DOI_ROOT / "raw" / "cryo_tem_mrc"
RAW_SEM_DIR = DOI_ROOT / "raw" / "sem_tiff"
ADDITIONAL_INFO_DIR = DOI_ROOT / "analysis" / "additional_information"
FIGURE_SCRIPTS_DIR = PAPER_ROOT / "scripts" / "figures"

