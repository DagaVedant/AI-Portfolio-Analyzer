"""
utils.py — Shared utilities for the AI Portfolio Analyzer.

Covers:
- Config loading from YAML
- Reproducible seed setting
- Device detection (CUDA / MPS / CPU)
- Colab vs local detection
- Path management
- W&B helper wrappers
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load a YAML config file and return it as a nested dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path.resolve()}")
    return cfg


def get_nested(cfg: Dict, *keys, default=None):
    """Safely retrieve a nested config value.

    Example:
        get_nested(cfg, 'model', 'hidden_size', default=128)
    """
    val = cfg
    for k in keys:
        if not isinstance(val, dict) or k not in val:
            return default
        val = val[k]
    return val


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True when running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def detect_device() -> str:
    """Return the best available compute device string."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except ImportError:
        device = "cpu"
    logger.info(f"Using device: {device}")
    return device


def get_project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    if is_colab():
        # When running on Colab with Drive mounted
        drive_path = Path("/content/drive/MyDrive/ai-portfolio-analyzer")
        if drive_path.exists():
            return drive_path
        return Path("/content/ai-portfolio-analyzer")
    # Walk up until we find configs/config.yaml
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "configs" / "config.yaml").exists():
            return current
        current = current.parent
    return Path.cwd()


def ensure_dirs(cfg: Dict) -> None:
    """Create all required data/output directories from config."""
    root = get_project_root()
    dirs = [
        cfg["data"]["raw_dir"],
        cfg["data"]["processed_dir"],
        cfg["training"]["checkpoint_dir"],
        "reports/figures",
        "dashboard",
    ]
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    logger.info("All directories ensured.")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Seed Python / NumPy / PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    logger.info(f"Global seed set to {seed}")


# ---------------------------------------------------------------------------
# W&B helpers
# ---------------------------------------------------------------------------

def init_wandb(cfg: Dict, job_type: str = "train") -> Optional[Any]:
    """Initialise a W&B run if enabled in config.

    Returns the wandb run object, or None if W&B is disabled / unavailable.
    """
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        logger.info("W&B disabled in config.")
        return None
    try:
        import wandb
        run = wandb.init(
            project=wcfg.get("project", "ai-portfolio-analyzer"),
            entity=wcfg.get("entity") or None,
            name=wcfg.get("run_name") or None,
            tags=wcfg.get("tags", []),
            config=cfg,
            job_type=job_type,
            reinit="finish previous",
        )
        logger.info(f"W&B run started: {run.url}")
        return run
    except ImportError:
        logger.warning("wandb not installed; skipping W&B logging.")
        return None
    except Exception as e:
        logger.warning(f"W&B init failed: {e}")
        return None


def wandb_log(run, metrics: Dict, step: Optional[int] = None) -> None:
    """Log a metrics dict to W&B, silently skip if run is None."""
    if run is None:
        return
    try:
        run.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"W&B log failed: {e}")


def finish_wandb(run) -> None:
    """Finish a W&B run gracefully."""
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def load_env() -> None:
    """Load .env file if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
        root = get_project_root()
        env_file = root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info(".env loaded")
        else:
            logger.info("No .env file found; relying on system environment.")
    except ImportError:
        pass


def format_number(value: float, decimals: int = 2) -> str:
    """Format a float for display."""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{decimals}f}M"
    if abs(value) >= 1e3:
        return f"{value/1e3:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def pct(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"
