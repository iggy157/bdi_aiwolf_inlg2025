# -*- coding: utf-8 -*-
"""Configuration file loading utilities.

設定ファイルロードユーティリティ.
"""

from pathlib import Path
import yaml
from typing import Any, Dict


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        path: Path to config YAML file
        
    Returns:
        Configuration dictionary
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}