"""Common writer utilities for macro belief system.

マクロ信念システム共通書き込みユーティリティ.
"""

from __future__ import annotations

import os
import io
import yaml
import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict


def git_short_sha_or_unknown() -> str:
    """Get git short SHA or 'unknown' if not available.
    
    Returns:
        Git short SHA or 'unknown'
    """
    try:
        result = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return result
    except Exception:
        return "unknown"


def write_macro_belief(data: Dict[str, Any], out_dir: Path) -> Path:
    """Write macro belief data to YAML file atomically.
    
    Args:
        data: Macro belief data dictionary
        out_dir: Output directory path
        
    Returns:
        Path to written file
        
    Raises:
        Exception: If writing fails
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "macro_belief.yml"
    tmp_path = out_dir / "macro_belief.yml.tmp"
    
    try:
        with io.open(tmp_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        
        # Atomic replacement
        os.replace(tmp_path, final_path)
        return final_path
        
    except Exception as e:
        # Clean up temporary file on error
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise e


def create_meta_dict(game_id: str, agent: str) -> Dict[str, Any]:
    """Create metadata dictionary.
    
    Args:
        game_id: Game ID
        agent: Agent name
        
    Returns:
        Metadata dictionary
    """
    now_iso = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return {
        "game_id": game_id,
        "agent": agent, 
        "generated_at": now_iso,
        "code_version": git_short_sha_or_unknown(),
    }