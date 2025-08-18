#!/usr/bin/env python3
"""Adjust credibility scores in analysis files based on affinity/trust from talk history.

analysis.ymlの信頼度スコアをtalk_historyのliking/creditabilityで調整するモジュール.
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import yaml

TALK_DIR_CANDIDATES = ("talk_history", "トーク履歴")
ANALYSIS_CANDIDATES = ("analysis.yml", "analysis_test.yml")
DEFAULT_SCORE = 0.5

logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """Sanitize agent name for use as filename."""
    if not name:
        return "unknown"
    s = re.sub(r'[/\\:*?"<>|\0]', "", str(name)).strip()
    return s or "unknown"


def find_talk_dir(agent_dir: Path, candidates: tuple[str, ...] = TALK_DIR_CANDIDATES) -> Optional[Path]:
    """Find existing talk directory from candidates."""
    for dirname in candidates:
        talk_dir = agent_dir / dirname
        if talk_dir.exists():
            return talk_dir
    return None


def find_analysis_file(agent_dir: Path, candidates: tuple[str, ...] = ANALYSIS_CANDIDATES) -> Optional[Path]:
    """Find existing analysis file from candidates."""
    for filename in candidates:
        analysis_file = agent_dir / filename
        if analysis_file.exists():
            return analysis_file
    return None


def load_yaml_file(file_path: Path) -> Optional[dict[str, Any]]:
    """Load YAML file and return dict, or None if failed."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Invalid YAML format (expected dict) at %s", file_path)
    except Exception as e:
        logger.warning("Failed to load YAML file %s: %s", file_path, e)
    return None


def get_affinity_scores(talk_dir: Path, from_name: str) -> tuple[float, float]:
    """Get liking and creditability scores for a speaker from talk history.
    
    Returns:
        tuple: (liking, creditability) scores, defaults to 0.5 each if not found
    """
    if not talk_dir or not from_name:
        return DEFAULT_SCORE, DEFAULT_SCORE
    
    # Sanitize the from_name for filename
    clean_name = sanitize_name(from_name)
    talk_file = talk_dir / f"{clean_name}.yml"
    
    if not talk_file.exists():
        logger.debug("Talk history file not found for %s: %s", from_name, talk_file)
        return DEFAULT_SCORE, DEFAULT_SCORE
    
    data = load_yaml_file(talk_file)
    if not data:
        return DEFAULT_SCORE, DEFAULT_SCORE
    
    # Extract scores with defaults
    liking = float(data.get("liking", DEFAULT_SCORE))
    creditability = float(data.get("creditability", DEFAULT_SCORE))
    
    return liking, creditability


def extract_credibility(entry: dict[str, Any]) -> float:
    """Extract credibility score from entry, handling spelling variations."""
    for key in ("credibility", "creditability"):
        if key in entry:
            try:
                return float(entry[key])
            except (TypeError, ValueError):
                logger.debug("Invalid %s value: %r", key, entry[key])
    return 0.0


def is_pending_entry(entry: dict[str, Any]) -> bool:
    """Check if entry is pending analysis and should be skipped."""
    if entry.get("type") == "pending_analysis":
        return True
    
    content = entry.get("content", "")
    if isinstance(content, str) and content.strip().startswith("[PENDING]"):
        return True
    
    return False


def sort_analysis_keys(data: dict[str, Any]) -> dict[str, Any]:
    """Sort analysis data by numeric keys first, then string keys."""
    def sort_key(key: Any) -> tuple[int, Any]:
        s = str(key)
        # Numeric keys come first (0), then string keys (1)
        return (0, int(s)) if s.isdigit() else (1, s)
    
    sorted_items = sorted(data.items(), key=lambda x: sort_key(x[0]))
    return dict(sorted_items)


def apply_affinity_to_analysis(
    base_dir: Path,
    game_id: str,
    agent: str,
    talk_dir_candidates: tuple[str, ...] = TALK_DIR_CANDIDATES,
    analysis_candidates: tuple[str, ...] = ANALYSIS_CANDIDATES,
    decimals: int = 3,
    logger_obj: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Apply affinity/trust adjustments to analysis credibility scores.
    
    Args:
        base_dir: Base directory for BDI info
        game_id: Game ID
        agent: Agent name
        talk_dir_candidates: Directory name candidates for talk history
        analysis_candidates: File name candidates for analysis
        decimals: Number of decimal places for rounding
        logger_obj: Logger instance to use
    
    Returns:
        Path to the updated analysis file, or None if no changes made
    """
    log = logger_obj or logger
    
    agent_dir = base_dir / game_id / agent
    if not agent_dir.exists():
        log.warning("Agent directory not found: %s", agent_dir)
        return None
    
    # Find analysis file
    analysis_file = find_analysis_file(agent_dir, analysis_candidates)
    if not analysis_file:
        log.debug("No analysis file found for %s/%s", game_id, agent)
        return None
    
    # Find talk directory
    talk_dir = find_talk_dir(agent_dir, talk_dir_candidates)
    if not talk_dir:
        log.debug("No talk directory found for %s/%s", game_id, agent)
        # We can still proceed, will use default scores
    
    # Load analysis data
    analysis_data = load_yaml_file(analysis_file)
    if not analysis_data:
        log.warning("Failed to load analysis file: %s", analysis_file)
        return None
    
    log.info("Processing analysis file: %s", analysis_file.name)
    
    # Process each entry
    modified_count = 0
    
    for key, entry in analysis_data.items():
        if not isinstance(entry, dict):
            continue
        
        # Skip pending entries
        if is_pending_entry(entry):
            log.debug("Skipping pending entry: %s", key)
            continue
        
        # Get speaker
        from_name = entry.get("from")
        if not from_name:
            log.debug("No 'from' field in entry %s, skipping", key)
            continue
        
        # Get affinity scores
        liking, creditability = get_affinity_scores(talk_dir, str(from_name))
        
        # Calculate factor
        factor = (liking + creditability) / 2.0
        
        # Get original credibility
        credibility_raw = extract_credibility(entry)
        
        # Calculate adjusted credibility
        credibility_adjusted = round(credibility_raw * factor, decimals)
        
        # Update entry
        entry["credibility_raw"] = credibility_raw
        entry["credibility"] = credibility_adjusted
        entry["adjust_meta"] = {
            "method": "multiply_mean",
            "factor": round(factor, decimals),
            "liking": liking,
            "creditability": creditability,
            "from": str(from_name)
        }
        
        modified_count += 1
        log.debug(
            "Adjusted entry %s: from=%s, raw=%.3f, factor=%.3f, adjusted=%.3f",
            key, from_name, credibility_raw, factor, credibility_adjusted
        )
    
    if modified_count == 0:
        log.info("No entries to modify in %s", analysis_file.name)
        return None
    
    # Sort keys numerically
    sorted_data = sort_analysis_keys(analysis_data)
    
    # Save updated analysis
    try:
        with analysis_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                sorted_data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False
            )
        log.info("Updated %d entries in %s", modified_count, analysis_file.name)
        return analysis_file
    except Exception as e:
        log.error("Failed to save updated analysis file %s: %s", analysis_file, e)
        return None


def main() -> int:
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply affinity adjustments to analysis credibility")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
                       help="Base directory")
    parser.add_argument("--decimals", type=int, default=3, help="Decimal places for rounding")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        result_path = apply_affinity_to_analysis(
            base_dir=args.base_dir,
            game_id=args.game_id,
            agent=args.agent,
            decimals=args.decimals,
            logger_obj=logger
        )
        
        if result_path:
            print(f"Updated analysis file: {result_path}")
            return 0
        else:
            print("No updates made")
            return 1
    except Exception as e:
        logger.error("Failed to apply affinity adjustments: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())