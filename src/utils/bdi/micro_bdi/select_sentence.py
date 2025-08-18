#!/usr/bin/env python3
"""Select one utterance from the latest N analysis entries and save to select_sentence.yml.

analysis.ymlの最新N件から1件の発話を選択してselect_sentence.ymlに保存するモジュール.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_WINDOW = 5
ANALYSIS_CANDIDATES = ("analysis.yml", "analysis_test.yml")

logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """Sanitize agent name for consistent handling."""
    if not name:
        return "unknown"
    s = re.sub(r'[/\\:*?"<>|\0]', "", str(name)).strip()
    return s or "unknown"


def find_analysis_file(agent_dir: Path, candidates: tuple[str, ...] = ANALYSIS_CANDIDATES) -> Optional[Path]:
    """Find existing analysis file from candidates."""
    for filename in candidates:
        analysis_file = agent_dir / filename
        if analysis_file.exists():
            return analysis_file
    return None


def safe_load_yaml(file_path: Path) -> Optional[dict[str, Any]]:
    """Load YAML file safely, return None if failed."""
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
        logger.warning("Invalid YAML format (expected dict) at %s", file_path)
    except Exception as e:
        logger.warning("Failed to load YAML file %s: %s", file_path, e)
    return None


def is_pending_entry(entry: dict[str, Any]) -> bool:
    """Check if entry is pending analysis and should be skipped."""
    if entry.get("type") == "pending_analysis":
        return True
    
    content = entry.get("content", "")
    if isinstance(content, str) and content.strip().startswith("[PENDING]"):
        return True
    
    return False


def extract_credibility_with_field(entry: dict[str, Any]) -> tuple[float, str]:
    """Extract credibility score from entry, handling spelling variations.
    
    Returns:
        tuple: (credibility_value, field_used)
    """
    for key in ("credibility", "creditability"):
        if key in entry:
            try:
                return float(entry[key]), key
            except (TypeError, ValueError):
                logger.debug("Invalid %s value: %r", key, entry[key])
    return 0.0, "derived"


def parse_to_targets(to_value: Any) -> set[str]:
    """Parse 'to' field value and return set of target agent names."""
    if to_value is None:
        return set()
    
    if isinstance(to_value, list):
        return {str(x).strip() for x in to_value if str(x).strip()}
    
    to_str = str(to_value).strip()
    if not to_str:
        return set()
    
    # Handle comma-separated values
    if "," in to_str:
        return {target.strip() for target in to_str.split(",") if target.strip()}
    
    return {to_str}


def is_addressed_to_self(to_value: Any, agent_name: str) -> bool:
    """Check if the 'to' field addresses the current agent.
    
    Args:
        to_value: Value from entry["to"]
        agent_name: Current agent's name
    
    Returns:
        bool: True if addressed to self (excludes 'all' and 'null')
    """
    targets = parse_to_targets(to_value)
    
    # Exclude 'all' and 'null' from self-addressing
    special_targets = {"all", "null"}
    if targets <= special_targets:  # Only contains special targets
        return False
    
    # Check if agent name is in targets
    clean_agent_name = sanitize_name(agent_name)
    for target in targets:
        if sanitize_name(target) == clean_agent_name or target == agent_name:
            return True
    
    return False


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def sort_analysis_keys(keys: list[Any]) -> list[int]:
    """Sort analysis keys numerically."""
    def numeric_sort_key(key: Any) -> int:
        try:
            return int(key)
        except (TypeError, ValueError):
            return 10**12  # Put non-numeric keys at the end
    
    return sorted([int(k) for k in keys if str(k).isdigit()], key=numeric_sort_key)


def select_single_entry(
    candidates: list[tuple[int, dict[str, Any]]],
    agent_name: str,
    prefer_to_self: bool = True,
) -> tuple[int, dict[str, Any], str]:
    """Select single entry from candidates based on selection criteria.
    
    Args:
        candidates: List of (key, entry) tuples
        agent_name: Current agent name
        prefer_to_self: Whether to prefer entries addressed to self
    
    Returns:
        tuple: (selected_key, selected_entry, reason)
    """
    if not candidates:
        raise ValueError("No candidates provided")
    
    # Filter for self-addressed entries if preferred
    if prefer_to_self:
        self_addressed = []
        for key, entry in candidates:
            if is_addressed_to_self(entry.get("to"), agent_name):
                self_addressed.append((key, entry))
        
        if self_addressed:
            # Sort by credibility (ascending), then by key (descending for latest)
            self_addressed.sort(key=lambda x: (extract_credibility_with_field(x[1])[0], -x[0]))
            selected_key, selected_entry = self_addressed[0]
            return selected_key, selected_entry, "to_self"
    
    # Select from all candidates: lowest credibility, then latest key
    candidates.sort(key=lambda x: (extract_credibility_with_field(x[1])[0], -x[0]))
    selected_key, selected_entry = candidates[0]
    return selected_key, selected_entry, "min_in_window"


def load_select_sentence_state(file_path: Path) -> dict[str, Any]:
    """Load existing select_sentence.yml state."""
    if not file_path.exists():
        return {}
    
    data = safe_load_yaml(file_path)
    return data if data else {}


def save_select_sentence(
    file_path: Path,
    agent_name: str,
    game_id: str,
    last_seen_key: int,
    selected_item: dict[str, Any],
    existing_items: list[dict[str, Any]],
) -> None:
    """Save select_sentence.yml with updated data."""
    # Check for duplicate keys in existing items
    existing_keys = {item.get("key") for item in existing_items if isinstance(item, dict)}
    
    # Only add if not already present
    items = list(existing_items)  # Copy existing items
    if selected_item.get("key") not in existing_keys:
        items.append(selected_item)
    
    output_data = {
        "agent": agent_name,
        "game_id": game_id,
        "last_seen_key": last_seen_key,
        "items": items,
    }
    
    try:
        with file_path.open("w", encoding="utf-8") as f:
            f.write("# generated by utils.bdi.micro_bdi.select_sentence\n")
            yaml.safe_dump(
                output_data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False
            )
        logger.info("Saved select_sentence update -> %s", file_path)
    except Exception as e:
        logger.error("Failed to save select_sentence file %s: %s", file_path, e)
        raise


def update_select_sentence_for_agent(
    base_dir: Path,
    game_id: str,
    agent: str,
    *,
    window_size: int = DEFAULT_WINDOW,
    prefer_to_self: bool = True,
    analysis_candidates: tuple[str, ...] = ANALYSIS_CANDIDATES,
    skip_pending: bool = True,
    logger_obj: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Update select_sentence.yml for an agent by selecting one utterance from latest analysis entries.
    
    Args:
        base_dir: Base directory for BDI info
        game_id: Game ID
        agent: Agent name
        window_size: Maximum number of latest entries to consider
        prefer_to_self: Whether to prefer entries addressed to self
        analysis_candidates: File name candidates for analysis
        skip_pending: Whether to skip pending entries
        logger_obj: Logger instance to use
    
    Returns:
        Path to the updated select_sentence file, or None if no changes made
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
    
    # Load analysis data
    analysis_data = safe_load_yaml(analysis_file)
    if not analysis_data:
        log.warning("Failed to load analysis file: %s", analysis_file)
        return None
    
    log.info("Processing analysis file: %s", analysis_file.name)
    
    # Get sorted numeric keys
    numeric_keys = sort_analysis_keys(list(analysis_data.keys()))
    if not numeric_keys:
        log.info("No valid numeric keys in analysis for %s/%s", game_id, agent)
        return None
    
    # Get the latest window
    recent_keys = numeric_keys[-window_size:]
    max_analysis_key = numeric_keys[-1]
    
    # Load existing select_sentence state
    select_file = agent_dir / "select_sentence.yml"
    existing_state = load_select_sentence_state(select_file)
    last_seen_key = int(existing_state.get("last_seen_key", 0))
    existing_items = existing_state.get("items", [])
    if not isinstance(existing_items, list):
        existing_items = []
    
    # Check if there are new entries to process
    if max_analysis_key <= last_seen_key:
        log.debug("No new analysis entries since last_seen_key=%d for %s/%s", 
                 last_seen_key, game_id, agent)
        return select_file  # File exists but no update needed
    
    # Build candidates from recent entries
    candidates: list[tuple[int, dict[str, Any]]] = []
    
    for key in recent_keys:
        # Try both integer key and string key for YAML compatibility
        entry = analysis_data.get(key) or analysis_data.get(str(key))
        if not isinstance(entry, dict):
            continue
        
        # Skip pending entries if requested
        if skip_pending and is_pending_entry(entry):
            log.debug("Skipping pending entry: key=%d", key)
            continue
        
        # Skip entries without valid content
        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            log.debug("Skipping entry without content: key=%d", key)
            continue
        
        candidates.append((key, entry))
    
    # If no valid candidates, just update last_seen_key
    if not candidates:
        log.info("No valid candidates in window for %s/%s, updating last_seen_key only", 
                game_id, agent)
        save_select_sentence(
            select_file, agent, game_id, max_analysis_key, {}, existing_items
        )
        return select_file
    
    # Select single entry
    try:
        selected_key, selected_entry, reason = select_single_entry(
            candidates, agent, prefer_to_self
        )
    except ValueError as e:
        log.warning("Failed to select entry for %s/%s: %s", game_id, agent, e)
        return None
    
    # Extract credibility information
    credibility_value, credibility_field = extract_credibility_with_field(selected_entry)
    
    # Create selected item
    selected_item = {
        "key": selected_key,
        "from": selected_entry.get("from"),
        "to": selected_entry.get("to"),
        "content": selected_entry.get("content"),
        "credibility": float(credibility_value),
        "credibility_field": credibility_field,
        "reason": reason,
        "selected_at": get_current_timestamp(),
    }
    
    # Save updated select_sentence.yml
    try:
        save_select_sentence(
            select_file, agent, game_id, max_analysis_key, selected_item, existing_items
        )
        log.info("Selected 1 entry (key=%d, reason=%s) for %s/%s", 
                selected_key, reason, game_id, agent)
        return select_file
    except Exception as e:
        log.error("Failed to save select_sentence for %s/%s: %s", game_id, agent, e)
        return None


def main() -> int:
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Select and save sentence from analysis")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--base-dir", type=Path,
                       default=Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
                       help="Base directory")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW,
                       help="Window size for recent entries")
    parser.add_argument("--no-prefer-self", action="store_true",
                       help="Don't prefer entries addressed to self")
    parser.add_argument("--include-pending", action="store_true",
                       help="Include pending entries in selection")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    try:
        result_path = update_select_sentence_for_agent(
            base_dir=args.base_dir,
            game_id=args.game_id,
            agent=args.agent,
            window_size=args.window_size,
            prefer_to_self=(not args.no_prefer_self),
            skip_pending=(not args.include_pending),
            logger_obj=logger
        )
        
        if result_path:
            print(f"Updated select_sentence file: {result_path}")
            return 0
        else:
            print("No updates made")
            return 1
    except Exception as e:
        logger.error("Failed to update select_sentence: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())