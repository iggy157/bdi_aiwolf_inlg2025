#!/usr/bin/env python3
"""Micro intention generator - generates specific intention for next utterance.

状況適応的意図（micro_intention）生成モジュール.
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
    """Safely load YAML file, return empty dict if file doesn't exist or fails to load."""
    if not file_path.exists():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return {}


def get_file_sha1(file_path: Path) -> str:
    """Get SHA1 hash of file content, return empty string if file doesn't exist."""
    if not file_path.exists():
        return ""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get SHA1 for {file_path}: {e}")
        return ""


def get_file_mtime(file_path: Path) -> float:
    """Get file modification time, return 0.0 if file doesn't exist."""
    if not file_path.exists():
        return 0.0
    try:
        return file_path.stat().st_mtime
    except Exception as e:
        logger.warning(f"Failed to get mtime for {file_path}: {e}")
        return 0.0


def should_skip_regeneration(
    micro_intention_path: Path,
    micro_desire_path: Path,
    trigger: str
) -> bool:
    """Check if we should skip regeneration based on SHA1 hash comparison."""
    if trigger != "talk_fallback":
        return False
    
    if not micro_intention_path.exists():
        return False
    
    try:
        intention_data = load_yaml_safe(micro_intention_path)
        existing_sha1 = intention_data.get("meta", {}).get("source_micro_desire_sha1", "")
        current_sha1 = get_file_sha1(micro_desire_path)
        
        if existing_sha1 and current_sha1 and existing_sha1 == current_sha1:
            return True
    except Exception as e:
        logger.warning(f"Failed to check SHA1 for skip logic: {e}")
    
    return False


def generate_micro_intention_for_agent(
    game_id: str,
    agent: str,
    *,
    base_micro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
    base_macro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi"),
    agent_obj=None,
    logger_obj=None,
    trigger: Optional[str] = None
) -> Optional[Path]:
    """Generate micro_intention YAML for agent based on micro_desire and behavior tendencies.
    
    Args:
        game_id: Game ID
        agent: Agent name
        base_micro_dir: Base directory for micro BDI data
        base_macro_dir: Base directory for macro BDI data
        agent_obj: Agent object for LLM calls
        logger_obj: Logger object for logging
        trigger: Trigger event name
        
    Returns:
        Path to generated micro_intention.yml or None if failed
    """
    if logger_obj:
        logger_obj.logger.info(f"Generating micro_intention for {agent} (trigger: {trigger})")
    
    # Setup paths
    agent_micro_dir = base_micro_dir / game_id / agent
    agent_macro_dir = base_macro_dir / game_id / agent
    
    micro_desire_path = agent_macro_dir / "micro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"
    micro_intention_path = agent_micro_dir / "micro_intention.yml"
    
    # Check if we should skip regeneration
    if should_skip_regeneration(micro_intention_path, micro_desire_path, trigger or ""):
        if logger_obj:
            logger_obj.logger.info(f"Skipping regeneration for {agent}: SHA1 unchanged")
        return micro_intention_path
    
    # Collect data
    try:
        # Load micro_desire
        micro_desire_text = ""
        if micro_desire_path.exists():
            micro_desire_text = micro_desire_path.read_text(encoding='utf-8')
        else:
            if logger_obj:
                logger_obj.logger.warning(f"micro_desire.yml not found: {micro_desire_path}")
        
        # Load behavior_tendency from macro_belief
        behavior_tendency = {}
        macro_belief_data = load_yaml_safe(macro_belief_path)
        if "macro_belief" in macro_belief_data:
            mb = macro_belief_data["macro_belief"]
            behavior_tendency = mb.get("behavior_tendency", {})
            if isinstance(behavior_tendency, dict) and "behavior_tendencies" in behavior_tendency:
                behavior_tendency = behavior_tendency["behavior_tendencies"]
        
        if not behavior_tendency and logger_obj:
            logger_obj.logger.warning(f"No behavior_tendency found in: {macro_belief_path}")
        
        if logger_obj:
            logger_obj.logger.info(f"Collected data: micro_desire_len={len(micro_desire_text)}, "
                                 f"behavior_tendency_keys={len(behavior_tendency)}")
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to collect reference data: {e}")
        return None
    
    # Prepare LLM call
    if not agent_obj:
        if logger_obj:
            logger_obj.logger.error("No agent_obj provided for LLM call")
        return None
    
    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        "micro_desire_text": micro_desire_text,
        "behavior_tendency": behavior_tendency,
        "char_limits": {
            "base_length": 125,
            "mention_length": 125
        }
    }
    
    try:
        response = agent_obj.send_message_to_llm(
            "micro_intention",
            extra_vars=extra_vars,
            log_tag="micro_intention_generation"
        )
        
        if not response:
            if logger_obj:
                logger_obj.logger.warning("LLM returned empty response for micro_intention")
            response_data = _create_fallback_micro_intention()
        else:
            try:
                # Remove code blocks if present
                clean_response = response.strip()
                if clean_response.startswith("```yaml"):
                    clean_response = clean_response[7:]
                elif clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                response_data = yaml.safe_load(clean_response)
                if not isinstance(response_data, dict) or "micro_intention" not in response_data:
                    if logger_obj:
                        logger_obj.logger.warning("Invalid LLM response format, using fallback")
                    response_data = _create_fallback_micro_intention()
                else:
                    if logger_obj:
                        logger_obj.logger.info("Successfully parsed LLM response")
            except yaml.YAMLError as e:
                if logger_obj:
                    logger_obj.logger.warning(f"Failed to parse LLM response as YAML: {e}, using fallback")
                response_data = _create_fallback_micro_intention()
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"LLM call failed: {e}")
        response_data = _create_fallback_micro_intention()
    
    # Add metadata
    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    micro_desire_sha1 = get_file_sha1(micro_desire_path)
    micro_desire_mtime = get_file_mtime(micro_desire_path)
    
    response_data["meta"] = {
        "generated_at": datetime.now().isoformat(),
        "trigger": trigger or "unknown",
        "model": model_name,
        "source_micro_desire_mtime": micro_desire_mtime,
        "source_micro_desire_sha1": micro_desire_sha1,
        "game_id": game_id,
        "agent": agent
    }
    
    # Save to file
    try:
        micro_intention_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(micro_intention_path, 'w', encoding='utf-8') as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_intention_generator\n")
            yaml.safe_dump(response_data, f, allow_unicode=True, sort_keys=False)
        
        if logger_obj:
            logger_obj.logger.info(f"Saved micro_intention to: {micro_intention_path}")
        
        # Append to log
        log_path = agent_micro_dir / "micro_intention_log.yml"
        try:
            # micro_intention.yml に書いた内容（response_data）をそのまま1件としてログにも積む
            _append_to_log(log_path, response_data)
            if logger_obj:
                logger_obj.logger.info(f"Appended micro_intention to log: {log_path}")
        except Exception as e:
            if logger_obj:
                logger_obj.logger.warning(f"Failed to append micro_intention to log: {e}")
        
        return micro_intention_path
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_intention: {e}")
        return None


def _append_to_log(log_path: Path, entry: Dict[str, Any]) -> None:
    """Append one micro_intention entry to micro_intention_log.yml as items[]."""
    data: Dict[str, Any] = load_yaml_safe(log_path)
    if not isinstance(data, dict):
        data = {}
    items = data.get("items")
    if not isinstance(items, list):
        items = []
    # 最小限の整形（将来の後方互換のため shallow copy 推奨）
    items.append(entry)
    data["items"] = items

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# generated by utils.bdi.micro_bdi.micro_intention_generator (log)\n")
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _create_fallback_micro_intention() -> Dict[str, Any]:
    """Create fallback micro_intention structure when LLM fails."""
    return {
        "micro_intention": {
            "summary": "fallback intention",
            "structure": {
                "use_both": False,
                "char_limits": {
                    "base_length": 125,
                    "mention_length": 125
                },
                "base": {
                    "phases": [
                        {
                            "label": "communicate",
                            "objective": "Share basic information"
                        }
                    ]
                },
                "mention": {
                    "targets": [],
                    "phases": []
                }
            },
            "content": [
                {
                    "mode": "base",
                    "phase": "communicate",
                    "include": [
                        "Express current thoughts",
                        "Maintain group harmony"
                    ]
                }
            ],
            "priorities": [
                {
                    "item": "Basic communication",
                    "priority": 0.5
                }
            ],
            "alignment": {
                "behavior_tendency_ok": True
            },
            "rationale": "LLM generation failed, using safe fallback"
        }
    }


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate micro_intention for agent")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--trigger", default="cli", help="Trigger event")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    path = generate_micro_intention_for_agent(
        game_id=args.game_id,
        agent=args.agent,
        trigger=args.trigger
    )
    
    if path:
        print(f"Generated: {path}")
        print(path.read_text(encoding="utf-8"))
    else:
        print("Failed to generate micro_intention")


if __name__ == "__main__":
    main()