#!/usr/bin/env python3
"""Talk generation from micro_intention - generates situation-adaptive talk utterances.

micro_intention から状況適応的な発話を生成するモジュール.
"""

import logging
import re
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


def to_one_line(s: str) -> str:
    """Convert text to one line: replace newlines with spaces, collapse multiple spaces."""
    s = s.replace('\r', ' ').replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def remove_forbidden(s: str) -> str:
    """Remove forbidden characters: comma, quote, backtick."""
    return re.sub(r'[,>`]', '', s)


def effective_len(s: str) -> int:
    """Count length excluding spaces."""
    return len(s.replace(' ', ''))


def trim_ignoring_spaces(s: str, limit: int) -> str:
    """Trim string to limit non-space characters while preserving spaces."""
    out_chars = []
    count = 0
    for ch in s:
        non_space = (ch != ' ')
        if non_space:
            if count >= limit:
                break
            count += 1
        out_chars.append(ch)
    return ''.join(out_chars)


def craft_talk_from_intention(
    game_id: str,
    agent: str,
    *,
    base_macro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
    agent_obj=None,
    logger_obj=None,
) -> Optional[str]:
    """
    micro_bdi/<game_id>/<agent>/micro_intention.yml を読み込み、
    LLM（prompt: 'talk_from_intention'）でプレーンテキスト生成 → 検証・整形して
    最終1行の talk を返す。失敗時 None。
    """
    if logger_obj:
        logger_obj.logger.info(f"Crafting talk from intention for {agent}")
    
    # Load micro_intention.yml
    intention_path = base_macro_dir / game_id / agent / "micro_intention.yml"
    micro_intention_text = ""
    
    if intention_path.exists():
        try:
            micro_intention_text = intention_path.read_text(encoding='utf-8')
            if logger_obj:
                logger_obj.logger.info(f"Loaded micro_intention: {len(micro_intention_text)} chars")
        except Exception as e:
            if logger_obj:
                logger_obj.logger.warning(f"Failed to load micro_intention: {e}")
            micro_intention_text = ""
    else:
        if logger_obj:
            logger_obj.logger.info(f"No micro_intention found at: {intention_path}")
    
    # Prepare LLM call
    if not agent_obj:
        if logger_obj:
            logger_obj.logger.error("No agent_obj provided for LLM call")
        return None
    
    # Get actual agent names from agent_obj if available
    agent_names = []
    if agent_obj and hasattr(agent_obj, 'info') and agent_obj.info:
        info = agent_obj.info
        if hasattr(info, 'status_map') and info.status_map:
            agent_names = list(info.status_map.keys())
    
    # Get talk history if available
    talk_history = []
    if agent_obj and hasattr(agent_obj, 'talk_history'):
        talk_history = agent_obj.talk_history
    
    sent_talk_count = 0
    if agent_obj and hasattr(agent_obj, 'sent_talk_count'):
        sent_talk_count = agent_obj.sent_talk_count
    
    # Get setting object for max_length
    setting = None
    if agent_obj and hasattr(agent_obj, 'setting'):
        setting = agent_obj.setting
    
    # Get affinity/trust data from talk history files
    affinity_trust = {}
    try:
        base_micro_dir_path = base_macro_dir / game_id / agent
        talk_dir_path = None
        
        # Find talk_history directory
        potential_paths = [
            base_micro_dir_path / "talk_history",
            base_micro_dir_path.parent.parent / "micro_bdi" / game_id / agent / "talk_history"
        ]
        
        for path in potential_paths:
            if path.exists():
                talk_dir_path = path
                break
        
        if talk_dir_path and talk_dir_path.exists():
            for yml_file in talk_dir_path.glob("*.yml"):
                agent_name = yml_file.stem
                try:
                    data = load_yaml_safe(yml_file)
                    affinity_trust[agent_name] = {
                        "liking": float(data.get("liking", 0.5)),
                        "creditability": float(data.get("creditability", 0.5))
                    }
                except Exception as e:
                    if logger_obj:
                        logger_obj.logger.warning(f"Failed to load affinity data from {yml_file}: {e}")
        
        if logger_obj:
            logger_obj.logger.info(f"Loaded affinity/trust data for {len(affinity_trust)} agents")
    except Exception as e:
        if logger_obj:
            logger_obj.logger.warning(f"Failed to collect affinity/trust data: {e}")
    
    # Build phase context if possible
    phase_ctx = {}
    if agent_obj and hasattr(agent_obj, 'info') and hasattr(agent_obj, 'talk_history'):
        from .phase_context import build_phase_context
        try:
            phase_ctx = build_phase_context(agent_obj.info, agent_obj.talk_history)
        except Exception as e:
            if logger_obj:
                logger_obj.logger.warning(f"Failed to build phase context: {e}")
    
    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        "micro_intention_text": micro_intention_text,
        "agent_names": agent_names,  # Add actual agent names
        "talk_history": talk_history,
        "sent_talk_count": sent_talk_count,
        "setting": setting,
        "phase_ctx": phase_ctx,  # Add phase context
        "affinity_trust": affinity_trust,  # Add affinity/trust data
        "role": getattr(agent_obj, 'role', None) if agent_obj else None
    }
    
    try:
        response = agent_obj.send_message_to_llm(
            "talk_from_intention",
            extra_vars=extra_vars,
            log_tag="talk_from_intention"
        )
        
        if not response:
            if logger_obj:
                logger_obj.logger.warning("LLM returned empty response for talk_from_intention")
            return None
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"LLM call failed: {e}")
        return None
    
    # Analyze micro_intention structure to optimize talk generation
    micro_desire_elements_achieved = 0
    try:
        if micro_intention_text:
            import yaml
            intention_data = yaml.safe_load(micro_intention_text) or {}
            
            # Extract content items for priority implementation
            content_items = intention_data.get("micro_intention", {}).get("content", [])
            accusation_items = []
            evidence_items = []
            question_items = []
            
            for content in content_items:
                include_items = content.get("include", [])
                for item in include_items:
                    item_str = str(item).upper()
                    if "ACCUSATION:" in item_str:
                        accusation_items.append(item.replace("ACCUSATION:", "").strip())
                    elif "EVIDENCE:" in item_str:
                        evidence_items.append(item.replace("EVIDENCE:", "").strip())
                    elif "QUESTION:" in item_str:
                        question_items.append(item.replace("QUESTION:", "").strip())
            
            if logger_obj:
                logger_obj.logger.info(f"micro_intention analysis: accusations={len(accusation_items)}, evidence={len(evidence_items)}, questions={len(question_items)}")
    except Exception as e:
        if logger_obj:
            logger_obj.logger.warning(f"Failed to analyze micro_intention structure: {e}")
    
    # Process plain text response
    one = to_one_line(response)
    one = remove_forbidden(one)
    
    # Get per_talk limit from agent_obj.setting (default: 125)
    limit = 125
    try:
        s = getattr(agent_obj, 'setting', None)
        mx = getattr(getattr(getattr(s, 'talk', None), 'max_length', None), 'per_talk', None) if s else None
        if mx is not None:
            limit = int(mx)
    except Exception:
        pass
    
    # Check length (excluding spaces) and trim if necessary
    if effective_len(one) > limit:
        one = trim_ignoring_spaces(one, limit)
    
    # Final cleanup: ensure no newlines or forbidden chars
    if '\n' in one or '\r' in one or any(c in one for c in [',', '>', '`']):
        one = remove_forbidden(to_one_line(one))
    
    # Reject if too short (effective length < 3)
    if effective_len(one) < 3:
        if logger_obj:
            logger_obj.logger.warning("Final talk too short, rejecting")
        return None
    
    # Analyze achieved micro_desire elements in final talk
    if logger_obj:
        achieved_elements = []
        talk_upper = one.upper()
        
        # Check for agent names (suspicion/accusation)
        for agent_name in agent_names:
            if agent_name.upper() in talk_upper:
                achieved_elements.append(f"named_{agent_name}")
        
        # Check for question patterns
        if '?' in one:
            achieved_elements.append("question_asked")
        
        # Check for accusation patterns
        accusation_patterns = ["WEREWOLF", "SUSPICIOUS", "SUSPECT", "LYING", "VOTE"]
        for pattern in accusation_patterns:
            if pattern in talk_upper:
                achieved_elements.append(f"accusation_{pattern}")
        
        logger_obj.logger.info(f"Generated talk: '{one}' (effective_len={effective_len(one)}, elements={len(achieved_elements)})")
        if achieved_elements:
            logger_obj.logger.info(f"Achieved elements: {achieved_elements}")
    
    return one


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate talk from micro_intention")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    result = craft_talk_from_intention(
        game_id=args.game_id,
        agent=args.agent
    )
    
    if result:
        print(f"Generated talk: {result}")
    else:
        print("Failed to generate talk")


if __name__ == "__main__":
    main()