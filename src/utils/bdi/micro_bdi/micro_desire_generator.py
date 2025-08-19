#!/usr/bin/env python3
"""Micro desire generator - generates situation-adaptive desires for next utterance.

状況適応的願望（micro_desire）生成モジュール.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .talk_history_init import determine_talk_dir

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


def collect_affinity_trust_data(talk_dir: Path, logger_obj=None) -> Dict[str, Dict[str, float]]:
    """Collect liking/creditability data from all talk history files."""
    result = {}
    
    if not talk_dir.exists():
        if logger_obj:
            logger_obj.logger.warning(f"Talk directory not found: {talk_dir}")
        return result
    
    for yml_file in talk_dir.glob("*.yml"):
        agent_name = yml_file.stem
        data = load_yaml_safe(yml_file)
        
        result[agent_name] = {
            "liking": float(data.get("liking", 0.5)),
            "creditability": float(data.get("creditability", 0.5))
        }
    
    if logger_obj:
        logger_obj.logger.info(f"Collected affinity/trust data for {len(result)} agents")
    
    return result


def extract_analysis_tail(analysis_path: Path, max_items: int = 12) -> str:
    """Extract the last max_items from analysis.yml items."""
    data = load_yaml_safe(analysis_path)
    items = data.get("items", [])
    
    if not items:
        return ""
    
    # Get tail items
    tail_items = items[-max_items:] if len(items) > max_items else items
    
    # Convert to readable text
    lines = []
    for i, item in enumerate(tail_items, 1):
        content = item.get("content", "")
        from_agent = item.get("from", "unknown")
        creditability = item.get("creditability", 0.0)
        lines.append(f"{i}. {from_agent}: {content} (cred: {creditability:.2f})")
    
    return "\n".join(lines)


def get_latest_select_sentence(select_sentence_path: Path) -> str:
    """Get the latest sentence from select_sentence.yml."""
    data = load_yaml_safe(select_sentence_path)
    
    # Handle different possible structures
    if isinstance(data, dict):
        if "sentence" in data:
            return str(data["sentence"])
        elif "latest_sentence" in data:
            return str(data["latest_sentence"])
        elif "content" in data:
            return str(data["content"])
        elif "selected" in data:
            selected = data["selected"]
            if isinstance(selected, dict) and "content" in selected:
                return str(selected["content"])
    
    return ""


def generate_micro_desire_for_agent(
    game_id: str,
    agent: str,
    *,
    base_micro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
    base_macro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi"),
    agent_obj=None,
    logger_obj=None,
    trigger: Optional[str] = None,
    max_analysis_items: int = 12
) -> Optional[Path]:
    """Generate micro_desire YAML for agent based on current situation.
    
    Args:
        game_id: Game ID
        agent: Agent name
        base_micro_dir: Base directory for micro BDI data
        base_macro_dir: Base directory for macro BDI data
        agent_obj: Agent object for LLM calls
        logger_obj: Logger object for logging
        trigger: Trigger event name
        max_analysis_items: Maximum analysis items to include
        
    Returns:
        Path to generated micro_desire.yml or None if failed
    """
    if logger_obj:
        logger_obj.logger.info(f"Generating micro_desire for {agent} (trigger: {trigger})")
    
    # Setup paths
    agent_micro_dir = base_micro_dir / game_id / agent
    agent_macro_dir = base_macro_dir / game_id / agent
    
    analysis_path = agent_micro_dir / "analysis.yml"
    select_sentence_path = agent_micro_dir / "select_sentence.yml"
    macro_desire_path = agent_macro_dir / "macro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"
    macro_plan_path = agent_macro_dir / "macro_plan.yml"
    
    # Determine talk directory
    talk_dir = determine_talk_dir(agent_micro_dir)
    
    # Collect data
    try:
        # Analysis tail
        analysis_tail = extract_analysis_tail(analysis_path, max_analysis_items)
        if not analysis_tail and logger_obj:
            logger_obj.logger.warning(f"No analysis data found: {analysis_path}")
        
        # Latest select sentence
        latest_sentence = get_latest_select_sentence(select_sentence_path)
        if not latest_sentence and logger_obj:
            logger_obj.logger.info(f"No select sentence found: {select_sentence_path}")
        
        # Affinity trust data
        affinity_trust = collect_affinity_trust_data(talk_dir, logger_obj)
        
        # Macro data
        macro_desire_data = load_yaml_safe(macro_desire_path)
        macro_desire_summary = ""
        macro_desire_description = ""
        if "macro_desire" in macro_desire_data:
            md = macro_desire_data["macro_desire"]
            macro_desire_summary = str(md.get("summary", ""))
            macro_desire_description = str(md.get("description", ""))
        
        macro_belief_data = load_yaml_safe(macro_belief_path)
        desire_tendency = {}
        if "macro_belief" in macro_belief_data:
            mb = macro_belief_data["macro_belief"]
            desire_tendency = mb.get("desire_tendency", {})
            if isinstance(desire_tendency, dict) and "desire_tendencies" in desire_tendency:
                desire_tendency = desire_tendency["desire_tendencies"]
        
        macro_plan_data = load_yaml_safe(macro_plan_path)
        macro_plan_text = yaml.safe_dump(macro_plan_data, allow_unicode=True, sort_keys=False)
        
        if logger_obj:
            analysis_count = len(analysis_tail.split('\n')) if analysis_tail else 0
            logger_obj.logger.info(f"Collected data: analysis_items={analysis_count}, "
                                 f"sentence_len={len(latest_sentence)}, affinity_agents={len(affinity_trust)}")
        
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
        "macro_desire_summary": macro_desire_summary,
        "macro_desire_description": macro_desire_description,
        "desire_tendency": desire_tendency,
        "macro_plan_text": macro_plan_text,
        "analysis_tail": analysis_tail,
        "latest_sentence": latest_sentence,
        "affinity_trust": affinity_trust,
        "max_analysis_items": max_analysis_items
    }
    
    try:
        response = agent_obj.send_message_to_llm(
            "micro_desire",
            extra_vars=extra_vars,
            log_tag="micro_desire_generation"
        )
        
        if not response:
            if logger_obj:
                logger_obj.logger.warning("LLM returned empty response for micro_desire")
            response_data = _create_fallback_micro_desire()
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
                if not isinstance(response_data, dict) or "micro_desire" not in response_data:
                    if logger_obj:
                        logger_obj.logger.warning("Invalid LLM response format, using fallback")
                    response_data = _create_fallback_micro_desire()
                else:
                    if logger_obj:
                        logger_obj.logger.info("Successfully parsed LLM response")
            except yaml.YAMLError as e:
                if logger_obj:
                    logger_obj.logger.warning(f"Failed to parse LLM response as YAML: {e}, using fallback")
                response_data = _create_fallback_micro_desire()
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"LLM call failed: {e}")
        response_data = _create_fallback_micro_desire()
    
    # Add metadata
    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    response_data["meta"] = {
        "generated_at": datetime.now().isoformat(),
        "trigger": trigger or "unknown",
        "model": model_name,
        "game_id": game_id,
        "agent": agent
    }
    
    # Save to file
    output_path = agent_micro_dir / "micro_desire.yml"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_desire_generator\n")
            yaml.safe_dump(response_data, f, allow_unicode=True, sort_keys=False)
        
        if logger_obj:
            logger_obj.logger.info(f"Saved micro_desire to: {output_path}")
        
        return output_path
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_desire: {e}")
        return None


def _create_fallback_micro_desire() -> Dict[str, Any]:
    """Create fallback micro_desire structure when LLM fails."""
    return {
        "micro_desire": {
            "summary": "fallback",
            "intent": {
                "label": "meta",
                "priority": 0.5,
                "targets": []
            },
            "next_utterance": {
                "objective": "Maintain communication flow",
                "key_points": [
                    "Respond appropriately to current situation"
                ],
                "ask_if_not_addressed": [
                    "What are others thinking?"
                ]
            },
            "consistency": {
                "align_macro_desire": True,
                "align_desire_tendency": True,
                "align_macro_plan": True
            },
            "rationale": "LLM generation failed, using safe fallback"
        }
    }


def main():
    """CLI entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate micro_desire for agent")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--trigger", default="cli", help="Trigger event")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    path = generate_micro_desire_for_agent(
        game_id=args.game_id,
        agent=args.agent,
        trigger=args.trigger
    )
    
    if path:
        print(f"Generated: {path}")
        print(path.read_text(encoding="utf-8"))
    else:
        print("Failed to generate micro_desire")


if __name__ == "__main__":
    main()