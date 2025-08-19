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

    # micro_desire は micro_bdi 側に保存される
    micro_desire_path = agent_micro_dir / "micro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"
    macro_desire_path = agent_macro_dir / "macro_desire.yml"
    macro_plan_path = agent_macro_dir / "macro_plan.yml"
    analysis_path = agent_micro_dir / "analysis.yml"
    micro_intention_path = agent_micro_dir / "micro_intention.yml"
    
    # Check if we should skip regeneration
    if should_skip_regeneration(micro_intention_path, micro_desire_path, trigger or ""):
        if logger_obj:
            logger_obj.logger.info(f"Skipping regeneration for {agent}: SHA1 unchanged")
        return micro_intention_path
    
    # Collect data
    try:
        # Load latest micro_desire (append-only format expected)
        micro_desire_text = ""
        latest_micro_desire: Dict[str, Any] = {}
        if micro_desire_path.exists():
            micro_desire_text = micro_desire_path.read_text(encoding='utf-8')
            try:
                md_all = yaml.safe_load(micro_desire_text) or {}
                if isinstance(md_all, dict) and isinstance(md_all.get("micro_desires"), list) and md_all["micro_desires"]:
                    latest_micro_desire = md_all["micro_desires"][-1]
            except Exception:
                pass
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

        # Load macro_desire/macro_plan for fallback
        macro_desire_data = load_yaml_safe(macro_desire_path)
        macro_plan_data = load_yaml_safe(macro_plan_path)
        macro_desire_summary = ""
        macro_desire_description = ""
        if isinstance(macro_desire_data.get("macro_desire"), dict):
            macro_desire_summary = str(macro_desire_data["macro_desire"].get("summary", ""))
            macro_desire_description = str(macro_desire_data["macro_desire"].get("description", ""))
        macro_plan_text = yaml.safe_dump(macro_plan_data, allow_unicode=True, sort_keys=False)

        # analysis tail (末尾12件) を簡易抽出
        analysis_tail = ""
        try:
            if analysis_path.exists():
                adata = load_yaml_safe(analysis_path)
                if isinstance(adata, dict) and "items" in adata:
                    items = adata.get("items") or []
                    tail = items[-12:] if len(items) > 12 else items
                    lines = []
                    for i, it in enumerate(tail, 1):
                        if isinstance(it, dict):
                            lines.append(f"{i}. {it.get('from','unknown')}: {it.get('content','')}")
                    analysis_tail = "\n".join(lines)
                else:
                    # 数値キー型にも緩く対応
                    keys = [int(k) for k in adata.keys() if str(k).isdigit()]
                    keys.sort()
                    tail_keys = keys[-12:] if len(keys) > 12 else keys
                    lines = []
                    for i, k in enumerate(tail_keys, 1):
                        it = adata.get(k) or adata.get(str(k)) or {}
                        if isinstance(it, dict):
                            lines.append(f"{i}. {it.get('from','unknown')}: {it.get('content','')}")
                    analysis_tail = "\n".join(lines)
        except Exception:
            analysis_tail = ""

        if logger_obj:
            logger_obj.logger.info(
                f"Collected data: latest_micro_desire={bool(latest_micro_desire)}, "
                f"behavior_tendency_keys={len(behavior_tendency)}, "
                f"analysis_tail_len={len(analysis_tail)}"
            )

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
        "latest_micro_desire": latest_micro_desire,
        "behavior_tendency": behavior_tendency,
        "macro_desire_summary": macro_desire_summary,
        "macro_desire_description": macro_desire_description,
        "macro_plan_text": macro_plan_text,
        "analysis_tail": analysis_tail,
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
    
    # 構築：保存用の最小エントリ（2項目のみ）
    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    micro_desire_sha1 = get_file_sha1(micro_desire_path)
    micro_desire_mtime = get_file_mtime(micro_desire_path)
    mi_obj = response_data.get("micro_intention", {}) if isinstance(response_data, dict) else {}
    entry = {
        "consist": str(mi_obj.get("consist", "Mention target briefly then state one-line plan.")),
        "content": str(mi_obj.get("content", ""))
    }
    
    # Save to file
    try:
        micro_intention_path.parent.mkdir(parents=True, exist_ok=True)
        # 既存読込
        existing: Dict[str, Any] = load_yaml_safe(micro_intention_path) if micro_intention_path.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        if "micro_intentions" not in existing or not isinstance(existing.get("micro_intentions"), list):
            existing["micro_intentions"] = []
        if "meta" not in existing or not isinstance(existing.get("meta"), dict):
            existing["meta"] = {
                "game_id": game_id,
                "agent": agent,
                "char_limits": {"mention": 125, "base": 125}
            }
        # SHA1/mtime を meta に更新（skip 判定のため最新を保持）
        existing["meta"]["source_micro_desire_sha1"] = micro_desire_sha1
        existing["meta"]["source_micro_desire_mtime"] = micro_desire_mtime
        existing["meta"]["generated_at"] = datetime.now().isoformat()
        existing["meta"]["trigger"] = (trigger or "unknown")
        existing["meta"]["model"] = model_name
        # 追記
        existing["micro_intentions"].append(entry)
        # 原子的書込み
        tmp = micro_intention_path.with_suffix(".yml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_intention_generator (append-only)\n")
            yaml.safe_dump(existing, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, micro_intention_path)

        if logger_obj:
            logger_obj.logger.info(f"Appended micro_intention -> {micro_intention_path} (total={len(existing['micro_intentions'])})")

        return micro_intention_path
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_intention: {e}")
        return None



def _create_fallback_micro_intention() -> Dict[str, Any]:
    """Create fallback micro_intention structure when LLM fails."""
    return {
        "micro_intention": {
            "consist": "Direct communication without specific targeting",
            "content": "I will share my current thoughts and observe others"
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