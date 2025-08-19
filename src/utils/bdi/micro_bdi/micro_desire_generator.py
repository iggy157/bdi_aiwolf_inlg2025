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


def get_latest_micro_desire(micro_desire_path: Path) -> Dict[str, Any]:
    """Get the latest micro_desire entry from append-only format file."""
    data = load_yaml_safe(micro_desire_path)
    
    if isinstance(data, dict) and "micro_desires" in data and data["micro_desires"]:
        return data["micro_desires"][-1]  # Return latest entry
    
    return {}


def _is_pending_text(text: str) -> bool:
    """Check if text starts with [PENDING]."""
    return isinstance(text, str) and text.strip().startswith("[PENDING]")


def _extract_cred(entry: dict) -> float:
    """Extract creditability/credibility score from entry."""
    for k in ("creditability", "credibility"):
        v = entry.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return 0.5  # Default


def _parse_to_targets(val) -> set[str]:
    """Parse 'to' field value into set of target names."""
    if val is None:
        return set()
    if isinstance(val, list):
        return {str(x).strip() for x in val if str(x).strip()}
    s = str(val).strip()
    if not s:
        return set()
    if "," in s:
        return {t.strip() for t in s.split(",") if t.strip()}
    return {s}


def _is_addressed_to_self(to_val, agent_name: str) -> bool:
    """Check if message is addressed to self (excluding 'all'/'null')."""
    targets = _parse_to_targets(to_val)
    if not targets:  # Empty is not considered addressed to self
        return False
    specials = {"all", "null"}
    if targets <= specials:
        return False
    # Normalize names for comparison
    def _norm(s: str) -> str:
        return str(s).strip()
    a = _norm(agent_name)
    return any(_norm(t) == a for t in targets)


def _load_latest5_entries(analysis_path: Path) -> list[dict]:
    """Load latest 5 entries from analysis.yml, supporting both formats."""
    data = load_yaml_safe(analysis_path)
    entries: list[dict] = []
    if not data:
        return entries
    
    if isinstance(data.get("items"), list):
        # Format A: {"items": [...]}
        entries = [e for e in data["items"] if isinstance(e, dict)]
    else:
        # Format B: numeric key dictionary {1:{...}, 2:{...}}
        numeric_keys = []
        for k in data.keys():
            try:
                numeric_keys.append(int(k))
            except Exception:
                continue
        numeric_keys.sort()
        for k in numeric_keys:
            e = data.get(k) or data.get(str(k))
            if isinstance(e, dict):
                entries.append(e)
    
    # Return latest 5 entries
    return entries[-5:]


def select_sentence_content_from_analysis(analysis_path: Path, agent_name: str) -> str:
    """Select sentence content from latest 5 analysis entries.
    
    Priority:
    1. Messages addressed to self (excluding 'all'/'null')
    2. If none, messages with minimum creditability (newer if tied)
    
    Returns:
        Content string (guaranteed non-empty if analysis has any entries)
    """
    latest5 = _load_latest5_entries(analysis_path)
    if not latest5:
        return ""  # Only return empty if no analysis exists
    
    # Filter valid utterances (non-empty content, not pending)
    candidates = []
    for e in latest5:
        content = e.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _is_pending_text(content):
            continue
        candidates.append(e)
    
    if not candidates:
        # If no valid candidates, pick last non-empty content from any entry
        for e in reversed(latest5):
            c = e.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
        # Worst case: empty (rare if analysis exists)
        return ""
    
    # Priority 1: Messages addressed to self
    to_self = [e for e in candidates if _is_addressed_to_self(e.get("to"), agent_name)]
    
    def sort_key(e, recency_idx):
        # Sort by creditability (ascending), then by recency (descending)
        return (_extract_cred(e), -recency_idx)
    
    if to_self:
        # Sort messages to self by creditability, prefer newer if tied
        ranked = sorted([(e, i) for i, e in enumerate(candidates) if e in to_self], 
                       key=lambda x: sort_key(x[0], x[1]))
        chosen = ranked[0][0]
        return str(chosen.get("content", "")).strip()
    
    # Priority 2: All candidates by minimum creditability
    ranked = sorted([(e, i) for i, e in enumerate(candidates)], 
                   key=lambda x: sort_key(x[0], x[1]))
    chosen = ranked[0][0]
    return str(chosen.get("content", "")).strip()


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
        
        # Affinity trust data
        affinity_trust = collect_affinity_trust_data(talk_dir, logger_obj)
        
        # ★ Simple sentence selection from latest 5 analysis entries
        selected_sentence_text = select_sentence_content_from_analysis(analysis_path, agent)
        # Ensure we never pass None (always string)
        if selected_sentence_text is None:
            selected_sentence_text = ""
        
        if logger_obj:
            logger_obj.logger.info(f"Selected sentence content: '{selected_sentence_text[:50]}...'" if selected_sentence_text else "No content selected")
        
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
                                 f"selected_sentence_len={len(selected_sentence_text)}, affinity_agents={len(affinity_trust)}")
        
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
        "selected_sentence_text": selected_sentence_text,  # ← Important: content only
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
            new_desire = _create_fallback_minimal_desire()
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
                if not isinstance(response_data, dict):
                    if logger_obj:
                        logger_obj.logger.warning("Invalid LLM response format, using fallback")
                    new_desire = _create_fallback_minimal_desire()
                else:
                    # Extract minimal fields directly from response
                    # Handle null values properly for content and response_to_selected
                    content = response_data.get("content")
                    response_to_selected = response_data.get("response_to_selected")
                    current_desire = response_data.get("current_desire", "")
                    
                    # Ensure current_desire is never null/empty
                    if not current_desire or str(current_desire).strip() == "":
                        current_desire = "状況に応じて適切に行動する"
                    
                    new_desire = {
                        "content": None if content is None or str(content).lower() in ["null", "none", ""] else str(content),
                        "response_to_selected": None if response_to_selected is None or str(response_to_selected).lower() in ["null", "none", ""] else str(response_to_selected),
                        "current_desire": str(current_desire),
                        "timestamp": datetime.now().isoformat(),
                        "trigger": trigger or "unknown"
                    }
                    if logger_obj:
                        logger_obj.logger.info(f"Successfully parsed minimal micro_desire (content: {'found' if new_desire['content'] else 'null'})")
            except yaml.YAMLError as e:
                if logger_obj:
                    logger_obj.logger.warning(f"Failed to parse LLM response as YAML: {e}, using fallback")
                new_desire = _create_fallback_minimal_desire()
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"LLM call failed: {e}")
        new_desire = _create_fallback_minimal_desire()
    
    # Add metadata to the new desire entry
    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    new_desire.update({
        "model": model_name,
        "game_id": game_id,
        "agent": agent
    })
    
    # Save to file with append-only format
    output_path = agent_micro_dir / "micro_desire.yml"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data or create new structure
        existing_data = {}
        if output_path.exists():
            existing_data = load_yaml_safe(output_path)
        
        # Ensure micro_desires list exists
        if "micro_desires" not in existing_data:
            existing_data["micro_desires"] = []
        
        # Append new desire
        existing_data["micro_desires"].append(new_desire)
        
        # Keep only last 20 entries to prevent file bloat
        if len(existing_data["micro_desires"]) > 20:
            existing_data["micro_desires"] = existing_data["micro_desires"][-20:]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_desire_generator (append-only format)\n")
            yaml.safe_dump(existing_data, f, allow_unicode=True, sort_keys=False)
        
        if logger_obj:
            logger_obj.logger.info(f"Appended micro_desire to: {output_path} (total entries: {len(existing_data['micro_desires'])})")
        
        return output_path
        
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_desire: {e}")
        return None


def _create_fallback_minimal_desire() -> Dict[str, Any]:
    """Create fallback minimal micro_desire structure when LLM fails."""
    return {
        "content": None,  # No content found
        "response_to_selected": None,  # No response since no content
        "current_desire": "ゲーム進行への貢献",  # Always required
        "timestamp": datetime.now().isoformat(),
        "trigger": "fallback"
    }




def _create_fallback_micro_desire() -> Dict[str, Any]:
    """Create fallback micro_desire structure when LLM fails (legacy compatibility)."""
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