#!/usr/bin/env python3
"""Micro desire generator - generates situation-adaptive desires for next utterance."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .talk_history_init import determine_talk_dir

logger = logging.getLogger(__name__)

def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return {}

def collect_affinity_trust_data(talk_dir: Path, logger_obj=None) -> Dict[str, Dict[str, float]]:
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
    data = load_yaml_safe(analysis_path)
    items = data.get("items", [])
    if not items:
        return ""
    tail_items = items[-max_items:] if len(items) > max_items else items
    lines = []
    for i, item in enumerate(tail_items, 1):
        content = item.get("content", "")
        from_agent = item.get("from", "unknown")
        creditability = item.get("creditability", 0.0)
        lines.append(f"{i}. {from_agent}: {content} (cred: {creditability:.2f})")
    return "\n".join(lines)

def count_analysis_items(analysis_path: Path) -> int:
    data = load_yaml_safe(analysis_path)
    items = data.get("items")
    if isinstance(items, list):
        return len(items)
    # support legacy numeric-key dict
    try:
        keys = [int(k) for k in data.keys() if str(k).isdigit()]
        return len(keys)
    except Exception:
        return 0

def derive_phase_label(agent_obj, analysis_items_count: int) -> Dict[str, Any]:
    """
    Return dict with:
      phase_label: "early" | "mid" | "late"
      meta: thresholds and counts for logging
    Rules:
      - late: alive_count <= round(total_count/3)
      - mid : day >= 1
      - early: (day == 0) and (analysis_items_count <= early_threshold_by_size)
        where early_threshold_by_size = 7 (5p) / 16 (13p) / linear scale otherwise
    """
    day = 0
    total_count = 0
    alive_count = 0
    if agent_obj and getattr(agent_obj, "info", None) and getattr(agent_obj.info, "status_map", None):
        smap = agent_obj.info.status_map
        total_count = len(smap)
        alive_count = sum(1 for v in smap.values() if str(v) == "ALIVE" or getattr(v, "name", "") == "ALIVE")
        day = int(getattr(agent_obj.info, "day", 0) or 0)

    # late
    late_cut = max(1, round(total_count / 3)) if total_count else 0
    if total_count and alive_count <= late_cut:
        return {"phase_label": "late", "meta": {"day": day, "alive": alive_count, "total": total_count, "rule": "alive<=~1/3"}}

    # mid (day1+)
    if day >= 1:
        return {"phase_label": "mid", "meta": {"day": day, "alive": alive_count, "total": total_count, "rule": "day>=1"}}

    # early (day==0 & within threshold)
    if total_count <= 5:
        early_th = 7
    elif total_count >= 13:
        early_th = 16
    else:
        # linear between 5->7 and 13->16
        early_th = round(7 + (total_count - 5) * (16 - 7) / (13 - 5))
    if day == 0 and analysis_items_count <= early_th:
        return {"phase_label": "early", "meta": {"day": day, "alive": alive_count, "total": total_count, "early_threshold": early_th, "items": analysis_items_count}}

    # default to mid if unknown
    return {"phase_label": "mid", "meta": {"day": day, "alive": alive_count, "total": total_count, "items": analysis_items_count, "rule": "default->mid"}}

def _is_pending_text(text: str) -> bool:
    return isinstance(text, str) and text.strip().startswith("[PENDING]")

def _extract_cred(entry: dict) -> float:
    for k in ("creditability", "credibility"):
        v = entry.get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return 0.5

def _parse_to_targets(val) -> set[str]:
    if val is None: return set()
    if isinstance(val, list): return {str(x).strip() for x in val if str(x).strip()}
    s = str(val).strip()
    if not s: return set()
    if "," in s: return {t.strip() for t in s.split(",") if t.strip()}
    return {s}

def _is_addressed_to_self(to_val, agent_name: str) -> bool:
    targets = _parse_to_targets(to_val)
    if not targets: return False
    specials = {"all", "null"}
    if targets <= specials: return False
    def _norm(s: str) -> str: return str(s).strip()
    a = _norm(agent_name)
    return any(_norm(t) == a for t in targets)

def _load_latest5_entries(analysis_path: Path) -> list[dict]:
    data = load_yaml_safe(analysis_path)
    entries: list[dict] = []
    if not data: return entries
    if isinstance(data.get("items"), list):
        entries = [e for e in data["items"] if isinstance(e, dict)]
    else:
        numeric_keys = []
        for k in data.keys():
            try: numeric_keys.append(int(k))
            except Exception: continue
        numeric_keys.sort()
        for k in numeric_keys:
            e = data.get(k) or data.get(str(k))
            if isinstance(e, dict): entries.append(e)
    return entries[-5:]

def select_sentence_content_from_analysis(analysis_path: Path, agent_name: str) -> str:
    latest5 = _load_latest5_entries(analysis_path)
    if not latest5: return ""
    candidates = []
    for e in latest5:
        content = e.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _is_pending_text(content): continue
        candidates.append(e)
    if not candidates:
        for e in reversed(latest5):
            c = e.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
        return ""
    to_self = [e for e in candidates if _is_addressed_to_self(e.get("to"), agent_name)]
    def sort_key(e, recency_idx): return (_extract_cred(e), -recency_idx)
    if to_self:
        ranked = sorted([(e, i) for i, e in enumerate(candidates) if e in to_self],
                        key=lambda x: sort_key(x[0], x[1]))
        chosen = ranked[0][0]
        return str(chosen.get("content", "")).strip()
    ranked = sorted([(e, i) for i, e in enumerate(candidates)],
                    key=lambda x: sort_key(x[0], x[1]))
    return str(ranked[0][0].get("content", "")).strip()

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
    if logger_obj:
        logger_obj.logger.info(f"Generating micro_desire for {agent} (trigger: {trigger})")

    agent_micro_dir = base_micro_dir / game_id / agent
    agent_macro_dir = base_macro_dir / game_id / agent

    analysis_path = agent_micro_dir / "analysis.yml"
    macro_desire_path = agent_macro_dir / "macro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"
    macro_plan_path = agent_macro_dir / "macro_plan.yml"

    talk_dir = determine_talk_dir(agent_micro_dir)

    try:
        analysis_tail = extract_analysis_tail(analysis_path, max_analysis_items)
        analysis_items_count = count_analysis_items(analysis_path)
        affinity_trust = collect_affinity_trust_data(talk_dir, logger_obj)

        selected_sentence_text = select_sentence_content_from_analysis(analysis_path, agent) or ""

        # macro desire / belief / plan
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
        # extract current phase goal from macro_plan (early/mid/late)
        # 1) derive phase label using info + analysis length
        phase = derive_phase_label(agent_obj, analysis_items_count)
        phase_label = phase.get("phase_label", "mid")
        mp = macro_plan_data.get("macro_plan", macro_plan_data)
        phase_goal = ""
        if phase_label == "early":
            phase_goal = str(mp.get("early_game", ""))
        elif phase_label == "late":
            phase_goal = str(mp.get("late_game", ""))
        else:
            phase_goal = str(mp.get("mid_game", ""))

        macro_plan_text = yaml.safe_dump(macro_plan_data, allow_unicode=True, sort_keys=False)

        if logger_obj:
            logger_obj.logger.info(f"Phase={phase_label} meta={phase.get('meta')}, selected='{selected_sentence_text[:48]}...'")

    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to collect reference data: {e}")
        return None

    if not agent_obj:
        if logger_obj:
            logger_obj.logger.error("No agent_obj provided for LLM call")
        return None

    # Pass phase info explicitly so LLM doesn’t have to infer
    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        "macro_desire_summary": macro_desire_summary,
        "macro_desire_description": macro_desire_description,
        "desire_tendency": desire_tendency,
        "macro_plan_text": macro_plan_text,
        "analysis_tail": analysis_tail,
        "selected_sentence_text": selected_sentence_text,
        "affinity_trust": affinity_trust,
        "max_analysis_items": max_analysis_items,
        "phase_label": phase_label,
        "phase_goal": phase_goal,
    }

    try:
        response = agent_obj.send_message_to_llm(
            "micro_desire",
            extra_vars=extra_vars,
            log_tag="micro_desire_generation"
        )
        if not response:
            new_desire = _create_fallback_minimal_desire()
        else:
            try:
                clean = response.strip()
                if clean.startswith("```yaml"): clean = clean[7:]
                elif clean.startswith("```"):   clean = clean[3:]
                if clean.endswith("```"):       clean = clean[:-3]
                clean = clean.strip()
                data = yaml.safe_load(clean)
                if not isinstance(data, dict):
                    new_desire = _create_fallback_minimal_desire()
                else:
                    content = data.get("content")
                    response_to_selected = data.get("response_to_selected")
                    current_desire = data.get("current_desire", "")
                    if not current_desire or str(current_desire).strip() == "":
                        current_desire = "Act appropriately for the current phase."
                    new_desire = {
                        "content": None if content is None or str(content).lower() in ["null", "none", ""] else str(content),
                        "response_to_selected": None if response_to_selected is None or str(response_to_selected).lower() in ["null", "none", ""] else str(response_to_selected),
                        "current_desire": str(current_desire),
                        "phase_label": phase_label,
                        "phase_goal": phase_goal,
                        "timestamp": datetime.now().isoformat(),
                        "trigger": trigger or "unknown"
                    }
            except yaml.YAMLError:
                new_desire = _create_fallback_minimal_desire()
    except Exception:
        new_desire = _create_fallback_minimal_desire()

    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    new_desire.update({"model": model_name, "game_id": game_id, "agent": agent})

    output_path = agent_micro_dir / "micro_desire.yml"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_yaml_safe(output_path) if output_path.exists() else {}
        if "micro_desires" not in existing:
            existing["micro_desires"] = []
        existing["micro_desires"].append(new_desire)
        if len(existing["micro_desires"]) > 20:
            existing["micro_desires"] = existing["micro_desires"][-20:]
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_desire (append-only)\n")
            yaml.safe_dump(existing, f, allow_unicode=True, sort_keys=False)
        return output_path
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_desire: {e}")
        return None

def _create_fallback_minimal_desire() -> Dict[str, Any]:
    return {
        "content": None,
        "response_to_selected": None,
        "current_desire": "Contribute to game progress.",
        "phase_label": "mid",
        "phase_goal": "",
        "timestamp": datetime.now().isoformat(),
        "trigger": "fallback"
    }

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate micro_desire for agent")
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--trigger", default="cli", help="Trigger event")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    path = generate_micro_desire_for_agent(game_id=args.game_id, agent=args.agent, trigger=args.trigger)
    if path:
        print(f"Generated: {path}")
        print(path.read_text(encoding="utf-8"))
    else:
        print("Failed to generate micro_desire")

if __name__ == "__main__":
    main()
