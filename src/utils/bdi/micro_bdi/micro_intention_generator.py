#!/usr/bin/env python3
"""Micro intention generator - generates specific intention for next utterance.

状況適応的意図（micro_intention）生成モジュール.
- macro_belief.yml の role_social_duties（role, duties）と behavior/desire を参照
- 役職CO方針を算出して LLM に渡す（村人=真CO、人狼/狂人=真COしない、村特殊=開示度で可変）
- consist：micro_desire の response_to_selected / current_desire を両方達成する構成（1文）
- content：手持ち情報のうち「今回どれを開示するか」を具体に（1文）
- intention_log.yml は出力しない（micro_intention.yml のみ append する）
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
    """Skip regeneration only for talk_fallback if source micro_desire unchanged."""
    if trigger != "talk_fallback":
        return False
    if not micro_intention_path.exists():
        return False
    try:
        data = load_yaml_safe(micro_intention_path)
        existing_sha1 = data.get("meta", {}).get("source_micro_desire_sha1", "")
        current_sha1 = get_file_sha1(micro_desire_path)
        return bool(existing_sha1 and current_sha1 and existing_sha1 == current_sha1)
    except Exception as e:
        logger.warning(f"Skip check failed: {e}")
        return False


def _one_line(s: Any, limit: int = 140) -> str:
    """Make a single-line, trimmed string (~limit chars)."""
    if s is None:
        return ""
    t = str(s).replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    return t[:limit]


def _get_float(d: Dict[str, float], key: str, default: float = 0.5) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def _derive_disclosure_level(behavior_tendency: Dict[str, float], desire_tendency: Dict[str, float]) -> float:
    """
    簡易な開示度(0-1)を推定：外向/主張/自由/承認/刺激 (+)、内向/回避/安定/保守 (-)、適応(+小)、共感(-小)
    """
    def pick(key: str) -> float:
        return _get_float(behavior_tendency, key, _get_float(desire_tendency, key, 0.5))

    pos = (pick("extroversion") + pick("assertiveness") +
           pick("freedom_independence") + pick("social_approval") +
           pick("adventure_stimulation")) / 5.0
    neg = (pick("introversion") + pick("avoidant_behavior") +
           pick("stability") + pick("risk_averse") + pick("conformity")) / 5.0
    empathy = pick("empathy")
    adaptability = pick("adaptability")

    disclosure_raw = 0.5 + (pos - neg) * 0.8 + (adaptability - 0.5) * 0.1 - (empathy - 0.5) * 0.05
    return max(0.0, min(1.0, disclosure_raw))


def _compute_role_co_policy(role_name: str, disclosure: float) -> Dict[str, Any]:
    """
    役職ごとのCO方針:
      - Werewolf/Possessed: 真COしない。開示度高→偽CO可、低→CO回避
      - Villager: 真COする
      - Seer/Medium/Knight: 開示度でCO判断（高→CO寄り、低→回避）
      - Unknown: 慎重（高開示なら状況によりCO）
    """
    rn = (role_name or "").strip().lower()
    wolf_side = {"werewolf", "人狼", "possessed", "狂人"}
    villager = {"villager", "村人"}
    village_special = {"seer", "占い師", "medium", "霊媒師", "knight", "騎士"}

    should_true_co = False
    allow_fake_co = False
    fake_role_suggestion = "Seer"
    policy_note = ""

    if rn in wolf_side:
        should_true_co = False
        allow_fake_co = disclosure >= 0.6
        policy_note = "Wolf-side: do not reveal true role; consider fake claim if talkative."
    elif rn in villager:
        should_true_co = True
        allow_fake_co = False
        policy_note = "Villager: claim villager to reduce confusion."
    elif rn in village_special:
        should_true_co = disclosure >= 0.65
        allow_fake_co = False
        policy_note = "Village special: choose claim timing by disclosure; avoid fake claims."
    else:
        should_true_co = disclosure >= 0.8
        allow_fake_co = False
        policy_note = "Unknown role: default cautious; claim only if highly talkative and safe."

    return {
        "role_name": role_name,
        "should_true_co": should_true_co,
        "allow_fake_co": allow_fake_co,
        "fake_role_suggestion": fake_role_suggestion,
        "policy_note": policy_note,
    }


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
    if logger_obj:
        logger_obj.logger.info(f"Generating micro_intention for {agent} (trigger: {trigger})")
    
    agent_micro_dir = base_micro_dir / game_id / agent
    agent_macro_dir = base_macro_dir / game_id / agent

    micro_desire_path = agent_micro_dir / "micro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"
    macro_desire_path = agent_macro_dir / "macro_desire.yml"
    macro_plan_path = agent_macro_dir / "macro_plan.yml"
    analysis_path = agent_micro_dir / "analysis.yml"
    micro_intention_path = agent_micro_dir / "micro_intention.yml"

    # Skip check（talk_fallback時のみ）
    if should_skip_regeneration(micro_intention_path, micro_desire_path, trigger or ""):
        if logger_obj:
            logger_obj.logger.info(f"Skipping regeneration for {agent}: SHA1 unchanged")
        return micro_intention_path

    try:
        # ----- latest micro_desire -----
        micro_desire_text = ""
        latest_micro_desire: Dict[str, Any] = {}
        md_response_to_selected = ""
        md_current_desire = ""

        if micro_desire_path.exists():
            micro_desire_text = micro_desire_path.read_text(encoding='utf-8')
            try:
                md_all = yaml.safe_load(micro_desire_text) or {}
                if isinstance(md_all, dict) and isinstance(md_all.get("micro_desires"), list) and md_all["micro_desires"]:
                    latest_micro_desire = md_all["micro_desires"][-1] or {}
                    md_response_to_selected = _one_line(latest_micro_desire.get("response_to_selected"), 200)
                    md_current_desire = _one_line(latest_micro_desire.get("current_desire"), 200)
            except Exception:
                pass

        # ----- macro_belief: role/duties + tendencies -----
        role_name = ""
        role_duties = ""
        behavior_tendency: Dict[str, float] = {}
        desire_tendency: Dict[str, float] = {}

        mb_data = load_yaml_safe(macro_belief_path)
        if "macro_belief" in mb_data:
            mb = mb_data["macro_belief"]
            rsd = mb.get("role_social_duties", {}) or {}
            role_name = str(rsd.get("role", "")) or str(mb.get("role", ""))
            role_duties = str(rsd.get("duties", "")) or str(rsd.get("definition", ""))
            bt = mb.get("behavior_tendency", {}) or {}
            behavior_tendency = bt.get("behavior_tendencies") or bt or {}
            dt = mb.get("desire_tendency", {}) or {}
            desire_tendency = dt.get("desire_tendencies") or dt or {}

        disclosure = _derive_disclosure_level(behavior_tendency, desire_tendency)
        co_policy = _compute_role_co_policy(role_name, disclosure)

        # ----- optional: macro_desire / macro_plan / analysis tail -----
        mdz = load_yaml_safe(macro_desire_path)
        mplan = load_yaml_safe(macro_plan_path)
        macro_desire_summary = ""
        macro_desire_description = ""
        if isinstance(mdz.get("macro_desire"), dict):
            macro_desire_summary = str(mdz["macro_desire"].get("summary", ""))
            macro_desire_description = str(mdz["macro_desire"].get("description", ""))
        macro_plan_text = yaml.safe_dump(mplan, allow_unicode=True, sort_keys=False)

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
                "Collected for MI: "
                f"role='{role_name}', disclosure={disclosure:.2f}, "
                f"policy={co_policy.get('policy_note','')}, "
                f"md_resp='{md_response_to_selected[:60]}', md_cur='{md_current_desire[:60]}'"
            )

    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to collect reference data: {e}")
        return None

    # ----- LLM call -----
    if not agent_obj:
        if logger_obj:
            logger_obj.logger.error("No agent_obj provided for LLM call")
        return None

    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        # MUST targets
        "micro_desire_text": micro_desire_text,
        "latest_micro_desire": latest_micro_desire,
        "md_response_to_selected": md_response_to_selected,
        "md_current_desire": md_current_desire,
        # role/duties & tendencies
        "role_name": role_name,
        "role_duties": role_duties,
        "behavior_tendency": behavior_tendency,
        "desire_tendency": desire_tendency,
        "disclosure_level": f"{disclosure:.2f}",
        "role_co_policy": co_policy,
        # context
        "macro_desire_summary": macro_desire_summary,
        "macro_desire_description": macro_desire_description,
        "macro_plan_text": macro_plan_text,
        "analysis_tail": analysis_tail,
        "char_limits": {"base_length": 125, "mention_length": 125},
    }

    try:
        response = agent_obj.send_message_to_llm(
            "micro_intention",
            extra_vars=extra_vars,
            log_tag="micro_intention_generation"
        )
        if not response:
            response_data = _create_fallback_micro_intention()
        else:
            try:
                clean = response.strip()
                if clean.startswith("```yaml"): clean = clean[7:]
                elif clean.startswith("```"):   clean = clean[3:]
                if clean.endswith("```"):       clean = clean[:-3]
                clean = clean.strip()
                response_data = yaml.safe_load(clean) or {}
                if not isinstance(response_data, dict) or "micro_intention" not in response_data:
                    response_data = _create_fallback_micro_intention()
            except yaml.YAMLError:
                response_data = _create_fallback_micro_intention()
    except Exception:
        response_data = _create_fallback_micro_intention()

    # ----- Save entry to micro_intention.yml (no separate log file) -----
    model_name = type(agent_obj.llm_model).__name__ if agent_obj and hasattr(agent_obj, 'llm_model') and agent_obj.llm_model else "unknown"
    micro_desire_sha1 = get_file_sha1(micro_desire_path)
    micro_desire_mtime = get_file_mtime(micro_desire_path)

    mi_obj = response_data.get("micro_intention", {}) if isinstance(response_data, dict) else {}
    entry = {
        "consist": _one_line(mi_obj.get("consist"), 140),
        "content": _one_line(mi_obj.get("content"), 140),
        "md_response_to_selected": md_response_to_selected,
        "md_current_desire": md_current_desire,
        "role_name": role_name,
        "role_co_policy": co_policy,
    }

    try:
        micro_intention_path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = load_yaml_safe(micro_intention_path) if micro_intention_path.exists() else {}
        if not isinstance(existing, dict):
            existing = {}
        if "micro_intentions" not in existing or not isinstance(existing.get("micro_intentions"), list):
            existing["micro_intentions"] = []
        if "meta" not in existing or not isinstance(existing.get("meta"), dict):
            existing["meta"] = {"game_id": game_id, "agent": agent, "char_limits": {"mention": 125, "base": 125}}

        existing["meta"]["source_micro_desire_sha1"] = micro_desire_sha1
        existing["meta"]["source_micro_desire_mtime"] = micro_desire_mtime
        existing["meta"]["generated_at"] = datetime.now().isoformat()
        existing["meta"]["trigger"] = (trigger or "unknown")
        existing["meta"]["model"] = model_name

        existing["micro_intentions"].append(entry)

        tmp = micro_intention_path.with_suffix(".yml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_intention_generator (append-only)\n")
            yaml.safe_dump(existing, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, micro_intention_path)

        if logger_obj:
            logger_obj.logger.info(
                f"Appended micro_intention -> {micro_intention_path} "
                f"(total={len(existing['micro_intentions'])})"
            )

        return micro_intention_path
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_intention: {e}")
        return None


def _create_fallback_micro_intention() -> Dict[str, Any]:
    """Fallback minimal structure when LLM fails."""
    return {
        "micro_intention": {
            "consist": "Plan: address the selected content then state a one-line plan to achieve both targets within limits.",
            "content": "Info policy: disclose only minimal concrete facts now (e.g., 1 safe claim or a verified read); avoid risky role claims."
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
        print(Path(path).read_text(encoding="utf-8"))
    else:
        print("Failed to generate micro_intention")
