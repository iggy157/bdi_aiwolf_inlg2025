#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_plan 生成器（プロンプトは config.yml 管理／禁止キーは絶対に出力しない）

変更点（CO方針の決定を LLM 主導に）:
- 「真CO > COしない > 偽CO」の信頼度優先順位（evidence/credibility priority）を LLM に提示。
- ROLE_POLICY は“既定（ソフト）”として渡すのみで、コード側で co_policy を強制修正しない。
- LLM が behavior_tendencies / desire_tendencies / macro_desire(summary|description) を総合し、
  真CO / COしない / 偽CO の三択から最適を選んだ上で、policy文として表現する。

出力仕様（禁止キーは出力しない）:
  macro_plan:
    strategy_summary: "<non-conversational strategic summary>"
    policies:
      co_policy: "<role-claim policy reflecting a chosen option: true_co / no_co / fake_co>"
      results_policy: "<how/when to share special ability results>"
      analysis_policy: "<how to analyze>"
      persuasion_policy: "<how to persuade / drive consensus>"
      vote_policy: "<how to decide/finalize vote>"

禁止キー:
  - self_introduction, information_sharing, reasoning_analysis,
    discussion_persuasion, voting_decision
  - early_game, mid_game, late_game
"""

from __future__ import annotations

import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from jinja2 import Template

# ==== パス定義 ====
BASE = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
MB_PATH_FMT = BASE / "info/bdi_info/macro_bdi" / "{game_id}" / "{agent}" / "macro_belief.yml"
MD_PATH_FMT = BASE / "info/bdi_info/macro_bdi" / "{game_id}" / "{agent}" / "macro_desire.yml"
CFG_PATH    = BASE / "config" / "config.yml"

# ==== 役職正規化（日本語→英語）と陣営 ====
ROLE_JA2CANON = {
    "占い師": "seer",
    "霊媒師": "medium",
    "狩人": "bodyguard", "騎士": "bodyguard", "守護": "bodyguard",
    "村人": "villager",
    "人狼": "werewolf",
    "狂人": "possessed",
}
ROLE_CANON_SET = {"seer","medium","bodyguard","villager","werewolf","possessed"}

def _canonical_role(role_ja_or_en: str | None) -> str | None:
    if not role_ja_or_en:
        return None
    s = str(role_ja_or_en).strip().lower()
    for ja, en in ROLE_JA2CANON.items():
        if ja in s:
            return en
    if s in ROLE_CANON_SET:
        return s
    return None

def _alignment_for_role(canon: str | None) -> str | None:
    if canon is None:
        return None
    return "werewolf" if canon in ("werewolf","possessed") else "villager"

# ==== 既定（ソフト）COポリシー ====
ROLE_POLICY_DEFAULTS = {
    "seer":       dict(should_true_co=True,  allow_fake_co=False),
    "medium":     dict(should_true_co=True,  allow_fake_co=False),
    "bodyguard":  dict(should_true_co=False, allow_fake_co=False),
    "villager":   dict(should_true_co=False, allow_fake_co=False),
    "werewolf":   dict(should_true_co=False, allow_fake_co=True ),
    "possessed":  dict(should_true_co=False, allow_fake_co=True ),
}

# ==== IO ヘルパ ====
def _load_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _atomic_write_yaml(obj: Dict[str, Any], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        f.flush()
    tmp.replace(dst)

def _render(tmpl: str, vars: Dict[str, Any]) -> str:
    return Template(tmpl).render(**vars).strip()

# ==== パースユーティリティ ====
def _extract_yaml(resp: str) -> Dict[str, Any]:
    if not resp:
        return {}
    m = re.search(r"```(?:yaml|yml)?\s*\r?\n([\s\S]*?)\r?\n```", resp, re.IGNORECASE)
    if m:
        try:
            return yaml.safe_load(m.group(1).strip()) or {}
        except yaml.YAMLError:
            pass
    try:
        return yaml.safe_load(resp.strip()) or {}
    except yaml.YAMLError:
        pass
    i = resp.lower().find("macro_plan:")
    if i >= 0:
        try:
            return yaml.safe_load(resp[i:]) or {}
        except yaml.YAMLError:
            pass
    return {}

FORBIDDEN_KEYS = {
    "self_introduction", "information_sharing", "reasoning_analysis",
    "discussion_persuasion", "voting_decision", "early_game", "mid_game", "late_game"
}

def _strip_forbidden_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """誤って含まれた禁止キーを除去（深さ1想定）。"""
    mp = d.get("macro_plan", {})
    if isinstance(mp, dict):
        for k in list(mp.keys()):
            if k in FORBIDDEN_KEYS:
                mp.pop(k, None)
        pol = mp.get("policies", {})
        if isinstance(pol, dict):
            for k in list(pol.keys()):
                if k in FORBIDDEN_KEYS:
                    pol.pop(k, None)
        d["macro_plan"] = mp
    return d

def _ensure_shape(d: Dict[str, Any], role: str, alignment: str) -> Dict[str, Any]:
    """期待スキーマに整形しつつ、最低限の保険を入れる（LLM決定は尊重）。"""
    mp = d.get("macro_plan", {}) if isinstance(d, dict) else {}
    pol = mp.get("policies", {}) if isinstance(mp, dict) else {}

    strategy_summary = mp.get("strategy_summary", "") if isinstance(mp, dict) else ""
    if not isinstance(strategy_summary, str):
        strategy_summary = ""

    def _get_str(key: str) -> str:
        v = pol.get(key, "")
        return v if isinstance(v, str) else ""

    result = {
        "macro_plan": {
            "strategy_summary": strategy_summary.strip(),
            "policies": {
                "co_policy": _get_str("co_policy"),
                "results_policy": _get_str("results_policy"),
                "analysis_policy": _get_str("analysis_policy"),
                "persuasion_policy": _get_str("persuasion_policy"),
                "vote_policy": _get_str("vote_policy"),
            }
        }
    }

    # Seer/Medium の結果共有方針が空/薄いときの軽い保険のみ（CO意思決定には介入しない）
    if role in ("seer", "medium"):
        rp = result["macro_plan"]["policies"]["results_policy"]
        if not isinstance(rp, str) or not rp.strip():
            result["macro_plan"]["policies"]["results_policy"] = (
                "When a valid result exists, publish the latest result (target + judgement) promptly and clearly."
            )

    # 陣営目標の最低限の明示（vote_policy の末尾に付加・重複は避ける）
    vp = result["macro_plan"]["policies"]["vote_policy"]
    low = (vp or "").lower()
    if alignment == "villager":
        if ("village" not in low) and ("werewolf" not in low) and ("wolf" not in low):
            result["macro_plan"]["policies"]["vote_policy"] = (vp.rstrip(".") + " Aim toward the village victory condition.").strip()
    else:
        if ("werewolf" not in low) and ("wolf" not in low) and ("village" not in low):
            result["macro_plan"]["policies"]["vote_policy"] = (vp.rstrip(".") + " Aim toward the werewolf victory condition.").strip()

    return result

# ==== メイン ====
def generate_macro_plan(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool=False,
    overwrite: bool=False
) -> Dict[str, Any]:
    """macro_belief.yml / macro_desire.yml を参照し、summary+policies だけの macro_plan を生成。"""
    out_path = BASE / "info/bdi_info/macro_bdi" / game_id / agent / "macro_plan.yml"
    if out_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output exists: {out_path}. Use overwrite=True.")

    # 入力
    mb_path = Path(str(MB_PATH_FMT).format(game_id=game_id, agent=agent))
    md_path = Path(str(MD_PATH_FMT).format(game_id=game_id, agent=agent))
    cfg = _load_yaml(CFG_PATH)

    mb_all = _load_yaml(mb_path)
    mb = mb_all.get("macro_belief", {}) or {}

    # 役職解決（日本語→英語）
    role_ja = None
    if isinstance(mb.get("role_social_duties"), dict):
        role_ja = mb["role_social_duties"].get("role")
    if not role_ja:
        role_ja = mb.get("role") or mb.get("role_name")
    role_canon = _canonical_role(role_ja) or "villager"
    alignment = _alignment_for_role(role_canon) or "villager"

    behavior = (mb.get("behavior_tendency", {}) or {}).get("behavior_tendencies") or {}
    desire_tendencies = (mb.get("desire_tendency", {}) or {}).get("desire_tendencies") or {}

    # macro_desire
    md_all = _load_yaml(md_path)
    md_obj = md_all.get("macro_desire", {}) or {}
    md_summary = str(md_obj.get("summary") or "")
    md_description = str(md_obj.get("description") or "")
    desire_loaded = bool(md_summary or md_description)

    # プロンプト取得
    prompt_tmpl = (cfg.get("prompt", {}) or {}).get("macro_plan", "")
    if not prompt_tmpl:
        raise RuntimeError("prompt.macro_plan is missing in config.yml. Please add it to config.")

    # CO意思決定フレーム（LLMに渡す）
    co_decision_framework = {
        "options": ["true_co", "no_co", "fake_co"],
        "credibility_priority": ["true_co", "no_co", "fake_co"],  # 高信頼 → 低信頼
        "notes": (
            "Prefer options that maximize expected credibility and strategic value "
            "while staying consistent with tendencies and macro_desire."
        ),
    }

    # 変数束
    prompt_vars = {
        "game_id": game_id,
        "agent": agent,
        "role": role_canon,
        "alignment": alignment,
        "md_summary": md_summary,
        "md_description": md_description,
        "behavior_tendencies": behavior or {},
        "desire_tendencies": desire_tendencies or {},
        # 既定は“参考情報”。LLM は tendencies / macro_desire を優先し、必要なら上書き判断してよい。
        "role_policy_defaults": ROLE_POLICY_DEFAULTS.get(role_canon, ROLE_POLICY_DEFAULTS["villager"]),
        # CO意思決定の優先順位（真CO>無CO>偽CO）
        "co_decision_framework": co_decision_framework,
    }
    prompt = _render(prompt_tmpl, prompt_vars)
    if dry_run:
        print("---- PROMPT (macro_plan) ----")
        print(prompt)

    if agent_obj is None:
        raise ValueError("agent_obj is required.")

    # LLM 呼び出し
    resp = agent_obj.send_message_to_llm(
        "macro_plan",
        extra_vars=prompt_vars,
        log_tag="MACRO_PLAN_POLICIES",
        use_shared_history=False
    )

    # パース＆正規化（LLM判断を尊重し、禁止キーは剥がす）
    parsed = _extract_yaml(resp or "")
    parsed = _strip_forbidden_keys(parsed)
    final_plan = _ensure_shape(parsed, role_canon, alignment)

    final = {
        **final_plan,
        "meta": {
            "game_id": game_id,
            "agent": agent,
            "role_raw": role_ja,
            "role_canon": role_canon,
            "alignment": alignment,
            "desire_loaded": desire_loaded,
            "model": (agent_obj.config.get("openai", {}).get("model")
                      or agent_obj.config.get("google", {}).get("model")
                      or agent_obj.config.get("ollama", {}).get("model")),
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
            "source_macro_belief": str(mb_path),
            "source_macro_desire": str(md_path),
            "format": "summary_and_policies_only",
            "forbidden_keys_removed": True
        }
    }

    if dry_run:
        print("---- RESULT ----")
        print(yaml.safe_dump(final, allow_unicode=True, sort_keys=False))
        return final

    _atomic_write_yaml(final, out_path)
    print(f"Saved macro_plan: {out_path}")
    return final


def main():
    print("This module is intended to be called from the Agent runtime.")
    return 0


if __name__ == "__main__":
    exit(main())
