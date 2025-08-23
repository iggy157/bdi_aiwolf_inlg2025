#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_plan 生成器（プロンプトは config.yml 管理／コード内に埋め込まない版）

変更点:
- 出力フェーズを early/mid/late から以下の5区分に変更
  * self_introduction: Self-introduction (including role claims)
  * information_sharing: Information sharing and role-related updates (special ability results, etc.)
  * reasoning_analysis: Reasoning and analysis
  * discussion_persuasion: Discussion and persuasion
  * voting_decision: Voting decision
- 互換性維持のため、上記5区分から early_game/mid_game/late_game を合成して
  レガシー別名(legacy bridge)としても出力

生成方針:
- macro_belief.yml / macro_desire.yml を参照して5区分を各1文で生成
- 役職は macro_belief.role_social_duties.role（日本語）を最優先で英語に正規化
- LLM 出力の形式ズレ（辞書返却など）は一文に合成して救済
- Seer/Medium:
    * self_introduction では結果語（白/黒/人間/人狼/result/flip 等）を含めない
    * information_sharing では「結果があるときは最新の対象+判定を開示」を強制補完
- voting_decision: 陣営勝利条件への言及を強制補完
- 村/狩: 真CO抑止、狼/狂: 真CO禁止（偽COは条件付き）
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

# ==== COポリシー（統一・コード内） ====
ROLE_POLICY = {
    "seer":       dict(should_true_co=True,  allow_fake_co=False),
    "medium":     dict(should_true_co=True,  allow_fake_co=False),
    "bodyguard":  dict(should_true_co=False, allow_fake_co=False),
    "villager":   dict(should_true_co=False, allow_fake_co=False),
    "werewolf":   dict(should_true_co=False, allow_fake_co=True ),
    "possessed":  dict(should_true_co=False, allow_fake_co=True ),
}

# ==== 新5区分の骨子（ヒント & フォールバック合成に使用） ====
BASE5: Dict[str, Dict[str, str]] = {
    "seer": {
        "self_introduction":     "Brief self-intro; clarify analytical stance; state non-claim policy for now.",
        "information_sharing":   "When a seer result exists, true-claim and reveal the latest check with target and judgement promptly.",
        "reasoning_analysis":    "Name a suspicious player with explicit logic referencing result consistency and talk-history.",
        "discussion_persuasion": "Coordinate with likely allies to align checks, counter-claims, and consolidate narratives.",
        "voting_decision":       "Fix the elimination target based on confirmed info and push consensus toward the village victory condition.",
    },
    "medium": {
        "self_introduction":     "Brief self-intro; clarify careful verification stance; keep claim policy stated without outing.",
        "information_sharing":   "When a flip result exists, true-claim and reveal it promptly with the concrete wording of target and judgement.",
        "reasoning_analysis":    "Advance a suspect using flip-consistent logic and contradiction spotting.",
        "discussion_persuasion": "Resolve claim conflicts efficiently and coordinate confirmations.",
        "voting_decision":       "Select the elimination target based on confirmed flips and secure votes toward the village victory condition.",
    },
    "bodyguard": {
        "self_introduction":     "Brief self-intro emphasizing cautious protection; explicitly prefer non-claim policy.",
        "information_sharing":   "Do not reveal protection details; keep patterns opaque while sharing safe inferences.",
        "reasoning_analysis":    "Name a suspect while prioritizing power-role safety and risk assessment.",
        "discussion_persuasion": "Quietly steer discussion to reduce PR exposure and support stable verification.",
        "voting_decision":       "Allocate votes that maximize PR survival and progress toward the village victory condition.",
    },
    "villager": {
        "self_introduction":     "Brief self-intro clarifying approach; prefer non-claim policy.",
        "information_sharing":   "Synthesize public information rather than special results; share testable observations.",
        "reasoning_analysis":    "Present a concrete suspect with explicit reasons grounded in claims and consistency.",
        "discussion_persuasion": "Drive consensus by comparing alternatives and addressing counter-arguments.",
        "voting_decision":       "Lock in the elimination target with crisp rationale toward the village victory condition.",
    },
    "werewolf": {
        "self_introduction":     "Light self-intro to build cover; do not true-claim; consider conditional fake-CO only if beneficial.",
        "information_sharing":   "No true results; if feigning, keep it plausible and reactive to public info.",
        "reasoning_analysis":    "Redirect suspicion plausibly by presenting an alternative narrative.",
        "discussion_persuasion": "Shape the talk flow to shield allies and create favorable momentum.",
        "voting_decision":       "Engineer votes to maximize wolf odds and secure the werewolf victory condition.",
    },
    "possessed": {
        "self_introduction":     "Calm self-intro to earn trust with low heat; do not true-claim; consider situational fake-CO.",
        "information_sharing":   "No true results; seed controlled misreads only when safe.",
        "reasoning_analysis":    "Insert a misread on a low-credibility target with some rationale.",
        "discussion_persuasion": "Assist wolves indirectly by steering frames and tempo.",
        "voting_decision":       "Steer votes to sustain misreads and enable the werewolf victory condition.",
    },
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

# ==== 文字列ユーティリティ ====
def _dedupe_tokens(s: str) -> str:
    if not s: return s
    s = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", s, flags=re.IGNORECASE)
    s = s.replace("non-state", "non-claim")
    s = re.sub(r"\bpolicy(\s+policy)+\b", "policy", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

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

NEW_KEYS = [
    "self_introduction",
    "information_sharing",
    "reasoning_analysis",
    "discussion_persuasion",
    "voting_decision",
]

def _normalize_to_five_lines(data: Dict[str, Any], role: str) -> Dict[str, str]:
    """LLM応答を5区分・各1文に正規化（辞書返却時は合成）。"""
    mp = data.get("macro_plan", {})
    out = {k: "" for k in NEW_KEYS}
    for k in NEW_KEYS:
        v = mp.get(k, "")
        if isinstance(v, str):
            out[k] = _dedupe_tokens(v.strip())
        elif isinstance(v, dict):
            parts = [str(x).strip().rstrip(".") for x in v.values() if str(x).strip()]
            out[k] = _dedupe_tokens(("; ".join(parts) + ".") if parts else "")
        else:
            out[k] = ""
        if not out[k]:
            base = BASE5.get(role, BASE5["villager"]).get(k, "")
            out[k] = _dedupe_tokens(base)
    return out

def _has_any(s: str, terms: tuple[str, ...]) -> bool:
    low = (s or "").lower()
    return any(t in low for t in terms)

RESULT_TERMS = (
    "reveal","result","flip","divine","seer","medium","判定","結果","白","黒","人狼","人間"
)

def _enforce_alignment(line: str, alignment: str) -> str:
    s = (line or "").lower()
    if alignment == "villager":
        if ("village" not in s) and ("villager" not in s) and ("werewolf" not in s):
            return (line.rstrip(".") + " toward the village victory condition.")
    else:
        if ("werewolf" not in s) and ("wolf" not in s) and ("village" not in s):
            return (line.rstrip(".") + " toward the werewolf victory condition.")
    return line

def _validate_and_fix5(plan5: Dict[str,str], role: str, alignment: str) -> Dict[str,str]:
    si = plan5["self_introduction"]
    is_ = plan5["information_sharing"]
    ra = plan5["reasoning_analysis"]
    dp = plan5["discussion_persuasion"]
    vd = plan5["voting_decision"]

    # Seer/Medium: self_introduction で結果語を消し、information_sharing に結果開示を保証
    if role in ("seer","medium"):
        if _has_any(si, RESULT_TERMS):
            si = re.sub(r"(?i)\b(reveal|result|flip|divine|seer|medium)\b.*?(?:\.|;|$)", "", si).strip()
            si = re.sub(r"\s{2,}", " ", si).rstrip(";").strip()
            if si and not si.endswith("."):
                si += "."
        if not _has_any(is_, ("reveal","result","flip","divine","判定","結果","白","黒","人狼","人間")):
            is_ = (is_.rstrip(".") + "; when a result exists, reveal the latest result (target and judgement) promptly.").strip()

    # 村/狩：不用意な真COを抑止
    if role in ("villager","bodyguard") and _has_any(si, ("true-claim","trueclaim"," claim")):
        si = re.sub(r"(?i)\btrue-?claim\b|\bclaim\b", "state non-claim policy", si)

    # 狼/狂：真CO禁止（偽COは条件付き）
    if role in ("werewolf","possessed") and _has_any(si, ("true-claim","trueclaim")):
        si = re.sub(r"(?i)\btrue-?claim\b", "do not true-claim", si)

    # voting_decision：陣営勝利条件の明示を保証
    vd = _enforce_alignment(vd, alignment)

    return {
        "self_introduction":     _dedupe_tokens(si),
        "information_sharing":   _dedupe_tokens(is_),
        "reasoning_analysis":    _dedupe_tokens(ra),
        "discussion_persuasion": _dedupe_tokens(dp),
        "voting_decision":       _dedupe_tokens(vd),
    }

def _compose_legacy_bridge(plan5: Dict[str,str], alignment: str) -> Dict[str,str]:
    """新5区分から early/mid/late を一文合成（互換レイヤ）。"""
    early = plan5["self_introduction"]
    mid   = "; ".join(
        [x.rstrip(".") for x in (plan5["information_sharing"], plan5["reasoning_analysis"]) if x.strip()]
    ).strip()
    if mid and not mid.endswith("."):
        mid += "."
    late  = "; ".join(
        [x.rstrip(".") for x in (plan5["discussion_persuasion"], plan5["voting_decision"]) if x.strip()]
    ).strip()
    if late and not late.endswith("."):
        late += "."
    late = _enforce_alignment(late, alignment)

    return {
        "early_game": _dedupe_tokens(early),
        "mid_game":   _dedupe_tokens(mid),
        "late_game":  _dedupe_tokens(late),
    }

# ==== メイン ====
def generate_macro_plan(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool=False,
    overwrite: bool=False
) -> Dict[str, Any]:
    """macro_belief.yml / macro_desire.yml を参照して macro_plan（5区分・各1文）を生成。"""
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
    role_source = "macro_belief.role_social_duties.role" if role_ja else "fallback(villager)"
    alignment = _alignment_for_role(role_canon) or "villager"

    behavior = (mb.get("behavior_tendency", {}) or {}).get("behavior_tendencies") or {}
    desire_tendencies = (mb.get("desire_tendency", {}) or {}).get("desire_tendencies") or {}

    # macro_desire
    md_all = _load_yaml(md_path)
    md_obj = md_all.get("macro_desire", {}) or {}
    md_summary = str(md_obj.get("summary") or "")
    md_description = str(md_obj.get("description") or "")
    desire_loaded = bool(md_summary or md_description)

    # プロンプト取得（コード内にフォールバック無し）
    prompt_tmpl = (cfg.get("prompt", {}) or {}).get("macro_plan", "")
    if not prompt_tmpl:
        raise RuntimeError("prompt.macro_plan is missing in config.yml. Please add it to config.")

    # 5区分骨子（ヒントとして渡す）
    section_hints = BASE5.get(role_canon, BASE5["villager"])

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
        "role_policy": ROLE_POLICY.get(role_canon, ROLE_POLICY["villager"]),
        # ヒント（LLMに渡すが、そのまま出力させない）
        "section_hints": section_hints,
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
        log_tag="MACRO_PLAN_FIVE_SECTIONS",
        use_shared_history=False
    )

    # パース＆正規化（5区分へ）
    parsed = _extract_yaml(resp or "")
    plan5 = _normalize_to_five_lines(parsed, role_canon)
    fixed5 = _validate_and_fix5(plan5, role_canon, alignment)

    # 互換レイヤ（legacy bridge）
    legacy3 = _compose_legacy_bridge(fixed5, alignment)

    final = {
        "macro_plan": {
            **fixed5,
            # 互換キー（既存モジュール用）
            **legacy3,
        },
        "meta": {
            "game_id": game_id,
            "agent": agent,
            "role_raw": role_ja,
            "role_canon": role_canon,
            "role_source": role_source,
            "alignment": alignment,
            "desire_loaded": desire_loaded,
            "model": (agent_obj.config.get("openai", {}).get("model")
                      or agent_obj.config.get("google", {}).get("model")
                      or agent_obj.config.get("ollama", {}).get("model")),
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat()+"Z",
            "source_macro_belief": str(mb_path),
            "source_macro_desire": str(md_path),
            "format": "five_sections + legacy_bridge",
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
