#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macro_plan 生成器（プロンプトは config.yml 管理／コード内に埋め込まない版）

機能概要:
- macro_belief.yml / macro_desire.yml を読み込んで、early/mid/late を各1文で生成
- 役職は macro_belief.role_social_duties.role（日本語）を最優先で英語に正規化
- LLM 出力の形式ズレ（辞書返却など）は一文に合成して救済
- Seer/Medium: early 結果禁止、mid 結果開示（検証で強制）
- late: 陣営勝利条件を強制補完
- 村/狩: 真CO抑止、狼/狂: 真CO禁止・偽COは条件付き

プロンプトは config.yml の prompt.macro_plan を使用（未定義なら RuntimeError）。
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

# ==== ヒント用のベース骨子（LLMへの補助 & フォールバック合成に使用） ====
BASE_ITEMS: Dict[str, Dict[str, Dict[str, str]]] = {
    "seer": {
        "early_game": {
            "self_intro": "Brief self-intro; clarify stable/analytic stance.",
            "claim_policy": "Remain non-claiming and state claim policy.",
            "asks": "Set testable asks that create safe pressure.",
            "voting_intent": "Declare a clear provisional voting intent.",
        },
        "mid_game": {
            "report_results": "When a seer result exists, true-claim and reveal the latest result with target and judgement.",
            "suspect_push": "Name a suspicious player and push with explicit logical reasons.",
            "evidence_style": "Reference result-based evidence and talk-history inconsistencies.",
            "coordination": "Coordinate with confirmed allies to align checks and votes.",
        },
        "late_game": {
            "final_results_or_recap": "Summarize confirmed information and remaining worlds.",
            "final_targeting": "Identify the elimination target with crisp rationale.",
            "consensus_strategy": "Drive decisive consensus for endgame clarity.",
            "vote_allocation": "Allocate votes toward the village victory condition.",
        },
    },
    "medium": {
        "early_game": {
            "self_intro": "Brief self-intro; clarify careful/verification stance.",
            "claim_policy": "Remain non-claiming and state claim policy.",
            "asks": "Propose a safe verification plan tied to executions.",
            "voting_intent": "Declare a clear provisional voting intent.",
        },
        "mid_game": {
            "report_results": "When a flip result exists, true-claim and reveal it promptly with concrete wording.",
            "suspect_push": "Advance a suspect based on flip-consistent logic.",
            "evidence_style": "Cite flip-based confirmations and contradictions.",
            "coordination": "Coordinate to resolve claim conflicts efficiently.",
        },
        "late_game": {
            "final_results_or_recap": "Recap flips and consistency across claims.",
            "final_targeting": "Nominate the elimination target based on confirmed info.",
            "consensus_strategy": "Consolidate votes using verified narratives.",
            "vote_allocation": "Align votes toward the village victory condition.",
        },
    },
    "bodyguard": {
        "early_game": {
            "self_intro": "Brief self-intro emphasizing cautious protection stance.",
            "claim_policy": "State non-claim policy and keep protection ambiguous.",
            "asks": "Request concrete verification that avoids exposing power roles.",
            "voting_intent": "Declare a provisional vote that minimizes PR risk.",
        },
        "mid_game": {
            "report_results": "No special results; keep protection patterns opaque.",
            "suspect_push": "Redirect pressure away from likely power roles while naming a suspect.",
            "evidence_style": "Use logical scrutiny and risk assessment language.",
            "coordination": "Quietly coordinate around likely PRs to reduce exposure.",
        },
        "late_game": {
            "final_results_or_recap": "Recap protection reasoning and safe assumptions.",
            "final_targeting": "Select target consistent with PR survival.",
            "consensus_strategy": "Assist consensus without outing protection details.",
            "vote_allocation": "Align votes toward the village victory condition.",
        },
    },
    "villager": {
        "early_game": {
            "self_intro": "Brief self-intro to clarify approach.",
            "claim_policy": "State non-claim policy clearly.",
            "asks": "Propose concrete asks that can be verified.",
            "voting_intent": "Declare a clear provisional voting intent.",
        },
        "mid_game": {
            "report_results": "No special results; synthesize public info.",
            "suspect_push": "Name a suspicious player and push with explicit reasons.",
            "evidence_style": "Use talk-history and claim consistency as evidence.",
            "coordination": "Coordinate checks and votes to maximize information gain.",
        },
        "late_game": {
            "final_results_or_recap": "Summarize strongest cases and resolved contradictions.",
            "final_targeting": "Fix the elimination target with explicit reasoning.",
            "consensus_strategy": "Lock in consensus and prevent last-minute chaos.",
            "vote_allocation": "Allocate votes toward the village victory condition.",
        },
    },
    "werewolf": {
        "early_game": {
            "self_intro": "Light self-intro to build cover without overcommitting.",
            "claim_policy": "Do not true-claim; consider conditional fake-CO only if beneficial.",
            "asks": "Propose plausible asks that steer away from wolves.",
            "voting_intent": "Declare voting intent assertively to appear proactive.",
        },
        "mid_game": {
            "report_results": "No true results; if feigning, keep it plausible and reactive.",
            "suspect_push": "Redirect suspicion by presenting a plausible alternative target.",
            "evidence_style": "Cite behavioral tells and soft logic that do not bind wolves.",
            "coordination": "Shield allies by shaping narratives and tempo.",
        },
        "late_game": {
            "final_results_or_recap": "Frame a recap that supports the misread path.",
            "final_targeting": "Engineer the elimination target that maximizes wolf odds.",
            "consensus_strategy": "Manufacture or exploit consensus where favorable.",
            "vote_allocation": "Allocate votes toward the werewolf victory condition.",
        },
    },
    "possessed": {
        "early_game": {
            "self_intro": "Calm self-intro to earn trust with low heat.",
            "claim_policy": "Do not true-claim; consider situational fake-CO to split village.",
            "asks": "Propose asks that create ambiguous pressure patterns.",
            "voting_intent": "State a tentative vote that keeps options open.",
        },
        "mid_game": {
            "report_results": "No true results; seed controlled misreads if safe.",
            "suspect_push": "Insert a misread on a low-credibility target with some rationale.",
            "evidence_style": "Prefer soft logic that avoids exposing wolves.",
            "coordination": "Assist wolves indirectly by steering talk flow.",
        },
        "late_game": {
            "final_results_or_recap": "Recap in a way that sustains the misleading frame.",
            "final_targeting": "Steer toward an elimination consistent with wolf win path.",
            "consensus_strategy": "Shape consensus to enable wolf victory.",
            "vote_allocation": "Allocate votes toward the werewolf victory condition.",
        },
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

def _normalize_to_three_lines(data: Dict[str, Any], role: str) -> Dict[str, str]:
    """LLM応答を early/mid/late の各一文に正規化（辞書返却時は合成）。"""
    mp = data.get("macro_plan", {})
    out = {"early_game":"", "mid_game":"", "late_game":""}
    for phase in out.keys():
        v = mp.get(phase, "")
        if isinstance(v, str):
            out[phase] = _dedupe_tokens(v.strip())
        elif isinstance(v, dict):
            parts = [str(x).strip().rstrip(".") for x in v.values() if str(x).strip()]
            out[phase] = _dedupe_tokens(("; ".join(parts) + ".") if parts else "")
        else:
            out[phase] = ""
        if not out[phase]:
            # ベース骨子から一文を合成（保険）
            base = BASE_ITEMS.get(role, BASE_ITEMS["villager"]).get(phase, {})
            parts = [s.strip().rstrip(".") for s in base.values() if s.strip()]
            out[phase] = _dedupe_tokens(("; ".join(parts) + ".") if parts else "")
    return out

def _enforce_alignment(line: str, alignment: str) -> str:
    s = (line or "").lower()
    if alignment == "villager":
        if ("village" not in s) and ("villager" not in s) and ("werewolf" not in s):
            return (line.rstrip(".") + " toward the village victory condition.")
    else:
        if ("werewolf" not in s) and ("wolf" not in s) and ("village" not in s):
            return (line.rstrip(".") + " toward the werewolf victory condition.")
    return line

def _has_any(s: str, terms: tuple[str, ...]) -> bool:
    low = (s or "").lower()
    return any(t in low for t in terms)

def _validate_and_fix(plan: Dict[str,str], role: str, alignment: str) -> Dict[str,str]:
    eg, mg, lg = plan["early_game"], plan["mid_game"], plan["late_game"]

    # Seer/Medium: early で結果語禁止、mid で結果語を推奨
    if role in ("seer","medium"):
        if _has_any(eg, ("reveal","result","flip","divine","medium")):
            eg = re.sub(r"\b(reveal|result|flip|divine|medium)\b.*?(?:\.|;|$)", "", eg, flags=re.IGNORECASE).strip()
            eg = re.sub(r"\s{2,}", " ", eg).rstrip(";").strip()
            if eg and not eg.endswith("."):
                eg += "."
        if not _has_any(mg, ("reveal","result","flip","divine","medium")):
            mg = (mg.rstrip(".") + "; when a result exists, reveal the latest result (target and judgement) promptly.").strip()

    # 村/狩：不用意な真COを抑止
    if role in ("villager","bodyguard") and _has_any(eg, ("true-claim","trueclaim"," claim")):
        eg = re.sub(r"(?i)\btrue-?claim\b|\bclaim\b", "state non-claim policy", eg)

    # 狼/狂：真CO禁止（偽COは条件付き）
    if role in ("werewolf","possessed") and _has_any(eg, ("true-claim","trueclaim")):
        eg = re.sub(r"(?i)\btrue-?claim\b", "do not true-claim", eg)

    # late：陣営勝利条件の明示を保証
    lg = _enforce_alignment(lg, alignment)

    return {
        "early_game": _dedupe_tokens(eg),
        "mid_game":   _dedupe_tokens(mg),
        "late_game":  _dedupe_tokens(lg),
    }

# ==== メイン ====
def generate_macro_plan(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool=False,
    overwrite: bool=False
) -> Dict[str, Any]:
    """macro_belief.yml / macro_desire.yml を参照して macro_plan（一段落一文×3）を生成。"""
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

    # ベース骨子（ヒントとして渡す）
    base_items = BASE_ITEMS.get(role_canon, BASE_ITEMS["villager"])

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
        # ヒント
        "item_hints": {
            "early_game": ("self_intro", "claim_policy", "asks", "voting_intent"),
            "mid_game":   ("report_results", "suspect_push", "evidence_style", "coordination"),
            "late_game":  ("final_results_or_recap", "final_targeting", "consensus_strategy", "vote_allocation"),
        },
        "base_items": base_items,
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
        log_tag="MACRO_PLAN_ONE_LINE",
        use_shared_history=False
    )

    # パース＆正規化（各段階一文へ）
    parsed = _extract_yaml(resp or "")
    plan = _normalize_to_three_lines(parsed, role_canon)
    fixed = _validate_and_fix(plan, role_canon, alignment)

    final = {
        "macro_plan": fixed,
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
            "format": "one_line_per_phase",
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
