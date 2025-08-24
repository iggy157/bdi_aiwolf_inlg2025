#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import yaml

from .talk_history_init import determine_talk_dir

logger = logging.getLogger(__name__)

# ====== 役職正規化 ======
ROLE_JA2CANON = {
    "占い師": "seer",
    "霊媒師": "medium",
    "狩人": "bodyguard", "騎士": "bodyguard", "守護": "bodyguard",
    "村人": "villager",
    "人狼": "werewolf",
    "狂人": "possessed",
}
ROLE_CANON_SET = {"seer", "medium", "bodyguard", "villager", "werewolf", "possessed"}


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


# ====== 安全YAMLローダ ======
def load_yaml_safe(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        return {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return {}


# ====== 既存ヘルパ（改修） ======
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
            "creditability": float(data.get("creditability", 0.5)),
        }
    if logger_obj:
        logger_obj.logger.info(f"Collected affinity/trust data for {len(result)} agents")
    return result


def extract_analysis_tail(analysis_path: Path, max_items: int = 12) -> str:
    data = load_yaml_safe(analysis_path)
    # 新形式: {1:{...},2:{...}} / 旧形式: items: [...]
    items: List[dict] = []
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        items = data["items"]
    elif isinstance(data, dict):
        keys = []
        for k in data.keys():
            try:
                keys.append(int(k))
            except Exception:
                pass
        keys.sort()
        for k in keys:
            v = data.get(k) or data.get(str(k))
            if isinstance(v, dict):
                items.append(v)

    if not items:
        return ""
    tail_items = items[-max_items:] if len(items) > max_items else items
    lines = []
    for i, item in enumerate(tail_items, 1):
        content = item.get("content", "")
        from_agent = item.get("from", "unknown")
        cred = item.get("credibility", item.get("creditability", 0.0))
        try:
            cred = float(cred)
        except Exception:
            cred = 0.0
        lines.append(f"{i}. {from_agent}: {content} (cred: {cred:.2f})")
    return "\n".join(lines)


def count_analysis_items(analysis_path: Path) -> int:
    data = load_yaml_safe(analysis_path)
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return len(data["items"])
    # support legacy numeric-key dict
    try:
        keys = [int(k) for k in data.keys() if str(k).isdigit()]
        return len(keys)
    except Exception:
        return 0


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
    targets = _parse_to_targets(to_val)
    if not targets:
        return False
    specials = {"all", "null"}
    if targets <= specials:
        return False
    def _norm(s: str) -> str:
        return str(s).strip()
    a = _norm(agent_name)
    return any(_norm(t) == a for t in targets)


def _load_latest5_entries(analysis_path: Path) -> list[dict]:
    data = load_yaml_safe(analysis_path)
    entries: list[dict] = []
    if not data:
        return entries
    if isinstance(data.get("items"), list):
        entries = [e for e in data["items"] if isinstance(e, dict)]
    else:
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
    return entries[-5:]


def select_sentence_content_from_analysis(analysis_path: Path, agent_name: str) -> str:
    latest5 = _load_latest5_entries(analysis_path)
    if not latest5:
        return ""
    candidates = []
    for e in latest5:
        content = e.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if _is_pending_text(content):
            continue
        candidates.append(e)
    if not candidates:
        for e in reversed(latest5):
            c = e.get("content")
            if isinstance(c, str) and c.strip():
                return c.strip()
        return ""
    to_self = [e for e in candidates if _is_addressed_to_self(e.get("to"), agent_name)]
    def sort_key(e, recency_idx):
        return (_extract_cred(e), -recency_idx)
    if to_self:
        ranked = sorted([(e, i) for i, e in enumerate(candidates) if e in to_self],
                        key=lambda x: sort_key(x[0], x[1]))
        chosen = ranked[0][0]
        return str(chosen.get("content", "")).strip()
    ranked = sorted([(e, i) for i, e in enumerate(candidates)],
                    key=lambda x: sort_key(x[0], x[1]))
    return str(ranked[0][0].get("content", "")).strip()


# ====== macro/micro 読み出し ======
def _load_macro_belief_role(base_macro_dir: Path, game_id: str, agent: str) -> str:
    p = base_macro_dir / game_id / agent / "macro_belief.yml"
    d = load_yaml_safe(p)
    role_ja = None
    mb = d.get("macro_belief", {}) if isinstance(d, dict) else {}
    if isinstance(mb.get("role_social_duties"), dict):
        role_ja = mb["role_social_duties"].get("role")
    if not role_ja:
        role_ja = mb.get("role") or mb.get("role_name")
    return _canonical_role(role_ja) or "villager"


def _load_macro_plan_text(base_macro_dir: Path, game_id: str, agent: str) -> Dict[str, Any]:
    """
    macro_plan は無効化されているが、プロンプト互換のため summary/policies を空で返す。
    """
    p = base_macro_dir / game_id / agent / "macro_plan.yml"
    d = load_yaml_safe(p)
    mp = d.get("macro_plan", {}) if isinstance(d, dict) else {}
    summary = mp.get("strategy_summary", "")
    policies = mp.get("policies", {}) if isinstance(mp, dict) else {}
    return {
        "strategy_summary": summary or "",
        "policies": {
            "co_policy": policies.get("co_policy", ""),
            "results_policy": policies.get("results_policy", ""),
            "analysis_policy": policies.get("analysis_policy", ""),
            "persuasion_policy": policies.get("persuasion_policy", ""),
            "vote_policy": policies.get("vote_policy", ""),
        },
    }


def _load_micro_belief(base_micro_dir: Path, game_id: str, agent: str) -> Dict[str, Any]:
    p = base_micro_dir / game_id / agent / "micro_belief.yml"
    return load_yaml_safe(p)


def _load_self_talk_stats(agent_micro_dir: Path) -> Tuple[int, Optional[str]]:
    """
    self_talk.yml（数値キー or items配列）から件数と最新contentを返す
    """
    p = agent_micro_dir / "self_talk.yml"
    d = load_yaml_safe(p)
    if not d:
        return 0, None

    items: List[dict] = []
    if isinstance(d.get("items"), list):
        items = [x for x in d["items"] if isinstance(x, dict)]
    else:
        # 数値キー形
        numeric_keys = []
        for k in d.keys():
            try:
                numeric_keys.append(int(k))
            except Exception:
                continue
        numeric_keys.sort()
        for k in numeric_keys:
            e = d.get(k) or d.get(str(k))
            if isinstance(e, dict):
                items.append(e)

    count = len(items)
    latest_content = None
    if items:
        last = items[-1]
        c = last.get("content")
        if isinstance(c, str) and c.strip():
            latest_content = c.strip()
    return count, latest_content


# ====== micro_desire 既存件数の取得 ======
def _count_existing_micro_desires(output_path: Path) -> int:
    """
    micro_desire.yml の既存件数を返す。
    - 新形式: { micro_desires: [ ... ] } の長さ
    - 旧形式: { micro_desire: {...} } があれば 1 とみなす
    - それ以外 / ファイル無し: 0
    """
    d = load_yaml_safe(output_path)
    if not isinstance(d, dict):
        return 0
    if isinstance(d.get("micro_desires"), list):
        return len(d["micro_desires"])
    if isinstance(d.get("micro_desire"), dict):
        return 1
    return 0


# ====== negatives 等の集計 ======
def _summarize_negatives_from_micro_belief(mb: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    total = 0
    per_target: Dict[str, int] = {}
    if not isinstance(mb, dict):
        return (0, {})
    for other, block in mb.items():
        if other == "_meta" or not isinstance(block, dict):
            continue
        for k, v in block.items():
            if isinstance(k, str) and k.startswith("negative_to_"):
                tgt = k[len("negative_to_") :]
                try:
                    c = int(v)
                except Exception:
                    continue
                if c > 0:
                    total += c
                    per_target[tgt] = per_target.get(tgt, 0) + c
    return (total, per_target)


def _collect_low_trust_candidates(mb: Dict[str, Any], topk: int = 3) -> List[Tuple[str, float, float, float]]:
    """
    return list of (agent, liking, creditability, score) sorted by worst score (低い順).
    score = 1 - 0.5*(liking + creditability)
    """
    cands: List[Tuple[str, float, float, float]] = []
    for other, block in mb.items():
        if other == "_meta" or not isinstance(block, dict):
            continue
        try:
            lk = float(block.get("liking", 0.5))
            cr = float(block.get("creditability", 0.5))
            score = 1.0 - 0.5 * (lk + cr)
            cands.append((other, lk, cr, score))
        except Exception:
            continue
    cands.sort(key=lambda x: x[3], reverse=True)  # score 大=悪い
    return cands[:topk]


def _count_seer_co_nonnull(mb: Dict[str, Any]) -> int:
    cnt = 0
    for other, block in mb.items():
        if other == "_meta" or not isinstance(block, dict):
            continue
        if block.get("seer_co") not in (None, "", "null"):
            cnt += 1
    return cnt


def _exists_self_co_seer_or_medium(mb: Dict[str, Any]) -> bool:
    for other, block in mb.items():
        if other == "_meta" or not isinstance(block, dict):
            continue
        val = str(block.get("self_co", "") or "").strip().lower()
        if val in {"seer", "medium"}:
            return True
    return False


def _info_has_valid_result(agent_role: str, agent_obj) -> Tuple[bool, Optional[str]]:
    """
    Info から divine_result / medium_result の有無を検査。
    文字列化も返す（プロンプト渡し用）。
    """
    if not agent_obj or not getattr(agent_obj, "info", None):
        return (False, None)
    info = agent_obj.info
    try:
        if agent_role == "seer":
            res = getattr(info, "divine_result", None)
            if res is not None:
                return (True, str(res))
        elif agent_role == "medium":
            res = getattr(info, "medium_result", None)
            if res is not None:
                return (True, str(res))
    except Exception:
        pass
    return (False, None)


def _should_enter_voting_decision(agent_obj, neg_total: int) -> bool:
    """簡易条件: day>=1 かつ 自分の残り発言が少ない（<=1） または negative が多い。"""
    try:
        day = int(getattr(agent_obj.info, "day", 0) or 0)
    except Exception:
        day = 0
    talk_limit = 0
    try:
        talk_limit = int(getattr(agent_obj.setting.talk.max_count, "per_agent", 0) or 0)
    except Exception:
        talk_limit = 0
    # 自発話残り数の概算
    own_talk = 0
    try:
        if hasattr(agent_obj, "talk_history") and agent_obj.talk_history:
            name = agent_obj.info.agent if hasattr(agent_obj.info, "agent") else getattr(agent_obj, "agent_name", "")
            own_talk = sum(1 for t in agent_obj.talk_history if getattr(t, "agent", None) == name)
    except Exception:
        pass
    remaining = max(0, talk_limit - own_talk) if talk_limit > 0 else 99
    alive_cnt = 0
    try:
        smap = getattr(agent_obj.info, "status_map", {}) or {}
        alive_cnt = sum(1 for v in smap.values() if str(v) == "ALIVE" or getattr(v, "name", "") == "ALIVE")
    except Exception:
        pass
    return (day >= 1 and remaining <= 1) or (day >= 1 and neg_total >= max(2, alive_cnt // 2)) or (alive_cnt in (3, 4))


# ====== 旧ヒューリスティック（フォールバックで使用） ======
def _decide_discussion_stage_fallback(
    agent_role: str,
    agent_obj,
    micro_belief: Dict[str, Any],
    *,
    is_first_desire: bool
) -> Tuple[str, Dict[str, Any]]:
    # 1) self_introduction
    if is_first_desire:
        return "self_introduction", {"reason": "first_micro_desire_entry"}

    day = int(getattr(agent_obj.info, "day", 0) or 0) if getattr(agent_obj, "info", None) else 0
    neg_total, neg_per_target = _summarize_negatives_from_micro_belief(micro_belief)
    seer_co_nonnull = _count_seer_co_nonnull(micro_belief)
    exists_self_co = _exists_self_co_seer_or_medium(micro_belief)
    has_result, result_str = _info_has_valid_result(agent_role, agent_obj)

    if agent_role in {"seer", "medium"}:
        if has_result:
            return "information_sharing", {"reason": "has_valid_result", "role": agent_role, "result": result_str}
    else:
        if seer_co_nonnull <= day:
            return "information_sharing", {"reason": "seer_co_nonnull<=day", "seer_co_nonnull": seer_co_nonnull, "day": day}
        if not exists_self_co:
            return "information_sharing", {"reason": "no_self_co_seer_or_medium_found"}

    if neg_total == 0:
        return "reasoning_analysis", {"reason": "no_negative_entries", "neg_total": neg_total}

    if neg_total > 0 and not _should_enter_voting_decision(agent_obj, neg_total):
        return "discussion_persuasion", {"reason": "negatives_present", "neg_total": neg_total, "neg_targets": neg_per_target}

    return "voting_decision", {"reason": "endgame_or_time_pressure", "neg_total": neg_total}


# ====== macro_desire.yml 読み込みと選択 ======
def _normalize_entries_from_macro_desire(md_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    macro_desire.yml からエントリ配列を抽出して正規化する。
    返す各要素は {title: str, desire: str, conditions: dict|None} を基本とする。
    """
    entries: List[Dict[str, Any]] = []

    if not md_data:
        return entries

    node = md_data.get("macro_desire", md_data)

    # 1) entries: [...]
    if isinstance(node, dict) and isinstance(node.get("entries"), list):
        for e in node["entries"]:
            if not isinstance(e, dict):
                continue
            title = str(e.get("title", "")).strip()
            desire = str(e.get("desire", "")).strip()
            cond = e.get("conditions") if isinstance(e.get("conditions"), dict) else None
            if title:
                entries.append({"title": title, "desire": desire, "conditions": cond})
        return entries

    # 2) stages: { title: {desire, conditions?}, ... }
    if isinstance(node, dict) and isinstance(node.get("stages"), dict):
        for title, body in node["stages"].items():
            if not isinstance(body, dict):
                continue
            desire = str(body.get("desire", body.get("description", "")) or "").strip()
            cond = body.get("conditions") if isinstance(body.get("conditions"), dict) else None
            title_s = str(title).strip()
            if title_s:
                entries.append({"title": title_s, "desire": desire, "conditions": cond})
        return entries

    # 3) 直に配列
    if isinstance(node, list):
        for e in node:
            if not isinstance(e, dict):
                continue
            title = str(e.get("title", "")).strip()
            desire = str(e.get("desire", "")).strip()
            cond = e.get("conditions") if isinstance(e.get("conditions"), dict) else None
            if title:
                entries.append({"title": title, "desire": desire, "conditions": cond})
        return entries

    # 4) summary / description のみ: 空配列（フォールバック側で扱う）
    return entries


def _build_context_for_selection(
    *,
    agent_role: str,
    agent_obj,
    analysis_items_count: int,
    micro_belief_data: Dict[str, Any],
    existing_desires: int,
    self_talk_count: int,
) -> Dict[str, Any]:
    neg_total, _ = _summarize_negatives_from_micro_belief(micro_belief_data)
    has_result, _ = _info_has_valid_result(agent_role, agent_obj)
    alive_cnt = 0
    day = 0
    try:
        day = int(getattr(agent_obj.info, "day", 0) or 0)
        smap = getattr(agent_obj.info, "status_map", {}) or {}
        alive_cnt = sum(1 for v in smap.values() if str(v) == "ALIVE" or getattr(v, "name", "") == "ALIVE")
    except Exception:
        pass
    time_pressure = _should_enter_voting_decision(agent_obj, neg_total)

    return {
        "day": day,
        "alive": alive_cnt,
        "neg_total": neg_total,
        "has_result": has_result,
        "first_desire": (existing_desires == 0),
        "analysis_items": analysis_items_count,
        "self_talk_count": self_talk_count,
        "role": agent_role,
        "time_pressure": time_pressure,
        "seer_co_nonnull": _count_seer_co_nonnull(micro_belief_data),
        "exists_self_co_seer_or_medium": _exists_self_co_seer_or_medium(micro_belief_data),
    }


def _conditions_match(conds: Dict[str, Any] | None, ctx: Dict[str, Any]) -> Tuple[bool, int, Dict[str, Any]]:
    """
    シンプルな条件評価器。
    True/False と 満たした条件数、および評価詳細を返す。
    サポート条件（すべて任意）:
      - day_min / day_max (int)
      - alive_min / alive_max (int)
      - neg_total_min / neg_total_max (int)
      - analysis_items_min / analysis_items_max (int)
      - self_talk_count_min / self_talk_count_max (int)
      - first_desire (bool)
      - has_result (bool)
      - time_pressure (bool)
      - role_is (str or [str])  # seer/medium/...（正規化済みを期待）
      - seer_co_nonnull_min / seer_co_nonnull_max (int)
      - exists_self_co_seer_or_medium (bool)
    """
    if not conds:
        return True, 0, {}

    matched = 0
    details: Dict[str, Any] = {}
    ok = True

    def _cmp_num(name: str, op: str, v: int) -> bool:
        cv = int(ctx.get(name, 0) or 0)
        return cv >= v if op == "min" else cv <= v

    for k, v in conds.items():
        try:
            if k.endswith("_min"):
                base = k[:-4]
                if not _cmp_num(base, "min", int(v)):
                    ok = False
                else:
                    matched += 1
                    details[k] = True
            elif k.endswith("_max"):
                base = k[:-4]
                if not _cmp_num(base, "max", int(v)):
                    ok = False
                else:
                    matched += 1
                    details[k] = True
            elif k in ("first_desire", "has_result", "time_pressure", "exists_self_co_seer_or_medium"):
                if bool(ctx.get(k, False)) == bool(v):
                    matched += 1
                    details[k] = True
                else:
                    ok = False
            elif k == "role_is":
                if isinstance(v, list):
                    cond = {str(_).strip().lower() for _ in v}
                else:
                    cond = {str(v).strip().lower()}
                if ctx.get("role") in cond:
                    matched += 1
                    details[k] = True
                else:
                    ok = False
            else:
                # 未知条件はスコアに影響させず詳細に残す
                details[k] = "ignored"
        except Exception:
            ok = False
    return ok, matched, details


def _select_stage_from_macro_desire(
    entries: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    *,
    agent_role: str,
    agent_obj,
    micro_belief_data: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    macro_desire の entries から最適な1件を条件で選ぶ。
    条件が無いエントリは常時候補。複数一致時は「一致条件数が多い順」→「先勝ち」。
    一致なしの場合はヒューリスティックでタイトル名に応じて推定する。
    """
    if not entries:
        return None, {"source": "macro_desire_missing"}

    # 1) 条件評価
    scored: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []  # (match_count, entry, detail)
    unconditional: List[Dict[str, Any]] = []
    for e in entries:
        conds = e.get("conditions") if isinstance(e.get("conditions"), dict) else None
        if not conds:
            unconditional.append(e)
            continue
        ok, cnt, detail = _conditions_match(conds, ctx)
        if ok:
            scored.append((cnt, e, {"match_details": detail}))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[0]
        e = top[1]
        meta = {"source": "macro_desire_conditions", **top[2]}
        return e, meta

    if unconditional:
        # 無条件エントリのうちタイトルに基づく推定（優先度）
        # time_pressure → voting_decision
        # neg_total>0 → discussion_persuasion
        # neg_total==0 → reasoning_analysis
        # has_result → information_sharing
        # first_desire → self_introduction
        title_pref = [
            ("voting_decision", ctx.get("time_pressure", False)),
            ("discussion_persuasion", ctx.get("neg_total", 0) > 0),
            ("reasoning_analysis", ctx.get("neg_total", 0) == 0),
            ("information_sharing", ctx.get("has_result", False)),
            ("self_introduction", ctx.get("first_desire", False)),
        ]
        for t, cond in title_pref:
            if not cond:
                continue
            for e in unconditional:
                if str(e.get("title", "")).strip().lower() == t:
                    return e, {"source": "macro_desire_unconditional_pref", "reason": t}

        # いずれもマッチしない場合は先頭
        return unconditional[0], {"source": "macro_desire_unconditional_first"}

    # 2) どれもマッチしない → タイトル文字列にヒューリスティックで近似
    #   旧ロジックにフォールバック
    stage, reason = _decide_discussion_stage_fallback(
        agent_role=agent_role,
        agent_obj=agent_obj,
        micro_belief=micro_belief_data,
        is_first_desire=ctx.get("first_desire", False),
    )
    # stage に最も近いタイトルを探す
    target = stage.strip().lower()
    for e in entries:
        if str(e.get("title", "")).strip().lower() == target:
            return e, {"source": "macro_desire_heuristic_title_match", "reason": reason}
    return None, {"source": "macro_desire_no_match_fallback", "reason": reason}


# ====== 生成メイン ======
def generate_micro_desire_for_agent(
    game_id: str,
    agent: str,
    *,
    base_micro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi"),
    base_macro_dir: Path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi"),
    agent_obj=None,
    logger_obj=None,
    trigger: Optional[str] = None,
    max_analysis_items: int = 12,
) -> Optional[Path]:
    if logger_obj:
        logger_obj.logger.info(f"Generating micro_desire for {agent} (trigger: {trigger})")

    agent_micro_dir = base_micro_dir / game_id / agent
    agent_macro_dir = base_macro_dir / game_id / agent

    analysis_path = agent_micro_dir / "analysis.yml"
    macro_desire_path = agent_macro_dir / "macro_desire.yml"
    macro_belief_path = agent_macro_dir / "macro_belief.yml"

    talk_dir = determine_talk_dir(agent_micro_dir)
    micro_desire_out_path = agent_micro_dir / "micro_desire.yml"

    try:
        analysis_tail = extract_analysis_tail(analysis_path, max_analysis_items)
        analysis_items_count = count_analysis_items(analysis_path)
        affinity_trust = collect_affinity_trust_data(talk_dir, logger_obj)

        selected_sentence_text = select_sentence_content_from_analysis(analysis_path, agent) or ""

        # macro desire / belief
        macro_desire_data = load_yaml_safe(macro_desire_path)
        macro_desire_entries = _normalize_entries_from_macro_desire(macro_desire_data)

        macro_desire_summary = ""
        macro_desire_description = ""
        if isinstance(macro_desire_data.get("macro_desire"), dict):
            md = macro_desire_data["macro_desire"]
            macro_desire_summary = str(md.get("summary", ""))
            macro_desire_description = str(md.get("description", ""))

        # 自役職
        agent_role = _load_macro_belief_role(base_macro_dir, game_id, agent)

        # macro_plan（互換のため空でも読み出して渡す）
        macro_plan_pack = _load_macro_plan_text(base_macro_dir, game_id, agent)

        # micro_belief
        micro_belief_data = _load_micro_belief(base_micro_dir, game_id, agent)
        neg_total, neg_per_target = _summarize_negatives_from_micro_belief(micro_belief_data)
        low_trust = _collect_low_trust_candidates(micro_belief_data, topk=3)

        # micro_desire 既存件数（self_introduction の判断などに使用）
        existing_desires = _count_existing_micro_desires(micro_desire_out_path)
        self_talk_count, self_talk_last = _load_self_talk_stats(agent_micro_dir)

        # 選択用コンテキストを構築
        ctx = _build_context_for_selection(
            agent_role=agent_role,
            agent_obj=agent_obj,
            analysis_items_count=analysis_items_count,
            micro_belief_data=micro_belief_data,
            existing_desires=existing_desires,
            self_talk_count=self_talk_count,
        )

        # macro_desire ベースで discussion_stage / current_desire を決定
        selected_entry, sel_meta = _select_stage_from_macro_desire(
            macro_desire_entries,
            ctx,
            agent_role=agent_role,
            agent_obj=agent_obj,
            micro_belief_data=micro_belief_data,
        )

        if selected_entry:
            stage = str(selected_entry.get("title", "")).strip() or "reasoning_analysis"
            fixed_current_desire = str(selected_entry.get("desire", "")).strip() or f"Proceed according to {stage}."
            stage_meta = {"selection": sel_meta, "ctx": {k: ctx[k] for k in ("day", "alive", "neg_total", "has_result", "first_desire", "time_pressure")}}
            stage_meta["macro_desire_entry_used"] = True
        else:
            # フォールバック（旧ルール）
            stage, fallback_meta = _decide_discussion_stage_fallback(
                agent_role=agent_role,
                agent_obj=agent_obj,
                micro_belief=micro_belief_data,
                is_first_desire=(existing_desires == 0),
            )
            fixed_current_desire = f"Proceed according to {stage} stage."
            stage_meta = {"selection": {"source": "fallback_only", "reason": fallback_meta}, "ctx": ctx}
            stage_meta["macro_desire_entry_used"] = False

        if logger_obj:
            logger_obj.logger.info(
                f"[micro_desire] stage='{stage}' current_desire='{fixed_current_desire[:60]}...' meta={stage_meta.get('selection',{})}"
            )

    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to collect reference data: {e}")
        return None

    if not agent_obj:
        if logger_obj:
            logger_obj.logger.error("No agent_obj provided for LLM call")
        return None

    # LLM 入力（従来どおり）。content/response_to_selected の生成を任せる。
    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        "agent_role": agent_role,
        "macro_desire_summary": macro_desire_summary,
        "macro_desire_description": macro_desire_description,
        "macro_plan": macro_plan_pack,  # 互換のため渡す（中身が空でも可）
        "analysis_tail": analysis_tail,
        "selected_sentence_text": selected_sentence_text,
        "affinity_trust": affinity_trust,
        "discussion_stage": stage,            # ← ここは本関数で確定
        "stage_meta": stage_meta,             # ← 参照情報
        "negatives": {
            "total": neg_total,
            "per_target": neg_per_target,
        },
        "low_trust_candidates": [
            {"agent": a, "liking": lk, "creditability": cr, "score": sc}
            for (a, lk, cr, sc) in low_trust
        ],
        "max_analysis_items": max_analysis_items,
        # 参考値（プロンプトは必須ではないが、LLMが内容整合しやすいようヒントとして渡す）
        "md_hint_current_desire": fixed_current_desire,
    }

    # 実行（content/response_to_selected はLLM結果を採用。current_desire は本関数で確定値を上書き）
    try:
        response = agent_obj.send_message_to_llm(
            "micro_desire",
            extra_vars=extra_vars,
            log_tag="micro_desire_generation",
        )
        if not response:
            new_desire = _create_fallback_minimal_desire(stage)
        else:
            try:
                clean = response.strip()
                if clean.startswith("```yaml"):
                    clean = clean[7:]
                elif clean.startswith("```"):
                    clean = clean[3:]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
                data = yaml.safe_load(clean)
                if not isinstance(data, dict):
                    new_desire = _create_fallback_minimal_desire(stage)
                else:
                    content = data.get("content")
                    response_to_selected = data.get("response_to_selected")
                    new_desire = {
                        "content": None if content is None or str(content).lower() in ["null", "none", ""] else str(content),
                        "response_to_selected": None if response_to_selected is None or str(response_to_selected).lower() in ["null", "none", ""] else str(response_to_selected),
                        # ← current_desire は LLM 出力に依存させず、確定値で上書き
                        "current_desire": fixed_current_desire,
                        "discussion_stage": stage,
                        "stage_meta": stage_meta,
                        "timestamp": datetime.now().isoformat(),
                        "trigger": trigger or "unknown",
                    }
            except yaml.YAMLError:
                new_desire = _create_fallback_minimal_desire(stage)
    except Exception:
        new_desire = _create_fallback_minimal_desire(stage)

    model_name = (
        type(agent_obj.llm_model).__name__
        if agent_obj and hasattr(agent_obj, "llm_model") and agent_obj.llm_model
        else "unknown"
    )
    new_desire.update({"model": model_name, "game_id": game_id, "agent": agent})

    output_path = agent_micro_dir / "micro_desire.yml"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing = load_yaml_safe(output_path) if output_path.exists() else {}
        if "micro_desires" not in existing or not isinstance(existing.get("micro_desires"), list):
            existing["micro_desires"] = []
        existing["micro_desires"].append(new_desire)
        if len(existing["micro_desires"]) > 20:
            existing["micro_desires"] = existing["micro_desires"][-20:]

        # ---- アトミック保存（破損耐性） ----
        tmp = output_path.with_suffix(".yml.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("# generated by utils.bdi.micro_bdi.micro_desire_generator (append-only)\n")
            yaml.safe_dump(existing, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, output_path)

        return output_path
    except Exception as e:
        if logger_obj:
            logger_obj.logger.exception(f"Failed to save micro_desire: {e}")
        return None


def _create_fallback_minimal_desire(stage: str) -> Dict[str, Any]:
    return {
        "content": None,
        "response_to_selected": None,
        "current_desire": f"Proceed according to {stage} stage.",
        "discussion_stage": stage,
        "stage_meta": {"fallback": True},
        "timestamp": datetime.now().isoformat(),
        "trigger": "fallback",
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
