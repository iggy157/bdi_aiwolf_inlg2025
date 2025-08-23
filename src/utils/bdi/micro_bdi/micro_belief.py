#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
micro_belief.yml 生成スクリプト（正規表現ベース・非LLM）

- 入力:
  - analysis.yml
    /home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi/{game_id}/{agent}/analysis.yml
  - talk_history/*.yml
    /home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi/{game_id}/{agent}/talk_history/{other}.yml

- 出力:
  - micro_belief.yml（同ディレクトリに生成・原子的に置換）

- 仕様（エージェント owner = {agent} 視点で、本人以外の全員について記述）:
  各 other エージェントのブロックに以下を出力する。
    * creditability: talk_history/{other}.yml の top-level "creditability"
    * liking:        talk_history/{other}.yml の top-level "liking"
    * self_co:       analysis.yml から (type=co, from=other, to に other を含む) 発話の本文を正規表現解析して
                     役職/陣営（下記いずれか）を1つ抽出・正規化
    * seer_co:       analysis.yml から (type=co, from!=other, to に other を含む) 発話の本文を正規表現解析して
                     役職/陣営（下記いずれか）を1つ抽出・正規化
    * negative_to_*: analysis.yml から (type=negative, from=other, to に target を含む) の
                     **行数カウント（整数）**。**0 件のキーは出力しない**
    * positive_to_*: 上記の positive 版。**0 件のキーは出力しない**

  役職/陣営の正規化（戻り値は下記のいずれか）:
    - 役職: villager, seer, werewolf, possessed, bodyguard, medium
            （対応語: Villager/村人, Seer/占い師, Werewolf/人狼, Possessed/狂人, Bodyguard/騎士, Medium/霊媒師）
    - 陣営: villager_side（human/人間/white/白 を含むとき）
            werewolf_side（black/黒 を含むとき）

  注記:
    - self_co は自己CO想定なのでロール優先で抽出。該当なしのときは human/white/black/白/黒 を陣営として使用。
    - seer_co は結果報告想定なので陣営語（human/white/black/白/黒）を優先。無ければロール語から抽出。
    - analysis.yml の "to" はカンマ区切り複数可。分割し、対象名に一致するか判定する。
    - negative/positive は **メッセージ（行）件数**でカウント（1行に同じ target が複数出現しても 1件）。
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


# ====== 定数・パス =================================================================

BASE_DIR = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
MICRO_BDI_DIR = BASE_DIR / "info" / "bdi_info" / "micro_bdi"

# 役職語（英/日）
ROLE_WORDS_EN = ["villager", "seer", "werewolf", "possessed", "bodyguard", "medium"]
ROLE_WORDS_JA = ["村人", "占い師", "人狼", "狂人", "騎士", "霊媒師"]

# 陣営語（結果報告）
SIDE_VILLAGER_EN = ["human", "white"]
SIDE_VILLAGER_JA = ["人間", "白"]
SIDE_WEREWOLF_EN = ["black"]
SIDE_WEREWOLF_JA = ["黒"]

# 正規化マップ（ロール）
ROLE_NORMALIZE: Dict[str, str] = {
    # English (lower)
    "villager": "villager",
    "seer": "seer",
    "werewolf": "werewolf",
    "possessed": "possessed",
    "bodyguard": "bodyguard",
    "medium": "medium",
    # Japanese
    "村人": "villager",
    "占い師": "seer",
    "人狼": "werewolf",
    "狂人": "possessed",
    "騎士": "bodyguard",
    "霊媒師": "medium",
    # Display variants (capitalized)
    "Villager": "villager",
    "Seer": "seer",
    "Werewolf": "werewolf",
    "Possessed": "possessed",
    "Bodyguard": "bodyguard",
    "Medium": "medium",
}

SIDE_VILLAGER_SET = set([s.lower() for s in SIDE_VILLAGER_EN]) | set(SIDE_VILLAGER_JA)
SIDE_WEREWOLF_SET = set([s.lower() for s in SIDE_WEREWOLF_EN]) | set(SIDE_WEREWOLF_JA)

# 役職抽出用の正規表現
ROLE_RE_EN = re.compile(r"\b(seer|villager|werewolf|possessed|bodyguard|medium)\b", re.IGNORECASE)
ROLE_RE_JA = re.compile(r"(占い師|霊媒師|村人|人狼|狂人|騎士)")

SIDE_V_RE_EN = re.compile(r"\b(human|white)\b", re.IGNORECASE)
SIDE_W_RE_EN = re.compile(r"\b(black)\b", re.IGNORECASE)
SIDE_V_RE_JA = re.compile(r"(人間|白)")
SIDE_W_RE_JA = re.compile(r"(黒)")

TO_SPLIT_RE = re.compile(r"\s*,\s*")


# ====== ユーティリティ =============================================================

def _load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except Exception:
            return None


def _atomic_write_yaml(obj: Any, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}-{int(time.time()*1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _split_to_field(val: Any) -> List[str]:
    """
    analysis.yml の to フィールドをリスト化。
    - "null"/"all"/None は空扱いにする（宛先特定なし）
    - "A,B" → ["A","B"]
    """
    if val is None:
        return []
    if isinstance(val, list):
        toks = [str(x).strip() for x in val if str(x).strip()]
        return [t for t in toks if t.lower() not in {"null", "all"}]
    s = str(val).strip()
    if not s or s.lower() in {"null", "all"}:
        return []
    return [t for t in TO_SPLIT_RE.split(s) if t]


def _collect_all_agent_names(
    talk_history_dir: Path, analysis_data: Dict[Any, Dict[str, Any]], owner: str
) -> List[str]:
    """観測可能なエージェント名を収集（talk_history のファイル名 + analysisのfrom/to）。ownerは除外しない。"""
    names = set()

    # talk_history ファイル名
    if talk_history_dir.exists():
        for p in talk_history_dir.glob("*.yml"):
            if p.is_file():
                names.add(p.stem)

    # analysis.yml から from/to
    if isinstance(analysis_data, dict):
        for _, row in analysis_data.items():
            if not isinstance(row, dict):
                continue
            frm = row.get("from")
            if isinstance(frm, str) and frm.strip():
                names.add(frm.strip())
            for t in _split_to_field(row.get("to")):
                names.add(t)

    # この時点で owner 自身も含まれる可能性があるが、後段で speaker!=target 等の条件で除外する
    return sorted(names)


# ====== 役職/陣営 抽出（正規表現のみ） =============================================

def _normalize_role_token(tok: str) -> Optional[str]:
    """役職語を正規化（見つからなければ None）"""
    if not tok:
        return None
    if tok in ROLE_NORMALIZE:
        return ROLE_NORMALIZE[tok]
    t = tok.strip()
    if not t:
        return None
    # 大文字小文字を詰めた英語の基本対応
    low = t.lower()
    if low in ROLE_NORMALIZE:
        return ROLE_NORMALIZE[low]
    return ROLE_NORMALIZE.get(t, None)


def _extract_role_self_claim(text: str) -> Optional[str]:
    """
    self_co 用。
    - まず役職語（villager/seer/.../medium, 村人/占い師/.../霊媒師）を探す
    - 見つからない場合のみ、陣営語（human/white/人間/白 → villager_side, black/黒 → werewolf_side）
    """
    if not text:
        return None

    # 英語ロール
    m = ROLE_RE_EN.search(text)
    if m:
        return _normalize_role_token(m.group(1))

    # 日本語ロール
    m = ROLE_RE_JA.search(text)
    if m:
        return _normalize_role_token(m.group(1))

    # 陣営語（村陣営）
    if SIDE_V_RE_EN.search(text) or SIDE_V_RE_JA.search(text):
        return "villager_side"

    # 陣営語（狼陣営）
    if SIDE_W_RE_EN.search(text) or SIDE_W_RE_JA.search(text):
        return "werewolf_side"

    return None


def _extract_role_report(text: str) -> Optional[str]:
    """
    seer_co 用（結果報告）。
    - まず陣営語（human/white/人間/白 → villager_side, black/黒 → werewolf_side）
    - 見つからない場合、役職語（villager/seer/.../medium, 村人/占い師/.../霊媒師）
    """
    if not text:
        return None

    # 陣営語（村陣営）
    if SIDE_V_RE_EN.search(text) or SIDE_V_RE_JA.search(text):
        return "villager_side"

    # 陣営語（狼陣営）
    if SIDE_W_RE_EN.search(text) or SIDE_W_RE_JA.search(text):
        return "werewolf_side"

    # 英語ロール
    m = ROLE_RE_EN.search(text)
    if m:
        return _normalize_role_token(m.group(1))

    # 日本語ロール
    m = ROLE_RE_JA.search(text)
    if m:
        return _normalize_role_token(m.group(1))

    return None


# ====== コア処理 ===================================================================

def _load_analysis_rows(analysis_path: Path) -> List[Dict[str, Any]]:
    """
    analysis.yml を番号キー順にリスト化。
    例: {1:{...}, 2:{...}} → [ {...}, {...} ]
    """
    data = _load_yaml(analysis_path)
    if not isinstance(data, dict):
        return []
    # 数字キー順に並べ替え（数字でないキーは後ろ）
    def _key_sort(k: Any) -> Tuple[int, str]:
        try:
            return (0, int(k))
        except Exception:
            return (1, str(k))
    rows = []
    for _, row in sorted(data.items(), key=lambda kv: _key_sort(kv[0])):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _get_credit_like_from_talk_history(talk_history_dir: Path, other: str) -> Tuple[Optional[float], Optional[float]]:
    p = talk_history_dir / f"{other}.yml"
    data = _load_yaml(p) or {}
    if not isinstance(data, dict):
        return (None, None)
    liking = data.get("liking", None)
    cred = data.get("creditability", None)
    try:
        liking = float(liking) if liking is not None else None
    except Exception:
        liking = None
    try:
        cred = float(cred) if cred is not None else None
    except Exception:
        cred = None
    return (cred, liking)


def _collect_negative_positive_to(
    rows: List[Dict[str, Any]],
    speaker: str,
    target_pool: Iterable[str],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    speaker が他者に向けて発した negative/positive を走査。
    各 target について **件数（int）** を返す（該当がなければ 0）。
    同一行に target が複数回出現しても、その行は 1件 としてカウントする。
    """
    neg_counts = {t: 0 for t in target_pool if t != speaker}
    pos_counts = {t: 0 for t in target_pool if t != speaker}

    for row in rows:
        if row.get("from") != speaker:
            continue
        rtype = str(row.get("type", "")).lower()
        if rtype not in {"negative", "positive"}:
            continue
        tos = set(_split_to_field(row.get("to")))  # 重複を 1 回に抑制
        if not tos:
            continue
        for t in tos:
            if t == speaker:
                continue
            if t in neg_counts and rtype == "negative":
                neg_counts[t] += 1
            if t in pos_counts and rtype == "positive":
                pos_counts[t] += 1
    return neg_counts, pos_counts


def _pick_last_role_from_rows(
    rows: List[Dict[str, Any]],
    predicate,
    extractor,
) -> Optional[str]:
    """
    rows を先頭から走査し、predicate(row) を満たす行について extractor(row["content"]) で
    役職/陣営文字列を抽出。最後に見つかった値（最新）を返す。
    """
    picked: Optional[str] = None
    for row in rows:
        try:
            if predicate(row):
                content = str(row.get("content", "") or "")
                role = extractor(content)
                if role:
                    picked = role
        except Exception:
            continue
    return picked


def build_micro_belief(game_id: str, owner: str) -> Dict[str, Any]:
    """owner 視点の micro_belief.yml を構築して返す（書き込みはしない）"""
    base_dir = MICRO_BDI_DIR / game_id / owner
    analysis_path = base_dir / "analysis.yml"
    talk_history_dir = base_dir / "talk_history"

    rows = _load_analysis_rows(analysis_path)
    # 利用可能な全員（owner を含む可能性あり）
    all_names = _collect_all_agent_names(talk_history_dir, dict(enumerate(rows, start=1)), owner)
    # 「本人以外のエージェント全員」を出力対象にする
    others = [n for n in all_names if n != owner]

    result: Dict[str, Any] = {}
    for other in others:
        cred, like = _get_credit_like_from_talk_history(talk_history_dir, other)

        # self_co: (type=co, from=other, to に other を含む)
        def _pred_self(row: Dict[str, Any]) -> bool:
            if str(row.get("type", "")).lower() != "co":
                return False
            if row.get("from") != other:
                return False
            return other in _split_to_field(row.get("to"))

        self_co = _pick_last_role_from_rows(rows, _pred_self, _extract_role_self_claim)

        # seer_co: (type=co, from!=other, to に other を含む)
        def _pred_report(row: Dict[str, Any]) -> bool:
            if str(row.get("type", "")).lower() != "co":
                return False
            if row.get("from") == other:
                return False
            return other in _split_to_field(row.get("to"))

        seer_co = _pick_last_role_from_rows(rows, _pred_report, _extract_role_report)

        # negative_to_*, positive_to_*（other 以外の全員に対して件数カウント）
        target_names = [n for n in all_names if n != other]  # owner も含む
        neg_counts, pos_counts = _collect_negative_positive_to(rows, other, target_names)

        # ブロックを作成
        block: Dict[str, Any] = {
            "creditability": cred,
            "liking": like,
            "self_co": self_co,
            "seer_co": seer_co,
        }

        # 件数が 1 以上のものだけ出力（0 はキーを作らない）
        for tgt, cnt in neg_counts.items():
            if int(cnt) > 0:
                block[f"negative_to_{tgt}"] = int(cnt)
        for tgt, cnt in pos_counts.items():
            if int(cnt) > 0:
                block[f"positive_to_{tgt}"] = int(cnt)

        result[other] = block

    # 付帯メタ（必要なければ削除してもよい）
    result["_meta"] = {
        "game_id": game_id,
        "owner": owner,
        "generated_at": _now_iso(),
        "source_analysis": str(MICRO_BDI_DIR / game_id / owner / "analysis.yml"),
    }

    return result


def write_micro_belief(game_id: str, owner: str) -> Path:
    """micro_belief.yml を生成し、原子的に保存してパスを返す。"""
    data = build_micro_belief(game_id, owner)
    out_path = MICRO_BDI_DIR / game_id / owner / "micro_belief.yml"
    _atomic_write_yaml(data, out_path)
    print(f"[micro_belief] saved: {out_path}")
    return out_path


# ====== 監視（ポーリング） =========================================================

def watch_and_build(game_id: str, owner: str, interval: float = 0.5) -> None:
    """
    analysis.yml の mtime を監視し、更新毎に micro_belief.yml を再生成。
    外部ライブラリ不使用の簡易ポーリング。
    """
    base_dir = MICRO_BDI_DIR / game_id / owner
    analysis_path = base_dir / "analysis.yml"

    last_mtime: Optional[float] = None
    if analysis_path.exists():
        last_mtime = analysis_path.stat().st_mtime

    print(f"[micro_belief] watching {analysis_path} (interval={interval}s)")
    while True:
        try:
            if analysis_path.exists():
                m = analysis_path.stat().st_mtime
                if last_mtime is None or m > last_mtime:
                    last_mtime = m
                    write_micro_belief(game_id, owner)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("[micro_belief] watch stopped by user")
            break


# ====== CLI =======================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate micro_belief.yml from analysis.yml (regex-based).")
    parser.add_argument("--game-id", required=True, help="Game ULID / ID")
    parser.add_argument("--agent", required=True, help="Owner agent name (視点のエージェント)")
    parser.add_argument("--watch", action="store_true", help="analysis.yml を監視して更新毎に再生成")
    parser.add_argument("--interval", type=float, default=0.5, help="監視間隔秒（--watch 時）")
    args = parser.parse_args()

    if args.watch:
        watch_and_build(args.game_id, args.agent, args.interval)
    else:
        write_micro_belief(args.game_id, args.agent)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
