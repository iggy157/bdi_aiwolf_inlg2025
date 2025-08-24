# -*- coding: utf-8 -*-
"""自分の発話のみを self_talk.yml に蓄積するユーティリティ.

- 保存先: project_root/info/bdi_info/micro_bdi/<game_id>/<agent_name>/self_talk.yml
- 構造  : 1始まりの連番をキーに、値は {"content": "<発話本文>"} のみ
- 方針  : 追記型・重複(同一content)は追加しない・空/Skip/Over/skip=True/over=True は弾く
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from tempfile import NamedTemporaryFile
import yaml
import os
import re

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Talk


# ---- path helpers ----
def _resolve_project_root(start: Path) -> Path:
    """config/.env を手がかりにプロジェクトルートを探索."""
    for p in [start] + list(start.parents):
        if (p / "config" / ".env").exists():
            return p
    parents = list(start.parents)
    return parents[3] if len(parents) >= 4 else (parents[-1] if parents else start)


# ---- domain helpers ----
def _is_self_valid_talk(t: Any, *, self_name: str) -> bool:
    """自分自身の有効な発話のみ許容."""
    try:
        if getattr(t, "agent", None) != self_name:
            return False
        txt = (getattr(t, "text", "") or "").strip()
        if not txt or txt in {"Skip", "Over"}:
            return False
        if getattr(t, "skip", False) or getattr(t, "over", False):
            return False
        return True
    except Exception:
        return False


def _load_yaml(path: Path) -> Dict[int, Dict[str, str]]:
    """既存の self_talk.yml を緩くロード（1: {content: "..."} 形式に正規化）."""
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    out: Dict[int, Dict[str, str]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, dict) and "content" in v:
                out[idx] = {"content": str(v["content"])}
            else:
                # 1: "text" のような簡易表現にも寛容に対応
                out[idx] = {"content": str(v)}
    return out


def _atomic_dump_yaml(path: Path, data: Dict[int, Dict[str, str]]) -> None:
    """YAMLを原子的に保存."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(path.parent),
                            prefix="self_talk_", suffix=".tmp") as tmp:
        yaml.safe_dump(
            {k: v for k, v in sorted(data.items(), key=lambda kv: kv[0])},
            tmp,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False
        )
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(str(tmp_path), str(path))


# ---- public API ----
def write_self_talk_for_agent(
    *,
    game_id: str,
    agent: str,
    talks: list["Talk"],
    filename: str = "self_talk.yml",
) -> Path:
    """Agent.talk_history から自分の発話だけを self_talk.yml に追記。

    Args:
        game_id: ゲームID
        agent  : 自分の表示名（Info.agent と同じ想定）
        talks  : Agent.talk_history のスナップショット
        filename: 出力ファイル名（既定 self_talk.yml）

    Returns:
        Path: 保存先ファイルパス
    """
    project_root = _resolve_project_root(Path(__file__).resolve())
    out_dir = project_root / "info" / "bdi_info" / "micro_bdi" / game_id / agent
    out_path = out_dir / filename

    # 既存データ読み込み
    data = _load_yaml(out_path)
    existing_contents = {v.get("content", "") for v in data.values() if isinstance(v, dict)}
    next_idx = (max(data.keys()) + 1) if data else 1

    # 自分の有効発話を抽出して追記（content重複はスキップ）
    added = 0
    for t in talks or []:
        if not _is_self_valid_talk(t, self_name=agent):
            continue
        content = str(getattr(t, "text", "")).strip()
        if content in existing_contents:
            continue
        data[next_idx] = {"content": content}
        existing_contents.add(content)
        next_idx += 1
        added += 1

    # 変更があるときのみ保存
    if added > 0 or not out_path.exists():
        _atomic_dump_yaml(out_path, data)

    return out_path
