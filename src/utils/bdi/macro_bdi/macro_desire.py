#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro desire generation (description only, ≤2 sentences).
- macro_belief.yml / role_social_duties を参照
- 出力は macro_desire.description のみ（summary は生成しない）
- description は最大 2 文に制限
- プロンプトは config.yml の prompt.macro_desire を使用（コード内に埋め込まない）
"""

from __future__ import annotations

import os
import re
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
from jinja2 import Template

# ====== パス ======
BASE = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
CFG_PATH = BASE / "config" / "config.yml"

# ====== IO ======
def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _atomic_write_text(text: str, dst: Path) -> None:
    _safe_mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
    os.replace(tmp, dst)

def _atomic_write_yaml(obj: Dict[str, Any], dst: Path) -> None:
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, default_flow_style=False)
    _atomic_write_text(text, dst)

# ====== 役職正規化（この6パターンのみ対応） ======
# 入力に対して返す表示名（英語タイトルケース）
ROLE_NORMALIZE_TABLE = {
    "villager":  "Villager",  "Villager": "Villager",  "村人":   "Villager",
    "seer":      "Seer",      "Seer":     "Seer",      "占い師": "Seer",
    "werewolf":  "Werewolf",  "Werewolf": "Werewolf",  "人狼":   "Werewolf",
    "possessed": "Possessed", "Possessed":"Possessed", "狂人":   "Possessed",
    "bodyguard": "Bodyguard", "Bodyguard":"Bodyguard", "騎士":   "Bodyguard",
    "medium":    "Medium",    "Medium":   "Medium",    "霊媒師": "Medium",
}
# config.role_social_duties でのキー対応（Bodyguard は Knight 配下に定義がある想定）
CONFIG_ROLE_KEY = {
    "Villager":  "Villager",
    "Seer":      "Seer",
    "Werewolf":  "Werewolf",
    "Possessed": "Possessed",
    "Bodyguard": "Knight",
    "Medium":    "Medium",
}

def _normalize_role_name(s: str | None) -> str:
    if not s:
        return "Villager"
    key = s.strip()
    return ROLE_NORMALIZE_TABLE.get(key, "Villager")

def _supplement_role_definition(display_role: str, role_definition: str,
                                config_data: Dict[str, Any]) -> Tuple[str, str]:
    """display_role を前提に、config.yml から役職定義を補完して返す。"""
    if role_definition:
        return display_role, role_definition
    rsd = (config_data.get("role_social_duties") or {}) if isinstance(config_data, dict) else {}
    cfg_key = CONFIG_ROLE_KEY.get(display_role, display_role)
    if cfg_key in rsd:
        role_definition = rsd[cfg_key].get("definition") or rsd[cfg_key].get("定義") or role_definition
    return display_role, (role_definition or "")

# ====== プロンプト構築（config.yml 管理） ======
def build_prompt(template: str, game_id: str, agent: str, role: str,
                 role_definition: str, desire_tendencies: Dict[str, float]) -> str:
    return Template(template).render(
        game_id=game_id,
        agent=agent,
        role=role,
        role_definition=role_definition,
        desire_tendencies=desire_tendencies or {},
    ).strip()

# ====== 抽出ロバスト化 ======
def _sanitize_text(s: str) -> str:
    s = (s or "").lstrip("\ufeff").strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("…", "...")
    s = re.sub(r"^(yaml|yml)\s*\r?\n", "", s, flags=re.IGNORECASE)
    return s

def _unfence(s: str) -> str | None:
    m = re.search(r"```(?:yaml|yml)?\s*\r?\n([\s\S]*?)\r?\n```", s, re.IGNORECASE)
    return m.group(1).strip() if m else None

def _slice_from_key(s: str, key: str = "macro_desire:") -> str | None:
    i = s.lower().find(key)
    return s[i:].strip() if i != -1 else None

def extract_yaml_from_response(response: str) -> Dict[str, Any]:
    """LLM応答から YAML を頑健に抽出（失敗時は description を救済）。"""
    s = _sanitize_text(response)
    body = _unfence(s)
    if body:
        try:
            return yaml.safe_load(body) or {}
        except yaml.YAMLError:
            pass
    try:
        return yaml.safe_load(s) or {}
    except yaml.YAMLError:
        pass
    tail = _slice_from_key(s)
    if tail:
        try:
            return yaml.safe_load(tail) or {}
        except yaml.YAMLError:
            pass
    # 最後の手段：description 行だけ抽出
    m = re.search(r"description\s*:\s*(.+)", s, re.IGNORECASE)
    desc = (m.group(1).strip()) if m else "Auto-generated description."
    return {"macro_desire": {"description": desc}}

# ====== 文数制御（最大2文） ======
def _limit_sentences(text: str, max_sents: int = 2) -> str:
    t = (text or "").strip()
    if not t:
        return t
    # 日英っぽい句点で区切って2文まで
    parts = re.split(r"(?<=[。．\.!?！？」])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    trimmed = " ".join(parts[:max_sents]).strip()
    return trimmed if re.search(r"[。．\.!?！？」]$", trimmed) else (trimmed + "." if trimmed else "")

# ====== 正規化（description のみ） ======
def normalize_macro_desire(data: Dict[str, Any]) -> Dict[str, Any]:
    """macro_desire.description を必須にし、最大2文に制限。"""
    if "macro_desire" not in data or not isinstance(data["macro_desire"], dict):
        data = {"macro_desire": (data if isinstance(data, dict) else {})}
    md = data["macro_desire"]
    desc = str(md.get("description") or "")
    desc = desc.replace("```", "").strip()
    desc = re.sub(r"^yaml\s*", "", desc, flags=re.IGNORECASE)
    desc = _limit_sentences(desc, 2)
    return {"macro_desire": {"description": desc}}

# ====== 決定論フォールバック（description のみ・最大2文） ======
def build_deterministic_macro_desire(agent: str, role: str,
                                     role_definition: str,
                                     desire_tendencies: Dict[str, float]) -> Dict[str, Any]:
    # 上位2傾向を文に織り込む（軽量）
    tops = sorted((desire_tendencies or {}).items(), key=lambda kv: kv[1], reverse=True)[:2]
    drivers = ", ".join(k.replace("_"," ") for k,_ in tops) if tops else "stability"
    s1 = f"{agent} aims to fulfill the {role} role while prioritizing {drivers}."
    s2 = f"Role duties: {role_definition or 'n/a'}."
    return {"macro_desire": {"description": _limit_sentences(f'{s1} {s2}', 2)}}

def _is_unusable(md_obj: Dict[str, Any]) -> bool:
    d = (md_obj or {}).get("description", "")
    return len((d or "").strip()) < 10

# ====== メイン ======
def generate_macro_desire(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Generate macro_desire (description only, ≤2 sentences) from macro_belief data."""
    mb_path = BASE / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_belief.yml"
    cfg_path = CFG_PATH
    out_path = BASE / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_desire.yml"

    if out_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {out_path}. Use --overwrite to overwrite.")

    try:
        macro_belief_data = load_yaml_file(mb_path)
        config_data = load_yaml_file(cfg_path)
    except Exception:
        macro_belief_data, config_data = {"macro_belief": {}}, {}

    # macro_belief から役職/定義/傾向
    m = macro_belief_data.get("macro_belief", {}) or {}
    role_data = m.get("role_social_duties", {}) or {}
    role_raw = role_data.get("role") or m.get("role") or "Villager"
    role_display = _normalize_role_name(role_raw)
    duties = role_data.get("duties", {}) if isinstance(role_data.get("duties", {}), dict) else {}
    role_def = duties.get("definition") or duties.get("定義") or m.get("role_definition") or ""
    role_display, role_def = _supplement_role_definition(role_display, role_def, config_data)

    dt = m.get("desire_tendency", {}) or {}
    desire_tendencies = dt.get("desire_tendencies") or dt
    if not isinstance(desire_tendencies, dict):
        desire_tendencies = {}

    # プロンプト（config.yml 必須）
    prompt_template = (config_data.get("prompt", {}) or {}).get("macro_desire", "")
    if not prompt_template:
        raise RuntimeError("prompt.macro_desire is missing in config.yml")

    prompt = build_prompt(prompt_template, game_id, agent, role_display, role_def, desire_tendencies)
    if dry_run:
        print("---- PROMPT ----")
        print(prompt)

    # LLM 呼び出し
    if agent_obj is None:
        raise ValueError("agent_obj is required. Direct API calls are not allowed.")
    extra_vars = {
        "game_id": game_id,
        "agent": agent,
        "role": role_display,
        "role_definition": role_def,
        "desire_tendencies": desire_tendencies
    }
    response = agent_obj.send_message_to_llm(
        "macro_desire",
        extra_vars=extra_vars,
        log_tag="MACRO_DESIRE_GENERATION_LITE",
        use_shared_history=False
    )
    if response is None:
        raise ValueError("Agent LLM call returned None")

    # パース & 正規化（description のみ、最大2文）
    parsed = extract_yaml_from_response(response)
    normalized = normalize_macro_desire(parsed)

    # フォールバック
    if _is_unusable(normalized.get("macro_desire", {})):
        normalized = build_deterministic_macro_desire(agent, role_display, role_def, desire_tendencies)

    # メタ付与
    final_data = {
        **normalized,
        "meta": {
            "game_id": game_id,
            "agent": agent,
            "model": (agent_obj.config.get("openai", {}).get("model")
                      or agent_obj.config.get("google", {}).get("model")
                      or agent_obj.config.get("ollama", {}).get("model")),
            "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "source_macro_belief": str(mb_path)
        }
    }

    if dry_run:
        print("---- RESULT ----")
        print(yaml.safe_dump(final_data, allow_unicode=True, sort_keys=False))
        return final_data

    _atomic_write_yaml(final_data, out_path)
    print(f"Saved macro_desire: {out_path}")
    return final_data

def main():
    """Deprecated CLI function."""
    print("❌ This CLI no longer calls LLM directly.")
    print("💡 Run from Agent runtime context instead.")
    print("   Example: agent.generate_macro_desire(game_id, agent_name, agent_obj=agent)")
    return 1

if __name__ == "__main__":
    exit(main())
