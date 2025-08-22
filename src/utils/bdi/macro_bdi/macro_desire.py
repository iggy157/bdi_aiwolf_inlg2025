#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro desire generation from role_social_duties and desire_tendency.

役職責任と欲求傾向からマクロ欲求を生成するスクリプト。
- プロンプト: フェンス禁止 / 先頭を `macro_desire:` に固定
- 抽出: サニタイズ → フェンス解除 → そのまま読み → キー以降切り出し → ぶっこ抜き救済
- 正規化: 欠損を必ず埋める
- 決定論フォールバック: 役職・傾向から要約/説明を機械生成
- 役職定義の穴埋め: config.yml の role_social_duties から補完
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jinja2 import Template

# NOTE: .env loading is handled by agent.py only

# ====== Fallback prompt (フェンス禁止・先頭固定版) ======
FALLBACK_PROMPT_TEMPLATE = """あなたは社会的役割と欲求傾向にもとづき、エージェントの上位欲求（macro_desire）を設計する専門家です。
以下の入力を読み、**YAMLのみ**で出力してください。**Markdownのコードフェンスや余計な文字は出力しない**でください。
**最初の行は必ず `macro_desire:` から開始**し、その下に `summary` と `description` の2項目を記述してください。

[context]
- game_id: {{ game_id }}
- agent: {{ agent }}

[role_social_duties]
- role: {{ role }}
- definition: {{ role_definition }}

[desire_tendency]
以下は {{ agent }} の欲求傾向（0–1）です。値が高いほど志向が強い想定です。
{% for key, value in desire_tendencies.items() -%}
  - {{ key }}: {{ "%.3f"|format(value) }}
{% endfor %}

[要件]
- 出力は **YAMLのみ**。フェンスや解説文を含めないこと。
- **最初の行は `macro_desire:`**。
- role_social_duties の達成と desire_tendencies の強弱を踏まえ、ゲーム全体での欲求を記述。
- role_social_duties をどれだけ重視するかは desire_tendencies に依存してよい。

[出力スキーマ]
macro_desire:
  summary: "<短い要約>"
  description: "<詳細な説明>"
"""


# ========= IO helpers =========
def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(text: str, dst: Path) -> None:
    _safe_mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}-{int(time.time()*1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def _atomic_write_yaml(obj: Dict[str, Any], dst: Path) -> None:
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, default_flow_style=False)
    _atomic_write_text(text, dst)


# ========= Role canonicalization / config補完 =========
ROLE_ALIASES = {
    "werewolf": ("werewolf", "狼", "人狼"),
    "villager": ("villager", "村", "村人"),
    "possessed": ("possessed", "madman", "狂人"),
    "seer": ("seer", "占い師", "占師", "占い"),
    "knight": ("knight", "guard", "bodyguard", "騎士", "狩人", "守護"),
    "medium": ("medium", "霊媒師", "霊能"),
}

def _canonical_role_key(s: str) -> str:
    t = (s or "").strip().lower()
    for k, aliases in ROLE_ALIASES.items():
        if t == k or any(a in t for a in aliases):
            return k
    # heuristics
    if "占" in t or "seer" in t: return "seer"
    if "霊" in t or "medium" in t: return "medium"
    if "騎" in t or "狩" in t or "guard" in t: return "knight"
    if "狼" in t or "werewolf" in t: return "werewolf"
    if "狂" in t or "mad" in t: return "possessed"
    if "村" in t or "vill" in t: return "villager"
    return "villager"


def _supplement_role_definition(role: str, role_definition: str, config_data: Dict[str, Any]) -> Tuple[str, str]:
    """config.yml の role_social_duties から定義を補完し、role 名も英字に正規化した表示名を返す。"""
    if role_definition and role and role != "不明":
        return role, role_definition

    canonical = _canonical_role_key(role or "")
    rsd = config_data.get("role_social_duties", {}) if isinstance(config_data, dict) else {}
    # role_social_duties は英字キー想定
    # キー存在時のみ採用
    for key in rsd.keys():
        if _canonical_role_key(key) == canonical:
            role_definition = role_definition or rsd[key].get("definition", "") or rsd[key].get("定義", "")
            break

    # 表示用のロール名（英字タイトルケース）
    display = {
        "werewolf": "Werewolf",
        "villager": "Villager",
        "possessed": "Possessed",
        "seer": "Seer",
        "knight": "Knight",
        "medium": "Medium",
    }.get(canonical, "Villager")

    return display, role_definition


# ========= Prompt builder =========
def build_prompt(template: str, game_id: str, agent: str, role: str,
                 role_definition: str, desire_tendencies: Dict[str, float]) -> str:
    jinja_template = Template(template)
    return jinja_template.render(
        game_id=game_id,
        agent=agent,
        role=role,
        role_definition=role_definition,
        desire_tendencies=desire_tendencies or {},
    ).strip()


# ========= Robust extraction =========
def _sanitize_text(s: str) -> str:
    s = (s or "")
    s = s.lstrip("\ufeff").strip()  # BOM + outer spaces
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("…", "...")
    # 裸の 'yaml' / 'yml' 行が先頭に来る事故を除去
    s = re.sub(r"^(yaml|yml)\s*\r?\n", "", s, flags=re.IGNORECASE)
    return s


def _unfence(s: str) -> Optional[str]:
    # ```yaml ... ``` / ```yml ... ``` / ``` ... ```
    m = re.search(r"```(?:yaml|yml)?\s*\r?\n([\s\S]*?)\r?\n```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _slice_from_key(s: str, key: str = "macro_desire:") -> Optional[str]:
    i = s.lower().find(key)
    return s[i:].strip() if i != -1 else None


def extract_yaml_from_response(response: str) -> Dict[str, Any]:
    """LLM応答から YAML を頑健に抽出。失敗時は salvage して最低限の形に復元する。"""
    s = _sanitize_text(response)

    # 1) フェンス優先
    body = _unfence(s)
    if body:
        try:
            return yaml.safe_load(body) or {}
        except yaml.YAMLError:
            pass

    # 2) 全体をそのまま
    try:
        return yaml.safe_load(s) or {}
    except yaml.YAMLError:
        pass

    # 3) 'macro_desire:' 以降を切り出し
    tail = _slice_from_key(s)
    if tail:
        try:
            return yaml.safe_load(tail) or {}
        except yaml.YAMLError:
            pass

    # 4) ブルータル救済: summary/description 行を掘り出して補完
    sum_m = re.search(r"summary\s*:\s*(.+)", s, re.IGNORECASE)
    desc_m = re.search(r"description\s*:\s*(.+)", s, re.IGNORECASE)
    summary = (sum_m.group(1).strip(' "\'')) if sum_m else "Auto-generated summary"
    description = (desc_m.group(1).strip()) if desc_m else "Auto-generated description."
    return {"macro_desire": {"summary": summary, "description": description}}


# ========= Normalizer =========
def normalize_macro_desire(data: Dict[str, Any]) -> Dict[str, Any]:
    """macro_desire の2フィールドを必ず埋め、余計なフェンス/接頭語を掃除。"""
    if "macro_desire" not in data or not isinstance(data["macro_desire"], dict):
        data = {"macro_desire": (data if isinstance(data, dict) else {})}

    md = data["macro_desire"]
    md["summary"] = str(md.get("summary") or "Auto-generated summary")
    md["description"] = str(md.get("description") or "No detailed description provided")

    for k in ("summary", "description"):
        v = md.get(k, "")
        v = v.replace("```", "").strip()
        v = re.sub(r"^yaml\s*", "", v, flags=re.IGNORECASE)
        md[k] = v

    return {"macro_desire": md}


# ========= Deterministic fallback =========
def build_deterministic_macro_desire(agent: str, role: str,
                                     role_definition: str,
                                     desire_tendencies: Dict[str, float]) -> Dict[str, Any]:
    tops = sorted((desire_tendencies or {}).items(), key=lambda kv: kv[1], reverse=True)[:3]
    drivers = ", ".join(f"{k}={v:.2f}" for k, v in tops) if tops else "n/a"
    focus = tops[0][0] if tops else "stability"
    focus_disp = focus.replace("_", " ")

    summary = f"Balance {focus_disp} with {role} duties."
    description = (
        f"{agent} aims to fulfill the {role} role while prioritizing {focus_disp}. "
        f"Role duties: {role_definition or 'n/a'}. "
        f"Key drivers: {drivers}. "
        f"The agent will adjust commitment to role expectations proportionally to these tendencies."
    )
    return {"macro_desire": {"summary": summary, "description": description}}


def _is_unusable(md_obj: Dict[str, Any]) -> bool:
    s = (md_obj or {}).get("summary", "")
    d = (md_obj or {}).get("description", "")
    if "Failed to parse LLM response" in s:
        return True
    if s.startswith("Auto-generated") and len(d) < 10:
        return True
    return False


# ========= Main generator =========
def generate_macro_desire(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Generate macro desire from macro_belief data."""
    base_path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
    macro_belief_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_belief.yml"
    config_path = base_path / "config" / "config.yml"
    output_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_desire.yml"

    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to overwrite.")

    try:
        # Load inputs
        try:
            macro_belief_data = load_yaml_file(macro_belief_path)
        except Exception:
            macro_belief_data = {"macro_belief": {}}

        try:
            config_data = load_yaml_file(config_path)
        except Exception:
            config_data = {}

        # Extract from macro_belief (robust)
        m = macro_belief_data.get("macro_belief", {}) or {}
        # role
        role_data = m.get("role_social_duties", {}) or {}
        role = role_data.get("role") or m.get("role") or "不明"
        # definition
        duties = role_data.get("duties", {})
        duties = duties if isinstance(duties, dict) else {}
        role_definition = duties.get("definition") or duties.get("定義") or m.get("role_definition") or ""

        # supplement from config if missing
        role, role_definition = _supplement_role_definition(role, role_definition, config_data)

        # desire tendencies
        dt = m.get("desire_tendency", {}) or {}
        desire_tendencies = dt.get("desire_tendencies") or dt
        if not isinstance(desire_tendencies, dict):
            desire_tendencies = {}

        # Template
        prompt_template = (config_data.get("prompt", {}) or {}).get("macro_desire", FALLBACK_PROMPT_TEMPLATE) or FALLBACK_PROMPT_TEMPLATE

        # Build prompt
        prompt = build_prompt(prompt_template, game_id, agent, role, role_definition, desire_tendencies)

        if dry_run:
            print("\n" + "="*50)
            print("DRY RUN - GENERATED PROMPT:")
            print("="*50)
            print(prompt)
            print("\n" + "="*50)

        # LLM call via agent
        if agent_obj is None:
            raise ValueError("agent_obj is required. Direct API calls are not allowed.")

        extra_vars = {
            "game_id": game_id,
            "agent": agent,
            "role": role,
            "role_definition": role_definition,
            "desire_tendencies": desire_tendencies
        }
        response = agent_obj.send_message_to_llm(
            "macro_desire",
            extra_vars=extra_vars,
            log_tag="MACRO_DESIRE_GENERATION",
            use_shared_history=False
            # 温度を下げたい場合は、agent側でこのタグを見て下げる実装にする
        )
        if response is None:
            raise ValueError("Agent LLM call returned None")

        if dry_run:
            print("RAW LLM RESPONSE:")
            print("="*50)
            print(response)
            print("\n" + "="*50)

        # Parse
        parsed_data = extract_yaml_from_response(response)
        normalized_data = normalize_macro_desire(parsed_data)

        # Deterministic fallback if unusable
        if _is_unusable(normalized_data.get("macro_desire", {})):
            normalized_data = build_deterministic_macro_desire(agent, role, role_definition, desire_tendencies)

        # Meta
        final_data = {
            **normalized_data,
            "meta": {
                "game_id": game_id,
                "agent": agent,
                "model": (agent_obj.config.get("openai", {}).get("model")
                          or agent_obj.config.get("google", {}).get("model")
                          or agent_obj.config.get("ollama", {}).get("model")),
                "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                "source_macro_belief": str(macro_belief_path)
            }
        }

        if dry_run:
            print("PARSED AND NORMALIZED RESULT:")
            print("="*50)
            print(yaml.safe_dump(final_data, allow_unicode=True, sort_keys=False))
            return final_data

        _atomic_write_yaml(final_data, output_path)
        print(f"Saved macro_desire: {output_path}")
        return final_data

    except Exception as e:
        error_msg = str(e)
        print(f"Error generating macro_desire: {error_msg}")

        if not dry_run:
            # Best-effort deterministic fallback on hard error
            try:
                # try to re-read what we can to build a reasonable fallback
                role_fb, role_def_fb = "Villager", ""
                try:
                    config_data = load_yaml_file(config_path)
                except Exception:
                    config_data = {}
                role_fb, role_def_fb = _supplement_role_definition(role_fb, role_def_fb, config_data)

                macro_belief_data = macro_belief_data if "macro_belief_data" in locals() else {"macro_belief": {}}
                m = macro_belief_data.get("macro_belief", {}) or {}
                dt = m.get("desire_tendency", {}) or {}
                desire_tendencies = dt.get("desire_tendencies") or dt
                if not isinstance(desire_tendencies, dict):
                    desire_tendencies = {}

                fb_md = build_deterministic_macro_desire(agent, role_fb, role_def_fb, desire_tendencies)

                fallback_data = {
                    **fb_md,
                    "meta": {
                        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "source": "macro_desire.py fallback",
                        "game_id": game_id,
                        "agent": agent,
                        "model": (agent_obj.config.get("openai", {}).get("model")
                                  or agent_obj.config.get("google", {}).get("model")
                                  or agent_obj.config.get("ollama", {}).get("model"))
                                 if agent_obj else "unknown",
                        "fallback": True,
                        "error": error_msg[:200]
                    }
                }
                _atomic_write_yaml(fallback_data, output_path)
                print(f"Created fallback macro_desire: {output_path}")
                return fallback_data

            except Exception as fallback_error:
                print(f"Failed to write fallback macro_desire: {fallback_error}")
                raise e  # Re-raise original error
        else:
            raise e


def main():
    """Deprecated CLI function."""
    print("❌ This CLI no longer calls LLM directly.")
    print("💡 Run from Agent runtime context instead.")
    print("   Example: agent.generate_macro_desire(game_id, agent_name, agent_obj=agent)")
    return 1


if __name__ == "__main__":
    exit(main())
