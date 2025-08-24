# -*- coding: utf-8 -*-
"""
macro_desire generator (role-aware, situation-wise, one-line desire injection)

- Reads role / tendencies from macro_belief.yml
- Loads role-specific "conventional_wisdom/*.yml" (situations with title/derivation_method)
- For each situation, asks LLM (via agent.send_message_to_llm) to infer a **single-sentence desire**
  based on behavior_tendency + desire_tendency (+ situation context if available)
- Writes macro_desire.yml:
    macro_desire:
      role: "<RoleName>"
      items:
        - title: "<situation title>"
          derivation_method: "<how to derive>"
          desire: "<one-line desire>"

LLM prompt policy:
- Primary prompt key: "macro_desire_one_liner" (recommended to be defined in config/config.yml).
- Fallback: "macro_desire" (existing template). In this case we parse YAML and take the first sentence
  of `macro_desire.description` as the one-line desire.

Notes:
- No direct access to API keys here; agent_obj.send_message_to_llm handles the LLM call using config/.env.
- Robust parsing against multiple possible macro_belief.yml shapes.

Author: (you)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import yaml


# ---------- datatypes ----------

@dataclass
class Situation:
    title: str
    derivation_method: str


@dataclass
class RoleConventional:
    role: str
    situations: List[Situation]


# ---------- file utils ----------

def _load_yaml_safe(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _dump_yaml_safe(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


# ---------- macro_belief parsing ----------

def _extract_role_from_macro_belief(mb: dict, fallback_role_name: Optional[str] = None) -> Optional[str]:
    """
    Try multiple known shapes:
    - {"macro_belief": {"role": "Villager"}}
    - {"macro_belief": {"role": {"name": "Villager"} } }
    - {"role": "Villager"}
    """
    if not mb:
        return fallback_role_name

    if "macro_belief" in mb and isinstance(mb["macro_belief"], dict):
        role_val = mb["macro_belief"].get("role")
        if isinstance(role_val, dict):
            for k in ("name", "value", "role", "en", "role_name"):
                if k in role_val and isinstance(role_val[k], str):
                    return role_val[k]
        if isinstance(role_val, str):
            return role_val

    # Flat fallback
    for k in ("role", "role_name"):
        if isinstance(mb.get(k), str):
            return mb[k]

    return fallback_role_name


def _extract_tendencies(mb: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Return (behavior_tendency, desire_tendency) as dicts of str -> float.
    Accept multiple possible shapes:
    - mb["macro_belief"]["behavior_tendency"] or ["behavior_tendencies"]
    - top-level variants
    """
    def _pick(d: dict, *paths: Tuple[str, ...]) -> Optional[dict]:
        for p in paths:
            if p in d and isinstance(d[p], dict):
                return d[p]
        return None

    root = mb.get("macro_belief", {}) if isinstance(mb.get("macro_belief"), dict) else mb

    beh = _pick(root, "behavior_tendency", "behavior_tendencies") or {}
    des = _pick(root, "desire_tendency", "desire_tendencies") or {}

    # normalize numeric values (clip 0..1 softly)
    def _norm_map(x: dict) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in (x or {}).items():
            try:
                fv = float(v)
                if fv < 0.0: fv = 0.0
                if fv > 1.0: fv = 1.0
                out[str(k)] = fv
            except Exception:
                # ignore non-numeric
                continue
        return out

    return _norm_map(beh), _norm_map(des)


# ---------- role ↔ file mapping ----------

_ROLE_FILE_MAP = {
    # Town
    "villager": "villager.yml",
    "medium": "medium.yml",
    "seer": "seer.yml",
    "bodyguard": "bodyguard.yml",
    "knight": "bodyguard.yml",   # alias
    # Wolf side
    "werewolf": "werewolf.yml",
    "possessed": "possessed.yml",
    "madman": "possessed.yml",   # alias
}

def _map_role_to_file(role_name: str) -> Optional[str]:
    key = (role_name or "").strip().lower()
    return _ROLE_FILE_MAP.get(key)


# ---------- conventional wisdom loading ----------

def _load_role_conventional(path: Path) -> Optional[RoleConventional]:
    data = _load_yaml_safe(path)
    if not data:
        return None
    role = str(data.get("role") or data.get("Role") or "")
    raw_situ = data.get("situations") or []
    situations: List[Situation] = []
    if isinstance(raw_situ, list):
        for item in raw_situ:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("タイトル") or "").strip()
            deriv = str(item.get("derivation_method") or item.get("導出方法") or "").strip()
            if title:
                situations.append(Situation(title=title, derivation_method=deriv))
    return RoleConventional(role=role, situations=situations)


# ---------- LLM helpers ----------

_SENT_SPLIT_RE = re.compile(r"(?<=[。．\.!?！？])\s+")

def _take_first_sentence(text: str) -> str:
    text = (text or "").strip().strip('"').strip("'")
    if not text:
        return text
    # Try to split on Japanese/English sentence enders
    parts = _SENT_SPLIT_RE.split(text)
    if parts:
        return parts[0].strip()
    return text


def _extract_one_liner_from_response(resp: Optional[str]) -> str:
    """
    Accepts:
    - A raw final line (e.g., "Final: <one line>")
    - Multi-line reasoning; take the last non-empty line
    - YAML (macro_desire: description: "...") -> take first sentence of description
    """
    if not resp:
        return ""

    s = str(resp).strip()

    # Case 1: YAML-like
    if s.startswith("macro_desire:"):
        try:
            y = yaml.safe_load(s)
            desc = ""
            if isinstance(y, dict):
                md = y.get("macro_desire") or {}
                if isinstance(md, dict):
                    desc = str(md.get("description") or "").strip()
            return _take_first_sentence(desc)
        except Exception:
            pass

    # Case 2: "Final: ..."
    m = re.search(r"Final\s*:\s*(.+)$", s, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return _take_first_sentence(m.group(1))

    # Case 3: take the last non-empty line
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        return _take_first_sentence(lines[-1])

    return ""


# ---------- main API ----------

def generate_macro_desire(
    *,
    game_id: str,
    agent: str,
    agent_obj: Any,     # instance of Agent (has send_message_to_llm)
    dry_run: bool = False,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Build macro_desire.yml from role-specific conventional wisdom + one-line LLM desires.

    Returns the output path if successful, else None.
    """
    # --- paths ---
    base = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
    macro_dir = base / "info/bdi_info/macro_bdi" / game_id / agent
    out_path = macro_dir / "macro_desire.yml"
    mb_path = macro_dir / "macro_belief.yml"

    if out_path.exists() and not overwrite:
        return out_path

    # --- load macro_belief (role + tendencies) ---
    mb = _load_yaml_safe(mb_path)
    # fallback role from agent_obj.role if available
    fallback_role = getattr(getattr(agent_obj, "role", None), "name", None)
    role_name = _extract_role_from_macro_belief(mb, fallback_role_name=fallback_role) or "Villager"
    behavior_tendency, desire_tendency = _extract_tendencies(mb)

    # --- map role to conventional file ---
    file_name = _map_role_to_file(role_name)
    if not file_name:
        # default to villager if unmapped role
        file_name = "villager.yml"

    conv_path = base / "info/conventional_wisdom" / file_name
    conv = _load_role_conventional(conv_path)
    if not conv or not conv.situations:
        # Write minimal file and return
        minimal = {
            "macro_desire": {
                "role": role_name,
                "items": []
            },
            "meta": {
                "note": f"No conventional wisdom found for role={role_name}",
                "source_conventional_file": str(conv_path),
            }
        }
        if not dry_run:
            _dump_yaml_safe(out_path, minimal)
        return out_path if out_path.exists() else None

    # --- prepare items by LLM ---
    items_out: List[Dict[str, str]] = []

    # Preferred prompt key (must be defined in config/prompt if you want the exact JP instruction)
    primary_prompt_key = "macro_desire_one_liner"
    fallback_prompt_key = "macro_desire"  # existing template that returns YAML with description

    # Iterate situations; for each, ask LLM for one-line desire
    for situ in conv.situations:
        desire_line = ""

        # Try primary prompt first
        extra_vars_primary = {
            "game_id": game_id,
            "agent": agent,
            # custom vars for your template
            "role_name": role_name,
            "situation_title": situ.title,
            "situation_derivation": situ.derivation_method,
            "behavior_tendency": behavior_tendency,
            "desire_tendency": desire_tendency,
        }
        resp = agent_obj.send_message_to_llm(
            primary_prompt_key,
            extra_vars=extra_vars_primary,
            log_tag="macro_desire_one_liner",
            use_shared_history=False,
        )

        if resp is None:
            # Fallback to the existing macro_desire prompt (YAML macro_desire: description: ...)
            # It expects: role, role_definition, desire_tendencies
            # We don't have role_definition here; pass a short default to avoid template errors.
            extra_vars_fallback = {
                "game_id": game_id,
                "agent": agent,
                "role": role_name,
                "role_definition": f"Role {role_name}",
                "desire_tendencies": desire_tendency or {},  # template expects this key
            }
            resp = agent_obj.send_message_to_llm(
                fallback_prompt_key,
                extra_vars=extra_vars_fallback,
                log_tag="macro_desire_fallback",
                use_shared_history=False,
            )

        desire_line = _extract_one_liner_from_response(resp) or "Act in a way that maximizes expected team equity."

        # keep single line; strip quotes
        desire_line = re.sub(r"\s+", " ", desire_line).strip().strip('"').strip("'")

        items_out.append({
            "title": situ.title,
            "derivation_method": situ.derivation_method,
            "desire": desire_line
        })

    # --- assemble output ---
    out_data = {
        "macro_desire": {
            "role": role_name,
            "items": items_out
        },
        "meta": {
            "source": "macro_desire.py",
            "conventional_source": str(conv_path),
            "macro_belief_source": str(mb_path),
        }
    }

    if not dry_run:
        _dump_yaml_safe(out_path, out_data)

    return out_path
