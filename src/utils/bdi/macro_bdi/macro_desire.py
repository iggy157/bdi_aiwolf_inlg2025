#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro desire generation from role_social_duties and desire_tendency.

役職責任と欲求傾向からマクロ欲求を生成するスクリプト.
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
from typing import Any, Dict, List, Optional

from jinja2 import Template

# NOTE: .env loading is handled by agent.py only

# Default fallback prompt template (used when config.yml is not available)
FALLBACK_PROMPT_TEMPLATE = """あなたは社会的役割と欲求傾向にもとづき、エージェントの上位欲求（macro_desire）をYAMLで設計する専門家です。  
以下の入力を読み、**YAMLのみ**で `macro_desire` を出力してください。

[context]
- game_id: {{ game_id }}
- agent: {{ agent }}

[role_social_duties]
- role: {{ role }}
- 定義: {{ role_definition }}

[desire_tendency]
次は {{ agent }} の欲求傾向（0–1）。値が高いほど志向が強い想定です。
{% for key, value in desire_tendencies.items() -%}
  - {{ key }}: {{ "%.3f"|format(value) }}
{% endfor %}

[要件]
1) 出力は**YAMLのみ**で、以下のスキーマに沿ってください。
2) 今回のゲームにおいてrole_social_dutiesを達成する上で、どのような願望を抱くと考えられるかdesire_tendencyを参考にして記述してください。
3) role_social_dutiesをどれだけ重視するかはdesire_tendencyの値に左右されます。

[出力フォーマット]
```yaml
macro_desire:
  summary: "<短い要約>"
  description: "<詳細な説明>"
```

厳守: 出力は**YAMLのみ**。余計なテキストやコードブロック記号は不要。"""


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load YAML file with UTF-8 encoding.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Loaded YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with file_path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _safe_mkdir(p: Path) -> None:
    """Safely create directory structure.
    
    Args:
        p: Path to create
    """
    p.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(text: str, dst: Path) -> None:
    """Atomically write text to file.
    
    Args:
        text: Text content to write
        dst: Destination file path
    """
    _safe_mkdir(dst.parent)
    tmp = dst.with_suffix(dst.suffix + f".tmp-{os.getpid()}-{int(time.time()*1000)}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)  # POSIXでは原子的に置換


def _atomic_write_yaml(obj: Dict[str, Any], dst: Path) -> None:
    """Atomically write YAML data to file.
    
    Args:
        obj: Dictionary data to write as YAML
        dst: Destination file path
    """
    text = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, default_flow_style=False)
    _atomic_write_text(text, dst)


def build_prompt(
    template: str,
    game_id: str,
    agent: str,
    role: str,
    role_definition: str,
    desire_tendencies: Dict[str, float]
) -> str:
    """Build prompt using Jinja2 template.
    
    Args:
        template: Jinja2 template string
        game_id: Game ID
        agent: Agent name
        role: Role name (Japanese)
        role_definition: Role definition text
        desire_tendencies: Desire tendency values
        
    Returns:
        Built prompt string
    """
    jinja_template = Template(template)
    prompt = jinja_template.render(
        game_id=game_id,
        agent=agent,
        role=role,
        role_definition=role_definition,
        desire_tendencies=desire_tendencies
    ).strip()
    
    return prompt


def extract_numeric_value(value_str: str) -> float:
    """Extract and normalize numeric value from string.
    
    Args:
        value_str: String containing numeric value
        
    Returns:
        Normalized value (0-1 range)
    """
    # Clean the value string
    clean_value = re.sub(r'[^\d\.\-\+]', '', str(value_str).strip())
    
    if not clean_value:
        return 0.5  # Default
        
    try:
        value = float(clean_value)
        
        # Handle different scales
        if value > 1.0:
            if value <= 10.0:  # 0-10 scale
                value = value / 10.0
            elif value <= 100.0:  # 0-100 scale
                value = value / 100.0
            else:  # Assume percentage over 100
                value = min(value / 100.0, 1.0)
        
        # Clamp to 0-1 range
        return max(0.0, min(1.0, value))
        
    except ValueError:
        return 0.5  # Default


# Direct API calls removed - use agent.send_message_to_llm instead


# Direct API calls removed - use agent.send_message_to_llm instead


# Direct API calls removed - use agent.send_message_to_llm instead


def extract_yaml_from_response(response: str) -> Dict[str, Any]:
    """Extract YAML content from LLM response.
    
    Args:
        response: Raw LLM response
        
    Returns:
        Parsed YAML data
    """
    # Try to find YAML code blocks first
    yaml_patterns = [
        r'```yaml\n(.*?)\n```',
        r'```yml\n(.*?)\n```',
        r'```\n(.*?)\n```'
    ]
    
    for pattern in yaml_patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            yaml_content = match.group(1).strip()
            try:
                return yaml.safe_load(yaml_content)
            except yaml.YAMLError:
                continue
    
    # Try to parse entire response as YAML
    try:
        return yaml.safe_load(response.strip())
    except yaml.YAMLError:
        pass
    
    # Try to find JSON and convert
    json_patterns = [
        r'```json\n(.*?)\n```',
        r'\{.*\}',
    ]
    
    for pattern in json_patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            json_content = match.group(1) if '```' in pattern else match.group(0)
            try:
                return json.loads(json_content.strip())
            except json.JSONDecodeError:
                continue
    
    # Fallback: create minimal structure
    return {
        "macro_desire": {
            "summary": "Failed to parse LLM response",
            "description": f"Original response: {response[:200]}..."
        }
    }


def normalize_macro_desire(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize macro_desire data and fix numeric values.
    
    Args:
        data: Raw macro desire data
        
    Returns:
        Normalized data
    """
    if "macro_desire" not in data:
        data = {"macro_desire": data}
    
    macro_desire = data["macro_desire"]
    
    # Ensure required fields exist for new format
    if "summary" not in macro_desire:
        macro_desire["summary"] = "Generated macro desires"
    
    if "description" not in macro_desire:
        macro_desire["description"] = "No detailed description provided"
    
    return data


def generate_macro_desire(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Generate macro desire from macro_belief data.
    
    Args:
        game_id: Game ID
        agent: Agent name
        model: LLM model name
        dry_run: If True, only show prompt and response without saving
        overwrite: If True, overwrite existing files
        
    Returns:
        Generated macro desire data
    """
    import logging
    
    # Define paths
    base_path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
    macro_belief_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_belief.yml"
    config_path = base_path / "config" / "config.yml"
    output_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_desire.yml"
    
    # Check if output already exists
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to overwrite.")
    
    try:
        # Load input files
        try:
            macro_belief_data = load_yaml_file(macro_belief_path)
        except FileNotFoundError:
            print(f"Warning: macro_belief file not found: {macro_belief_path}")
            macro_belief_data = {"macro_belief": {}}
        except Exception as e:
            print(f"Warning: Failed to load macro_belief: {e}")
            macro_belief_data = {"macro_belief": {}}
        
        # Load config
        try:
            config_data = load_yaml_file(config_path)
        except FileNotFoundError:
            print("Config file not found, using fallback prompt")
            config_data = {}
        
        # Extract required data with safe fallbacks
        m = macro_belief_data.get("macro_belief", {})
        role_data = m.get("role_social_duties", {}) or {}
        role = role_data.get("role") or m.get("role") or "不明"
        duties = role_data.get("duties", {}) if isinstance(role_data.get("duties", {}), dict) else {}
        role_definition = duties.get("定義") or duties.get("definition") or m.get("role_definition") or ""
        dt = m.get("desire_tendency", {}) or {}
        desire_tendencies = dt.get("desire_tendencies") or dt  # どちらでも受ける
        if not isinstance(desire_tendencies, dict):
            desire_tendencies = {}
        
        # Debug: extracted data loaded
        
        # Get prompt template from config
        prompt_template = config_data.get("prompt", {}).get("macro_desire", FALLBACK_PROMPT_TEMPLATE)
        if not prompt_template:
            prompt_template = FALLBACK_PROMPT_TEMPLATE
            print("Warning: Using fallback prompt template")
        
        # Build prompt
        prompt = build_prompt(prompt_template, game_id, agent, role, role_definition, desire_tendencies)
        
        if dry_run:
            print("\n" + "="*50)
            print("DRY RUN - GENERATED PROMPT:")
            print("="*50)
            print(prompt)
            print("\n" + "="*50)
        
        # Call LLM via agent only
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
        )
        if response is None:
            raise ValueError("Agent LLM call returned None")
        
        if dry_run:
            print("RAW LLM RESPONSE:")
            print("="*50)
            print(response)
            print("\n" + "="*50)
        
        # Parse response
        parsed_data = extract_yaml_from_response(response)
        normalized_data = normalize_macro_desire(parsed_data)
        
        # Add metadata
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
            print(yaml.dump(final_data, allow_unicode=True, sort_keys=False))
            return final_data
        
        # Save result atomically
        _atomic_write_yaml(final_data, output_path)
        print(f"Saved macro_desire: {output_path}")
        
        return final_data
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating macro_desire: {error_msg}")
        
        if not dry_run:
            # Write fallback minimal YAML structure
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fallback_data = {
                    "macro_desire": {
                        "summary": "Auto-generated (fallback)",
                        "description": f"Fallback due to error: {error_msg[:100]}"
                    },
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