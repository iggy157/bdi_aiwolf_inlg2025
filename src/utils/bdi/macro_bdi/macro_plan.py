#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Macro plan generation from behavior_tendencies.

行動傾向からマクロプランを生成するスクリプト.
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
FALLBACK_PROMPT_TEMPLATE = """あなたは人狼ゲームにおけるエージェントのマクロ視点(ゲーム全体を通して)の行動計画を設計する戦略プランナーです。
出力は有効な YAML のみとし、説明やマークダウンは含めないでください。

# Context
game_id: {{ game_id }}
agent: {{ agent }}

# Macro Desires
Summary: {{ desire_summary }}
Description: {{ desire_description }}

# Behavior Tendencies
{% for key, value in behavior_tendencies.items() -%}
{{ key }}: {{ "%.3f"|format(value) }}
{% endfor %}

# Task
上記のマクロ欲求（macro desires）を達成するための具体的なプランを考えてください。またその際には行動傾向（behavior tendencies）に矛盾することない現実的なプランとなるようにしてください。
人狼ゲームを通してのマクロな視点でのプランを YAML で作成してください。

[要件]
- 各プランは次の項目を含む:
  - ラベル: "プランの名前"
  - トリガ・イベント: プラン実行の条件
  - 前提条件: そのプランが選ばれる条件
  - 本体: 副目標あるいは基本行為の列（順に達成する）
    - 副目標: その目標を生成し、手段を再帰的に選んで達成
    - 基本行為: 実行
- 3〜6件程度のプランを提示
- 人狼ゲームにおける村陣営・人狼陣営どちらでも汎用的に解釈できる macro な目標にする
- 役割逸脱・反ゲーム的行動は禁止

[出力スキーマ]
```yaml
macro_plan:
  plans:
    - label: "<プラン名>"
      trigger_event: "<条件>"
      preconditions:
        - "<条件1>"
      body:
        - type: "subgoal" | "basic_action"
          description: "<説明>"
  notes: "<任意の補足>"
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
    behavior_tendencies: Dict[str, float],
    desire_summary: str,
    desire_description: str
) -> str:
    """Build prompt using Jinja2 template.
    
    Args:
        template: Jinja2 template string
        game_id: Game ID
        agent: Agent name
        behavior_tendencies: Behavior tendency values
        desire_summary: Summary of desires
        desire_description: Description of desires
        
    Returns:
        Built prompt string
    """
    jinja_template = Template(template)
    prompt = jinja_template.render(
        game_id=game_id,
        agent=agent,
        behavior_tendencies=behavior_tendencies,
        desire_summary=desire_summary,
        desire_description=desire_description
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
        "macro_plan": {
            "plans": [
                {
                    "label": "default_plan",
                    "trigger_event": "Default due to parse failure",
                    "preconditions": ["Parse failed"],
                    "body": [
                        {
                            "type": "basic_action",
                            "description": "Default action due to parsing failure"
                        }
                    ]
                }
            ],
            "notes": f"Original response: {response[:200]}..."
        }
    }


def normalize_macro_plan(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize macro_plan data and fix structure.
    
    Args:
        data: Raw macro plan data
        
    Returns:
        Normalized data
    """
    if "macro_plan" not in data:
        data = {"macro_plan": data}
    
    macro_plan = data["macro_plan"]
    
    # Ensure required fields exist
    if "plans" not in macro_plan:
        macro_plan["plans"] = []
    
    # Normalize plan entries
    for plan in macro_plan["plans"]:
        if "label" not in plan:
            plan["label"] = "unnamed_plan"
        
        if "trigger_event" not in plan:
            plan["trigger_event"] = "No trigger specified"
            
        if "preconditions" not in plan:
            plan["preconditions"] = []
            
        if "body" not in plan:
            plan["body"] = []
            
        # Normalize body entries
        for body_item in plan["body"]:
            if "type" not in body_item:
                body_item["type"] = "basic_action"
            if "description" not in body_item:
                body_item["description"] = "No description"
    
    # Ensure notes exist
    if "notes" not in macro_plan:
        macro_plan["notes"] = "Generated automatically"
    
    return data


def generate_macro_plan(
    game_id: str,
    agent: str,
    agent_obj,
    dry_run: bool = False,
    overwrite: bool = False
) -> Dict[str, Any]:
    """Generate macro plan from macro_belief behavior_tendencies and macro_desire data.
    
    Args:
        game_id: Game ID
        agent: Agent name
        model: LLM model name
        dry_run: If True, only show prompt and response without saving
        overwrite: If True, overwrite existing files
        
    Returns:
        Generated macro plan data
    """
    import logging
    
    # Define paths
    base_path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
    macro_belief_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_belief.yml"
    macro_desire_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_desire.yml"
    config_path = base_path / "config" / "config.yml"
    output_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_plan.yml"
    
    # Check if output already exists
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to overwrite.")
    
    try:
        # Load input files with fallbacks
        try:
            macro_belief_data = load_yaml_file(macro_belief_path)
        except FileNotFoundError:
            print(f"Warning: macro_belief file not found: {macro_belief_path}")
            macro_belief_data = {"macro_belief": {}}
        except Exception as e:
            print(f"Warning: Failed to load macro_belief: {e}")
            macro_belief_data = {"macro_belief": {}}
        
        # Load macro_desire
        try:
            macro_desire_data = load_yaml_file(macro_desire_path)
        except FileNotFoundError:
            print(f"Warning: macro_desire file not found, generating it first: {macro_desire_path}")
            # Try to generate macro_desire first
            try:
                from utils.bdi.macro_bdi.macro_desire import generate_macro_desire
                macro_desire_data = generate_macro_desire(
                    game_id=game_id,
                    agent=agent,
                    model=model,
                    dry_run=False,
                    overwrite=False,
                    agent_obj=agent_obj
                )
            except Exception as desire_error:
                print(f"Failed to generate macro_desire: {desire_error}")
                macro_desire_data = {"macro_desire": {"summary": "", "description": ""}}
        except Exception as e:
            print(f"Warning: Failed to load macro_desire: {e}")
            macro_desire_data = {"macro_desire": {"summary": "", "description": ""}}
        
        # Load config
        try:
            config_data = load_yaml_file(config_path)
        except FileNotFoundError:
            print("Config file not found, using fallback prompt")
            config_data = {}
        
        # Extract required data with safe fallbacks
        m = macro_belief_data.get("macro_belief", {})
        behavior_tendency = m.get("behavior_tendency", {}) or {}
        behavior_tendencies = behavior_tendency.get("behavior_tendencies") or behavior_tendency or {}
        if not isinstance(behavior_tendencies, dict):
            behavior_tendencies = {}
        
        md = macro_desire_data.get("macro_desire", {})
        desire_summary = md.get("summary", "")
        desire_description = md.get("description", "")
        
        # Debug: extracted data loaded
        
        # Get prompt template from config
        prompt_template = config_data.get("prompt", {}).get("macro_plan", FALLBACK_PROMPT_TEMPLATE)
        if not prompt_template:
            prompt_template = FALLBACK_PROMPT_TEMPLATE
            print("Warning: Using fallback prompt template")
        
        # Build prompt
        prompt = build_prompt(prompt_template, game_id, agent, behavior_tendencies, desire_summary, desire_description)
        
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
            "behavior_tendencies": behavior_tendencies,
            "desire_summary": desire_summary,
            "desire_description": desire_description
        }
        response = agent_obj.send_message_to_llm(
            "macro_plan",
            extra_vars=extra_vars,
            log_tag="MACRO_PLAN_GENERATION",
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
        normalized_data = normalize_macro_plan(parsed_data)
        
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
                "source_macro_belief": str(macro_belief_path),
                "source_macro_desire": str(macro_desire_path)
            }
        }
        
        if dry_run:
            print("PARSED AND NORMALIZED RESULT:")
            print("="*50)
            print(yaml.dump(final_data, allow_unicode=True, sort_keys=False))
            return final_data
        
        # Save result atomically
        _atomic_write_yaml(final_data, output_path)
        print(f"Saved macro_plan: {output_path}")
        
        return final_data
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating macro_plan: {error_msg}")
        
        if not dry_run:
            # Write fallback minimal YAML structure
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fallback_data = {
                    "macro_plan": {
                        "plans": [
                            {
                                "label": "fallback_plan",
                                "trigger_event": "startup",
                                "preconditions": [],
                                "body": [
                                    {
                                        "type": "basic_action",
                                        "description": "Fallback due to error"
                                    }
                                ]
                            }
                        ],
                        "notes": "auto-generated fallback"
                    },
                    "meta": {
                        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "source": "macro_plan.py fallback",
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
                print(f"Created fallback macro_plan: {output_path}")
                return fallback_data
                
            except Exception as fallback_error:
                print(f"Failed to write fallback macro_plan: {fallback_error}")
                raise e  # Re-raise original error
        else:
            raise e


def main():
    """Deprecated CLI function."""
    print("❌ This CLI no longer calls LLM directly.")
    print("💡 Run from Agent runtime context instead.")
    print("   Example: agent.generate_macro_plan(game_id, agent_name, agent_obj=agent)")
    return 1


if __name__ == "__main__":
    exit(main())