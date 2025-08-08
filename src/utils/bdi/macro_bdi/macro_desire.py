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

import requests
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / "config" / ".env")

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
2) `desires[].name` は **snake_case** の短い動詞句（例: `protect_citizens`, `coordinate_with_allies`）。
3) `desires[].strength` は 0–1 に正規化された数値。
4) `rationale` に役割定義・傾向値からの根拠を簡潔に記載。
5) 役割と定義に整合する上位欲求（2〜6個）を出力。
6) 可能なら `constraints` と `notes` も含める。

[出力フォーマット]
```yaml
macro_desire:
  summary: "<短い要約>"
  desires:
    - name: "<snake_case>"
      strength: <0.0-1.0>
      rationale: "<根拠>"
  constraints:
    - "<任意>"
  notes: "<任意>"
```

[作成の指針]
* 村人(role={{ role }})であれば「観察・推論・社会的意思決定への参加」を重視。
* 高い `freedom_independence` や `adventure_stimulation` は探索・リスク許容の欲求を強める。
* 低い `stability` / `stable_relationships` は過度な保守に偏らない姿勢。
* 反社会的・役割逸脱的な目標は禁止。

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


def call_openai_api(prompt: str, model: str, max_retries: int = 3) -> str:
    """Call OpenAI API with exponential backoff retry.
    
    Args:
        prompt: Input prompt
        model: Model name (e.g., gpt-4o)
        max_retries: Maximum number of retries
        
    Returns:
        API response text
        
    Raises:
        Exception: If all retries fail
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def call_anthropic_api(prompt: str, model: str, max_retries: int = 3) -> str:
    """Call Anthropic API with exponential backoff retry.
    
    Args:
        prompt: Input prompt
        model: Model name (e.g., claude-3-5-sonnet)
        max_retries: Maximum number of retries
        
    Returns:
        API response text
        
    Raises:
        Exception: If all retries fail
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            return data["content"][0]["text"]
            
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def call_llm_api(prompt: str, model: str) -> str:
    """Call appropriate LLM API based on model name.
    
    Args:
        prompt: Input prompt
        model: Model name
        
    Returns:
        API response text
    """
    model_lower = model.lower()
    
    if any(provider in model_lower for provider in ["gpt", "openai"]):
        return call_openai_api(prompt, model)
    elif any(provider in model_lower for provider in ["claude", "anthropic"]):
        return call_anthropic_api(prompt, model)
    else:
        # Default to OpenAI if unclear
        print(f"Unknown model provider for '{model}', defaulting to OpenAI API")
        return call_openai_api(prompt, model)


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
            "desires": [
                {
                    "name": "default_desire",
                    "strength": 0.5,
                    "rationale": "Default due to parse failure"
                }
            ],
            "constraints": [],
            "notes": f"Original response: {response[:200]}..."
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
    
    # Ensure required fields exist
    if "desires" not in macro_desire:
        macro_desire["desires"] = []
    
    if "summary" not in macro_desire:
        macro_desire["summary"] = "Generated macro desires"
    
    # Normalize desire entries
    for desire in macro_desire["desires"]:
        if "strength" in desire:
            desire["strength"] = extract_numeric_value(str(desire["strength"]))
        else:
            desire["strength"] = 0.5
        
        if "name" not in desire:
            desire["name"] = "unnamed_desire"
        
        if "rationale" not in desire:
            desire["rationale"] = "No rationale provided"
    
    # Ensure constraints and notes exist
    if "constraints" not in macro_desire:
        macro_desire["constraints"] = []
    
    if "notes" not in macro_desire:
        macro_desire["notes"] = "Generated automatically"
    
    return data


def generate_macro_desire(
    game_id: str,
    agent: str,
    model: str,
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
    # Define paths
    base_path = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025")
    macro_belief_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_belief.yml"
    config_path = base_path / "config" / "config.yml"
    output_path = base_path / "info" / "bdi_info" / "macro_bdi" / game_id / agent / "macro_desire.yml"
    
    # Check if output already exists
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to overwrite.")
    
    # Load input files
    print(f"Loading macro_belief from: {macro_belief_path}")
    macro_belief_data = load_yaml_file(macro_belief_path)
    
    print(f"Loading config from: {config_path}")
    try:
        config_data = load_yaml_file(config_path)
    except FileNotFoundError:
        print("Config file not found, using fallback prompt")
        config_data = {}
    
    # Extract required data
    role_data = macro_belief_data["macro_belief"]["role_social_duties"]
    role = role_data["role"]
    role_definition = role_data["duties"]["定義"]
    desire_tendencies = macro_belief_data["macro_belief"]["desire_tendency"]["desire_tendencies"]
    
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
    
    # Call LLM
    print(f"Calling {model} API...")
    response = call_llm_api(prompt, model)
    
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
            "model": model,
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


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate macro_desire from role_social_duties and desire_tendency"
    )
    parser.add_argument("--game_id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--model", required=True, help="LLM model name (e.g., gpt-4o, claude-3-5-sonnet)")
    parser.add_argument("--dry-run", action="store_true", help="Show prompt and response only, don't save")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    
    try:
        result = generate_macro_desire(
            game_id=args.game_id,
            agent=args.agent,
            model=args.model,
            dry_run=args.dry_run,
            overwrite=args.overwrite
        )
        
        if args.dry_run:
            print("\n✅ Dry run completed successfully!")
        else:
            print("✅ Macro desire generation completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())