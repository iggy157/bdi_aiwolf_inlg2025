#!/usr/bin/env python3
"""Update affinity (liking) and trust (creditability) scores in talk history files."""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# Constants
BASE_INFO_DIR = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
BASE_MACRO_DIR = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_yaml_safe(file_path: Path) -> Optional[Any]:
    """Safely load YAML file."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def load_talk_history(file_path: Path) -> tuple[list[dict], Optional[float], Optional[float]]:
    """Load talk history from YAML file.
    
    Returns:
        (items, existing_liking, existing_creditability)
    """
    data = load_yaml_safe(file_path)
    if not data:
        return [], None, None
    
    # Handle list format
    if isinstance(data, list):
        items = []
        for item in data:
            if isinstance(item, dict):
                content = item.get('content', '')
                # Handle both creditability and credibility
                cred = item.get('creditability', item.get('credibility', 0.0))
                items.append({'content': str(content), 'creditability': float(cred)})
        return items, None, None
    
    # Handle dict format with items
    if isinstance(data, dict):
        existing_liking = data.get('liking')
        existing_cred = data.get('creditability')
        
        items_data = data.get('items', [])
        items = []
        for item in items_data:
            if isinstance(item, dict):
                content = item.get('content', '')
                cred = item.get('creditability', item.get('credibility', 0.0))
                items.append({'content': str(content), 'creditability': float(cred)})
        return items, existing_liking, existing_cred
    
    return [], None, None


def load_macro_belief(game_id: str, agent: str) -> tuple[dict[str, float], dict[str, float]]:
    """Load cognitive bias weights from macro_belief.yml.
    
    Returns:
        (liking_weights, trust_weights)
    """
    macro_path = BASE_MACRO_DIR / game_id / agent / "macro_belief.yml"
    data = load_yaml_safe(macro_path)
    
    liking_weights = {}
    trust_weights = {}
    
    if data and isinstance(data, dict):
        cognitive_bias = data.get('cognitive_bias', {})
        
        # Extract liking tendency weights
        liking_tendency = cognitive_bias.get('liking_tendency', {})
        if isinstance(liking_tendency, dict):
            for k, v in liking_tendency.items():
                try:
                    liking_weights[k] = float(v)
                except (TypeError, ValueError):
                    liking_weights[k] = 0.5
        
        # Extract trust tendency weights
        trust_tendency = cognitive_bias.get('trust_tendency', {})
        if isinstance(trust_tendency, dict):
            for k, v in trust_tendency.items():
                try:
                    trust_weights[k] = float(v)
                except (TypeError, ValueError):
                    trust_weights[k] = 0.5
    
    # If no weights found, use equal weights for common keys
    if not liking_weights:
        liking_weights = {
            'friendliness': 1.0/3,
            'emotional_resonance': 1.0/3,
            'attractive_expression': 1.0/3
        }
    
    if not trust_weights:
        trust_weights = {
            'social_proof': 1.0/3,
            'honesty': 1.0/3,
            'consistency': 1.0/3
        }
    
    return liking_weights, trust_weights


def initialize_llm(config: dict) -> Any:
    """Initialize LLM client based on config."""
    load_dotenv(Path(__file__).parent.parent.parent.parent.parent / "config" / ".env")
    
    model_type = str(config.get("llm", {}).get("type", "openai"))
    logger.info(f"Initializing LLM with type: {model_type}")
    
    try:
        match model_type:
            case "openai":
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found")
                return ChatOpenAI(
                    model=str(config.get("openai", {}).get("model", "gpt-4o")),
                    temperature=float(config.get("openai", {}).get("temperature", 0.7)),
                    api_key=SecretStr(api_key),
                )
            case "google":
                api_key = os.environ.get("GOOGLE_API_KEY", "")
                if not api_key:
                    logger.warning("GOOGLE_API_KEY not found")
                return ChatGoogleGenerativeAI(
                    model=str(config.get("google", {}).get("model", "gemini-1.5-pro")),
                    temperature=float(config.get("google", {}).get("temperature", 0.7)),
                    google_api_key=SecretStr(api_key),
                )
            case "ollama":
                return ChatOllama(
                    model=str(config.get("ollama", {}).get("model", "llama3.1")),
                    temperature=float(config.get("ollama", {}).get("temperature", 0.7)),
                )
            case _:
                raise ValueError(f"Unknown LLM type: {model_type}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def extract_json_from_response(response: str) -> Optional[dict]:
    """Extract JSON from LLM response, handling various formats."""
    if not response:
        return None
    
    # Remove code blocks
    response = re.sub(r'```json?\s*', '', response)
    response = re.sub(r'```\s*', '', response)
    
    # Try to find JSON structure
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Try direct parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    return None


def evaluate_with_llm(
    llm_model: Any,
    config: dict,
    from_agent: str,
    agent: str,
    talks: list[str],
    liking_keys: list[str],
    trust_keys: list[str],
    agent_logger: Optional[Any] = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Evaluate affinity and trust scores using LLM."""
    
    # Get prompt template from config
    prompt_template = config.get("prompt", {}).get("affinity_trust_eval", "")
    if not prompt_template:
        # Default prompt if not in config
        prompt_template = """
あなたは会話分析の評価者です。対象人物「{{ from_agent }}」の発話履歴（時系列）に基づき、
下記の評価項目ごとに 0〜1 の実数でスコア化してください（0=非常に低い, 1=非常に高い）。
出力は **厳密な JSON** のみ。余計な説明やコメントは不要。

[評価方針]
- スコアは発話の内容・一貫性・態度・推論可能な意図を総合的に考慮
- 0〜1の範囲にクリップ
- 不明な場合は 0.5 とする

[liking 側の項目]
{% for k in liking_keys %}- {{ k }}
{% endfor %}

[trust 側の項目]
{% for k in trust_keys %}- {{ k }}
{% endfor %}

[対象の発話（古い→新しい順）]
{% for t in talks %}
- {{ t }}
{% endfor %}

[出力フォーマット（厳密JSON）]
{
  "liking": {
    {% for k in liking_keys %}"{{ k }}": 0.5{% if not loop.last %},{% endif %}
    {% endfor %}
  },
  "trust": {
    {% for k in trust_keys %}"{{ k }}": 0.5{% if not loop.last %},{% endif %}
    {% endfor %}
  }
}
"""
    
    # Render prompt
    template = Template(prompt_template)
    prompt = template.render(
        from_agent=from_agent,
        agent=agent,
        talks=talks,
        liking_keys=liking_keys,
        trust_keys=trust_keys
    )
    
    # Call LLM
    try:
        messages = [HumanMessage(content=prompt)]
        response = (llm_model | StrOutputParser()).invoke(messages)
        
        # Log LLM interaction if agent_logger available
        if agent_logger and hasattr(agent_logger, 'llm_interaction'):
            model_info = str(type(llm_model).__name__)
            agent_logger.llm_interaction("affinity_trust_eval", prompt, response, model_info)
        
        # Parse response
        result = extract_json_from_response(response)
        if not result:
            logger.warning("Failed to parse JSON from LLM response")
            return {}, {}
        
        # Extract scores with validation
        liking_scores = {}
        trust_scores = {}
        
        if isinstance(result.get('liking'), dict):
            for k in liking_keys:
                val = result['liking'].get(k, 0.5)
                try:
                    liking_scores[k] = max(0.0, min(1.0, float(val)))
                except (TypeError, ValueError):
                    liking_scores[k] = 0.5
        
        if isinstance(result.get('trust'), dict):
            for k in trust_keys:
                val = result['trust'].get(k, 0.5)
                try:
                    trust_scores[k] = max(0.0, min(1.0, float(val)))
                except (TypeError, ValueError):
                    trust_scores[k] = 0.5
        
        # Fill missing keys with 0.5
        for k in liking_keys:
            if k not in liking_scores:
                liking_scores[k] = 0.5
        
        for k in trust_keys:
            if k not in trust_scores:
                trust_scores[k] = 0.5
        
        return liking_scores, trust_scores
        
    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")
        return {k: 0.5 for k in liking_keys}, {k: 0.5 for k in trust_keys}


def calculate_weighted_average(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Calculate weighted average of scores."""
    if not scores or not weights:
        return 0.5
    
    total_score = 0.0
    total_weight = 0.0
    
    for key, score in scores.items():
        weight = weights.get(key, 0.0)
        total_score += score * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.5
    
    return round(total_score / total_weight, 3)


def save_talk_history_with_headers(
    file_path: Path,
    items: list[dict],
    liking: float,
    creditability: float
) -> None:
    """Save talk history with liking and creditability headers."""
    data = {
        'liking': liking,
        'creditability': creditability,
        'items': items
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("# generated by utils.bdi.micro_bdi.affinity_trust_updater\n")
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        logger.info(f"Saved {file_path} with liking={liking}, creditability={creditability}")
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")


def update_affinity_trust_for_agent(
    config: dict,
    game_id: str,
    agent: str,
    agent_logger: Optional[Any] = None,
    talk_dir_name: str = "talk_history",
    agent_obj: Optional[Any] = None
) -> None:
    """Update affinity and trust scores for all talk history files of an agent."""
    
    # Find talk history directory (try talk_history first, then トーク履歴)
    base_agent_dir = BASE_INFO_DIR / game_id / agent
    talk_dir = base_agent_dir / talk_dir_name
    
    if not talk_dir.exists():
        # Fallback to トーク履歴
        talk_dir = base_agent_dir / "トーク履歴"
        if not talk_dir.exists():
            logger.warning(f"Talk history directory not found for {game_id}/{agent}")
            return
    
    logger.info(f"Processing talk history in {talk_dir}")
    
    # Load macro belief weights
    liking_weights, trust_weights = load_macro_belief(game_id, agent)
    liking_keys = list(liking_weights.keys())
    trust_keys = list(trust_weights.keys())
    
    logger.info(f"Loaded weights - liking: {liking_keys}, trust: {trust_keys}")
    
    # Initialize LLM for fallback (even if agent_obj is available)
    llm_model = None
    try:
        llm_model = initialize_llm(config)
        logger.info("LLM initialized for fallback")
    except Exception as e:
        logger.warning(f"Failed to initialize fallback LLM: {e}")
        llm_model = None
    
    # Process each talk history file
    for yml_file in talk_dir.glob("*.yml"):
        try:
            from_agent = yml_file.stem
            logger.info(f"Processing {from_agent}.yml")
            
            # Load existing data
            items, existing_liking, existing_cred = load_talk_history(yml_file)
            
            if not items:
                logger.warning(f"No items in {yml_file}, skipping")
                continue
            
            # Check if update needed (optional optimization)
            # For now, always update
            
            # Prepare talks for LLM
            talks = [item['content'] for item in items]
            
            # Initialize scores variables
            liking_scores = {}
            trust_scores = {}
            
            # Evaluate with LLM - try agent_obj first, then fallback to direct LLM
            if agent_obj is not None and talks:
                # Use Agent's centralized LLM calling method
                extra_vars = {
                    "from_agent": from_agent,
                    "talks": talks,
                    "liking_keys": liking_keys,
                    "trust_keys": trust_keys
                }
                logger.info(f"Trying agent_obj.send_message_to_llm for {from_agent} with {len(talks)} talks")
                
                try:
                    response = agent_obj.send_message_to_llm(
                        "affinity_trust_eval",
                        extra_vars=extra_vars,
                        log_tag="affinity_trust_eval"
                    )
                    
                    parsed = extract_json_from_response(response) if response else None
                    if parsed and isinstance(parsed.get('liking'), dict) and isinstance(parsed.get('trust'), dict):
                        logger.info(f"Successfully evaluated via agent_obj for {from_agent}")
                        
                        # Extract liking scores
                        for k in liking_keys:
                            val = parsed['liking'].get(k, 0.5)
                            try:
                                liking_scores[k] = max(0.0, min(1.0, float(val)))
                            except (TypeError, ValueError):
                                liking_scores[k] = 0.5
                        
                        # Extract trust scores
                        for k in trust_keys:
                            val = parsed['trust'].get(k, 0.5)
                            try:
                                trust_scores[k] = max(0.0, min(1.0, float(val)))
                            except (TypeError, ValueError):
                                trust_scores[k] = 0.5
                    else:
                        logger.warning(f"affinity_trust_eval via agent_obj failed; falling back to direct LLM for {from_agent}")
                        if llm_model is None:
                            try:
                                llm_model = initialize_llm(config)
                            except Exception as e:
                                logger.error(f"Failed to init llm_model: {e}")
                        
                        if llm_model:
                            liking_scores, trust_scores = evaluate_with_llm(
                                llm_model, config, from_agent, agent,
                                talks, liking_keys, trust_keys, agent_logger
                            )
                            logger.info(f"evaluated via direct LLM for {from_agent}")
                        else:
                            logger.error(f"No LLM available; falling back to 0.5 for {from_agent}")
                            liking_scores = {k: 0.5 for k in liking_keys}
                            trust_scores = {k: 0.5 for k in trust_keys}
                except Exception as e:
                    logger.warning(f"agent_obj.send_message_to_llm failed for {from_agent}: {e}; falling back to direct LLM")
                    if llm_model is None:
                        try:
                            llm_model = initialize_llm(config)
                        except Exception as init_e:
                            logger.error(f"Failed to init llm_model: {init_e}")
                    
                    if llm_model:
                        liking_scores, trust_scores = evaluate_with_llm(
                            llm_model, config, from_agent, agent,
                            talks, liking_keys, trust_keys, agent_logger
                        )
                        logger.info(f"evaluated via direct LLM for {from_agent}")
                    else:
                        logger.error(f"No LLM available; falling back to 0.5 for {from_agent}")
                        liking_scores = {k: 0.5 for k in liking_keys}
                        trust_scores = {k: 0.5 for k in trust_keys}
            
            elif llm_model and talks:
                # Direct LLM call when no agent_obj
                logger.info(f"Using direct LLM evaluation for {from_agent}")
                liking_scores, trust_scores = evaluate_with_llm(
                    llm_model, config, from_agent, agent,
                    talks, liking_keys, trust_keys, agent_logger
                )
            else:
                # Fallback to defaults when no LLM available
                logger.warning(f"No LLM available; falling back to 0.5 for {from_agent}")
                liking_scores = {k: 0.5 for k in liking_keys}
                trust_scores = {k: 0.5 for k in trust_keys}
            
            # Fill missing keys with 0.5
            for k in liking_keys:
                if k not in liking_scores:
                    liking_scores[k] = 0.5
            for k in trust_keys:
                if k not in trust_scores:
                    trust_scores[k] = 0.5
            
            # Calculate weighted averages
            liking_score = calculate_weighted_average(liking_scores, liking_weights)
            trust_score = calculate_weighted_average(trust_scores, trust_weights)
            
            logger.info(f"{from_agent}: liking={liking_score}, creditability={trust_score}")
            
            # Save updated file
            save_talk_history_with_headers(yml_file, items, liking_score, trust_score)
            
        except Exception as e:
            logger.error(f"Failed to process {yml_file}: {e}")
            # Save with defaults on error
            try:
                items, _, _ = load_talk_history(yml_file)
                if items:
                    save_talk_history_with_headers(yml_file, items, 0.5, 0.5)
            except Exception:
                pass


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Update affinity and trust scores in talk history files"
    )
    parser.add_argument('--game-id', required=True, help='Game ID')
    parser.add_argument('--agent', required=True, help='Agent name')
    parser.add_argument('--talk-dir', default='talk_history', help='Talk directory name')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load config
    config_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "config.yml"
    config = load_yaml_safe(config_path) or {}
    
    try:
        update_affinity_trust_for_agent(
            config=config,
            game_id=args.game_id,
            agent=args.agent,
            talk_dir_name=args.talk_dir
        )
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())