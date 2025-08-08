"""Macro belief orchestrator - consolidates all BDI inference results into single YAML.

マクロ信念オーケストレータ - 全てのBDI推論結果を単一YAMLに統合.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .mbti_inference import infer_mbti
from .enneagram_inference import infer_enneagram
from .cognitive_bias import infer_cognitive_bias
from .desire_tendency import infer_desire_tendency
from .behavior_tendency import infer_behavior_tendency
from ._writer import write_macro_belief, create_meta_dict


def generate_macro_belief(
    config: Dict[str, Any],
    profile: str,
    agent_name: str,
    game_id: str,
    out_root: str | Path = "info/bdi_info/macro_bdi/macro_belief"
) -> Path:
    """Generate consolidated macro belief YAML file.
    
    Args:
        config: Configuration dictionary
        profile: Profile text for MBTI inference
        agent_name: Agent name
        game_id: Game ID
        out_root: Output root directory
        
    Returns:
        Path to generated macro_belief.yml file
    """
    # Step 1: Infer MBTI parameters
    mbti_data = infer_mbti(config, profile, agent_name)
    
    # Step 2: Calculate Enneagram parameters
    enneagram_data = infer_enneagram(mbti_data)
    
    # Step 3: Calculate cognitive bias
    cognitive_bias_data = infer_cognitive_bias(mbti_data, enneagram_data)
    
    # Step 4: Calculate desire tendency
    desire_tendency_data = infer_desire_tendency(mbti_data, enneagram_data)
    
    # Step 5: Calculate behavior tendency
    behavior_tendency_data = infer_behavior_tendency(mbti_data, enneagram_data)
    
    # Step 6: Consolidate all data
    payload = {
        "macro_belief": {
            "mbti": mbti_data,
            "enneagram": enneagram_data,
            "cognitive_bias": cognitive_bias_data,
            "desire_tendency": desire_tendency_data,
            "behavior_tendency": behavior_tendency_data,
        },
        "meta": create_meta_dict(game_id, agent_name),
    }
    
    # Step 7: Write to file atomically
    out_dir = Path(out_root) / game_id / agent_name
    return write_macro_belief(payload, out_dir)


def main():
    """Command-line interface for macro belief generation."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated macro belief YAML from BDI inference"
    )
    parser.add_argument("--game-id", required=True, help="Game ID")
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--profile", required=True, help="Profile text for MBTI inference")
    parser.add_argument(
        "--out-root", 
        default="info/bdi_info/macro_bdi/macro_belief",
        help="Output root directory"
    )
    parser.add_argument("--config", help="Path to config YAML file (optional)")
    
    args = parser.parse_args()
    
    # Load config if provided, otherwise use minimal config
    if args.config:
        import yaml
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Minimal default config for CLI usage
        config = {
            "llm": {"type": "openai"},
            "openai": {"model": "gpt-3.5-turbo", "temperature": 0.7},
            "prompt": {
                "mbti_inference": "Analyze the following profile and provide MBTI parameters (0-1 range):\n{{ profile }}\n\nAgent: {{ agent_name }}\n\nProvide scores for: extroversion, introversion, sensing, intuition, thinking, feeling, judging, perceiving"
            }
        }
    
    try:
        output_path = generate_macro_belief(
            config=config,
            profile=args.profile,
            agent_name=args.agent,
            game_id=args.game_id,
            out_root=args.out_root
        )
        print(f"Generated macro belief: {output_path}")
        
    except Exception as e:
        print(f"Error generating macro belief: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())