"""Macro belief orchestrator - consolidates all BDI inference results into single YAML.

マクロ信念オーケストレータ - 全てのBDI推論結果を単一YAMLに統合.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from .mbti_inference import infer_mbti, get_game_timestamp
from .enneagram_inference import infer_enneagram
from .cognitive_bias import infer_cognitive_bias
from .desire_tendency import infer_desire_tendency
from .behavior_tendency import infer_behavior_tendency
from ._writer import write_macro_belief, create_meta_dict
from ._config import load_config
from ._roles import to_japanese_role


def generate_macro_belief(
    config: Dict[str, Any],
    profile: str,
    agent_name: str,
    game_id: str,
    out_root: str | Path = "info/bdi_info/macro_bdi",
    role_en: str | None = None
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
    
    # Step 6: Prepare role social duties
    role_social_duties = {}
    if role_en:
        role_ja = to_japanese_role(role_en)
        duties_cfg = config.get("role_social_duties", {})
        
        if role_ja and role_ja in duties_cfg:
            role_social_duties = {"role": role_ja, "duties": duties_cfg[role_ja]}
        elif role_ja:
            role_social_duties = {"role": role_ja, "duties": {"note": "role duties not found in config"}}
    else:
        role_social_duties = {"role": None, "duties": {"note": "role not provided"}}
    
    # Step 7: Consolidate all data (role_social_duties first)
    macro_belief_data = {
        "role_social_duties": role_social_duties,
        "mbti": mbti_data,
        "enneagram": enneagram_data,
        "cognitive_bias": cognitive_bias_data,
        "desire_tendency": desire_tendency_data,
        "behavior_tendency": behavior_tendency_data,
    }
    
    payload = {
        "macro_belief": macro_belief_data,
        "meta": create_meta_dict(game_id, agent_name),
    }
    
    # Step 8: Write to file atomically (using game_id directly)
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
        default="info/bdi_info/macro_bdi",
        help="Output root directory"
    )
    parser.add_argument("--config", default="/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/config/config.yml", help="Path to config YAML file")
    parser.add_argument("--role-en", default=None, help="SEER/WEREWOLF/VILLAGER/POSSESSED/BODYGUARD/MEDIUM")
    
    args = parser.parse_args()
    
    # Load config using the new loader
    config = load_config(Path(args.config))
    
    try:
        output_path = generate_macro_belief(
            config=config,
            profile=args.profile,
            agent_name=args.agent,
            game_id=args.game_id,
            out_root=args.out_root,
            role_en=args.role_en
        )
        print(f"Generated macro belief: {output_path}")
        
    except Exception as e:
        print(f"Error generating macro belief: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())