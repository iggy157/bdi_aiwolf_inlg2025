"""MBTI parameter inference from profile text using LLM.

プロフィール文からLLMを使用してMBTIパラメータを推測するモジュール.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any

from jinja2 import Template
from datetime import datetime, timezone
from ulid import ULID

from .enneagram_inference import infer_and_save_enneagram
from .cognitive_bias import calculate_and_save_cognitive_bias
from .desire_tendency import calculate_and_save_desire_tendency
from .behavior_tendency import calculate_and_save_behavior_tendency
from ._mbti_parser import parse_mbti_output_robust

# NOTE: .env loading is handled by agent.py only


def get_game_timestamp(game_id: str) -> str:
    """ULIDからタイムスタンプ文字列(yyyyMMddHHmmssfff)を取得"""
    try:
        ulid_obj = ULID.from_str(game_id)
        tz = datetime.now(timezone.utc).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(
            ulid_obj.timestamp / 1000, tz=tz
        ).strftime("%Y%m%d%H%M%S%f")[:-3]
        return game_timestamp
    except Exception:
        # Fallback to current timestamp if ULID parsing fails
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]


class MBTIInference:
    """MBTI parameter inference from profile text."""

    def __init__(self, config: dict[str, Any], agent_logger=None, agent_obj=None):
        """Initialize MBTI inference.

        Args:
            config: Configuration dictionary
            agent_logger: AgentLogger instance for LLM interaction logging
            agent_obj: Agent object with send_message_to_llm method
        """
        self.config = config
        self.agent_logger = agent_logger
        self.agent_obj = agent_obj

    # Direct LLM model creation removed - use agent.send_message_to_llm instead

    def infer_mbti_parameters(self, profile: str, agent_name: str) -> dict[str, float]:
        """Infer MBTI parameters from profile text.

        Args:
            profile: Profile text to analyze
            agent_name: Agent name for context

        Returns:
            Dictionary containing MBTI parameters (0-1 range)
        """
        if not profile:
            if self.agent_logger:
                self.agent_logger.llm_error("mbti_inference", "Profile empty")
            return self._get_default_mbti_parameters()
        
        if self.agent_obj is None:
            if self.agent_logger:
                self.agent_logger.llm_error("mbti_inference", "agent_obj not provided")
            return self._get_default_mbti_parameters()

        try:
            extra_vars = {
                "profile": profile,
                "agent_name": agent_name
            }
            response = self.agent_obj.send_message_to_llm(
                "mbti_inference",
                extra_vars=extra_vars,
                log_tag="MBTI_INFERENCE",
                use_shared_history=False
            )
            
            if response is None:
                if self.agent_logger:
                    self.agent_logger.llm_error("mbti_inference", "Agent LLM call returned None")
                return self._get_default_mbti_parameters()

            result = self._parse_mbti_response(response)
            
            if self.agent_logger:
                self.agent_logger.logger.info(f"MBTI inference result for {agent_name}: {result}")
            
            return result
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error("mbti_inference", str(e))
            print(f"Error during MBTI inference: {e}")
            return self._get_default_mbti_parameters()

    def _parse_mbti_response(self, response: str) -> dict[str, float]:
        """Parse LLM response to extract MBTI parameters using robust parser.

        Args:
            response: LLM response containing MBTI parameters

        Returns:
            Dictionary with MBTI parameters
        """
        try:
            return parse_mbti_output_robust(response)
        except Exception as e:
            print(f"Error parsing MBTI response: {e}")
            return self._get_default_mbti_parameters()

    def _get_default_mbti_parameters(self) -> dict[str, float]:
        """Get default MBTI parameters.

        Returns:
            Default MBTI parameters dictionary
        """
        return {
            "extroversion": 0.5,
            "introversion": 0.5,
            "sensing": 0.5,
            "intuition": 0.5,
            "thinking": 0.5,
            "feeling": 0.5,
            "judging": 0.5,
            "perceiving": 0.5
        }

    def save_mbti_parameters(self, mbti_params: dict[str, float], agent_name: str, game_id: str) -> None:
        """Save MBTI parameters to YAML file (new output directory structure).

        Args:
            mbti_params: MBTI parameters dictionary
            agent_name: Agent name for path
            game_id: ULID-based game ID (for timestamped directory)
        """
        try:
            game_timestamp = get_game_timestamp(game_id)
            output_dir = Path("info") / "bdi_info" / "macro_bdi" / "macro_belief" / game_timestamp / agent_name
            output_dir.mkdir(parents=True, exist_ok=True)
            file_path = output_dir / "mbti.yml"

            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(mbti_params, f, default_flow_style=False, allow_unicode=True)
            
            # Also calculate and save Enneagram parameters
            try:
                infer_and_save_enneagram(file_path)
                
                # Calculate and save cognitive bias if Enneagram exists
                enneagram_path = output_dir / "enneagram.yml"
                if enneagram_path.exists():
                    calculate_and_save_cognitive_bias(file_path, enneagram_path, output_dir)
                    calculate_and_save_desire_tendency(file_path, enneagram_path, output_dir)
                    calculate_and_save_behavior_tendency(file_path, enneagram_path, output_dir)
            except Exception as e:
                print(f"Error calculating Enneagram parameters or cognitive bias: {e}")

        except Exception as e:
            print(f"Error saving MBTI parameters: {e}")


def infer_mbti(config: dict[str, Any], profile: str, agent_name: str, agent_logger=None, agent_obj=None) -> dict[str, float]:
    """Pure function to infer MBTI parameters without file I/O.

    Args:
        config: Configuration dictionary
        profile: Profile text
        agent_name: Agent name
        agent_logger: AgentLogger instance for LLM interaction logging
        agent_obj: Agent object with send_message_to_llm method

    Returns:
        MBTI parameters dictionary
    """
    inference = MBTIInference(config, agent_logger, agent_obj)
    return inference.infer_mbti_parameters(profile, agent_name)


def infer_and_save_mbti(config: dict[str, Any], profile: str, agent_name: str, game_id: str, agent_logger=None) -> dict[str, float]:
    """Convenience function to infer and save MBTI parameters.

    Args:
        config: Configuration dictionary
        profile: Profile text
        agent_name: Agent name
        game_id: ULID-based game ID
        agent_logger: AgentLogger instance for LLM interaction logging

    Returns:
        MBTI parameters dictionary
    """
    inference = MBTIInference(config, agent_logger)
    mbti_params = inference.infer_mbti_parameters(profile, agent_name)
    
    inference.save_mbti_parameters(mbti_params, agent_name, game_id)
    
    return mbti_params
