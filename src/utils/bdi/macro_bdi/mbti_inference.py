"""MBTI parameter inference from profile text using LLM.

プロフィール文からLLMを使用してMBTIパラメータを推測するモジュール.
"""

from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from datetime import datetime, timezone
from ulid import ULID

from .enneagram_inference import infer_and_save_enneagram
from .cognitive_bias import calculate_and_save_cognitive_bias
from .desire_tendency import calculate_and_save_desire_tendency
from .behavior_tendency import calculate_and_save_behavior_tendency
from ._mbti_parser import parse_mbti_output_robust

# Load environment variables from config/.env
load_dotenv(Path(__file__).parent.parent.parent.parent.parent / "config" / ".env")


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

    def __init__(self, config: dict[str, Any], agent_logger=None):
        """Initialize MBTI inference.

        Args:
            config: Configuration dictionary
            agent_logger: AgentLogger instance for LLM interaction logging
        """
        self.config = config
        self.llm_model = None
        self.agent_logger = agent_logger
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LLM model based on configuration."""
        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(f"Unknown LLM type: {model_type}")

    def infer_mbti_parameters(self, profile: str, agent_name: str) -> dict[str, float]:
        """Infer MBTI parameters from profile text.

        Args:
            profile: Profile text to analyze
            agent_name: Agent name for context

        Returns:
            Dictionary containing MBTI parameters (0-1 range)
        """
        if not profile or self.llm_model is None:
            if self.agent_logger:
                self.agent_logger.llm_error("mbti_inference", "Profile empty or LLM model not available")
            return self._get_default_mbti_parameters()

        try:
            prompt_template = Template(self.config["prompt"]["mbti_inference"])
            prompt = prompt_template.render(profile=profile, agent_name=agent_name).strip()

            message = HumanMessage(content=prompt)
            response = (self.llm_model | StrOutputParser()).invoke([message])

            # LLMやり取りをログ出力
            if self.agent_logger:
                model_info = f"{type(self.llm_model).__name__}"
                self.agent_logger.llm_interaction("mbti_inference", prompt, response, model_info)

            result = self._parse_mbti_response(response)
            
            if self.agent_logger:
                self.agent_logger.logger.info(f"MBTI inference result for {agent_name}: {result}")
            
            return result
        except Exception as e:
            if self.agent_logger:
                self.agent_logger.llm_error("mbti_inference", str(e), prompt if 'prompt' in locals() else None)
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


def infer_mbti(config: dict[str, Any], profile: str, agent_name: str, agent_logger=None) -> dict[str, float]:
    """Pure function to infer MBTI parameters without file I/O.

    Args:
        config: Configuration dictionary
        profile: Profile text
        agent_name: Agent name
        agent_logger: AgentLogger instance for LLM interaction logging

    Returns:
        MBTI parameters dictionary
    """
    inference = MBTIInference(config, agent_logger)
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
