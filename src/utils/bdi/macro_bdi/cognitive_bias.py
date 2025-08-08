"""Cognitive bias calculation from MBTI and Enneagram parameters.

MBTIとエニアグラムパラメータから認知バイアスを算出するモジュール.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


class CognitiveBias:
    """Cognitive bias calculation from MBTI and Enneagram parameters."""

    def __init__(self):
        """Initialize cognitive bias calculator."""
        self.trust_tendencies: Dict[str, float] = {}
        self.liking_tendencies: Dict[str, float] = {}
        self.statement_bias_history: List[Dict[str, Any]] = []

    def calculate_statement_bias(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate bias towards statements/speech content.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing statement bias parameters
        """
        # Extract parameters with defaults
        thinking = mbti_params.get("thinking", 0.5)
        intuition = mbti_params.get("intuition", 0.5)
        sensing = mbti_params.get("sensing", 0.5)
        reformer = enneagram_params.get("reformer", 0.5)
        investigator = enneagram_params.get("investigator", 0.5)
        
        statement_bias = {
            "logical_consistency": (
                thinking * 0.4 + 
                intuition * 0.3 + 
                reformer * 0.3
            ),
            "specificity_and_detail": (
                sensing * 0.6 + 
                intuition * 0.2 + 
                investigator * 0.2
            ),
            "intuitive_depth": (
                intuition * 0.4 + 
                thinking * 0.3 + 
                investigator * 0.3
            ),
            "clarity_and_conciseness": (
                thinking * 0.5 + 
                intuition * 0.3 + 
                reformer * 0.2
            )
        }
        
        # Ensure all values are within 0-1 range
        for key in statement_bias:
            statement_bias[key] = max(0.0, min(1.0, statement_bias[key]))
            
        return statement_bias

    def calculate_trust_tendency(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate trust tendency towards other agents.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing trust tendency parameters
        """
        # Extract parameters with defaults
        extroversion = mbti_params.get("extroversion", 0.5)
        introversion = mbti_params.get("introversion", 0.5)
        judging = mbti_params.get("judging", 0.5)
        achiever = enneagram_params.get("achiever", 0.5)
        loyalist = enneagram_params.get("loyalist", 0.5)
        
        trust_tendency = {
            "social_proof": (
                extroversion * 0.6 + 
                achiever * 0.4
            ),
            "honesty": (
                judging * 0.7 + 
                introversion * 0.3 + 
                loyalist * 0.6
            ) / (0.7 + 0.3 + 0.6),
            "consistency": (
                judging * 0.7 + 
                introversion * 0.3 + 
                loyalist * 0.4
            ) / (0.7 + 0.3 + 0.4)
        }
        
        # Ensure all values are within 0-1 range
        for key in trust_tendency:
            trust_tendency[key] = max(0.0, min(1.0, trust_tendency[key]))
            
        return trust_tendency

    def calculate_liking_tendency(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate liking tendency towards other agents.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing liking tendency parameters
        """
        # Extract parameters with defaults
        feeling = mbti_params.get("feeling", 0.5)
        extroversion = mbti_params.get("extroversion", 0.5)
        helper = enneagram_params.get("helper", 0.5)
        
        liking_tendency = {
            "friendliness": (
                feeling * 0.5 + 
                extroversion * 0.3 + 
                helper * 0.4
            ) / (0.5 + 0.3 + 0.4),
            "emotional_resonance": (
                feeling * 0.6 + 
                helper * 0.4
            ),
            "attractive_expression": (
                extroversion * 0.5 + 
                helper * 0.5
            )
        }
        
        # Ensure all values are within 0-1 range
        for key in liking_tendency:
            liking_tendency[key] = max(0.0, min(1.0, liking_tendency[key]))
            
        return liking_tendency

    def update_agent_trust(self, agent_name: str, trust_score: float) -> None:
        """Update trust tendency for a specific agent.
        
        Args:
            agent_name: Name of the agent
            trust_score: Trust score (0-1 range)
        """
        if agent_name not in self.trust_tendencies:
            self.trust_tendencies[agent_name] = 0.5  # Initial value
        
        # Update with weighted average (new score has 30% weight)
        current_trust = self.trust_tendencies[agent_name]
        self.trust_tendencies[agent_name] = current_trust * 0.7 + trust_score * 0.3
        self.trust_tendencies[agent_name] = max(0.0, min(1.0, self.trust_tendencies[agent_name]))

    def update_agent_liking(self, agent_name: str, liking_score: float) -> None:
        """Update liking tendency for a specific agent.
        
        Args:
            agent_name: Name of the agent
            liking_score: Liking score (0-1 range)
        """
        if agent_name not in self.liking_tendencies:
            self.liking_tendencies[agent_name] = 0.5  # Initial value
        
        # Update with weighted average (new score has 30% weight)
        current_liking = self.liking_tendencies[agent_name]
        self.liking_tendencies[agent_name] = current_liking * 0.7 + liking_score * 0.3
        self.liking_tendencies[agent_name] = max(0.0, min(1.0, self.liking_tendencies[agent_name]))

    def add_statement_bias_record(
        self, 
        speaker: str, 
        statement: str, 
        bias_scores: Dict[str, float]
    ) -> None:
        """Add a statement bias record to history.
        
        Args:
            speaker: Name of the speaker
            statement: The statement content
            bias_scores: Calculated bias scores
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "speaker": speaker,
            "statement": statement,
            "bias_scores": bias_scores.copy()
        }
        self.statement_bias_history.append(record)

    def get_agent_trust(self, agent_name: str) -> float:
        """Get current trust tendency for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Trust tendency score (0-1 range)
        """
        return self.trust_tendencies.get(agent_name, 0.5)

    def get_agent_liking(self, agent_name: str) -> float:
        """Get current liking tendency for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Liking tendency score (0-1 range)
        """
        return self.liking_tendencies.get(agent_name, 0.5)

    def calculate_all_biases(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate all cognitive biases.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing all bias calculations
        """
        return {
            "statement_bias": self.calculate_statement_bias(mbti_params, enneagram_params),
            "trust_tendency": self.calculate_trust_tendency(mbti_params, enneagram_params),
            "liking_tendency": self.calculate_liking_tendency(mbti_params, enneagram_params),
            "agent_trust_scores": self.trust_tendencies.copy(),
            "agent_liking_scores": self.liking_tendencies.copy(),
            "statement_history_count": len(self.statement_bias_history)
        }

    def get_core_biases(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get only core cognitive biases (without agent scores and metadata).
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing only core bias calculations
        """
        return {
            "statement_bias": self.calculate_statement_bias(mbti_params, enneagram_params),
            "trust_tendency": self.calculate_trust_tendency(mbti_params, enneagram_params),
            "liking_tendency": self.calculate_liking_tendency(mbti_params, enneagram_params)
        }

    def save_cognitive_bias(self, bias_data: Dict[str, Any], output_path: Path) -> None:
        """Save cognitive bias data to YAML file.
        
        Args:
            bias_data: Cognitive bias data dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(bias_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving cognitive bias data: {e}")

    def save_statement_history(self, output_path: Path) -> None:
        """Save statement bias history to YAML file.
        
        Args:
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.statement_bias_history, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving statement history: {e}")

    def load_cognitive_bias(self, input_path: Path) -> Dict[str, Any] | None:
        """Load cognitive bias data from YAML file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Cognitive bias data dictionary or None if error
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Restore agent tendencies if available
            if "agent_trust_scores" in data:
                self.trust_tendencies = data["agent_trust_scores"]
            if "agent_liking_scores" in data:
                self.liking_tendencies = data["agent_liking_scores"]
                
            return data
        except Exception as e:
            print(f"Error loading cognitive bias data: {e}")
            return None

    def load_statement_history(self, input_path: Path) -> None:
        """Load statement bias history from YAML file.
        
        Args:
            input_path: Input file path
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.statement_bias_history = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Error loading statement history: {e}")
            self.statement_bias_history = []


def infer_cognitive_bias(
    mbti_params: Dict[str, float],
    enneagram_params: Dict[str, float]
) -> Dict[str, Any]:
    """Pure function to calculate cognitive bias from MBTI and Enneagram parameters without file I/O.
    
    Args:
        mbti_params: MBTI parameters dictionary
        enneagram_params: Enneagram parameters dictionary
        
    Returns:
        Cognitive bias data dictionary
    """
    bias_calculator = CognitiveBias()
    return bias_calculator.get_core_biases(mbti_params, enneagram_params)


def calculate_and_save_cognitive_bias(
    mbti_file_path: Path, 
    enneagram_file_path: Path, 
    output_dir: Path
) -> Dict[str, Any] | None:
    """Calculate and save cognitive bias from MBTI and Enneagram files.
    
    Args:
        mbti_file_path: Path to MBTI YAML file
        enneagram_file_path: Path to Enneagram YAML file
        output_dir: Output directory path
        
    Returns:
        Cognitive bias data dictionary or None if error
    """
    try:
        # Load MBTI parameters
        with open(mbti_file_path, 'r', encoding='utf-8') as f:
            mbti_params = yaml.safe_load(f)
            
        # Load Enneagram parameters
        with open(enneagram_file_path, 'r', encoding='utf-8') as f:
            enneagram_params = yaml.safe_load(f)
            
        # Calculate cognitive bias (core biases only)
        bias_calculator = CognitiveBias()
        bias_data = bias_calculator.get_core_biases(mbti_params, enneagram_params)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        bias_file_path = output_dir / "cognitive_bias.yml"
        bias_calculator.save_cognitive_bias(bias_data, bias_file_path)
        
        return bias_data
        
    except Exception as e:
        print(f"Error calculating cognitive bias: {e}")
        return None