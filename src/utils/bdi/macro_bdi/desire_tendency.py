"""Desire tendency calculation from MBTI and Enneagram parameters.

MBTIとエニアグラムパラメータから願望傾向を算出するモジュール.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


class DesireTendency:
    """Desire tendency calculation from MBTI and Enneagram parameters."""

    def __init__(self):
        """Initialize desire tendency calculator."""
        self.agent_desire_tendencies: Dict[str, float] = {}
        self.desire_tendency_history: List[Dict[str, Any]] = []

    def calculate_desire_tendency(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate self-realization tendency.
        
        自己実現 = (MBTI/直観性 * 0.6 + エニアグラム/改革する人 * 0.4)
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Self-realization tendency value (0-1 range)
        """
                # Extract parameters with defaults
        extroversion = mbti_params.get("extroversion", 0.5)
        introversion = mbti_params.get("introversion", 0.5)
        sensing = mbti_params.get("sensing", 0.5)
        intuition = mbti_params.get("intuition", 0.5)
        thinking = mbti_params.get("thinking", 0.5)
        feeling = mbti_params.get("feeling", 0.5)
        judging = mbti_params.get("judging", 0.5)
        peacemaker = enneagram_params.get("peacemaker", 0.5)
        
        reformer = enneagram_params.get("reformer", 0.5)
        helper = enneagram_params.get("helper", 0.5)
        achiever = enneagram_params.get("achiever", 0.5)
        indivisualist = enneagram_params.get("indivisualist", 0.5)
        investigator = enneagram_params.get("investigator", 0.5)
        
        desire_tendencies = {
            "self_realization": (
                intuition * 0.6 + 
                reformer * 0.4
            ),
            "social_approval": (
                sensing * 0.5 + 
                achiever * 0.5
            ),
            "stability": (
                introversion * 0.6 + 
                peacemaker * 0.4
            ),
            "love_intimacy": (
                introversion * 0.5 + 
                helper * 0.5
            ),
            "freedom_independence": (
                extroversion * 0.7 + 
                reformer * 0.3
            ),
            "adventure_stimulation": (
                extroversion * 0.6 + 
                intuition * 0.4
            ),
            "stable_relationships": (
                introversion * 0.6 + 
                peacemaker * 0.4
            ),
        }

        # Ensure all values are within 0-1 range    
        for key in desire_tendencies:
            desire_tendencies[key] = max(0.0, min(1.0, desire_tendencies[key]))
            
        return desire_tendencies

    def update_agent_desire(self, agent_name: str, desire_score: float) -> None:
        """Update desire tendency for a specific agent.
        
        Args:
            agent_name: Name of the agent
            desire_score: Desire score (0-1 range)
        """
        if agent_name not in self.agent_desire_tendencies:
            self.agent_desire_tendencies[agent_name] = 0.5  # Initial value
        
        # Update with weighted average (new score has 30% weight)
        current_desire = self.agent_desire_tendencies[agent_name]
        self.agent_desire_tendencies[agent_name] = current_desire * 0.7 + desire_score * 0.3
        self.agent_desire_tendencies[agent_name] = max(0.0, min(1.0, self.agent_desire_tendencies[agent_name]))

    def add_desire_record(
        self, 
        agent_name: str, 
        desire_type: str, 
        desire_scores: Dict[str, float]
    ) -> None:
        """Add a desire tendency record to history.
        
        Args:
            agent_name: Name of the agent
            desire_type: Type of desire being recorded
            desire_scores: Calculated desire scores
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "desire_type": desire_type,
            "desire_scores": desire_scores.copy()
        }
        self.desire_tendency_history.append(record)

    def get_agent_desire(self, agent_name: str) -> float:
        """Get current desire tendency for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Desire tendency score (0-1 range)
        """
        return self.agent_desire_tendencies.get(agent_name, 0.5)

    def calculate_all_desires(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate all desire tendencies.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing all desire calculations
        """
        return {
            "desire_tendencies": self.calculate_desire_tendency(mbti_params, enneagram_params),
            "agent_desire_scores": self.agent_desire_tendencies.copy(),
            "desire_history_count": len(self.desire_tendency_history)
        }

    def get_core_desires(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get only core desire tendencies (without agent scores and metadata).
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing only core desire calculations
        """
        return {
            "desire_tendencies": self.calculate_desire_tendency(mbti_params, enneagram_params)
        }

    def save_desire_tendency(self, desire_data: Dict[str, Any], output_path: Path) -> None:
        """Save desire tendency data to YAML file.
        
        Args:
            desire_data: Desire tendency data dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(desire_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving desire tendency data: {e}")

    def save_desire_history(self, output_path: Path) -> None:
        """Save desire tendency history to YAML file.
        
        Args:
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.desire_tendency_history, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving desire history: {e}")

    def load_desire_tendency(self, input_path: Path) -> Dict[str, Any] | None:
        """Load desire tendency data from YAML file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Desire tendency data dictionary or None if error
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Restore agent tendencies if available
            if "agent_desire_scores" in data:
                self.agent_desire_tendencies = data["agent_desire_scores"]
                
            return data
        except Exception as e:
            print(f"Error loading desire tendency data: {e}")
            return None

    def load_desire_history(self, input_path: Path) -> None:
        """Load desire tendency history from YAML file.
        
        Args:
            input_path: Input file path
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.desire_tendency_history = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Error loading desire history: {e}")
            self.desire_tendency_history = []


def calculate_and_save_desire_tendency(
    mbti_file_path: Path, 
    enneagram_file_path: Path, 
    output_dir: Path
) -> Dict[str, Any] | None:
    """Calculate and save desire tendency from MBTI and Enneagram files.
    
    Args:
        mbti_file_path: Path to MBTI YAML file
        enneagram_file_path: Path to Enneagram YAML file
        output_dir: Output directory path
        
    Returns:
        Desire tendency data dictionary or None if error
    """
    try:
        # Load MBTI parameters
        with open(mbti_file_path, 'r', encoding='utf-8') as f:
            mbti_params = yaml.safe_load(f)
            
        # Load Enneagram parameters
        with open(enneagram_file_path, 'r', encoding='utf-8') as f:
            enneagram_params = yaml.safe_load(f)
            
        # Calculate desire tendency (core desires only)
        desire_calculator = DesireTendency()
        desire_data = desire_calculator.get_core_desires(mbti_params, enneagram_params)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        desire_file_path = output_dir / "desire_tendency.yml"
        desire_calculator.save_desire_tendency(desire_data, desire_file_path)
        
        return desire_data
        
    except Exception as e:
        print(f"Error calculating desire tendency: {e}")
        return None

    