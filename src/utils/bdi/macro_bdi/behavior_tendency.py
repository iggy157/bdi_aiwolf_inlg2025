"""Behavior tendency calculation from MBTI and Enneagram parameters.

MBTIとエニアグラムパラメータから発話・行動傾向を算出するモジュール.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


class BehaviorTendency:
    """Behavior tendency calculation from MBTI and Enneagram parameters."""

    def __init__(self):
        """Initialize behavior tendency calculator."""
        self.agent_behavior_tendencies: Dict[str, float] = {}
        self.behavior_tendency_history: List[Dict[str, Any]] = []

    def calculate_behavior_tendency(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate behavior tendency.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Behavior tendency values (0-1 range)
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
        
        behavior_tendencies = {
            "avoidant_behavior": (
                introversion * 0.6 + 
                peacemaker * 0.4
            ),
            "aggressive_behavior": (
                extroversion * 0.4 + 
                achiever * 0.6
            ),
            "adaptability": (
                feeling * 0.5 + 
                thinking * 0.5
            ),
            "introversion": introversion,
            "extroversion": extroversion,
            "empathy": (
                feeling * 0.6 + 
                peacemaker * 0.4
            ),
            "assertiveness": (
                extroversion * 0.6 + 
                achiever * 0.4
            ),
        }

        # Ensure all values are within 0-1 range    
        for key in behavior_tendencies:
            behavior_tendencies[key] = max(0.0, min(1.0, behavior_tendencies[key]))
            
        return behavior_tendencies

    def update_agent_behavior(self, agent_name: str, behavior_score: float) -> None:
        """Update behavior tendency for a specific agent.
        
        Args:
            agent_name: Name of the agent
            behavior_score: Behavior score (0-1 range)
        """
        if agent_name not in self.agent_behavior_tendencies:
            self.agent_behavior_tendencies[agent_name] = 0.5  # Initial value
        
        # Update with weighted average (new score has 30% weight)
        current_behavior = self.agent_behavior_tendencies[agent_name]
        self.agent_behavior_tendencies[agent_name] = current_behavior * 0.7 + behavior_score * 0.3
        self.agent_behavior_tendencies[agent_name] = max(0.0, min(1.0, self.agent_behavior_tendencies[agent_name]))

    def add_behavior_record(
        self, 
        agent_name: str, 
        behavior_type: str, 
        behavior_scores: Dict[str, float]
    ) -> None:
        """Add a behavior tendency record to history.
        
        Args:
            agent_name: Name of the agent
            behavior_type: Type of behavior being recorded
            behavior_scores: Calculated behavior scores
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "behavior_type": behavior_type,
            "behavior_scores": behavior_scores.copy()
        }
        self.behavior_tendency_history.append(record)

    def get_agent_behavior(self, agent_name: str) -> float:
        """Get current behavior tendency for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Behavior tendency score (0-1 range)
        """
        return self.agent_behavior_tendencies.get(agent_name, 0.5)

    def calculate_all_behaviors(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate all behavior tendencies.
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing all behavior calculations
        """
        return {
            "behavior_tendencies": self.calculate_behavior_tendency(mbti_params, enneagram_params),
            "agent_behavior_scores": self.agent_behavior_tendencies.copy(),
            "behavior_history_count": len(self.behavior_tendency_history)
        }

    def get_core_behaviors(
        self, 
        mbti_params: Dict[str, float], 
        enneagram_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get only core behavior tendencies (without agent scores and metadata).
        
        Args:
            mbti_params: MBTI parameters dictionary
            enneagram_params: Enneagram parameters dictionary
            
        Returns:
            Dictionary containing only core behavior calculations
        """
        return {
            "behavior_tendencies": self.calculate_behavior_tendency(mbti_params, enneagram_params)
        }

    def save_behavior_tendency(self, behavior_data: Dict[str, Any], output_path: Path) -> None:
        """Save behavior tendency data to YAML file.
        
        Args:
            behavior_data: Behavior tendency data dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(behavior_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving behavior tendency data: {e}")

    def save_behavior_history(self, output_path: Path) -> None:
        """Save behavior tendency history to YAML file.
        
        Args:
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.behavior_tendency_history, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving behavior history: {e}")

    def load_behavior_tendency(self, input_path: Path) -> Dict[str, Any] | None:
        """Load behavior tendency data from YAML file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Behavior tendency data dictionary or None if error
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Restore agent tendencies if available
            if "agent_behavior_scores" in data:
                self.agent_behavior_tendencies = data["agent_behavior_scores"]
                
            return data
        except Exception as e:
            print(f"Error loading behavior tendency data: {e}")
            return None

    def load_behavior_history(self, input_path: Path) -> None:
        """Load behavior tendency history from YAML file.
        
        Args:
            input_path: Input file path
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.behavior_tendency_history = yaml.safe_load(f) or []
        except Exception as e:
            print(f"Error loading behavior history: {e}")
            self.behavior_tendency_history = []


def infer_behavior_tendency(
    mbti_params: Dict[str, float],
    enneagram_params: Dict[str, float]
) -> Dict[str, Any]:
    """Pure function to calculate behavior tendency from MBTI and Enneagram parameters without file I/O.
    
    Args:
        mbti_params: MBTI parameters dictionary
        enneagram_params: Enneagram parameters dictionary
        
    Returns:
        Behavior tendency data dictionary
    """
    behavior_calculator = BehaviorTendency()
    return behavior_calculator.get_core_behaviors(mbti_params, enneagram_params)


def calculate_and_save_behavior_tendency(
    mbti_file_path: Path, 
    enneagram_file_path: Path, 
    output_dir: Path
) -> Dict[str, Any] | None:
    """Calculate and save behavior tendency from MBTI and Enneagram files.
    
    Args:
        mbti_file_path: Path to MBTI YAML file
        enneagram_file_path: Path to Enneagram YAML file
        output_dir: Output directory path
        
    Returns:
        Behavior tendency data dictionary or None if error
    """
    try:
        # Load MBTI parameters
        with open(mbti_file_path, 'r', encoding='utf-8') as f:
            mbti_params = yaml.safe_load(f)
            
        # Load Enneagram parameters
        with open(enneagram_file_path, 'r', encoding='utf-8') as f:
            enneagram_params = yaml.safe_load(f)
            
        # Calculate behavior tendency (core behaviors only)
        behavior_calculator = BehaviorTendency()
        behavior_data = behavior_calculator.get_core_behaviors(mbti_params, enneagram_params)
        
        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)
        behavior_file_path = output_dir / "behavior_tendency.yml"
        behavior_calculator.save_behavior_tendency(behavior_data, behavior_file_path)
        
        return behavior_data
        
    except Exception as e:
        print(f"Error calculating behavior tendency: {e}")
        return None

    