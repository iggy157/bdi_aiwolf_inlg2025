"""Enneagram parameter inference from MBTI parameters.

MBTIパラメータからエニアグラムパラメータを推測するモジュール.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class EnneagramInference:
    """Enneagram parameter inference from MBTI parameters."""

    def calculate_enneagram_parameters(self, mbti_params: dict[str, float]) -> dict[str, float]:
        """Calculate Enneagram parameters from MBTI parameters.
        
        Args:
            mbti_params: MBTI parameters dictionary
            
        Returns:
            Dictionary containing Enneagram parameters (0-1 range)
        """
        enneagram_params = {
            "reformer": (
                mbti_params.get("intuition", 0.5) * 0.4 +
                mbti_params.get("thinking", 0.5) * 0.4 +
                mbti_params.get("judging", 0.5) * 0.2
            ),
            "helper": (
                mbti_params.get("feeling", 0.5) * 0.5 +
                mbti_params.get("extroversion", 0.5) * 0.5
            ),
            "achiever": (
                mbti_params.get("extroversion", 0.5) * 0.4 +
                mbti_params.get("thinking", 0.5) * 0.4 +
                mbti_params.get("judging", 0.5) * 0.2
            ),
            "individualist": (
                mbti_params.get("feeling", 0.5) * 0.6 +
                mbti_params.get("intuition", 0.5) * 0.4
            ),
            "investigator": (
                mbti_params.get("intuition", 0.5) * 0.5 +
                mbti_params.get("thinking", 0.5) * 0.5 +
                mbti_params.get("introversion", 0.5) * 0.5
            ) / 1.5,
            "loyalist": (
                mbti_params.get("sensing", 0.5) * 0.6 +
                mbti_params.get("introversion", 0.5) * 0.4
            ),
            "enthusiast": (
                mbti_params.get("extroversion", 0.5) * 0.6 +
                mbti_params.get("intuition", 0.5) * 0.4
            ),
            "challenger": (
                mbti_params.get("extroversion", 0.5) * 0.5 +
                mbti_params.get("thinking", 0.5) * 0.5
            ),
            "peacemaker": (
                mbti_params.get("introversion", 0.5) * 0.6 +
                mbti_params.get("feeling", 0.5) * 0.4
            )
        }
        
        # Ensure all values are within 0-1 range
        for key in enneagram_params:
            enneagram_params[key] = max(0.0, min(1.0, enneagram_params[key]))
            
        return enneagram_params

    def load_mbti_parameters(self, mbti_file_path: Path) -> dict[str, float] | None:
        """Load MBTI parameters from YAML file.
        
        Args:
            mbti_file_path: Path to MBTI YAML file
            
        Returns:
            MBTI parameters dictionary or None if error
        """
        try:
            with open(mbti_file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading MBTI parameters: {e}")
            return None

    def save_enneagram_parameters(self, enneagram_params: dict[str, float], output_path: Path) -> None:
        """Save Enneagram parameters to YAML file.
        
        Args:
            enneagram_params: Enneagram parameters dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(enneagram_params, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving Enneagram parameters: {e}")

    def process_mbti_to_enneagram(self, mbti_file_path: Path) -> dict[str, float] | None:
        """Process MBTI file and generate Enneagram parameters.
        
        Args:
            mbti_file_path: Path to MBTI YAML file
            
        Returns:
            Enneagram parameters dictionary or None if error
        """
        # Load MBTI parameters
        mbti_params = self.load_mbti_parameters(mbti_file_path)
        if mbti_params is None:
            return None
            
        # Calculate Enneagram parameters
        enneagram_params = self.calculate_enneagram_parameters(mbti_params)
        
        # Save to same directory as MBTI file
        output_path = mbti_file_path.parent / "enneagram.yml"
        self.save_enneagram_parameters(enneagram_params, output_path)
        
        return enneagram_params


def process_all_mbti_files(base_dir: str = "info/bdi_info/macro_bdi/macro_belief") -> None:
    """Process all MBTI files in the directory structure.
    
    Args:
        base_dir: Base directory to search for MBTI files
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory does not exist: {base_dir}")
        return
        
    inference = EnneagramInference()
    
    # Find all mbti.yml files
    mbti_files = list(base_path.glob("**/mbti.yml"))
    
    for mbti_file in mbti_files:
        print(f"Processing: {mbti_file}")
        result = inference.process_mbti_to_enneagram(mbti_file)
        if result:
            print(f"  Created: {mbti_file.parent / 'enneagram.yml'}")
        else:
            print(f"  Failed to process: {mbti_file}")


def infer_and_save_enneagram(mbti_file_path: str | Path) -> dict[str, float] | None:
    """Convenience function to infer and save Enneagram parameters from MBTI file.
    
    Args:
        mbti_file_path: Path to MBTI YAML file
        
    Returns:
        Enneagram parameters dictionary or None if error
    """
    inference = EnneagramInference()
    return inference.process_mbti_to_enneagram(Path(mbti_file_path))