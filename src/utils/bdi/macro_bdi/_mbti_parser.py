# -*- coding: utf-8 -*-
"""Robust MBTI output parser with normalization and key mapping.

MBTI出力の正規化とキーマッピングによる堅牢なパーサー.
"""

from __future__ import annotations

import re
from typing import Dict


# MBTI key mapping - various representations to standard keys
MBTI_KEY_MAPPING = {
    # Standard keys
    "extroversion": "extroversion",
    "introversion": "introversion", 
    "sensing": "sensing",
    "intuition": "intuition",
    "thinking": "thinking",
    "feeling": "feeling",
    "judging": "judging",
    "perceiving": "perceiving",
    
    # Common variations with prefixes/suffixes
    "- extroversion": "extroversion",
    "- introversion": "introversion",
    "- sensing": "sensing", 
    "- intuition": "intuition",
    "- thinking": "thinking",
    "- feeling": "feeling",
    "- judging": "judging",
    "- perceiving": "perceiving",
    
    # Bullet points
    "• extroversion": "extroversion",
    "• introversion": "introversion",
    "• sensing": "sensing",
    "• intuition": "intuition", 
    "• thinking": "thinking",
    "• feeling": "feeling",
    "• judging": "judging",
    "• perceiving": "perceiving",
    
    # Numbered
    "1. extroversion": "extroversion",
    "2. introversion": "introversion",
    "3. sensing": "sensing",
    "4. intuition": "intuition",
    "5. thinking": "thinking",
    "6. feeling": "feeling",
    "7. judging": "judging", 
    "8. perceiving": "perceiving",
    
    # Alternative spellings/abbreviations
    "extraversion": "extroversion",
    "e": "extroversion",
    "i": "introversion",
    "s": "sensing",
    "n": "intuition",
    "t": "thinking",
    "f": "feeling",
    "j": "judging",
    "p": "perceiving",
    
    # With scores
    "extroversion score": "extroversion",
    "introversion score": "introversion",
    "sensing score": "sensing",
    "intuition score": "intuition",
    "thinking score": "thinking", 
    "feeling score": "feeling",
    "judging score": "judging",
    "perceiving score": "perceiving",
    
    # Japanese
    "外向性": "extroversion",
    "内向性": "introversion",
    "感覚": "sensing",
    "直観": "intuition",
    "直感": "intuition",
    "思考": "thinking", 
    "感情": "feeling",
    "判断": "judging",
    "知覚": "perceiving",
}


def normalize_mbti_key(key: str) -> str | None:
    """Normalize MBTI parameter key to standard format.
    
    Args:
        key: Raw key from LLM output
        
    Returns:
        Normalized standard key or None if not found
    """
    # Clean the key
    clean_key = key.strip().lower()
    
    # Remove common prefixes/suffixes
    clean_key = re.sub(r'^[\-\•\*\d+\.\)\]\s]+', '', clean_key)  # Remove bullets, numbers, etc.
    clean_key = re.sub(r'[\:\=\s]+$', '', clean_key)  # Remove trailing colons, equals, spaces
    clean_key = clean_key.strip()
    
    # Direct mapping
    if clean_key in MBTI_KEY_MAPPING:
        return MBTI_KEY_MAPPING[clean_key]
    
    # Fuzzy matching for partial matches
    for mapped_key, standard_key in MBTI_KEY_MAPPING.items():
        if clean_key in mapped_key or mapped_key in clean_key:
            return standard_key
    
    return None


def extract_numeric_value(value_str: str) -> float | None:
    """Extract numeric value from various string formats.
    
    Args:
        value_str: String containing numeric value
        
    Returns:
        Normalized float value (0-1 range) or None if not found
    """
    # Clean the value string
    clean_value = re.sub(r'[^\d\.\-\+]', '', value_str.strip())
    
    if not clean_value:
        return None
        
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
        return None


def parse_mbti_output_robust(response: str) -> Dict[str, float]:
    """Robustly parse MBTI parameters from LLM response.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Dictionary with normalized MBTI parameters
    """
    mbti_params = {}
    
    # Split response into lines for processing
    lines = response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Look for key-value patterns with various separators
        patterns = [
            r'(.+?)[:=]\s*(.+)',  # key: value or key = value
            r'(.+?)\s+(-?\d+\.?\d*)',  # key 0.5
            r'(.+?)\s*[-–—]\s*(.+)',  # key - value
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                raw_key = match.group(1).strip()
                raw_value = match.group(2).strip()
                
                # Normalize key
                standard_key = normalize_mbti_key(raw_key)
                if not standard_key:
                    continue
                
                # Extract numeric value
                numeric_value = extract_numeric_value(raw_value)
                if numeric_value is not None:
                    mbti_params[standard_key] = numeric_value
                    break
    
    # Ensure all required parameters exist with defaults
    required_params = {
        "extroversion": 0.5,
        "introversion": 0.5,
        "sensing": 0.5,
        "intuition": 0.5,
        "thinking": 0.5,
        "feeling": 0.5,
        "judging": 0.5,
        "perceiving": 0.5
    }
    
    for param, default_value in required_params.items():
        if param not in mbti_params:
            mbti_params[param] = default_value
    
    return mbti_params


def test_parser():
    """Test function for the robust parser."""
    test_cases = [
        # Current problematic case
        """- extroversion: 0.2
- introversion: 0.8
- sensing: 0.4
- intuition: 0.6
- thinking: 0.8
- feeling: 0.2
- judging: 0.6
- perceiving: 0.4""",
        
        # Standard format
        """extroversion: 0.3
introversion: 0.7
sensing: 0.6
intuition: 0.4
thinking: 0.8
feeling: 0.2
judging: 0.5
perceiving: 0.5""",
        
        # Numbered list
        """1. Extroversion: 0.4
2. Introversion: 0.6
3. Sensing: 0.3
4. Intuition: 0.7
5. Thinking: 0.9
6. Feeling: 0.1
7. Judging: 0.8
8. Perceiving: 0.2""",
    ]
    
    for i, case in enumerate(test_cases):
        print(f"Test case {i+1}:")
        result = parse_mbti_output_robust(case)
        print(result)
        print()


if __name__ == "__main__":
    test_parser()