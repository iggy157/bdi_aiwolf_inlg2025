# -*- coding: utf-8 -*-
"""Role name mapping utilities.

役職名のマッピングユーティリティ.
"""

from typing import Optional

ROLE_MAP_EN2JA = {
    "WEREWOLF": "人狼",
    "VILLAGER": "村人",
    "POSSESSED": "狂人",
    "SEER": "占い師",
    "BODYGUARD": "騎士",
    "MEDIUM": "霊媒師",
}


def to_japanese_role(role_en: Optional[str]) -> Optional[str]:
    """Convert English role name to Japanese.
    
    Args:
        role_en: English role name (e.g., 'SEER', 'WEREWOLF')
        
    Returns:
        Japanese role name or None if not found
    """
    if not role_en:
        return None
    return ROLE_MAP_EN2JA.get(role_en.strip().upper())