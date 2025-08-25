#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized agent for 13-player games.

13人村でのパフォーマンス問題を解決するための最適化エージェント。
"""

from __future__ import annotations
import time
from typing import Optional
from pathlib import Path

def get_simplified_whisper_response(role: str, day: int, agent_name: str) -> str:
    """
    Generate simplified whisper response without heavy LLM processing.
    
    13人村でのwhisper処理を軽量化するため、ロールベースの簡単な応答を生成。
    """
    # Role-based simple whisper responses
    if role.lower() == "werewolf":
        if day == 0:
            responses = [
                "Let's coordinate our strategy.",
                "We should be careful.",
                "I'll follow your lead.",
                "Agreed.",
                "Good plan.",
            ]
        elif day == 1:
            responses = [
                "Who should we target?",
                "I suggest we focus.",
                "Let's avoid suspicion.",
                "Be careful with claims.",
                "Watch the seer.",
            ]
        else:
            responses = [
                "We need to eliminate threats.",
                "The seer might be hiding.",
                "Stay consistent.",
                "Don't vote together.",
                "Keep our stories straight.",
            ]
    elif role.lower() == "possessed":
        if day == 0:
            responses = [
                "I'll help create confusion.",
                "Looking forward to working together.",
                "I'll support your claims.",
                "Understood.",
                "Ready to assist.",
            ]
        else:
            responses = [
                "I can claim if needed.",
                "Should I counter-claim?",
                "I'll back your story.",
                "Creating doubt now.",
                "Misdirecting villagers.",
            ]
    else:
        # Villager roles shouldn't whisper, but provide fallback
        responses = ["Skip", "Over", ">>"]
    
    # Simple rotation based on agent name and day
    import hashlib
    hash_val = int(hashlib.md5(f"{agent_name}{day}".encode()).hexdigest()[:8], 16)
    return responses[hash_val % len(responses)]


def get_simplified_talk_response(role: str, day: int, talk_count: int, agent_name: str) -> str:
    """
    Generate simplified talk response for 13-player games.
    
    13人村での負荷を軽減するための簡略化された発話生成。
    """
    # First talk of the day
    if talk_count == 0:
        if day == 0:
            if role.lower() == "seer":
                return "I'll observe carefully today."
            elif role.lower() == "werewolf":
                return "Let's work together to find the werewolves."
            elif role.lower() == "possessed":
                return "I'm here to help the village."
            else:
                return "Good morning everyone."
        else:
            if role.lower() == "seer" and day >= 2:
                return "I have information to share."
            else:
                return "Let's discuss yesterday's events."
    
    # Subsequent talks - keep it simple
    if talk_count < 3:
        simple_responses = [
            "I agree with the analysis.",
            "That's an interesting point.",
            "We should consider all possibilities.",
            "I'm watching the discussion.",
            "Let me think about that.",
            ">>",
            "Skip",
            "Over",
        ]
        import hashlib
        hash_val = int(hashlib.md5(f"{agent_name}{day}{talk_count}".encode()).hexdigest()[:8], 16)
        return simple_responses[hash_val % len(simple_responses)]
    else:
        # After 3 talks, mostly skip
        return "Skip" if talk_count % 3 == 0 else ">>"


class OptimizedAgentMixin:
    """
    Mixin class to optimize agent performance for 13-player games.
    
    既存のAgentクラスに追加する最適化機能。
    """
    
    def is_large_game(self) -> bool:
        """Check if this is a large game (>= 10 players)."""
        if hasattr(self, "info") and self.info and hasattr(self.info, "status_map"):
            return len(self.info.status_map) >= 10
        if hasattr(self, "config") and "agent" in self.config:
            return self.config["agent"].get("num", 5) >= 10
        return False
    
    def should_use_simplified_whisper(self) -> bool:
        """Determine if simplified whisper should be used."""
        # Use simplified whisper for large games or when specified in config
        if self.is_large_game():
            return True
        if hasattr(self, "config") and "agent" in self.config:
            return self.config["agent"].get("use_simplified_whisper", False)
        return False
    
    def should_use_simplified_talk(self) -> bool:
        """Determine if simplified talk should be used."""
        # Use simplified talk after certain number of talks in large games
        if not self.is_large_game():
            return False
        
        if hasattr(self, "sent_talk_count"):
            # Use simplified after 5 talks in large games
            return self.sent_talk_count >= 5
        
        return False
    
    def get_quick_whisper(self) -> str:
        """Get quick whisper response without LLM."""
        role = str(self.role) if hasattr(self, "role") else "villager"
        day = self.info.day if hasattr(self, "info") and self.info else 0
        agent_name = self.agent_name if hasattr(self, "agent_name") else "Agent"
        
        return get_simplified_whisper_response(role, day, agent_name)
    
    def get_quick_talk(self) -> str:
        """Get quick talk response without LLM."""
        role = str(self.role) if hasattr(self, "role") else "villager"
        day = self.info.day if hasattr(self, "info") and self.info else 0
        talk_count = self.sent_talk_count if hasattr(self, "sent_talk_count") else 0
        agent_name = self.agent_name if hasattr(self, "agent_name") else "Agent"
        
        return get_simplified_talk_response(role, day, talk_count, agent_name)
    
    def optimize_for_large_game(self) -> None:
        """Apply optimizations for large games."""
        if not self.is_large_game():
            return
        
        # Reduce LLM processing for large games
        if hasattr(self, "config") and "llm" in self.config:
            # Increase sleep time to avoid rate limiting
            self.config["llm"]["sleep_time"] = max(0.5, float(self.config["llm"].get("sleep_time", 0)))
            
            # Reduce temperature for more deterministic responses
            if "openai" in self.config:
                self.config["openai"]["temperature"] = min(0.5, float(self.config["openai"].get("temperature", 0.7)))
            if "google" in self.config:
                self.config["google"]["temperature"] = min(0.5, float(self.config["google"].get("temperature", 0.7)))
    
    def precompute_responses(self) -> None:
        """Precompute some responses to reduce processing time."""
        if not self.is_large_game():
            return
        
        # Cache directory for precomputed responses
        cache_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/response_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # This can be extended to precompute common responses


def apply_13player_optimization(agent_class):
    """
    Decorator to apply 13-player optimizations to an agent class.
    
    使用例:
    @apply_13player_optimization
    class Agent:
        ...
    """
    # Store original methods
    original_whisper = agent_class.whisper if hasattr(agent_class, "whisper") else None
    original_talk = agent_class.talk if hasattr(agent_class, "talk") else None
    original_initialize = agent_class.initialize if hasattr(agent_class, "initialize") else None
    
    def optimized_whisper(self) -> str:
        """Optimized whisper for large games."""
        # Check if we should use simplified whisper
        mixin = OptimizedAgentMixin()
        mixin.__dict__ = self.__dict__  # Share state
        
        if mixin.should_use_simplified_whisper():
            response = mixin.get_quick_whisper()
            if hasattr(self, "sent_whisper_count") and hasattr(self, "whisper_history"):
                self.sent_whisper_count = len(self.whisper_history)
            return response
        
        # Fall back to original whisper
        if original_whisper:
            return original_whisper(self)
        return "Skip"
    
    def optimized_talk(self) -> str:
        """Optimized talk for large games."""
        # Check if we should use simplified talk
        mixin = OptimizedAgentMixin()
        mixin.__dict__ = self.__dict__  # Share state
        
        if mixin.should_use_simplified_talk():
            response = mixin.get_quick_talk()
            if hasattr(self, "sent_talk_count"):
                self.sent_talk_count += 1
            return response
        
        # Fall back to original talk
        if original_talk:
            return original_talk(self)
        return "Skip"
    
    def optimized_initialize(self) -> None:
        """Optimized initialization for large games."""
        # Apply optimizations
        mixin = OptimizedAgentMixin()
        mixin.__dict__ = self.__dict__  # Share state
        mixin.optimize_for_large_game()
        mixin.precompute_responses()
        
        # Call original initialize
        if original_initialize:
            original_initialize(self)
    
    # Replace methods
    agent_class.whisper = optimized_whisper
    agent_class.talk = optimized_talk
    agent_class.initialize = optimized_initialize
    
    return agent_class