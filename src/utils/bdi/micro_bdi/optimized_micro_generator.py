#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized micro_bdi generator to prevent fallbacks.

フォールバックを防ぐための最適化されたmicro_bdi生成器。
LLM呼び出しを効率化し、タイムアウトを回避する。
"""

import os
import time
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading

# Import existing modules
from utils.bdi.micro_bdi.micro_desire_generator import (
    _decide_discussion_stage_fallback,
    _create_fallback_minimal_desire,
    load_yaml_safe,
)
from utils.bdi.micro_bdi.micro_intention_generator import (
    _create_fallback_micro_intention,
)


class OptimizedMicroGenerator:
    """Optimized generator with parallel processing and intelligent fallbacks."""
    
    def __init__(self, max_workers: int = 3, timeout_seconds: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = timeout_seconds
        self.cache = {}
        self.cache_lock = threading.Lock()
        
    def generate_micro_desire_optimized(
        self,
        agent_obj,
        game_id: str,
        agent: str,
        trigger: Optional[str] = None,
        logger_obj=None,
    ) -> Optional[Path]:
        """
        Generate micro_desire with optimization to prevent fallbacks.
        
        主な最適化:
        1. コンテキストの事前簡略化
        2. 並列処理による高速化
        3. インテリジェントキャッシング
        4. 段階的フォールバック
        """
        try:
            # Prepare paths
            agent_micro_dir = Path(f"/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi/{game_id}/{agent}")
            
            # Quick cache check
            cache_key = f"{game_id}_{agent}_desire_{trigger}"
            if cache_key in self.cache:
                cached_time, cached_result = self.cache[cache_key]
                if time.time() - cached_time < 300:  # 5 minutes cache
                    return cached_result
            
            # Load required data with optimization
            context = self._load_context_optimized(agent_obj, agent_micro_dir, game_id, agent)
            
            if not context:
                # Quick fallback without heavy processing
                return self._quick_fallback_desire(agent_micro_dir, agent, game_id, trigger)
            
            # Try LLM generation with timeout
            future = self.executor.submit(
                self._generate_desire_with_llm,
                agent_obj,
                context,
                agent_micro_dir,
                agent,
                game_id,
                trigger,
                logger_obj
            )
            
            try:
                result = future.result(timeout=self.timeout)
                
                # Cache successful result
                with self.cache_lock:
                    self.cache[cache_key] = (time.time(), result)
                
                return result
                
            except TimeoutError:
                # Intelligent fallback based on context
                if logger_obj:
                    logger_obj.logger.warning(f"Micro desire generation timeout for {agent}, using intelligent fallback")
                
                return self._intelligent_fallback_desire(
                    context, agent_micro_dir, agent, game_id, trigger, logger_obj
                )
                
        except Exception as e:
            if logger_obj:
                logger_obj.logger.exception(f"Failed to generate micro_desire: {e}")
            return None
    
    def generate_micro_intention_optimized(
        self,
        agent_obj,
        game_id: str,
        agent: str,
        trigger: Optional[str] = None,
        logger_obj=None,
    ) -> Optional[Path]:
        """Generate micro_intention with optimization."""
        try:
            # Prepare paths
            agent_micro_dir = Path(f"/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi/{game_id}/{agent}")
            
            # Quick cache check
            cache_key = f"{game_id}_{agent}_intention_{trigger}"
            if cache_key in self.cache:
                cached_time, cached_result = self.cache[cache_key]
                if time.time() - cached_time < 300:  # 5 minutes cache
                    return cached_result
            
            # Load micro_desire as base
            micro_desire_path = agent_micro_dir / "micro_desire.yml"
            if not micro_desire_path.exists():
                return self._quick_fallback_intention(agent_micro_dir, agent, game_id, trigger)
            
            desire_data = load_yaml_safe(micro_desire_path)
            if not desire_data or "micro_desires" not in desire_data:
                return self._quick_fallback_intention(agent_micro_dir, agent, game_id, trigger)
            
            latest_desire = desire_data["micro_desires"][-1] if desire_data["micro_desires"] else {}
            
            # Try LLM generation with timeout
            future = self.executor.submit(
                self._generate_intention_with_llm,
                agent_obj,
                latest_desire,
                agent_micro_dir,
                agent,
                game_id,
                trigger,
                logger_obj
            )
            
            try:
                result = future.result(timeout=self.timeout)
                
                # Cache successful result
                with self.cache_lock:
                    self.cache[cache_key] = (time.time(), result)
                
                return result
                
            except TimeoutError:
                if logger_obj:
                    logger_obj.logger.warning(f"Micro intention generation timeout for {agent}, using intelligent fallback")
                
                return self._intelligent_fallback_intention(
                    latest_desire, agent_micro_dir, agent, game_id, trigger, logger_obj
                )
                
        except Exception as e:
            if logger_obj:
                logger_obj.logger.exception(f"Failed to generate micro_intention: {e}")
            return None
    
    def _load_context_optimized(
        self, agent_obj, agent_micro_dir: Path, game_id: str, agent: str
    ) -> Optional[Dict[str, Any]]:
        """Load context with optimization - only essential data."""
        try:
            context = {}
            
            # Load only essential data
            analysis_path = agent_micro_dir / "analysis.yml"
            if analysis_path.exists():
                analysis = load_yaml_safe(analysis_path) or {}
                # Only last 10 items to reduce token count
                if analysis:
                    sorted_keys = sorted(k for k in analysis.keys() if isinstance(k, int))
                    last_10_keys = sorted_keys[-10:] if len(sorted_keys) > 10 else sorted_keys
                    context["analysis_tail"] = [analysis[k] for k in last_10_keys]
            
            # Load macro desire summary only
            macro_desire_path = Path(f"/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/macro_bdi/{game_id}/{agent}/macro_desire.yml")
            if macro_desire_path.exists():
                macro = load_yaml_safe(macro_desire_path) or {}
                if "macro_desire" in macro:
                    context["macro_desire_summary"] = macro["macro_desire"].get("summary", "")
            
            # Basic agent info
            if agent_obj and hasattr(agent_obj, "info"):
                context["agent_role"] = str(agent_obj.role) if hasattr(agent_obj, "role") else "villager"
                context["day"] = agent_obj.info.day if agent_obj.info else 1
            
            context["agent"] = agent
            context["game_id"] = game_id
            
            return context
            
        except Exception:
            return None
    
    def _generate_desire_with_llm(
        self, agent_obj, context, agent_micro_dir, agent, game_id, trigger, logger_obj
    ):
        """Generate desire using LLM with simplified prompt."""
        try:
            # Determine discussion stage quickly
            agent_role = context.get("agent_role", "villager")
            day = context.get("day", 1)
            
            if day == 0:
                stage = "opening"
            elif agent_obj and hasattr(agent_obj, "info") and agent_obj.info:
                alive_count = len([a for a in agent_obj.info.status_map.values() if a == "ALIVE"])
                if alive_count <= 5:
                    stage = "voting"
                else:
                    stage = "discussion"
            else:
                stage = "discussion"
            
            # Simplified prompt
            simplified_vars = {
                "agent": agent,
                "agent_role": agent_role,
                "discussion_stage": stage,
                "macro_desire_summary": context.get("macro_desire_summary", "Play optimally"),
                "analysis_tail": context.get("analysis_tail", [])[-5:],  # Only last 5
            }
            
            # Use agent's LLM with simplified context
            response = agent_obj.send_message_to_llm(
                "micro_desire_simple",  # Need to add this prompt template
                extra_vars=simplified_vars,
                log_tag="micro_desire_optimized",
            )
            
            if response:
                return self._save_desire_result(response, agent_micro_dir, stage, trigger)
            else:
                return self._quick_fallback_desire(agent_micro_dir, agent, game_id, trigger)
                
        except Exception:
            return self._quick_fallback_desire(agent_micro_dir, agent, game_id, trigger)
    
    def _generate_intention_with_llm(
        self, agent_obj, latest_desire, agent_micro_dir, agent, game_id, trigger, logger_obj
    ):
        """Generate intention using LLM."""
        try:
            simplified_vars = {
                "agent": agent,
                "current_desire": latest_desire.get("current_desire", "Continue playing"),
                "discussion_stage": latest_desire.get("discussion_stage", "discussion"),
            }
            
            response = agent_obj.send_message_to_llm(
                "micro_intention_simple",  # Need to add this prompt template
                extra_vars=simplified_vars,
                log_tag="micro_intention_optimized",
            )
            
            if response:
                return self._save_intention_result(response, agent_micro_dir, trigger)
            else:
                return self._quick_fallback_intention(agent_micro_dir, agent, game_id, trigger)
                
        except Exception:
            return self._quick_fallback_intention(agent_micro_dir, agent, game_id, trigger)
    
    def _intelligent_fallback_desire(
        self, context, agent_micro_dir, agent, game_id, trigger, logger_obj
    ):
        """Create intelligent fallback based on context."""
        agent_role = context.get("agent_role", "villager")
        day = context.get("day", 1)
        
        # Role and day specific fallback
        if agent_role == "werewolf":
            if day <= 1:
                content = "Establish credibility early in the game"
                response = "I'm interested in hearing everyone's thoughts"
            else:
                content = "Maintain cover while steering discussion"
                response = "We should focus on suspicious behavior patterns"
        elif agent_role == "seer":
            if day <= 1:
                content = "Gather information before revealing"
                response = "Let's observe carefully today"
            else:
                content = "Consider timing for role reveal"
                response = "I have some observations to share"
        else:  # Villager types
            content = "Analyze discussions to find werewolves"
            response = "I'm watching for inconsistencies"
        
        stage = "opening" if day == 0 else "discussion" if day <= 2 else "voting"
        
        new_desire = {
            "content": content,
            "response_to_selected": response,
            "current_desire": f"Proceed according to {stage} stage.",
            "discussion_stage": stage,
            "stage_meta": {"intelligent_fallback": True},
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger or "unknown",
            "game_id": game_id,
            "agent": agent,
        }
        
        return self._save_fallback_desire(new_desire, agent_micro_dir)
    
    def _intelligent_fallback_intention(
        self, latest_desire, agent_micro_dir, agent, game_id, trigger, logger_obj
    ):
        """Create intelligent fallback intention based on desire."""
        stage = latest_desire.get("discussion_stage", "discussion")
        current_desire = latest_desire.get("current_desire", "Continue playing")
        
        if "vote" in stage.lower() or "voting" in stage.lower():
            summary = "Execute voting strategy"
            description = "Vote based on accumulated evidence and suspicions"
        elif "opening" in stage.lower():
            summary = "Establish initial position"
            description = "Share opening thoughts while observing others"
        else:
            summary = "Engage in discussion"
            description = "Participate actively while gathering information"
        
        new_intention = {
            "micro_intention": {
                "summary": summary,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "intelligent_fallback": True,
            },
            "trigger": trigger or "unknown",
            "game_id": game_id,
            "agent": agent,
        }
        
        return self._save_fallback_intention(new_intention, agent_micro_dir)
    
    def _quick_fallback_desire(self, agent_micro_dir, agent, game_id, trigger):
        """Quick fallback without any processing."""
        new_desire = {
            "content": None,
            "response_to_selected": None,
            "current_desire": "Continue playing the game",
            "discussion_stage": "discussion",
            "stage_meta": {"quick_fallback": True},
            "timestamp": datetime.now().isoformat(),
            "trigger": trigger or "unknown",
            "game_id": game_id,
            "agent": agent,
        }
        return self._save_fallback_desire(new_desire, agent_micro_dir)
    
    def _quick_fallback_intention(self, agent_micro_dir, agent, game_id, trigger):
        """Quick fallback intention."""
        new_intention = {
            "micro_intention": {
                "summary": "Continue",
                "description": "Adapt to the current situation",
                "timestamp": datetime.now().isoformat(),
                "quick_fallback": True,
            },
            "trigger": trigger or "unknown",
            "game_id": game_id,
            "agent": agent,
        }
        return self._save_fallback_intention(new_intention, agent_micro_dir)
    
    def _save_desire_result(self, response, agent_micro_dir, stage, trigger):
        """Save desire result from LLM response."""
        try:
            # Parse response
            clean = response.strip()
            if clean.startswith("```yaml"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            
            data = yaml.safe_load(clean.strip())
            
            new_desire = {
                "content": data.get("content"),
                "response_to_selected": data.get("response_to_selected"),
                "current_desire": data.get("current_desire", f"Proceed according to {stage} stage."),
                "discussion_stage": stage,
                "timestamp": datetime.now().isoformat(),
                "trigger": trigger or "unknown",
            }
            
            return self._save_fallback_desire(new_desire, agent_micro_dir)
            
        except Exception:
            return None
    
    def _save_intention_result(self, response, agent_micro_dir, trigger):
        """Save intention result from LLM response."""
        try:
            # Parse response
            clean = response.strip()
            if clean.startswith("```yaml"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            
            data = yaml.safe_load(clean.strip())
            
            new_intention = {
                "micro_intention": data.get("micro_intention", {
                    "summary": "Continue",
                    "description": "Adapt to situation"
                }),
                "timestamp": datetime.now().isoformat(),
                "trigger": trigger or "unknown",
            }
            
            return self._save_fallback_intention(new_intention, agent_micro_dir)
            
        except Exception:
            return None
    
    def _save_fallback_desire(self, new_desire, agent_micro_dir):
        """Save desire to file."""
        output_path = agent_micro_dir / "micro_desire.yml"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing = load_yaml_safe(output_path) if output_path.exists() else {}
            if "micro_desires" not in existing:
                existing["micro_desires"] = []
            existing["micro_desires"].append(new_desire)
            
            # Keep only last 20
            if len(existing["micro_desires"]) > 20:
                existing["micro_desires"] = existing["micro_desires"][-20:]
            
            # Atomic write
            tmp = output_path.with_suffix(".yml.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(existing, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, output_path)
            
            return output_path
        except Exception:
            return None
    
    def _save_fallback_intention(self, new_intention, agent_micro_dir):
        """Save intention to file."""
        output_path = agent_micro_dir / "micro_intention.yml"
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic write
            tmp = output_path.with_suffix(".yml.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                yaml.safe_dump(new_intention, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp, output_path)
            
            return output_path
        except Exception:
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=False)


# Global instance
_global_generator = None

def get_optimized_generator() -> OptimizedMicroGenerator:
    """Get or create global optimized generator."""
    global _global_generator
    if _global_generator is None:
        _global_generator = OptimizedMicroGenerator()
    return _global_generator