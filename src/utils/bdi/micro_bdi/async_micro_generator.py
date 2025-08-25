#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous micro_bdi generator with caching and optimization.

LLM処理の負荷を軽減し、フォールバックを避けるための非同期生成器。
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import yaml

class MicroBDICache:
    """Cache for micro_bdi generation to reduce LLM calls."""
    
    def __init__(self, cache_dir: Path, ttl_minutes: int = 5):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(minutes=ttl_minutes)
        
    def _get_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key from context."""
        # Extract key elements for caching
        key_elements = {
            "agent": context.get("agent"),
            "role": context.get("agent_role"),
            "day": context.get("day"),
            "discussion_stage": context.get("discussion_stage"),
            "analysis_count": len(context.get("analysis_tail", [])),
            "negatives": context.get("negatives", {}).get("total", 0),
        }
        key_str = json.dumps(key_elements, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached result if valid."""
        cache_key = self._get_cache_key(context)
        cache_file = self.cache_dir / f"{cache_key}.yml"
        
        if not cache_file.exists():
            return None
            
        # Check TTL
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            cache_file.unlink()  # Delete expired cache
            return None
            
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            return None
    
    def set(self, context: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Cache the result."""
        cache_key = self._get_cache_key(context)
        cache_file = self.cache_dir / f"{cache_key}.yml"
        
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(result, f, allow_unicode=True)
        except Exception:
            pass  # Ignore cache write errors


class AsyncMicroBDIGenerator:
    """Asynchronous generator for micro_bdi with optimizations."""
    
    def __init__(self, agent_obj, game_id: str, agent_name: str):
        self.agent_obj = agent_obj
        self.game_id = game_id
        self.agent_name = agent_name
        self.cache = MicroBDICache(
            Path(f"/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/cache/{game_id}/{agent_name}")
        )
        self.generation_queue = asyncio.Queue()
        self.results = {}
        
    async def generate_micro_desire_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate micro_desire asynchronously with caching."""
        # Check cache first
        cached = self.cache.get(context)
        if cached and "micro_desire" in cached:
            return cached["micro_desire"]
        
        # Simplify context for LLM (reduce token count)
        simplified_context = self._simplify_context(context)
        
        # Add to generation queue
        task_id = f"desire_{time.time()}"
        await self.generation_queue.put((task_id, "micro_desire", simplified_context))
        
        # Wait for result (with timeout)
        start_time = time.time()
        timeout = 10.0  # 10 seconds timeout for individual generation
        
        while task_id not in self.results:
            if time.time() - start_time > timeout:
                # Return simplified fallback instead of complex one
                return self._create_optimized_fallback_desire(context)
            await asyncio.sleep(0.1)
        
        result = self.results.pop(task_id)
        
        # Cache the result
        self.cache.set(context, {"micro_desire": result})
        
        return result
    
    async def generate_micro_intention_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate micro_intention asynchronously with caching."""
        # Check cache first
        cached = self.cache.get(context)
        if cached and "micro_intention" in cached:
            return cached["micro_intention"]
        
        # Simplify context for LLM
        simplified_context = self._simplify_context(context)
        
        # Add to generation queue
        task_id = f"intention_{time.time()}"
        await self.generation_queue.put((task_id, "micro_intention", simplified_context))
        
        # Wait for result (with timeout)
        start_time = time.time()
        timeout = 10.0
        
        while task_id not in self.results:
            if time.time() - start_time > timeout:
                return self._create_optimized_fallback_intention(context)
            await asyncio.sleep(0.1)
        
        result = self.results.pop(task_id)
        
        # Cache the result
        self.cache.set(context, {"micro_intention": result})
        
        return result
    
    def _simplify_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Simplify context to reduce LLM token usage."""
        simplified = {
            "agent": context.get("agent"),
            "agent_role": context.get("agent_role"),
            "discussion_stage": context.get("discussion_stage"),
            "current_desire": context.get("md_hint_current_desire", ""),
            "macro_desire_summary": context.get("macro_desire_summary", ""),
        }
        
        # Include only recent analysis (last 5 items)
        analysis_tail = context.get("analysis_tail", [])
        if analysis_tail:
            simplified["recent_analysis"] = analysis_tail[-5:]
        
        # Include only significant negative/positive counts
        negatives = context.get("negatives", {})
        if negatives.get("total", 0) > 0:
            simplified["negative_count"] = negatives["total"]
            
        # Include only low trust candidates with significant scores
        low_trust = context.get("low_trust_candidates", [])
        if low_trust:
            simplified["suspicious_agents"] = [
                candidate["agent"] for candidate in low_trust[:3]
            ]
        
        return simplified
    
    def _create_optimized_fallback_desire(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized fallback desire based on context."""
        stage = context.get("discussion_stage", "opening")
        role = context.get("agent_role", "villager")
        
        # Role-specific fallback strategies
        if role == "seer":
            content = "Gather information carefully while protecting my identity."
            response = "I'm observing the discussion patterns."
        elif role == "werewolf":
            content = "Blend in with villagers and avoid suspicion."
            response = "I agree with the village consensus."
        elif role == "possessed":
            content = "Create confusion without being too obvious."
            response = "There are interesting possibilities to consider."
        else:  # villager, bodyguard, medium
            content = "Find werewolves through logical analysis."
            response = "Let's discuss our observations."
        
        return {
            "content": content,
            "response_to_selected": response,
            "current_desire": f"Proceed according to {stage} stage.",
            "discussion_stage": stage,
            "stage_meta": {"optimized_fallback": True},
            "timestamp": datetime.now().isoformat(),
        }
    
    def _create_optimized_fallback_intention(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimized fallback intention based on context."""
        stage = context.get("discussion_stage", "opening")
        role = context.get("agent_role", "villager")
        
        # Stage-specific intentions
        if stage == "opening":
            summary = "Establish initial position"
            description = "Share initial thoughts and observe others"
        elif stage == "discussion":
            summary = "Analyze and discuss"
            description = "Engage in meaningful discussion based on observations"
        elif stage == "voting":
            summary = "Make voting decision"
            description = "Decide on vote target based on available information"
        else:
            summary = "Continue game play"
            description = "Adapt to current game state"
        
        return {
            "micro_intention": {
                "summary": summary,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "optimized_fallback": True,
            }
        }
    
    async def process_generation_queue(self):
        """Process generation queue in background."""
        while True:
            try:
                task_id, gen_type, context = await self.generation_queue.get()
                
                # Batch multiple requests if possible
                batch = [(task_id, gen_type, context)]
                
                # Try to get more items from queue (non-blocking)
                try:
                    for _ in range(2):  # Batch up to 3 items
                        task = self.generation_queue.get_nowait()
                        batch.append(task)
                except asyncio.QueueEmpty:
                    pass
                
                # Process batch
                for tid, gtype, ctx in batch:
                    if gtype == "micro_desire":
                        result = await self._generate_desire_with_llm(ctx)
                    else:
                        result = await self._generate_intention_with_llm(ctx)
                    self.results[tid] = result
                    
            except Exception as e:
                print(f"Error in generation queue: {e}")
                await asyncio.sleep(1)
    
    async def _generate_desire_with_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate desire using LLM with simplified context."""
        try:
            # Use simplified prompt
            prompt = self._create_simplified_prompt("micro_desire", context)
            
            # Call LLM asynchronously (simulated here, replace with actual async call)
            response = await self._async_llm_call(prompt)
            
            if response:
                return self._parse_llm_response(response, "desire")
            else:
                return self._create_optimized_fallback_desire(context)
        except Exception:
            return self._create_optimized_fallback_desire(context)
    
    async def _generate_intention_with_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate intention using LLM with simplified context."""
        try:
            prompt = self._create_simplified_prompt("micro_intention", context)
            response = await self._async_llm_call(prompt)
            
            if response:
                return self._parse_llm_response(response, "intention")
            else:
                return self._create_optimized_fallback_intention(context)
        except Exception:
            return self._create_optimized_fallback_intention(context)
    
    def _create_simplified_prompt(self, prompt_type: str, context: Dict[str, Any]) -> str:
        """Create simplified prompt to reduce token usage."""
        if prompt_type == "micro_desire":
            return f"""
Role: {context.get('agent_role')}
Stage: {context.get('discussion_stage')}
Goal: {context.get('current_desire', 'Play optimally')}

Generate a brief desire and response:
- content: Your internal thought (1 sentence)
- response: What you might say (1 sentence)

Format as YAML.
"""
        else:  # micro_intention
            return f"""
Role: {context.get('agent_role')}
Stage: {context.get('discussion_stage')}
Desire: {context.get('current_desire', 'Play optimally')}

Generate a brief intention:
- summary: Your plan (few words)
- description: How to execute (1 sentence)

Format as YAML.
"""
    
    async def _async_llm_call(self, prompt: str) -> Optional[str]:
        """Async LLM call (placeholder - integrate with actual LLM)."""
        # This is a placeholder for actual async LLM call
        # In production, use aiohttp or async OpenAI client
        await asyncio.sleep(0.1)  # Simulate async call
        
        # For now, return None to trigger optimized fallback
        # Replace with actual LLM integration
        return None
    
    def _parse_llm_response(self, response: str, response_type: str) -> Dict[str, Any]:
        """Parse LLM response into structured format."""
        try:
            # Clean response
            clean = response.strip()
            if clean.startswith("```yaml"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            
            data = yaml.safe_load(clean.strip())
            
            if response_type == "desire":
                return {
                    "content": data.get("content"),
                    "response_to_selected": data.get("response"),
                    "current_desire": data.get("current_desire", "Continue playing"),
                    "timestamp": datetime.now().isoformat(),
                }
            else:  # intention
                return {
                    "micro_intention": {
                        "summary": data.get("summary", "Continue"),
                        "description": data.get("description", "Adapt to situation"),
                        "timestamp": datetime.now().isoformat(),
                    }
                }
        except Exception:
            if response_type == "desire":
                return self._create_optimized_fallback_desire({})
            else:
                return self._create_optimized_fallback_intention({})


# Singleton instance management
_generator_instances: Dict[Tuple[str, str], AsyncMicroBDIGenerator] = {}

def get_async_generator(agent_obj, game_id: str, agent_name: str) -> AsyncMicroBDIGenerator:
    """Get or create async generator instance."""
    key = (game_id, agent_name)
    if key not in _generator_instances:
        _generator_instances[key] = AsyncMicroBDIGenerator(agent_obj, game_id, agent_name)
        # Start background processor
        asyncio.create_task(_generator_instances[key].process_generation_queue())
    return _generator_instances[key]