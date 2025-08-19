#!/usr/bin/env python3
"""Phase context builder for game stage awareness.

ゲーム進行段階（フェーズ）と履歴コンテキストを構築するユーティリティ.
"""

import re
from typing import Any, Dict, List, Optional


def build_phase_context(info: Any, talk_history: List[Any], *, tail: int = 8) -> Dict[str, Any]:
    """Build phase context from game info and talk history.
    
    Args:
        info: Game info object with day, agent, status_map etc.
        talk_history: List of talk history objects with agent and text.
        tail: Number of recent talks to include in summary.
        
    Returns:
        Dictionary containing phase context information.
    """
    # Extract basic info
    day = getattr(info, 'day', 1) or 1
    agent_name = getattr(info, 'agent', '')
    status_map = getattr(info, 'status_map', {}) or {}
    
    # Count alive agents
    alive_agents = [name for name, status in status_map.items() if status == 'ALIVE']
    num_alive = len(alive_agents)
    
    # Analyze talk history
    num_talks_total = len(talk_history) if talk_history else 0
    
    # Count talks from today (simplified: assume all recent talks are today if day > 1)
    num_talks_today = min(num_talks_total, 20) if day == 1 else min(15, num_talks_total)
    
    has_history = num_talks_total > 0
    
    # Check if agent has done self-introduction
    has_self_intro = False
    if talk_history and agent_name:
        for talk in talk_history:
            if hasattr(talk, 'agent') and talk.agent == agent_name:
                text = getattr(talk, 'text', '')
                # Simple pattern matching for self-intro
                if re.search(r"(I'm|I am|My name|私は|名前は|Hello|Hi|初めまして)", text, re.IGNORECASE):
                    has_self_intro = True
                    break
                # Also check if agent name is mentioned
                if agent_name.lower() in text.lower():
                    has_self_intro = True
                    break
    
    # Build recent tail text
    recent_tail_text = ""
    if talk_history and tail > 0:
        recent_talks = talk_history[-tail:] if len(talk_history) > tail else talk_history
        tail_parts = []
        for talk in recent_talks:
            agent = getattr(talk, 'agent', '?')
            text = getattr(talk, 'text', '')
            # Truncate long texts
            if len(text) > 50:
                text = text[:47] + "..."
            tail_parts.append(f"{agent}: {text}")
        recent_tail_text = " | ".join(tail_parts)
    
    # Determine game phase
    phase = "opening"
    if day == 1 and num_talks_total < 15:
        phase = "opening"
    elif day >= 3 or num_alive <= 5:
        phase = "endgame"
    else:
        phase = "midgame"
    
    # Set early guardrails
    early_guardrails = (phase == "opening" and not has_self_intro)
    
    return {
        "day": day,
        "alive_agents": alive_agents,
        "num_alive": num_alive,
        "num_talks_total": num_talks_total,
        "num_talks_today": num_talks_today,
        "has_history": has_history,
        "has_self_intro": has_self_intro,
        "recent_tail_text": recent_tail_text,
        "phase": phase,
        "early_guardrails": early_guardrails
    }


def main():
    """Test phase context builder."""
    from types import SimpleNamespace as NS
    
    # Test case 1: Opening with no history
    info1 = NS(day=1, agent="TestAgent", status_map={"A": "ALIVE", "B": "ALIVE", "C": "ALIVE"})
    talks1 = []
    ctx1 = build_phase_context(info1, talks1)
    print("Test 1 (opening, no history):", ctx1)
    assert ctx1["phase"] == "opening"
    assert ctx1["early_guardrails"] == True
    assert ctx1["has_history"] == False
    
    # Test case 2: Midgame with history (need >5 alive for midgame)
    info2 = NS(day=2, agent="TestAgent", status_map={"A": "ALIVE", "B": "ALIVE", "C": "ALIVE", "D": "ALIVE", "E": "ALIVE", "F": "ALIVE"})
    talks2 = [
        NS(agent="A", text="I claim Seer"),
        NS(agent="B", text="Vote for A"),
        NS(agent="TestAgent", text="I'm TestAgent, villager")
    ] * 10  # Add more talks to get past opening threshold
    ctx2 = build_phase_context(info2, talks2)
    print("Test 2 (midgame with history):", ctx2)
    assert ctx2["phase"] == "midgame"
    assert ctx2["early_guardrails"] == False
    assert ctx2["has_self_intro"] == True
    
    print("✓ All tests passed")


if __name__ == "__main__":
    main()