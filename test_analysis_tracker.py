#!/usr/bin/env python3
"""AnalysisTrackerのテストスクリプト."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
from aiwolf_nlp_common.packet import Talk, Info, Status

def test_analysis_tracker():
    """AnalysisTrackerの動作テスト."""
    print("=== AnalysisTracker Test Start ===")
    
    # 設定
    config = {
        "llm": {"type": "openai"},
        "openai": {"model": "gpt-4o", "temperature": 0.7},
        "prompt": {
            "analyze_message_type": "Analyze the type of this message: {{content}}. Reply with one of: co, question, negative, positive, null",
            "analyze_target_agents": "Who is this message directed to? {{content}}. Agents: {{agent_names}}. Reply with agent names or 'all' or 'null'",
            "analyze_credibility": "Rate credibility:\nlogical_consistency: 0-1\nspecificity_and_detail: 0-1\nintuitive_depth: 0-1\nclarity_and_conciseness: 0-1"
        }
    }
    
    # Trackerの初期化
    game_id = "01K2VKY29MTA8CGEXDZRFPHK0A"  # 実際のgame_idを使用
    agent_name = "TestAgent"
    
    print(f"Initializing AnalysisTracker with game_id={game_id}, agent={agent_name}")
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # ダミーのTalkデータ作成（自分の発話、Over、Skip含む）
    talk_history = []
    
    # 他者の発話（解析対象）
    talk1 = Talk(
        idx=1,
        day=1,
        turn=0,
        agent="Agent01",
        text="I think Agent02 is suspicious."
    )
    talk_history.append(talk1)
    
    # Over発話（除外される）
    talk2 = Talk(
        idx=2,
        day=1,
        turn=1,
        agent="Agent02",
        text="Over"
    )
    talk_history.append(talk2)
    
    # Skip発話（除外される）
    talk3 = Talk(
        idx=3,
        day=1,
        turn=2,
        agent="Agent03",
        text="Skip"
    )
    talk_history.append(talk3)
    
    # 自分の発話（除外される）
    talk4 = Talk(
        idx=4,
        day=1,
        turn=3,
        agent=agent_name,
        text="I agree with Agent01."
    )
    talk_history.append(talk4)
    
    # 他者の有意味な発話（解析対象）
    talk5 = Talk(
        idx=5,
        day=1,
        turn=4,
        agent="Agent04",
        text="Let's vote for Agent01."
    )
    talk_history.append(talk5)
    
    # Info作成
    info = Info(
        game_id=game_id,
        day=1,
        agent=agent_name,
        profile=None,
        medium_result=None,
        divine_result=None,
        executed_agent=None,
        attacked_agent=None,
        vote_list=[],
        attack_vote_list=[],
        status_map={
            "Agent01": Status.ALIVE,
            "Agent02": Status.ALIVE,
            "Agent03": Status.ALIVE,
            "Agent04": Status.ALIVE,
            agent_name: Status.ALIVE
        },
        role_map={}
    )
    
    # 第1回分析実行
    print(f"\n=== First Analysis: {len(talk_history)} talks ===")
    added1 = tracker.analyze_talk(talk_history, info, request_count=1)
    print(f"Added in first call: {added1}")
    
    if added1 > 0:
        tracker.save_analysis()
        print(f"Saved analysis with {added1} entries")
    
    # 第2回分析実行（同じデータ + 新しい1件）
    talk6 = Talk(
        idx=6,
        day=1,
        turn=5,
        agent="Agent05",
        text="I'm the medium. Agent02 is human."
    )
    talk_history.append(talk6)
    
    print(f"\n=== Second Analysis: {len(talk_history)} talks (1 new) ===")
    added2 = tracker.analyze_talk(talk_history, info, request_count=2)
    print(f"Added in second call: {added2}")
    
    if added2 > 0:
        tracker.save_analysis()
        print(f"Saved analysis with {added2} new entries")
    
    # 第3回分析実行（同じデータのみ、重複チェック）
    print(f"\n=== Third Analysis: {len(talk_history)} talks (no new) ===")
    added3 = tracker.analyze_talk(talk_history, info, request_count=3)
    print(f"Added in third call: {added3}")
    
    if added3 > 0:
        tracker.save_analysis()
        print(f"Saved analysis with {added3} new entries")
    else:
        print("No new entries, no save needed")
    
    # 保存確認
    print(f"\n=== Final File Check ===")
    if tracker.analysis_file.exists():
        print(f"✓ File exists: {tracker.analysis_file}")
        size = tracker.analysis_file.stat().st_size
        print(f"✓ File size: {size} bytes")
        
        # ファイル内容を表示
        with open(tracker.analysis_file, 'r') as f:
            content = f.read()
            print(f"\n--- File content ---")
            print(content[:1000] if len(content) > 1000 else content)
            print(f"--- End of content ---")
    else:
        print(f"✗ File not found: {tracker.analysis_file}")
    
    print(f"\n=== Expected Results ===")
    print(f"- First call should add 2 entries (Agent01, Agent04)")
    print(f"- Second call should add 1 entry (Agent05)")
    print(f"- Third call should add 0 entries (duplicates)")
    print(f"- Skip/Over/Self talks should be excluded")
    print(f"- Final file should contain 3 analysis entries")
    
    print("\n=== AnalysisTracker Test End ===")

if __name__ == "__main__":
    test_analysis_tracker()