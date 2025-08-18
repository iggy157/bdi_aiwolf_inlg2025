#!/usr/bin/env python3
"""Fixture modeのテストスクリプト."""

from pathlib import Path
import sys
import os
import yaml
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_fixture_disabled():
    """デフォルト（Fixture無効）の動作確認."""
    print("="*60)
    print("TEST 1: Fixture DISABLED (default)")
    print("="*60)
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    config = {
        "llm": {"type": "openai"},
        "openai": {"model": "gpt-4o", "temperature": 0.7},
        "prompt": {
            "analyze_message_type": "Type: {{content}}",
            "analyze_target_agents": "Target: {{content}}",
            "analyze_credibility": "Score"
        }
    }
    
    game_id = "01K2FIXTURE_TEST_DISABLED"
    agent_name = "TestAgent"
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # Skip/Overを含むトーク
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Hello World"),
    ]
    
    info = Info(
        game_id=game_id, day=1, agent=agent_name,
        profile=None, medium_result=None, divine_result=None,
        executed_agent=None, attacked_agent=None,
        vote_list=[], attack_vote_list=[],
        status_map={"Agent1": Status.ALIVE, "Agent2": Status.ALIVE, 
                   "Agent3": Status.ALIVE, agent_name: Status.ALIVE},
        role_map={}
    )
    
    added = tracker.analyze_talk(talks, info)
    print(f"Added entries: {added}")
    
    if added > 0:
        tracker.save_analysis()
        
        # 通常のanalysis.ymlが生成されているか確認
        analysis_file = tracker.output_dir / "analysis.yml"
        if analysis_file.exists():
            print(f"✓ analysis.yml created: {analysis_file}")
            with open(analysis_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Entries: {len(data)}")
                for k, v in data.items():
                    print(f"    {k}: {v['from']} -> '{v['content'][:30]}...'")
        else:
            print("✗ analysis.yml not created")
    else:
        print("No entries added (expected: Skip/Over filtered out)")
    
    print()

def test_fixture_enabled():
    """Fixture有効時の動作確認."""
    print("="*60)
    print("TEST 2: Fixture ENABLED")
    print("="*60)
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    config = {
        "llm": {"type": "openai"},
        "openai": {"model": "gpt-4o", "temperature": 0.7},
        "prompt": {
            "analyze_message_type": "Type: {{content}}",
            "analyze_target_agents": "Target: {{content}}",
            "analyze_credibility": "Score"
        },
        "analysis": {
            "fixture_mode": {
                "enable": True,
                "output_file": "analysis_test.yml",
                "trace_file": "analysis_fixture_trace.yml",
                "max_per_call": 3,
                "apply_to_agents": "others",
                "utterances": {
                    "default": [
                        "占いCOの有無を確認したいです。",
                        "便乗と早い同調を重く見ます。",
                        "初期の白黒各1名と根拠を提示ください。"
                    ]
                }
            }
        }
    }
    
    game_id = "01K2FIXTURE_TEST_ENABLED"
    agent_name = "TestAgent"
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # Skip/Overを含むトーク
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Skip"),
        Talk(idx=4, day=1, turn=3, agent="Agent4", text="Real message"),
        Talk(idx=5, day=1, turn=4, agent=agent_name, text="Skip"),  # 自分のSkip
    ]
    
    info = Info(
        game_id=game_id, day=1, agent=agent_name,
        profile=None, medium_result=None, divine_result=None,
        executed_agent=None, attacked_agent=None,
        vote_list=[], attack_vote_list=[],
        status_map={"Agent1": Status.ALIVE, "Agent2": Status.ALIVE,
                   "Agent3": Status.ALIVE, "Agent4": Status.ALIVE,
                   agent_name: Status.ALIVE},
        role_map={}
    )
    
    added = tracker.analyze_talk(talks, info)
    print(f"Added entries: {added}")
    
    if added > 0:
        tracker.save_analysis()
        
        # analysis_test.ymlが生成されているか確認
        test_file = tracker.output_dir / "analysis_test.yml"
        if test_file.exists():
            print(f"✓ analysis_test.yml created: {test_file}")
            with open(test_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Entries: {len(data)}")
                for k, v in data.items():
                    print(f"    {k}: {v['from']} -> '{v['content'][:50]}...'")
        else:
            print("✗ analysis_test.yml not created")
        
        # トレースファイルの確認
        trace_file = tracker.output_dir / "analysis_fixture_trace.yml"
        if trace_file.exists():
            print(f"✓ Trace file created: {trace_file}")
            with open(trace_file, 'r') as f:
                traces = yaml.safe_load(f)
                print(f"  Traces: {len(traces)}")
                for k, v in traces.items():
                    print(f"    {k}: '{v['original']}' -> '{v['replaced'][:30]}...'")
        else:
            print("✗ Trace file not created")
    
    print()

def test_fixture_with_env():
    """環境変数による設定上書きテスト."""
    print("="*60)
    print("TEST 3: Fixture with Environment Variables")
    print("="*60)
    
    # 環境変数を設定
    os.environ["ANALYSIS_FIXTURE_ENABLE"] = "1"
    os.environ["ANALYSIS_FIXTURE_OUTPUT_FILE"] = "analysis_env_test.yml"
    os.environ["ANALYSIS_FIXTURE_MAX_PER_CALL"] = "2"
    os.environ["ANALYSIS_FIXTURE_UTTERANCES_DEFAULT"] = "環境変数からの置換テキスト1|環境変数からの置換テキスト2"
    os.environ["ANALYSIS_UPDATE_SELECT_SENTENCE"] = "0"  # 下流処理を無効化
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    config = {
        "llm": {"type": "openai"},
        "openai": {"model": "gpt-4o", "temperature": 0.7},
        "prompt": {
            "analyze_message_type": "Type: {{content}}",
            "analyze_target_agents": "Target: {{content}}",
            "analyze_credibility": "Score"
        }
    }
    
    game_id = "01K2FIXTURE_TEST_ENV"
    agent_name = "TestAgent"
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Skip"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Skip"),
    ]
    
    info = Info(
        game_id=game_id, day=1, agent=agent_name,
        profile=None, medium_result=None, divine_result=None,
        executed_agent=None, attacked_agent=None,
        vote_list=[], attack_vote_list=[],
        status_map={"Agent1": Status.ALIVE, "Agent2": Status.ALIVE,
                   "Agent3": Status.ALIVE, agent_name: Status.ALIVE},
        role_map={}
    )
    
    added = tracker.analyze_talk(talks, info)
    print(f"Added entries: {added} (max_per_call=2, so max 2 expected)")
    
    if added > 0:
        tracker.save_analysis()
        
        # 環境変数で指定したファイル名で生成されているか
        env_file = tracker.output_dir / "analysis_env_test.yml"
        if env_file.exists():
            print(f"✓ analysis_env_test.yml created: {env_file}")
            with open(env_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Entries: {len(data)}")
                for k, v in data.items():
                    print(f"    {k}: {v['from']} -> '{v['content']}'")
        else:
            print("✗ analysis_env_test.yml not created")
    
    # 環境変数をクリーンアップ
    for key in ["ANALYSIS_FIXTURE_ENABLE", "ANALYSIS_FIXTURE_OUTPUT_FILE", 
                "ANALYSIS_FIXTURE_MAX_PER_CALL", "ANALYSIS_FIXTURE_UTTERANCES_DEFAULT",
                "ANALYSIS_UPDATE_SELECT_SENTENCE"]:
        os.environ.pop(key, None)
    
    print()

def main():
    """全テストを実行."""
    print("\n" + "="*60)
    print("FIXTURE MODE TEST SUITE")
    print("="*60 + "\n")
    
    test_fixture_disabled()
    test_fixture_enabled()
    test_fixture_with_env()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()