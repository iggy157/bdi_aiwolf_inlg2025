#!/usr/bin/env python3
"""Fixtureモードの動作確認スクリプト（本番config.ymlを使用）."""

from pathlib import Path
import sys
import os
import yaml
sys.path.insert(0, str(Path(__file__).parent / "src"))

def load_config():
    """本番のconfig.ymlをロード."""
    config_path = Path(__file__).parent / "config" / "config.yml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def test_fixture_default_off():
    """デフォルト（Fixture OFF）の動作確認."""
    print("="*70)
    print("TEST 1: Default Configuration (Fixture OFF)")
    print("="*70)
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    # 本番設定をロード
    config = load_config()
    
    game_id = "01K2VERIFY_DEFAULT_OFF"
    agent_name = "TestAgent"
    
    print(f"Config fixture_mode.enable: {config.get('analysis', {}).get('fixture_mode', {}).get('enable', False)}")
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # Skip/Overを含むトーク
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="こんにちは"),
        Talk(idx=4, day=1, turn=3, agent=agent_name, text="自分の発話"),
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
    print(f"\\nAdded entries: {added}")
    print("Expected: 1 (only Agent3's meaningful utterance)")
    
    if added > 0:
        tracker.save_analysis()
        
        # 通常のanalysis.ymlが生成されているか確認
        analysis_file = tracker.output_dir / "analysis.yml"
        if analysis_file.exists():
            print(f"✓ analysis.yml created at expected location")
            with open(analysis_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Total entries in file: {len(data) if data else 0}")
        else:
            print("✗ analysis.yml not created")
        
        # Fixture用ファイルが生成されていないことを確認
        test_file = tracker.output_dir / "analysis_test.yml"
        if test_file.exists():
            print("✗ Unexpected: analysis_test.yml created when Fixture is OFF")
        else:
            print("✓ analysis_test.yml not created (expected)")
    
    print()

def test_fixture_enabled_via_env():
    """環境変数でFixture ONにした動作確認."""
    print("="*70)
    print("TEST 2: Fixture Enabled via Environment Variable")
    print("="*70)
    
    # 環境変数でFixtureモードを有効化
    os.environ["ANALYSIS_FIXTURE_ENABLE"] = "1"
    os.environ["ANALYSIS_FIXTURE_OUTPUT_FILE"] = "analysis_fixture_test.yml"
    os.environ["ANALYSIS_FIXTURE_MAX_PER_CALL"] = "3"
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    # 本番設定をロード（config.ymlではenable: false）
    config = load_config()
    
    game_id = "01K2VERIFY_FIXTURE_ON"
    agent_name = "TestAgent"
    
    print(f"Config fixture_mode.enable: {config.get('analysis', {}).get('fixture_mode', {}).get('enable', False)}")
    print(f"ENV ANALYSIS_FIXTURE_ENABLE: {os.environ.get('ANALYSIS_FIXTURE_ENABLE')}")
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # Skip/Overばかりのトーク（実際のゲームでよくある状況）
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Skip"),
        Talk(idx=4, day=1, turn=3, agent="Agent4", text="Over"),
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
    print(f"\\nAdded entries: {added}")
    print(f"Expected: 3 (max_per_call=3, 4 others available but limited)")
    
    if added > 0:
        tracker.save_analysis()
        
        # Fixture用ファイルが生成されているか確認
        fixture_file = tracker.output_dir / "analysis_fixture_test.yml"
        if fixture_file.exists():
            print(f"✓ analysis_fixture_test.yml created")
            with open(fixture_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Entries: {len(data) if data else 0}")
                if data:
                    for k, v in list(data.items())[:2]:  # 最初の2件を表示
                        content_preview = v['content'][:40] + "..." if len(v['content']) > 40 else v['content']
                        print(f"    {k}: {v['from']} -> '{content_preview}'")
        else:
            print("✗ analysis_fixture_test.yml not created")
        
        # トレースファイルの確認
        trace_file = tracker.output_dir / "analysis_fixture_trace.yml"
        if trace_file.exists():
            print(f"✓ Trace file created")
            with open(trace_file, 'r') as f:
                traces = yaml.safe_load(f)
                print(f"  Trace entries: {len(traces) if traces else 0}")
                if traces:
                    for k, v in list(traces.items())[:2]:  # 最初の2件を表示
                        print(f"    {k}: '{v['original']}' -> '{v['replaced'][:30]}...'")
        else:
            print("✗ Trace file not created")
        
        # 通常のanalysis.ymlが作成されていないことを確認
        normal_file = tracker.output_dir / "analysis.yml"
        if normal_file.exists():
            print("✗ Unexpected: analysis.yml created when Fixture is ON")
        else:
            print("✓ analysis.yml not created (expected in Fixture mode)")
    
    # 環境変数をクリーンアップ
    for key in ["ANALYSIS_FIXTURE_ENABLE", "ANALYSIS_FIXTURE_OUTPUT_FILE", 
                "ANALYSIS_FIXTURE_MAX_PER_CALL"]:
        os.environ.pop(key, None)
    
    print()

def test_downstream_control():
    """下流処理の制御確認."""
    print("="*70)
    print("TEST 3: Downstream Process Control")
    print("="*70)
    
    # Fixtureモード有効 + 下流処理の制御
    os.environ["ANALYSIS_FIXTURE_ENABLE"] = "1"
    os.environ["ANALYSIS_UPDATE_SELECT_SENTENCE"] = "0"  # 明示的に無効
    os.environ["ANALYSIS_UPDATE_INTENTION"] = "1"        # 明示的に有効
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    config = load_config()
    
    game_id = "01K2VERIFY_DOWNSTREAM"
    agent_name = "TestAgent"
    
    print(f"ENV ANALYSIS_FIXTURE_ENABLE: {os.environ.get('ANALYSIS_FIXTURE_ENABLE')}")
    print(f"ENV ANALYSIS_UPDATE_SELECT_SENTENCE: {os.environ.get('ANALYSIS_UPDATE_SELECT_SENTENCE')}")
    print(f"ENV ANALYSIS_UPDATE_INTENTION: {os.environ.get('ANALYSIS_UPDATE_INTENTION')}")
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
    ]
    
    info = Info(
        game_id=game_id, day=1, agent=agent_name,
        profile=None, medium_result=None, divine_result=None,
        executed_agent=None, attacked_agent=None,
        vote_list=[], attack_vote_list=[],
        status_map={"Agent1": Status.ALIVE, agent_name: Status.ALIVE},
        role_map={}
    )
    
    print("\\nExpected log output:")
    print("  - 'Downstream(select_sentence): SKIP' (explicitly disabled)")
    print("  - 'Downstream(intention): RUN' (explicitly enabled)\\n")
    
    added = tracker.analyze_talk(talks, info)
    if added > 0:
        tracker.save_analysis()
    
    # 環境変数をクリーンアップ
    for key in ["ANALYSIS_FIXTURE_ENABLE", "ANALYSIS_UPDATE_SELECT_SENTENCE",
                "ANALYSIS_UPDATE_INTENTION"]:
        os.environ.pop(key, None)
    
    print()

def main():
    """全テストを実行."""
    print("\\n" + "="*70)
    print("FIXTURE MODE VERIFICATION")
    print("="*70 + "\\n")
    
    # 設定ファイルの存在確認
    config_path = Path(__file__).parent / "config" / "config.yml"
    if not config_path.exists():
        print(f"ERROR: config.yml not found at {config_path}")
        return
    
    print(f"Using config: {config_path}\\n")
    
    # 各テストを実行
    test_fixture_default_off()
    test_fixture_enabled_via_env()
    test_downstream_control()
    
    print("="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)
    print("\\nEnvironment variable examples for manual testing:")
    print("  export ANALYSIS_FIXTURE_ENABLE=1")
    print("  export ANALYSIS_FIXTURE_OUTPUT_FILE=analysis_test.yml")
    print("  export ANALYSIS_FIXTURE_TRACE_FILE=trace.yml")
    print("  export ANALYSIS_FIXTURE_TARGETS='Skip,Over'")
    print("  export ANALYSIS_FIXTURE_MAX_PER_CALL=5")
    print("  export ANALYSIS_FIXTURE_APPLY_TO=others")
    print('  export ANALYSIS_FIXTURE_UTTERANCES_DEFAULT="text1|text2|text3"')
    print("  export ANALYSIS_UPDATE_SELECT_SENTENCE=1")
    print("  export ANALYSIS_UPDATE_INTENTION=1")

if __name__ == "__main__":
    main()