#!/usr/bin/env python3
"""AnalysisTrackerのデバッグ用スクリプト（実際のゲームディレクトリで確認）."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_analysis_files():
    """既存のanalysis.ymlファイルを確認."""
    print("=== Existing analysis.yml files ===")
    
    micro_bdi_dir = Path("/home/bi23056/lab/inlg2025/bdi_aiwolf_inlg2025/info/bdi_info/micro_bdi")
    
    if not micro_bdi_dir.exists():
        print("micro_bdi directory does not exist")
        return
    
    for analysis_file in micro_bdi_dir.glob("**/analysis.yml"):
        try:
            size = analysis_file.stat().st_size
            print(f"\nFile: {analysis_file}")
            print(f"Size: {size} bytes")
            
            # ファイル内容の先頭部分を表示
            if size > 0:
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    content = f.read(200)  # 最初の200文字
                    print(f"Content preview: {content!r}")
            else:
                print("Content: EMPTY FILE")
                
        except Exception as e:
            print(f"Error reading {analysis_file}: {e}")

def create_test_tracker():
    """テスト用Trackerを作成して最新コードが使われているか確認."""
    print("\n=== Testing Latest Code ===")
    
    try:
        from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
        from aiwolf_nlp_common.packet import Talk, Info, Status
        
        # 設定
        config = {
            "llm": {"type": "openai"},
            "openai": {"model": "gpt-4o", "temperature": 0.7},
            "prompt": {
                "analyze_message_type": "Type: {{content}}",
                "analyze_target_agents": "Target: {{content}}",
                "analyze_credibility": "Score"
            }
        }
        
        # Trackerの初期化
        game_id = "01K2VTEST123456789ABCDEFGH"
        agent_name = "DebugAgent"
        
        print(f"Creating AnalysisTracker with game_id={game_id}, agent={agent_name}")
        tracker = AnalysisTracker(config, agent_name, game_id)
        
        # Skip/Overテスト用のTalkデータ
        talks = [
            Talk(idx=1, day=1, turn=0, agent="Other1", text="Hello"),
            Talk(idx=2, day=1, turn=1, agent="Other2", text="Skip"),
            Talk(idx=3, day=1, turn=2, agent="Other3", text="Over"),
            Talk(idx=4, day=1, turn=3, agent=agent_name, text="I think so"),  # 自分
            Talk(idx=5, day=1, turn=4, agent="Other4", text="Real talk"),
        ]
        
        # Info作成
        info = Info(
            game_id=game_id, day=1, agent=agent_name, profile=None,
            medium_result=None, divine_result=None, executed_agent=None,
            attacked_agent=None, vote_list=[], attack_vote_list=[],
            status_map={"Other1": Status.ALIVE, "Other2": Status.ALIVE, 
                       "Other3": Status.ALIVE, "Other4": Status.ALIVE,
                       agent_name: Status.ALIVE},
            role_map={}
        )
        
        print("Running analyze_talk...")
        added = tracker.analyze_talk(talks, info)
        print(f"Added entries: {added}")
        
        if added > 0:
            tracker.save_analysis()
            print("Analysis saved")
            
            # 結果確認
            if tracker.analysis_file.exists():
                size = tracker.analysis_file.stat().st_size
                print(f"Resulting file size: {size} bytes")
                
                with open(tracker.analysis_file, 'r') as f:
                    content = f.read()
                    print(f"Content:\n{content}")
            else:
                print("ERROR: File was not created!")
        else:
            print("No entries to save")
            
        # コードバージョン確認
        if hasattr(tracker, 'seen_talk_keys'):
            print("✓ Latest code: seen_talk_keys feature available")
        else:
            print("✗ Old code: seen_talk_keys feature missing")
            
        if hasattr(tracker, '_is_meaningful_other_utterance'):
            print("✓ Latest code: _is_meaningful_other_utterance available")
        else:
            print("✗ Old code: _is_meaningful_other_utterance missing")
            
    except Exception as e:
        print(f"Error creating test tracker: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_analysis_files()
    create_test_tracker()