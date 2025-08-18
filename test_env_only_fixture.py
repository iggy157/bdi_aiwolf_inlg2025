#!/usr/bin/env python3
"""環境変数のみでFixtureモードを動作させるテスト."""

from pathlib import Path
import sys
import os
import yaml
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_env_only_fixture():
    """config.yml無しで環境変数のみでFixtureモードが動くか確認."""
    print("="*80)
    print("TEST: Environment Variables Only (No config.yml prompts)")
    print("="*80)
    
    # Fixture関連の環境変数を設定
    os.environ["ANALYSIS_FIXTURE_ENABLE"] = "1"
    os.environ["ANALYSIS_FIXTURE_MAX_PER_CALL"] = "999"
    os.environ["ANALYSIS_FIXTURE_TARGETS"] = "Skip,Over"
    os.environ["ANALYSIS_FIXTURE_OUTPUT_FILE"] = "env_only_test.yml"
    os.environ["ANALYSIS_FIXTURE_TRACE_FILE"] = "env_only_trace.yml"
    os.environ["ANALYSIS_FIXTURE_APPLY_TO"] = "others"
    os.environ["ANALYSIS_FIXTURE_UTTERANCES_DEFAULT"] = "Are there any seer claims?|Please give a tentative vote with a reason.|Share one town and one wolf read."
    
    print("Environment variables set:")
    for key in ["ANALYSIS_FIXTURE_ENABLE", "ANALYSIS_FIXTURE_MAX_PER_CALL", 
                "ANALYSIS_FIXTURE_TARGETS", "ANALYSIS_FIXTURE_OUTPUT_FILE",
                "ANALYSIS_FIXTURE_TRACE_FILE", "ANALYSIS_FIXTURE_APPLY_TO",
                "ANALYSIS_FIXTURE_UTTERANCES_DEFAULT"]:
        print(f"  {key}={os.environ.get(key)}")
    
    from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
    from aiwolf_nlp_common.packet import Talk, Info, Status
    
    # Empty config (simulate no config.yml or missing prompts)
    config = {
        "llm": {"type": "openai"},
        "openai": {"model": "gpt-4o", "temperature": 0.7}
        # Note: No "prompt" section, no "analysis" section
    }
    
    game_id = "01K2ENV_ONLY_TEST"
    agent_name = "EnvTestAgent"
    
    print(f"\\nCreating AnalysisTracker with minimal config")
    print(f"Config has prompt section: {'prompt' in config}")
    print(f"Config has analysis section: {'analysis' in config}")
    
    tracker = AnalysisTracker(config, agent_name, game_id)
    
    # Skip/Over only talks (realistic scenario)
    talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Skip"),
        Talk(idx=4, day=1, turn=3, agent="Agent4", text="Over"),
        Talk(idx=5, day=1, turn=4, agent="Agent5", text="Skip"),
        Talk(idx=6, day=1, turn=5, agent=agent_name, text="Skip"),  # Self skip
    ]
    
    info = Info(
        game_id=game_id, day=1, agent=agent_name,
        profile=None, medium_result=None, divine_result=None,
        executed_agent=None, attacked_agent=None,
        vote_list=[], attack_vote_list=[],
        status_map={
            "Agent1": Status.ALIVE, "Agent2": Status.ALIVE,
            "Agent3": Status.ALIVE, "Agent4": Status.ALIVE,
            "Agent5": Status.ALIVE, agent_name: Status.ALIVE
        },
        role_map={}
    )
    
    print(f"\\nAnalyzing {len(talks)} talks (all Skip/Over)")
    print("Expected: 5 replacements for others, 0 for self")
    
    added = tracker.analyze_talk(talks, info)
    print(f"\\nAdded entries: {added}")
    
    if added > 0:
        tracker.save_analysis()
        
        # Check generated files
        output_file = tracker.output_dir / "env_only_test.yml"
        trace_file = tracker.output_dir / "env_only_trace.yml"
        
        print(f"\\n=== Results ===")
        
        if output_file.exists():
            print(f"✓ Output file created: {output_file}")
            with open(output_file, 'r') as f:
                data = yaml.safe_load(f)
                print(f"  Entries: {len(data) if data else 0}")
                if data:
                    for k, v in list(data.items())[:3]:  # Show first 3
                        content_preview = v['content'][:50] + "..." if len(v['content']) > 50 else v['content']
                        print(f"    {k}: {v['from']} -> '{content_preview}' (type: {v['type']}, credibility: {v['credibility']})")
        else:
            print("✗ Output file not created")
        
        if trace_file.exists():
            print(f"✓ Trace file created: {trace_file}")
            with open(trace_file, 'r') as f:
                traces = yaml.safe_load(f)
                print(f"  Trace entries: {len(traces) if traces else 0}")
                if traces:
                    for k, v in list(traces.items())[:3]:  # Show first 3
                        print(f"    {k}: '{v['original']}' -> '{v['replaced'][:30]}...'")
        else:
            print("✗ Trace file not created")
        
        # Verify content diversity (not all null)
        if output_file.exists():
            with open(output_file, 'r') as f:
                data = yaml.safe_load(f)
                if data:
                    types = [v['type'] for v in data.values()]
                    credibilities = [v['credibility'] for v in data.values()]
                    print(f"\\n=== Quality Check ===")
                    print(f"Types found: {set(types)}")
                    print(f"Non-zero credibility count: {sum(1 for c in credibilities if c > 0.0)}")
                    print(f"Expected: Some non-null types, some non-zero credibilities (using default prompts)")
    else:
        print("✗ No entries added")
    
    # Test carryover (max_per_call limitation)
    print(f"\\n=== Testing Carryover (max_per_call=2) ===")
    os.environ["ANALYSIS_FIXTURE_MAX_PER_CALL"] = "2"
    
    # Create new tracker for clean test
    tracker2 = AnalysisTracker(config, "CarryoverTest", game_id + "_CARRYOVER")
    
    # More talks than max_per_call
    more_talks = [
        Talk(idx=1, day=1, turn=0, agent="Agent1", text="Skip"),
        Talk(idx=2, day=1, turn=1, agent="Agent2", text="Over"),
        Talk(idx=3, day=1, turn=2, agent="Agent3", text="Skip"),
        Talk(idx=4, day=1, turn=3, agent="Agent4", text="Over"),
    ]
    
    # First call - should process only 2
    print("First analyze_talk call (max_per_call=2):")
    added1 = tracker2.analyze_talk(more_talks, info)
    print(f"Added: {added1}")
    
    # Second call - should process remaining 2
    print("Second analyze_talk call (same talks):")
    added2 = tracker2.analyze_talk(more_talks, info)
    print(f"Added: {added2}")
    
    total_added = added1 + added2
    print(f"Total added across calls: {total_added}")
    print(f"Expected: 4 (2 + 2, carryover working)")
    
    if total_added == 4:
        print("✓ Carryover mechanism working correctly")
    else:
        print("✗ Carryover mechanism not working properly")
    
    # Cleanup environment variables
    for key in ["ANALYSIS_FIXTURE_ENABLE", "ANALYSIS_FIXTURE_MAX_PER_CALL", 
                "ANALYSIS_FIXTURE_TARGETS", "ANALYSIS_FIXTURE_OUTPUT_FILE",
                "ANALYSIS_FIXTURE_TRACE_FILE", "ANALYSIS_FIXTURE_APPLY_TO",
                "ANALYSIS_FIXTURE_UTTERANCES_DEFAULT"]:
        os.environ.pop(key, None)
    
    print(f"\\n=== Test Complete ===")

if __name__ == "__main__":
    test_env_only_fixture()