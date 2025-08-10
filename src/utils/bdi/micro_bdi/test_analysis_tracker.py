#!/usr/bin/env python3
"""Test script for AnalysisTracker functionality.

AnalysisTrackerの機能をテストするスクリプト.
"""

import sys
from pathlib import Path
import tempfile
import shutil
from typing import Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.bdi.micro_bdi.analysis_tracker import AnalysisTracker
from aiwolf_nlp_common.packet import Info, Talk


def create_mock_info() -> Info:
    """Create a mock Info object for testing."""
    # Create a simple mock info object with required attributes
    class MockInfo:
        def __init__(self):
            self.game_id = "TEST_GAME_ID_123"
            self.agent = "テストエージェント"
            self.status_map = {
                "テストエージェント": "ALIVE",
                "エージェント2": "ALIVE",
                "エージェント3": "ALIVE"
            }
    
    return MockInfo()


def create_mock_talk(agent: str, text: str) -> Talk:
    """Create a mock Talk object for testing."""
    class MockTalk:
        def __init__(self, agent: str, text: str):
            self.agent = agent
            self.text = text
    
    return MockTalk(agent, text)


def test_analysis_tracker():
    """Test AnalysisTracker functionality."""
    print("Testing AnalysisTracker functionality...")
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test config
        config = {
            "llm": {"type": "openai"},
            "openai": {"model": "gpt-4o", "temperature": 0.7},
            "prompt": {
                "analyze_message_type": "Analyze the message type: {{ content }}",
                "analyze_target_agents": "Find target agents in: {{ content }}",
                "analyze_credibility": "Rate credibility: {{ content }}"
            }
        }
        
        # Change working directory to temp for test
        original_cwd = Path.cwd()
        try:
            # Create info directory structure in temp
            info_dir = temp_path / "info" / "bdi_info" / "micro_bdi"
            info_dir.mkdir(parents=True, exist_ok=True)
            
            import os
            os.chdir(temp_path)
            
            # Initialize AnalysisTracker
            tracker = AnalysisTracker(
                config=config,
                agent_name="テストエージェント",
                game_id="TEST_GAME_ID_123"
            )
            
            print(f"✓ AnalysisTracker initialized")
            print(f"  Output file: {tracker.analysis_file}")
            
            # Test 1: Save empty analysis (should create file with diagnostic info)
            print("\n--- Test 1: Empty analysis save ---")
            tracker.save_analysis()
            
            if tracker.analysis_file.exists():
                content = tracker.analysis_file.read_text(encoding="utf-8")
                print(f"✓ File created successfully")
                print(f"  File size: {tracker.analysis_file.stat().st_size} bytes")
                print(f"  Content preview:")
                for i, line in enumerate(content.split('\n')[:5]):
                    print(f"    {i+1}: {line}")
            else:
                print("✗ File was not created")
                
            # Test 2: Add some mock talk data and save
            print("\n--- Test 2: Analysis with mock data ---")
            mock_info = create_mock_info()
            
            # Create mock talk history
            talk_history = [
                create_mock_talk("テストエージェント", "こんにちは、皆さん！"),
                create_mock_talk("エージェント2", "よろしくお願いします。"),
                create_mock_talk("テストエージェント", "Over")
            ]
            
            # Analyze talk (this will use mock data since LLM is not available)
            tracker.analyze_talk(talk_history, mock_info, request_count=1)
            
            # Force save with status information
            tracker.force_save_with_status()
            
            if tracker.analysis_file.exists():
                content = tracker.analysis_file.read_text(encoding="utf-8")
                print(f"✓ File updated successfully")
                print(f"  File size: {tracker.analysis_file.stat().st_size} bytes")
                print(f"  Content preview:")
                for i, line in enumerate(content.split('\n')[:10]):
                    print(f"    {i+1}: {line}")
            else:
                print("✗ File was not updated")
            
            print("\n=== Test completed successfully ===")
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_analysis_tracker()