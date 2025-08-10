#!/usr/bin/env python3
"""Fix empty analysis.yml files by adding diagnostic information.

空のanalysis.ymlファイルを診断情報で修正するスクリプト.
"""

from pathlib import Path
import sys
import os

def fix_empty_analysis_files(micro_bdi_root: str = "info/bdi_info/micro_bdi"):
    """Fix all empty analysis.yml files in the micro_bdi directory structure.
    
    Args:
        micro_bdi_root: Root directory for micro_bdi files
    """
    root_path = Path(micro_bdi_root)
    
    if not root_path.exists():
        print(f"Directory {micro_bdi_root} does not exist")
        return
    
    # Find all analysis.yml files
    analysis_files = list(root_path.rglob("analysis.yml"))
    
    if not analysis_files:
        print("No analysis.yml files found")
        return
    
    fixed_count = 0
    
    for analysis_file in analysis_files:
        try:
            # Check if file is empty (0 bytes)
            if analysis_file.stat().st_size == 0:
                print(f"Fixing empty file: {analysis_file}")
                
                # Extract agent name and game_id from path
                # Path structure: micro_bdi/game_id/agent_name/analysis.yml
                parts = analysis_file.parts
                if len(parts) >= 3:
                    agent_name = parts[-2]  # agent_name directory
                    game_id = parts[-3]     # game_id directory
                else:
                    agent_name = "unknown"
                    game_id = "unknown"
                
                # Write diagnostic information
                with open(analysis_file, "w", encoding="utf-8") as f:
                    f.write("# No analysis entries found - Fixed by repair script\n")
                    f.write("# This could be due to:\n")
                    f.write("#   - No talk history available during game\n")
                    f.write("#   - LLM initialization failed\n")
                    f.write("#   - Analysis processing errors\n")
                    f.write("#   - File was created but never written to\n")
                    f.write(f"# Agent: {agent_name}\n")
                    f.write(f"# Game ID: {game_id}\n")
                    f.write(f"# File fixed at: {analysis_file}\n")
                
                fixed_count += 1
                print(f"  -> Fixed with diagnostic information")
            else:
                print(f"Skipping non-empty file: {analysis_file} ({analysis_file.stat().st_size} bytes)")
                
        except Exception as e:
            print(f"Error processing {analysis_file}: {e}")
    
    print(f"\nFixed {fixed_count} empty analysis.yml files")

if __name__ == "__main__":
    # Change to the script directory's parent to find the info directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent.parent.parent.parent  # Go up to project root
    os.chdir(root_dir)
    
    fix_empty_analysis_files()