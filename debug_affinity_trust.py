#!/usr/bin/env python3
"""Debug script for affinity trust updater with agent_obj."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
import yaml
from utils.bdi.micro_bdi.affinity_trust_updater import update_affinity_trust_for_agent

# Mock agent object with send_message_to_llm method
class MockAgent:
    def __init__(self, config):
        self.config = config
        
    def send_message_to_llm(self, prompt_key, extra_vars=None, log_tag=None):
        """Mock LLM call that returns a simple JSON response."""
        print(f"MockAgent.send_message_to_llm called with:")
        print(f"  prompt_key: {prompt_key}")
        print(f"  extra_vars: {extra_vars}")
        print(f"  log_tag: {log_tag}")
        
        # Return a simple test response
        if prompt_key == "affinity_trust_eval":
            return '''
            {
              "liking": {
                "friendliness": 0.8,
                "emotional_resonance": 0.7,
                "attractive_expression": 0.6
              },
              "trust": {
                "social_proof": 0.9,
                "honesty": 0.8,
                "consistency": 0.7
              }
            }
            '''
        return None

def main():
    # Load config
    config_path = Path('config/config.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Create mock agent
    mock_agent = MockAgent(config)
    
    print('Testing affinity_trust_updater with mock agent_obj...')
    try:
        update_affinity_trust_for_agent(
            config=config,
            game_id='01K303S0Q4R1VX3VF4Q65BG1FX',
            agent='ユミ',
            agent_logger=None,
            talk_dir_name='talk_history',
            agent_obj=mock_agent
        )
        print('Update completed successfully')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()