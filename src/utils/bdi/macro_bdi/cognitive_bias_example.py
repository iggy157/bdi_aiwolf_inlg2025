"""Example usage of cognitive bias calculation.

認知バイアス算出機能の使用例.
"""

from pathlib import Path
from cognitive_bias import CognitiveBias


def example_usage():
    """Example of how to use cognitive bias functionality."""
    
    # Sample MBTI parameters
    mbti_params = {
        "extroversion": 0.7,
        "introversion": 0.3,
        "sensing": 0.4, 
        "intuition": 0.6,
        "thinking": 0.8,
        "feeling": 0.2,
        "judging": 0.6,
        "perceiving": 0.4
    }
    
    # Sample Enneagram parameters
    enneagram_params = {
        "reformer": 0.7,
        "helper": 0.3,
        "achiever": 0.8,
        "individualist": 0.2,
        "investigator": 0.6,
        "loyalist": 0.4,
        "enthusiast": 0.7,
        "challenger": 0.5,
        "peacemaker": 0.3
    }
    
    # Initialize cognitive bias calculator
    bias_calculator = CognitiveBias()
    
    # Calculate statement bias
    print("=== Statement Bias ===")
    statement_bias = bias_calculator.calculate_statement_bias(mbti_params, enneagram_params)
    for bias_type, score in statement_bias.items():
        print(f"{bias_type}: {score:.3f}")
    
    # Calculate trust tendency
    print("\n=== Trust Tendency ===")
    trust_tendency = bias_calculator.calculate_trust_tendency(mbti_params, enneagram_params)
    for tendency_type, score in trust_tendency.items():
        print(f"{tendency_type}: {score:.3f}")
    
    # Calculate liking tendency  
    print("\n=== Liking Tendency ===")
    liking_tendency = bias_calculator.calculate_liking_tendency(mbti_params, enneagram_params)
    for tendency_type, score in liking_tendency.items():
        print(f"{tendency_type}: {score:.3f}")
    
    # Example of updating agent trust/liking
    print("\n=== Agent Interaction Example ===")
    bias_calculator.update_agent_trust("Agent1", 0.8)
    bias_calculator.update_agent_liking("Agent1", 0.6)
    bias_calculator.update_agent_trust("Agent2", 0.3)
    bias_calculator.update_agent_liking("Agent2", 0.7)
    
    print(f"Trust in Agent1: {bias_calculator.get_agent_trust('Agent1'):.3f}")
    print(f"Liking for Agent1: {bias_calculator.get_agent_liking('Agent1'):.3f}")
    print(f"Trust in Agent2: {bias_calculator.get_agent_trust('Agent2'):.3f}")
    print(f"Liking for Agent2: {bias_calculator.get_agent_liking('Agent2'):.3f}")
    
    # Add statement bias record
    bias_calculator.add_statement_bias_record(
        speaker="Agent1",
        statement="I think we should vote for Agent3 because they seem suspicious.",
        bias_scores=statement_bias
    )
    
    # Calculate core biases (without agent scores and metadata)
    print("\n=== Core Biases Summary ===")
    core_biases = bias_calculator.get_core_biases(mbti_params, enneagram_params)
    print(f"Statement bias keys: {list(core_biases['statement_bias'].keys())}")
    print(f"Trust tendency keys: {list(core_biases['trust_tendency'].keys())}")
    print(f"Liking tendency keys: {list(core_biases['liking_tendency'].keys())}")
    
    # Save to file (example) - core biases only
    output_path = Path("example_cognitive_bias.yml")
    bias_calculator.save_cognitive_bias(core_biases, output_path)
    print(f"\nSaved core cognitive bias data to: {output_path}")
    
    # Also show all biases example (with agent data)
    print("\n=== All Biases (with agent data) ===")
    all_biases = bias_calculator.calculate_all_biases(mbti_params, enneagram_params)
    print(f"Agent trust scores: {all_biases['agent_trust_scores']}")
    print(f"Agent liking scores: {all_biases['agent_liking_scores']}")
    print(f"Statement history count: {all_biases['statement_history_count']}")
    
    # Load back from file
    loaded_bias = bias_calculator.load_cognitive_bias(output_path)
    if loaded_bias:
        print("Successfully loaded cognitive bias data from file")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    example_usage()