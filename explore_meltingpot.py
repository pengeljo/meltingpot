"""Explore Melting Pot scenarios and capabilities."""

from meltingpot.python import scenario
import traceback

def explore_scenarios():
    """List and explore available scenarios."""
    print("=== MELTING POT SCENARIO EXPLORER ===\n")
    
    print("1. Available Scenarios:")
    scenarios = list(scenario.SCENARIOS.keys())
    for i, name in enumerate(scenarios):
        print(f"   {i+1:2d}. {name}")
    
    print(f"\n2. Scenarios by Substrate:")
    for substrate_name, scenario_list in scenario.SCENARIOS_BY_SUBSTRATE.items():
        print(f"   {substrate_name}:")
        for s in scenario_list:
            print(f"      - {s}")
    
    print(f"\n3. Testing first scenario:")
    if scenarios:
        test_scenario = scenarios[0]
        print(f"   Testing: {test_scenario}")
        try:
            env = scenario.build(test_scenario)
            print(f"   ✓ Built successfully!")
            print(f"   ✓ Type: {type(env)}")
            
            # Get specs
            action_spec = env.action_spec()
            obs_spec = env.observation_spec()
            
            print(f"   ✓ Action spec: {len(action_spec)} agents")
            print(f"   ✓ Observation spec: {len(obs_spec)} agents")
            
            # Try a reset
            timestep = env.reset()
            print(f"   ✓ Reset successful!")
            print(f"   ✓ Timestep type: {type(timestep)}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            traceback.print_exc()
    
    return scenarios

if __name__ == "__main__":
    scenarios = explore_scenarios()
