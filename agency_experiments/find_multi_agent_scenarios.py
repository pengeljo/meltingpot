"""
Quick script to find scenarios with multiple agents for fusion experiments.
"""

from meltingpot.python import scenario

def find_multi_agent_scenarios(min_agents=3):
    """Find scenarios with at least min_agents."""
    multi_agent_scenarios = []
    
    print(f"Searching for scenarios with at least {min_agents} agents...")
    
    # Test a subset of scenarios (testing all 312 would take too long)
    test_scenarios = [
        "commons_harvest__open_0",
        "clean_up_0", 
        "collaborative_cooking__ring_0",
        "collaborative_cooking__crowded_0",
        "territory__rooms_0",
        "predator_prey__open_0",
        "paintball__king_of_the_hill_0",
        "fruit_market__concentric_rivers_0",
        "boat_race__eight_races_0",
        "coop_mining_0",
        "daycare_0"
    ]
    
    for scenario_name in test_scenarios:
        try:
            env = scenario.build(scenario_name)
            num_agents = len(env.action_spec())
            print(f"  {scenario_name}: {num_agents} agents")
            
            if num_agents >= min_agents:
                multi_agent_scenarios.append((scenario_name, num_agents))
                
        except Exception as e:
            print(f"  {scenario_name}: ERROR - {e}")
    
    print(f"\nFound {len(multi_agent_scenarios)} suitable scenarios:")
    for scenario_name, num_agents in multi_agent_scenarios:
        print(f"  ✓ {scenario_name}: {num_agents} agents")
    
    return multi_agent_scenarios

if __name__ == "__main__":
    scenarios = find_multi_agent_scenarios(min_agents=3)
    
    if scenarios:
        best_scenario, num_agents = scenarios[0]
        print(f"\nRecommended scenario for fusion experiment: {best_scenario} ({num_agents} agents)")
    else:
        print("\nNo suitable scenarios found!")
