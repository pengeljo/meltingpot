"""
Quick test to verify MeltingPot neural training is working.
Uses minimal parameters for fast validation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agency_experiments.collective_agency_experiment import CollectiveAgencyExperiment
from agency_experiments.neural_collective.training import GenerationalTrainer  
from meltingpot.python import scenario

def test_meltingpot_neural_quick():
    """Quick test with minimal parameters."""
    print("=== QUICK MELTINGPOT NEURAL TEST ===")
    
    # Create trainer with minimal settings for fast test
    trainer = GenerationalTrainer(
        environment_builder=lambda: scenario.build('commons_harvest__open_0'),
        population_size=3,           # Very small population
        num_episodes_per_eval=1,     # Single episode evaluation
        max_steps_per_episode=20,    # Very short episodes
        generations=2                # Just 2 generations
    )
    
    print("Starting minimal neural training test...")
    print("Population: 3 agents, 1 episode each, 20 steps per episode, 2 generations")
    
    try:
        results = trainer.train(observation_size=64, agent_state_size=16, action_size=7)
        
        print("\n🎉 SUCCESS! Neural training completed:")
        print(f"- Best fitness: {results.get('best_fitness_ever', 0):.3f}")
        print(f"- Training time: {results.get('total_training_time', 0):.1f} seconds")
        print(f"- Generations: {results.get('generations_completed', 0)}")
        
        collective_analysis = results.get('collective_behavior_analysis', {})
        if collective_analysis:
            final_ratio = collective_analysis.get('final_collective_ratio', 0)
            print(f"- Final collective ratio: {final_ratio:.3f}")
            print(f"- Strategy evolution: {collective_analysis.get('strategy_evolution', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_experiment():
    """Test the full experiment interface."""
    print("\n=== TESTING FULL EXPERIMENT INTERFACE ===")
    
    try:
        # Test neural mode with minimal settings
        experiment = CollectiveAgencyExperiment(use_neural_networks=True)
        
        # Override trainer settings for faster test
        experiment.neural_trainer = GenerationalTrainer(
            environment_builder=lambda: scenario.build('commons_harvest__open_0'),
            population_size=3,
            num_episodes_per_eval=1,
            max_steps_per_episode=20,
            generations=2
        )
        
        print("Testing neural generational training...")
        results = experiment.run_neural_generational_training()
        
        print("✅ Full experiment interface works!")
        print(f"Best fitness: {results.get('best_fitness_ever', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full experiment test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing MeltingPot neural network integration...")
    
    # Test 1: Quick trainer test
    success1 = test_meltingpot_neural_quick()
    
    # Test 2: Full experiment interface
    success2 = test_full_experiment()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ MeltingPot compatibility issue is RESOLVED!")
        print("✅ Neural networks are training successfully!")
        print("✅ Full experiment interface works!")
        print("\nYou can now run:")
        print("  python -c \"from agency_experiments.collective_agency_experiment import main; main(mode='neural')\"")
    else:
        print("\n❌ Some tests failed - check the output above.")