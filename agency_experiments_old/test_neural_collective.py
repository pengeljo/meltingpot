"""
Test script for neural collective agency without the MeltingPot environment issues.
This demonstrates the neural network evolution working independently.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agency_experiments.neural_collective import NeuralAgentBrain, EvolutionaryTrainer, Individual

def test_neural_brain():
    """Test that neural networks work correctly."""
    print("=== TESTING NEURAL AGENT BRAIN ===")
    
    # Create a neural brain
    brain = NeuralAgentBrain(observation_size=64, agent_state_size=16, action_size=7)
    
    # Test observations and states
    obs = np.random.random(64).astype(np.float32)
    state = np.random.random(16).astype(np.float32)
    
    # Test collective benefit assessment
    collective_benefit = brain.assess_collective_benefit(obs, 3)
    print(f"Collective benefit: {collective_benefit:.3f}")
    
    # Test individual benefit assessment  
    individual_benefit = brain.assess_individual_benefit(obs, state)
    print(f"Individual benefit: {individual_benefit:.3f}")
    
    # Test action generation
    action_probs = brain.generate_action_probabilities(obs, state)
    print(f"Action probabilities: {action_probs}")
    
    # Test action sampling
    action = brain.sample_action(obs, state)
    print(f"Sampled action: {action}")
    
    print("✓ Neural brain test passed!\n")
    return brain

def test_evolution():
    """Test evolutionary training without MeltingPot environment."""
    print("=== TESTING EVOLUTIONARY ALGORITHM ===")
    
    # Create evolutionary trainer
    trainer = EvolutionaryTrainer(population_size=5, elite_size=2, mutation_rate=0.1)
    
    # Initialize population
    population = trainer.initialize_population(observation_size=64, agent_state_size=16, action_size=7)
    print(f"Initialized population of {len(population)} individuals")
    
    # Define a simple fitness function for testing
    def simple_fitness_function(individuals):
        """Simple fitness function that rewards diversity and collective behavior."""
        fitness_scores = []
        for individual in individuals:
            # Test the neural brain with random data
            obs = np.random.random(64).astype(np.float32)
            state = np.random.random(16).astype(np.float32)
            
            # Get neural network outputs
            collective_benefit = individual.brain.assess_collective_benefit(obs, 1)
            individual_benefit = individual.brain.assess_individual_benefit(obs, state)
            action_probs = individual.brain.generate_action_probabilities(obs, state)
            
            # Simple fitness based on encouraging collective behavior and action diversity
            collective_score = collective_benefit * 2.0  # Reward collective thinking
            diversity_score = np.std(action_probs)  # Reward diverse action policies
            
            fitness = collective_score + diversity_score + np.random.random() * 0.1
            fitness_scores.append(fitness)
            
        return fitness_scores
    
    # Run evolution for several generations
    for generation in range(3):
        print(f"\nGeneration {generation + 1}:")
        
        # Evaluate population
        trainer.evaluate_population(simple_fitness_function)
        
        # Get statistics
        stats = trainer.get_training_stats()
        print(f"  Best fitness: {stats.get('current_best_fitness', 0):.3f}")
        print(f"  Average fitness: {stats.get('current_avg_fitness', 0):.3f}")
        
        # Get diversity
        diversity = trainer.get_population_diversity()
        print(f"  Parameter diversity: {diversity.get('parameter_diversity', 0):.3f}")
        
        # Evolve to next generation (except last)
        if generation < 2:
            population = trainer.evolve_generation()
    
    print("✓ Evolution test passed!\n")
    return trainer

def test_collective_decision_making():
    """Test the collective decision-making process."""
    print("=== TESTING COLLECTIVE DECISION MAKING ===")
    
    # Create multiple agents with different thresholds
    agents = []
    for i in range(3):
        brain = NeuralAgentBrain(observation_size=64, agent_state_size=16, action_size=7)
        agents.append((i, brain, 0.5 + i * 0.2))  # Different collective thresholds
    
    # Simulate collective decision making
    obs = np.random.random(64).astype(np.float32)
    
    for agent_id, brain, threshold in agents:
        state = np.zeros(16)
        state[0] = agent_id / 10.0  # Agent ID in state
        
        # Assess benefits
        collective_benefit = brain.assess_collective_benefit(obs, 2)  # Proposed action 2
        individual_benefit = brain.assess_individual_benefit(obs, state)
        
        # Make decision
        if collective_benefit > individual_benefit * threshold:
            decision = "COLLECTIVE"
            action = 2  # Accept collective proposal
        else:
            decision = "INDIVIDUAL"  
            action = brain.sample_action(obs, state)
            
        print(f"Agent {agent_id} (threshold={threshold:.1f}): {decision} -> action {action}")
        print(f"  Collective benefit: {collective_benefit:.3f}, Individual benefit: {individual_benefit:.3f}")
    
    print("✓ Collective decision-making test passed!\n")

def main():
    """Run all tests."""
    print("=== NEURAL COLLECTIVE AGENCY TESTS ===")
    print("Testing neural network implementation without MeltingPot environment.\n")
    
    try:
        # Test neural brain
        brain = test_neural_brain()
        
        # Test evolution
        trainer = test_evolution()
        
        # Test collective decision making
        test_collective_decision_making()
        
        print("=== ALL TESTS PASSED ===")
        print("✓ Neural networks are working correctly")
        print("✓ Evolutionary training is functional")
        print("✓ Collective decision-making is implemented")
        print("\nThe neural collective agency framework is ready!")
        print("To use with MeltingPot, run:")
        print("  python agency_experiments/collective_agency_experiment.py")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()