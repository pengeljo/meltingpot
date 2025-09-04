"""
Experiment 1: Collective Agency in Commons Harvest
Testing novel forms of collective decision-making and identity flexibility.
"""

from meltingpot.python import scenario
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time
from .neural_collective import NeuralAgentBrain, GenerationalTrainer

class CollectiveAgent:
    """
    An agent that can participate in collective decision-making.
    Implements novel agency concepts from the thesis.
    Now uses neural networks for decision-making instead of random functions.
    """

    def __init__(self, agent_id: int, collective_threshold: float = 0.7, 
                 use_neural_networks: bool = False, neural_brain: NeuralAgentBrain = None):
        self.agent_id = agent_id
        self.collective_threshold = collective_threshold
        self.individual_preferences = {}
        self.collective_memberships = []
        self.decision_history = []
        
        # Neural network components
        self.use_neural_networks = use_neural_networks
        self.neural_brain = neural_brain
        self.agent_state = self._initialize_agent_state()

    def evaluate_collective_action(self, proposed_action, collective_context):
        """
        Decide whether to act individually or as part of collective.
        This is where we implement novel agency concepts.
        """
        # Novel agency: flexible identity boundaries
        collective_benefit = self._assess_collective_benefit(proposed_action, collective_context)
        individual_benefit = self._assess_individual_benefit(proposed_action)

        # Decision threshold based on collective vs individual agency
        if collective_benefit > individual_benefit * self.collective_threshold:
            return "collective", proposed_action
        else:
            return "individual", self._generate_individual_action()

    def _assess_collective_benefit(self, action, context):
        """Assess collective benefit using neural networks or fallback to random."""
        if self.use_neural_networks and self.neural_brain is not None:
            # Use neural network to assess collective benefit
            observation = self._extract_observation_from_context(context)
            return self.neural_brain.assess_collective_benefit(observation, action)
        else:
            # Fallback to random for backward compatibility
            return np.random.random()

    def _assess_individual_benefit(self, action):
        """Assess individual benefit using neural networks or fallback to random."""
        if self.use_neural_networks and self.neural_brain is not None:
            # Use neural network to assess individual benefit
            # Use a simple observation for now - in practice this would be more sophisticated
            dummy_observation = np.random.random(64)  # Placeholder observation
            return self.neural_brain.assess_individual_benefit(dummy_observation, self.agent_state)
        else:
            # Fallback to random for backward compatibility
            return np.random.random()

    def _generate_individual_action(self):
        """Generate individual action using neural networks or fallback to random."""
        if self.use_neural_networks and self.neural_brain is not None:
            # Use neural network to generate action
            dummy_observation = np.random.random(64)  # Placeholder observation
            return self.neural_brain.sample_action(dummy_observation, self.agent_state)
        else:
            # Fallback to random for backward compatibility
            return np.random.randint(0, 7)  # Assuming 7 possible actions
            
    def _initialize_agent_state(self) -> np.ndarray:
        """Initialize agent's internal state representation."""
        state = np.zeros(16)
        state[0] = self.agent_id / 10.0  # Normalized agent ID
        state[1] = self.collective_threshold
        state[2:] = np.random.normal(0, 0.1, 14)  # Random internal features
        return state
        
    def _extract_observation_from_context(self, context) -> np.ndarray:
        """Extract and flatten observation from collective context."""
        # Simplified observation extraction
        # In practice, this would be more sophisticated based on the actual context structure
        if hasattr(context, 'shape'):
            # If context is already an array, flatten it
            flat_obs = context.flatten()
        else:
            # If context is complex, create a simplified representation
            flat_obs = np.random.random(64)  # Placeholder
            
        # Ensure fixed size
        if len(flat_obs) > 64:
            return flat_obs[:64].astype(np.float32)
        else:
            padded = np.zeros(64, dtype=np.float32)
            padded[:len(flat_obs)] = flat_obs
            return padded
            
    def update_agent_state(self, reward: float, action_taken: int, collective_action: bool):
        """Update agent's internal state based on experience."""
        # Simple state update based on recent experience
        self.agent_state[12] = reward / 10.0  # Normalized recent reward
        self.agent_state[13] = float(collective_action)  # Whether last action was collective
        self.agent_state[14] = action_taken / 7.0  # Normalized last action
        self.agent_state[15] = len(self.decision_history) / 100.0  # Experience level

class CollectiveAgencyExperiment:
    """
    Main experiment class for testing collective agency concepts.
    Now supports both random baseline and neural network evolution modes.
    """

    def __init__(self, scenario_name: str = "commons_harvest__open_0", 
                 use_neural_networks: bool = False):
        self.scenario_name = scenario_name
        self.use_neural_networks = use_neural_networks
        self.env = scenario.build(scenario_name)
        self.agents = self._initialize_collective_agents()
        self.results = {
            'individual_actions': [],
            'collective_actions': [],
            'collective_formations': [],
            'outcomes': []
        }
        
        # Neural training components
        self.neural_trainer = None
        if use_neural_networks:
            self.neural_trainer = GenerationalTrainer(
                environment_builder=lambda: scenario.build(scenario_name),
                population_size=10,
                num_episodes_per_eval=3,
                max_steps_per_episode=100,
                generations=20
            )

    def _initialize_collective_agents(self):
        """Initialize agents with collective agency capabilities."""
        num_agents = len(self.env.action_spec())
        agents = {}
        for i in range(num_agents):
            # Vary collective thresholds to test different agency types
            threshold = 0.5 + (i * 0.1)  # Different agency preferences
            
            # Create neural brain if using neural networks
            neural_brain = None
            if self.use_neural_networks:
                neural_brain = NeuralAgentBrain(
                    observation_size=64,
                    agent_state_size=16,
                    action_size=7
                )
            
            agents[i] = CollectiveAgent(
                agent_id=i, 
                collective_threshold=threshold,
                use_neural_networks=self.use_neural_networks,
                neural_brain=neural_brain
            )
        return agents

    def run_episode(self, max_steps: int = 100):
        """Run a single episode testing collective agency."""
        timestep = self.env.reset()
        episode_data = {
            'steps': [],
            'collective_formations': [],
            'agency_decisions': []
        }

        for step in range(max_steps):
            if timestep.last():
                break

            # Get actions from our collective agents
            actions = self._get_collective_actions(timestep)

            # Record agency decisions
            episode_data['agency_decisions'].append(self._analyze_agency_decisions())

            # Step the environment
            timestep = self.env.step(actions)

            # Record step data
            step_data = {
                'step': step,
                'rewards': timestep.reward,
                'observations': timestep.observation
            }
            episode_data['steps'].append(step_data)

        return episode_data

    def _get_collective_actions(self, timestep):
        """
        Core method: decide actions using collective agency concepts.
        This is where the novel agency concepts are tested.
        """
        observations = timestep.observation
        actions = {}

        # Simulate collective deliberation
        collective_proposals = self._generate_collective_proposals(observations)

        for agent_id, agent in self.agents.items():
            # Each agent decides: individual or collective action?
            agency_type, action = agent.evaluate_collective_action(
                collective_proposals.get(agent_id, 0),
                observations
            )

            actions[agent_id] = action

            # Track agency decisions for analysis
            self.results['individual_actions' if agency_type == 'individual' else 'collective_actions'].append({
                'agent_id': agent_id,
                'step': len(self.results['individual_actions']) + len(self.results['collective_actions']),
                'action': action
            })

        return actions

    def _generate_collective_proposals(self, observations):
        """Generate collective action proposals based on shared observations."""
        # Simplified collective proposal mechanism
        # In practice, this would implement more sophisticated collective reasoning
        proposals = {}
        for agent_id in self.agents.keys():
            # Proposal based on collective benefit (simplified)
            proposals[agent_id] = np.random.randint(0, 7)
        return proposals

    def _analyze_agency_decisions(self):
        """Analyze the types of agency decisions being made."""
        individual_count = len([a for a in self.results['individual_actions'] if a['step'] == len(self.results['individual_actions'])])
        collective_count = len([a for a in self.results['collective_actions'] if a['step'] == len(self.results['collective_actions'])])

        return {
            'individual_ratio': individual_count / (individual_count + collective_count + 1e-10),
            'collective_ratio': collective_count / (individual_count + collective_count + 1e-10)
        }

    def run_multiple_episodes(self, num_episodes: int = 5):
        """Run multiple episodes to gather data on collective agency patterns."""
        all_results = []

        print(f"Running {num_episodes} episodes to test collective agency concepts...")

        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            episode_data = self.run_episode()
            all_results.append(episode_data)

        return all_results

    def run_neural_generational_training(self) -> Dict[str, Any]:
        """
        Run generational training using neural networks and evolutionary algorithms.
        This is the new main method for neural collective agency experiments.
        """
        if not self.use_neural_networks:
            raise ValueError("Neural generational training requires use_neural_networks=True")
            
        print("=== NEURAL COLLECTIVE AGENCY GENERATIONAL TRAINING ===")
        print("Training agents to learn collective decision-making through evolution...")
        
        # Run the generational training
        training_results = self.neural_trainer.train(
            observation_size=64,
            agent_state_size=16, 
            action_size=7
        )
        
        return training_results

    def run_baseline_comparison(self, neural_generations: int = 20) -> Dict[str, Any]:
        """
        Run both random baseline and neural network experiments for comparison.
        """
        print("=== BASELINE vs NEURAL COLLECTIVE AGENCY COMPARISON ===")
        
        # Run random baseline first
        print("\n1. Running random baseline experiments...")
        baseline_results = self.run_multiple_episodes(num_episodes=5)
        baseline_analysis = self.analyze_results(baseline_results)
        
        # Switch to neural network mode
        print(f"\n2. Running neural network training for {neural_generations} generations...")
        original_mode = self.use_neural_networks
        self.use_neural_networks = True
        
        # Reinitialize agents with neural networks
        self.agents = self._initialize_collective_agents()
        self.neural_trainer = GenerationalTrainer(
            environment_builder=lambda: scenario.build(self.scenario_name),
            population_size=10,
            num_episodes_per_eval=3,
            generations=neural_generations
        )
        
        # Run neural training
        neural_results = self.run_neural_generational_training()
        
        # Restore original mode
        self.use_neural_networks = original_mode
        
        # Combine results
        comparison_results = {
            'baseline_results': baseline_analysis,
            'neural_results': neural_results,
            'comparison_summary': self._generate_comparison_summary(baseline_analysis, neural_results)
        }
        
        return comparison_results
        
    def _generate_comparison_summary(self, baseline: Dict[str, Any], neural: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary comparing baseline vs neural results."""
        baseline_collective_ratio = baseline.get('collective_ratio', 0)
        neural_collective_ratio = neural.get('collective_behavior_analysis', {}).get('final_collective_ratio', 0)
        
        neural_improvement = neural.get('fitness_improvement', 0)
        neural_best_fitness = neural.get('best_fitness_ever', 0)
        
        return {
            'baseline_collective_ratio': baseline_collective_ratio,
            'neural_collective_ratio': neural_collective_ratio,
            'collective_behavior_improvement': neural_collective_ratio - baseline_collective_ratio,
            'neural_learning_demonstrated': neural_improvement > 0.1,
            'neural_best_fitness': neural_best_fitness,
            'training_time': neural.get('total_training_time', 0),
            'generations_completed': neural.get('generations_completed', 0)
        }

    def analyze_results(self, all_results):
        """Analyze results to understand collective agency patterns."""
        print("\n=== COLLECTIVE AGENCY ANALYSIS ===")

        total_individual = len(self.results['individual_actions'])
        total_collective = len(self.results['collective_actions'])
        total_actions = total_individual + total_collective

        if total_actions > 0:
            print(f"Individual agency actions: {total_individual} ({total_individual/total_actions*100:.1f}%)")
            print(f"Collective agency actions: {total_collective} ({total_collective/total_actions*100:.1f}%)")

            # Analyze by agent
            print(f"\nAgency preferences by agent:")
            for agent_id, agent in self.agents.items():
                agent_individual = len([a for a in self.results['individual_actions'] if a['agent_id'] == agent_id])
                agent_collective = len([a for a in self.results['collective_actions'] if a['agent_id'] == agent_id])
                agent_total = agent_individual + agent_collective

                if agent_total > 0:
                    print(f"  Agent {agent_id} (threshold={agent.collective_threshold:.1f}): "
                          f"{agent_collective/agent_total*100:.1f}% collective")

        return {
            'individual_ratio': total_individual / total_actions if total_actions > 0 else 0,
            'collective_ratio': total_collective / total_actions if total_actions > 0 else 0
        }

def main(mode: str = "neural", neural_generations: int = 15):
    """
    Run the collective agency experiment.
    
    Args:
        mode: "random" for random baseline, "neural" for neural networks, "comparison" for both
        neural_generations: Number of generations for neural training
    """
    print("=== COLLECTIVE AGENCY EXPERIMENT ===")
    print("Testing novel forms of agency from thesis concepts...")
    print(f"Mode: {mode}")

    if mode == "random":
        # Run original random baseline experiment
        print("\nRunning random baseline experiment...")
        experiment = CollectiveAgencyExperiment(use_neural_networks=False)
        results = experiment.run_multiple_episodes(num_episodes=3)
        experiment.analyze_results(results)
        
    elif mode == "neural":
        # Run neural network evolution experiment
        print(f"\nRunning neural network evolution experiment ({neural_generations} generations)...")
        experiment = CollectiveAgencyExperiment(use_neural_networks=True)
        results = experiment.run_neural_generational_training()
        
        # Print results summary
        from .neural_collective.utils import create_experiment_summary
        summary = create_experiment_summary(results)
        print(summary)
        
    elif mode == "comparison":
        # Run comparison between random and neural approaches
        print(f"\nRunning comparison between random baseline and neural networks...")
        experiment = CollectiveAgencyExperiment(use_neural_networks=False)
        comparison_results = experiment.run_baseline_comparison(neural_generations=neural_generations)
        
        # Print comparison summary
        summary = comparison_results['comparison_summary']
        print(f"\n=== COMPARISON RESULTS ===")
        print(f"Random baseline collective ratio: {summary['baseline_collective_ratio']:.3f}")
        print(f"Neural network collective ratio: {summary['neural_collective_ratio']:.3f}")
        print(f"Improvement in collective behavior: {summary['collective_behavior_improvement']:+.3f}")
        print(f"Neural learning demonstrated: {'Yes' if summary['neural_learning_demonstrated'] else 'No'}")
        print(f"Best fitness achieved: {summary['neural_best_fitness']:.3f}")
        print(f"Training time: {summary['training_time']:.1f} seconds")
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'random', 'neural', or 'comparison'")

    print(f"\n=== CONCLUSIONS ===")
    print(f"This experiment tested the idea that machine agents can explore")
    print(f"novel forms of collective agency beyond human limitations.")
    if mode in ["neural", "comparison"]:
        print(f"Neural networks enable agents to learn and evolve collective")
        print(f"decision-making strategies across generations, potentially")
        print(f"discovering novel forms of cooperation not accessible to humans.")

if __name__ == "__main__":
    main()
