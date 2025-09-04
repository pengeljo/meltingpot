"""
Training Infrastructure for Generational Learning

Manages the training loop that integrates evolutionary algorithms
with the MeltingPot environment for collective agency experiments.
"""

import numpy as np
import time
from typing import List, Dict, Any, Tuple, Callable
from .evolution import EvolutionaryTrainer, Individual
from .networks import NeuralAgentBrain


class GenerationalTrainer:
    """
    Manages generational training of collective agents using evolutionary algorithms.
    
    Handles:
    - Population evaluation in MeltingPot environments
    - Fitness calculation based on individual and collective performance
    - Evolution across multiple generations
    - Progress tracking and statistics
    """
    
    def __init__(self, environment_builder: Callable, population_size: int = 10,
                 num_episodes_per_eval: int = 5, max_steps_per_episode: int = 100,
                 generations: int = 50, random_seed: int = None):
        self.environment_builder = environment_builder
        self.population_size = population_size
        self.num_episodes_per_eval = num_episodes_per_eval
        self.max_steps_per_episode = max_steps_per_episode
        self.generations = generations
        
        # Initialize evolutionary trainer
        self.evolution_trainer = EvolutionaryTrainer(
            population_size=population_size,
            random_seed=random_seed
        )
        
        self.training_history = []
        self.best_agents_history = []
        
    def evaluate_population_fitness(self, population: List[Individual]) -> List[float]:
        """
        Evaluate fitness of entire population by running episodes in MeltingPot.
        
        Returns fitness scores for each individual in the population.
        """
        fitness_scores = []
        
        for individual in population:
            total_fitness = 0.0
            total_individual_reward = 0.0
            total_collective_reward = 0.0
            
            # Run multiple episodes for more stable evaluation
            for episode in range(self.num_episodes_per_eval):
                episode_fitness, individual_reward, collective_reward = self._evaluate_individual_episode(
                    individual, episode_seed=episode
                )
                total_fitness += episode_fitness
                total_individual_reward += individual_reward
                total_collective_reward += collective_reward
                
            # Average across episodes
            avg_fitness = total_fitness / self.num_episodes_per_eval
            avg_individual = total_individual_reward / self.num_episodes_per_eval
            avg_collective = total_collective_reward / self.num_episodes_per_eval
            
            # Update individual's scores
            individual.individual_score = avg_individual
            individual.collective_score = avg_collective
            
            fitness_scores.append(avg_fitness)
            
        return fitness_scores
        
    def _evaluate_individual_episode(self, individual: Individual, 
                                   episode_seed: int = None) -> Tuple[float, float, float]:
        """
        Evaluate a single individual in one episode.
        
        Returns:
            (episode_fitness, individual_reward_sum, collective_reward_estimate)
        """
        # Build environment
        env = self.environment_builder()
        
        if episode_seed is not None:
            np.random.seed(episode_seed)
            
        # Initialize episode
        timestep = env.reset()
        individual_reward_sum = 0.0
        collective_actions_count = 0
        total_actions_count = 0
        cooperation_events = 0
        
        # Get number of agents from action spec
        num_agents = len(env.action_spec())
        
        # Create agent state (simplified representation)
        agent_state = self._create_agent_state(individual.agent_id, timestep)
        
        for step in range(self.max_steps_per_episode):
            if timestep.last():
                break
                
            # Get observations - MeltingPot returns tuple of observations
            observations = timestep.observation
            
            # Convert tuple observations to dict-like structure for easier access
            if isinstance(observations, tuple):
                observations_dict = {i: obs for i, obs in enumerate(observations)}
            else:
                observations_dict = {0: observations}
            
            # Create collective action proposals (simplified)
            collective_proposals = self._generate_collective_proposals(observations_dict, num_agents)
            
            # Generate actions for all agents - MeltingPot expects a tuple/sequence
            actions_list = []
            
            for agent_id in range(num_agents):
                try:
                    if agent_id == individual.agent_id and agent_id < len(observations_dict):
                        # Use our trained individual
                        agent_obs = self._flatten_observation(observations_dict[agent_id])
                        
                        # Assess collective vs individual benefit
                        collective_benefit = individual.brain.assess_collective_benefit(
                            agent_obs, collective_proposals.get(agent_id, 0)
                        )
                        individual_benefit = individual.brain.assess_individual_benefit(
                            agent_obs, agent_state
                        )
                        
                        # Decide between collective and individual action
                        if collective_benefit > individual_benefit * 0.7:  # threshold
                            action = collective_proposals.get(agent_id, 0)
                            collective_actions_count += 1
                            cooperation_events += 1
                        else:
                            action = individual.brain.sample_action(agent_obs, agent_state)
                            
                    else:
                        # Simple random policy for other agents
                        action = np.random.randint(0, 7)
                        
                    actions_list.append(action)
                    total_actions_count += 1
                    
                except Exception as e:
                    # Fallback for any issues with observation processing
                    actions_list.append(np.random.randint(0, 7))
                
            # Step environment with tuple of actions (MeltingPot format)
            timestep = env.step(tuple(actions_list))
            
            # Accumulate individual reward - MeltingPot returns tuple of rewards
            try:
                if isinstance(timestep.reward, tuple) and individual.agent_id < len(timestep.reward):
                    reward = timestep.reward[individual.agent_id]
                    if reward is not None:
                        individual_reward_sum += float(reward)
                elif hasattr(timestep.reward, '__getitem__') and individual.agent_id < len(timestep.reward):
                    reward = timestep.reward[individual.agent_id]
                    if reward is not None:
                        individual_reward_sum += float(reward)
                elif timestep.reward is not None:
                    # Single reward value
                    individual_reward_sum += float(timestep.reward)
            except (IndexError, TypeError, ValueError):
                # Fallback - assume no reward this step
                pass
                
        # Calculate collective reward estimate (based on cooperation)
        cooperation_rate = cooperation_events / max(total_actions_count, 1)
        collective_reward_estimate = cooperation_rate * 10.0  # Cooperation bonus
        
        # Calculate final fitness
        episode_fitness = individual_reward_sum + 0.3 * collective_reward_estimate
        
        return episode_fitness, individual_reward_sum, collective_reward_estimate
        
    def _create_agent_state(self, agent_id: int, timestep) -> np.ndarray:
        """Create simplified agent state representation."""
        # For now, use a simple state based on agent_id and some random features
        # In practice, this could include memory, preferences, etc.
        state = np.zeros(16)
        state[0] = agent_id / 10.0  # Normalized agent ID
        state[1] = np.random.random()  # Random internal state
        state[2:] = np.random.normal(0, 0.1, 14)  # Random features
        return state
        
    def _flatten_observation(self, observation) -> np.ndarray:
        """Flatten MeltingPot observation into fixed-size vector."""
        # MeltingPot observations are immutabledict with RGB arrays and other features
        
        try:
            target_size = 64
            features = np.zeros(target_size, dtype=np.float32)
            
            if hasattr(observation, 'get') or isinstance(observation, dict):
                # Extract RGB observation and other features
                if 'RGB' in observation:
                    rgb = observation['RGB']
                    # Heavily downsample RGB to get key visual features
                    downsampled = rgb[::8, ::8, :].flatten()  # Downsample by factor of 8
                    # Use first portion for RGB features
                    rgb_features = min(50, len(downsampled))
                    features[:rgb_features] = (downsampled[:rgb_features] / 255.0)
                
                # Add other scalar features
                feature_idx = 50
                if 'READY_TO_SHOOT' in observation and feature_idx < target_size:
                    features[feature_idx] = float(observation['READY_TO_SHOOT'])
                    feature_idx += 1
                    
                if 'COLLECTIVE_REWARD' in observation and feature_idx < target_size:
                    features[feature_idx] = float(observation['COLLECTIVE_REWARD']) / 10.0  # Normalize
                    feature_idx += 1
                    
                # Fill remaining with simple derived features
                if 'RGB' in observation and feature_idx < target_size:
                    rgb = observation['RGB']
                    # Simple color statistics
                    features[feature_idx] = np.mean(rgb[:,:,0]) / 255.0  # Red mean
                    features[feature_idx+1] = np.mean(rgb[:,:,1]) / 255.0  # Green mean
                    features[feature_idx+2] = np.mean(rgb[:,:,2]) / 255.0  # Blue mean
                    if feature_idx + 3 < target_size:
                        features[feature_idx+3] = np.std(rgb) / 255.0  # Color variance
                        
            return features
                    
        except Exception as e:
            # Fallback to random observation if parsing fails
            return np.random.random(64).astype(np.float32)
        
    def _generate_collective_proposals(self, observations: Dict, num_agents: int) -> Dict[int, int]:
        """Generate collective action proposals based on observations."""
        # Simplified collective reasoning
        # In practice, this would implement more sophisticated collective deliberation
        
        proposals = {}
        
        # Simple strategy: all agents coordinate on similar actions
        base_action = np.random.randint(0, 7)
        
        for agent_id in range(num_agents):
            # Add some variation to collective proposals
            variation = np.random.randint(-1, 2)  # -1, 0, or 1
            proposals[agent_id] = max(0, min(6, base_action + variation))
            
        return proposals
        
    def train(self, observation_size: int = 64, agent_state_size: int = 16, 
              action_size: int = 7) -> Dict[str, Any]:
        """
        Run complete generational training process.
        
        Returns comprehensive training results and statistics.
        """
        print(f"Starting generational training for {self.generations} generations...")
        print(f"Population size: {self.population_size}")
        print(f"Episodes per evaluation: {self.num_episodes_per_eval}")
        
        start_time = time.time()
        
        # Initialize population
        print("Initializing population...")
        population = self.evolution_trainer.initialize_population(
            observation_size, agent_state_size, action_size
        )
        
        # Training loop
        for generation in range(self.generations):
            gen_start_time = time.time()
            
            print(f"\n--- Generation {generation + 1}/{self.generations} ---")
            
            # Evaluate population fitness
            print(f"Evaluating {len(population)} individuals...")
            self.evolution_trainer.evaluate_population(self.evaluate_population_fitness)
            
            # Get generation statistics
            stats = self.evolution_trainer.get_training_stats()
            gen_time = time.time() - gen_start_time
            
            print(f"Best fitness: {stats.get('current_best_fitness', 0):.3f}")
            print(f"Avg fitness: {stats.get('current_avg_fitness', 0):.3f}")
            print(f"Diversity: {stats.get('population_diversity', {}).get('parameter_diversity', 0):.3f}")
            print(f"Generation time: {gen_time:.2f}s")
            
            # Store training data
            generation_data = {
                'generation': generation,
                'stats': stats,
                'best_individual': self.evolution_trainer.get_best_individual().copy(),
                'generation_time': gen_time
            }
            self.training_history.append(generation_data)
            
            # Check for early stopping
            convergence_status = stats.get('convergence_status', 'improving')
            if convergence_status == 'converged' and generation > 20:
                print(f"\nEarly stopping: Population converged at generation {generation}")
                break
                
            # Evolve to next generation (except for last generation)
            if generation < self.generations - 1:
                population = self.evolution_trainer.evolve_generation()
                
        total_time = time.time() - start_time
        
        # Final results
        final_stats = self._compile_final_results(total_time)
        
        print(f"\n=== Training Complete ===")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best fitness achieved: {final_stats['best_fitness_ever']:.3f}")
        print(f"Generations completed: {len(self.training_history)}")
        
        return final_stats
        
    def _compile_final_results(self, total_time: float) -> Dict[str, Any]:
        """Compile comprehensive final training results."""
        if not self.training_history:
            return {}
            
        fitness_progression = [gen['stats'].get('current_best_fitness', 0) 
                             for gen in self.training_history]
        diversity_progression = [gen['stats'].get('population_diversity', {}).get('parameter_diversity', 0)
                               for gen in self.training_history]
        
        final_results = {
            'total_training_time': total_time,
            'generations_completed': len(self.training_history),
            'best_fitness_ever': max(fitness_progression) if fitness_progression else 0,
            'final_fitness': fitness_progression[-1] if fitness_progression else 0,
            'fitness_improvement': fitness_progression[-1] - fitness_progression[0] if len(fitness_progression) > 1 else 0,
            'fitness_progression': fitness_progression,
            'diversity_progression': diversity_progression,
            'final_population': self.evolution_trainer.population,
            'best_individual': self.evolution_trainer.get_best_individual(),
            'convergence_analysis': self._analyze_convergence(),
            'collective_behavior_analysis': self._analyze_collective_behaviors()
        }
        
        return final_results
        
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns in the training."""
        if len(self.training_history) < 5:
            return {'status': 'insufficient_data'}
            
        fitness_values = [gen['stats'].get('current_best_fitness', 0) for gen in self.training_history]
        
        # Calculate convergence metrics
        final_improvement = fitness_values[-1] - fitness_values[0]
        convergence_generation = len(fitness_values)
        
        # Find the generation where improvement started slowing
        for i in range(10, len(fitness_values)):
            recent_improvement = fitness_values[i] - fitness_values[i-10]
            if recent_improvement < 0.01:
                convergence_generation = i
                break
                
        return {
            'status': 'analyzed',
            'total_improvement': final_improvement,
            'convergence_generation': convergence_generation,
            'final_fitness_std': np.std(fitness_values[-5:]) if len(fitness_values) >= 5 else 0,
            'improvement_rate': final_improvement / len(fitness_values)
        }
        
    def _analyze_collective_behaviors(self) -> Dict[str, Any]:
        """Analyze emergent collective behaviors from training."""
        if not self.training_history:
            return {'status': 'no_data'}
            
        # Analysis of best individuals across generations
        best_individuals = [gen['best_individual'] for gen in self.training_history]
        
        collective_scores = [ind.collective_score for ind in best_individuals]
        individual_scores = [ind.individual_score for ind in best_individuals]
        
        return {
            'collective_score_trend': collective_scores,
            'individual_score_trend': individual_scores,
            'final_collective_ratio': collective_scores[-1] / (collective_scores[-1] + individual_scores[-1] + 1e-10),
            'collective_improvement': collective_scores[-1] - collective_scores[0] if len(collective_scores) > 1 else 0,
            'strategy_evolution': 'collective' if collective_scores[-1] > individual_scores[-1] else 'individual'
        }