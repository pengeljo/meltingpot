"""
Training system for collective agency experiment.

Implements evolutionary training with fitness evaluation for both individual and collective agents.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Tuple
import uuid
import copy
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.evolution import EvolutionaryTrainer
from environment import MultiAgentEnvironmentWrapper
from config import CollectiveAgencyConfig


class CollectiveAgencyFitness:
    """Fitness evaluator that measures both individual and collective performance."""
    
    def __init__(self, env_wrapper: MultiAgentEnvironmentWrapper, config: CollectiveAgencyConfig, scenario_config: Dict[str, Any]):
        self.env_wrapper = env_wrapper
        self.config = config
        self.scenario_config = scenario_config
        self.num_episodes = config.num_episodes_per_evaluation
        
        # Fitness components weights
        self.individual_performance_weight = 0.4
        self.collective_performance_weight = 0.3
        self.sustainability_weight = 0.2
        self.novelty_weight = 0.1
        
    def evaluate_population(self, agents: List) -> List[float]:
        """Evaluate fitness for entire population."""
        fitness_scores = []
        
        # Group agents for multi-agent episodes
        group_size = min(self.env_wrapper.env_wrapper.num_players, len(agents))
        
        for i in range(0, len(agents), group_size):
            # Get group of agents
            agent_group = agents[i:i + group_size]
            
            # If we don't have enough agents, fill with copies
            while len(agent_group) < group_size:
                # Duplicate a random agent from the group
                duplicate = copy.deepcopy(agent_group[np.random.randint(len(agent_group))])
                duplicate.agent_id = f"duplicate_{uuid.uuid4().hex[:8]}"
                agent_group.append(duplicate)
            
            # Evaluate this group
            group_fitness = self._evaluate_agent_group(agent_group)
            
            # Assign fitness scores
            for j, agent in enumerate(agents[i:i + group_size]):
                if j < len(group_fitness):
                    fitness_scores.append(group_fitness[j])
                else:
                    fitness_scores.append(0.0)
        
        return fitness_scores
    
    def _evaluate_agent_group(self, agents: List) -> List[float]:
        """Evaluate a group of agents playing together."""
        group_performance = []
        episode_histories = []
        
        for episode in range(self.num_episodes):
            # Run episode
            episode_rewards, episode_history, _ = self.env_wrapper.run_episode(agents)
            
            # Store results
            group_performance.append(episode_rewards)
            episode_histories.append(episode_history)
        
        # Calculate fitness for each agent
        agent_fitness = []
        
        for agent in agents:
            # Individual performance component
            individual_rewards = [ep_rewards[agent.agent_id] for ep_rewards in group_performance]
            individual_score = np.mean(individual_rewards)
            
            # Collective performance component (how well the group did overall)
            collective_scores = []
            for ep_rewards in group_performance:
                total_group_reward = sum(ep_rewards.values())
                collective_scores.append(total_group_reward)
            collective_score = np.mean(collective_scores)
            
            # Sustainability component (consistent performance over time)
            sustainability_score = self._calculate_sustainability(individual_rewards)
            
            # Novelty component (behavioral diversity)
            novelty_score = self._calculate_novelty(agent, episode_histories)
            
            # Combine components
            total_fitness = (
                self.individual_performance_weight * individual_score +
                self.collective_performance_weight * (collective_score / len(agents)) +
                self.sustainability_weight * sustainability_score +
                self.novelty_weight * novelty_score
            )
            
            agent_fitness.append(max(0.0, total_fitness))  # Ensure non-negative
        
        return agent_fitness
    
    def _calculate_sustainability(self, rewards: List[float]) -> float:
        """Calculate how sustainable the agent's performance is."""
        if len(rewards) < 2:
            return 0.0
        
        # Reward stability (lower variance is better)
        mean_reward = np.mean(rewards)
        reward_variance = np.var(rewards)
        
        # Sustainability score: high mean, low variance
        if mean_reward > 0:
            sustainability = mean_reward / (1.0 + reward_variance)
        else:
            sustainability = 0.0
        
        return sustainability
    
    def _calculate_novelty(self, agent, episode_histories: List[Dict]) -> float:
        """Calculate behavioral novelty score."""
        if not hasattr(agent, 'experience_buffer') or len(agent.experience_buffer) < 10:
            return 0.0
        
        # Look at action diversity
        recent_actions = [exp['action'] for exp in agent.experience_buffer[-20:]]
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate entropy of action distribution
        total_actions = len(recent_actions)
        entropy = 0.0
        for count in action_counts.values():
            prob = count / total_actions
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(agent.capabilities.action_size)
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        return normalized_entropy
    
    def get_population_stats(self, agents: List, fitness_scores: List[float]) -> Dict[str, Any]:
        """Get statistics about the current population."""
        individual_agents = [agent for agent in agents if hasattr(agent, 'get_genotype') and 
                           agent.get_genotype().get('agent_type') == 'individual']
        collective_agents = [agent for agent in agents if hasattr(agent, 'get_genotype') and 
                           agent.get_genotype().get('agent_type') == 'collective']
        
        individual_fitness = [fitness_scores[i] for i, agent in enumerate(agents) if agent in individual_agents]
        collective_fitness = [fitness_scores[i] for i, agent in enumerate(agents) if agent in collective_agents]
        
        stats = {
            'total_agents': len(agents),
            'individual_agents': len(individual_agents),
            'collective_agents': len(collective_agents),
            'mean_fitness': np.mean(fitness_scores),
            'max_fitness': np.max(fitness_scores),
            'min_fitness': np.min(fitness_scores),
            'fitness_std': np.std(fitness_scores)
        }
        
        if individual_fitness:
            stats.update({
                'individual_mean_fitness': np.mean(individual_fitness),
                'individual_max_fitness': np.max(individual_fitness),
                'individual_min_fitness': np.min(individual_fitness)
            })
        
        if collective_fitness:
            stats.update({
                'collective_mean_fitness': np.mean(collective_fitness),
                'collective_max_fitness': np.max(collective_fitness),
                'collective_min_fitness': np.min(collective_fitness)
            })
        
        # Agent-specific metrics
        if collective_agents:
            coordination_efficiencies = []
            component_diversities = []
            collective_coherences = []
            
            for agent in collective_agents:
                if hasattr(agent, '_calculate_coordination_efficiency'):
                    coordination_efficiencies.append(agent._calculate_coordination_efficiency())
                if hasattr(agent, '_calculate_component_diversity'):
                    component_diversities.append(agent._calculate_component_diversity())
                if hasattr(agent, '_calculate_collective_coherence'):
                    collective_coherences.append(agent._calculate_collective_coherence())
            
            if coordination_efficiencies:
                stats['mean_coordination_efficiency'] = np.mean(coordination_efficiencies)
            if component_diversities:
                stats['mean_component_diversity'] = np.mean(component_diversities)
            if collective_coherences:
                stats['mean_collective_coherence'] = np.mean(collective_coherences)
        
        return stats


class CollectiveAgencyEvolution(EvolutionaryTrainer):
    """Specialized evolutionary trainer for collective agency experiments."""
    
    def __init__(self, config, fitness_evaluator: CollectiveAgencyFitness):
        super().__init__(config)
        self.fitness_evaluator = fitness_evaluator
        
        # Track evolution statistics
        self.generation_stats = []
        self.best_agents_history = []
        
    def train(self, agent_factory: Callable, generations: int = None) -> Dict[str, Any]:
        """Train agents using evolutionary algorithm."""
        if generations is None:
            generations = self.config.generations
            
        print(f"Debug: received generations={generations}, config.generations={self.config.generations}")
        print(f"Starting evolutionary training for {generations} generations")
        
        # Initialize population
        population = agent_factory()
        print(f"Initial population: {len(population)} agents")
        
        # Track best fitness over generations
        best_fitness_ever = -float('inf')
        best_agent_ever = None
        
        for generation in range(generations):
            print(f"\n=== Generation {generation + 1}/{generations} ===")
            
            # Evaluate fitness
            fitness_scores = self.fitness_evaluator.evaluate_population(population)
            
            # Get statistics
            gen_stats = self.fitness_evaluator.get_population_stats(population, fitness_scores)
            gen_stats['generation'] = generation
            self.generation_stats.append(gen_stats)
            
            # Track best agent
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            best_agent = population[best_idx]
            
            if best_fitness > best_fitness_ever:
                best_fitness_ever = best_fitness
                best_agent_ever = copy.deepcopy(best_agent)
            
            # Print progress
            print(f"Best fitness: {best_fitness:.4f} | Mean: {gen_stats['mean_fitness']:.4f}")
            if 'individual_mean_fitness' in gen_stats and 'collective_mean_fitness' in gen_stats:
                print(f"Individual agents mean: {gen_stats['individual_mean_fitness']:.4f}")
                print(f"Collective agents mean: {gen_stats['collective_mean_fitness']:.4f}")
            
            # Save best agents from this generation
            top_agents = self._select_top_agents(population, fitness_scores, 3)
            self.best_agents_history.append({
                'generation': generation,
                'agents': [copy.deepcopy(agent) for agent in top_agents],
                'fitness_scores': [fitness_scores[population.index(agent)] for agent in top_agents]
            })
            
            # Evolve population for next generation
            if generation < generations - 1:  # Don't evolve after last generation
                population = self._evolve_population(population, fitness_scores, agent_factory)
        
        # Return training results
        results = {
            'generations_completed': generations,
            'best_fitness_ever': best_fitness_ever,
            'best_agent_ever': best_agent_ever,
            'generation_stats': self.generation_stats,
            'best_agents_history': self.best_agents_history,
            'final_population': population,
            'final_fitness_scores': fitness_scores
        }
        
        print(f"\nTraining complete! Best fitness achieved: {best_fitness_ever:.4f}")
        
        return results
    
    def _select_top_agents(self, population: List, fitness_scores: List[float], num_top: int) -> List:
        """Select top performing agents."""
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        top_indices = sorted_indices[:num_top]
        return [population[i] for i in top_indices]
    
    def _evolve_population(self, population: List, fitness_scores: List[float], agent_factory: Callable) -> List:
        """Evolve population to next generation."""
        # Selection: keep elite agents
        elite_size = min(self.config.elite_size, len(population))
        elite_agents = self._select_top_agents(population, fitness_scores, elite_size)
        
        # Create new population
        new_population = []
        
        # Add elite agents
        for agent in elite_agents:
            new_agent = copy.deepcopy(agent)
            new_agent.agent_id = f"{agent.agent_id}_gen_{uuid.uuid4().hex[:6]}"
            new_population.append(new_agent)
        
        # Generate offspring through mutation and crossover
        while len(new_population) < len(population):
            if np.random.random() < self.config.crossover_rate and len(elite_agents) >= 2:
                # Crossover
                parent1, parent2 = np.random.choice(elite_agents, 2, replace=False)
                offspring = self._crossover(parent1, parent2, agent_factory)
            else:
                # Mutation
                parent = np.random.choice(elite_agents)
                offspring = self._mutate(parent)
            
            new_population.append(offspring)
        
        return new_population[:len(population)]  # Ensure same size
    
    def _crossover(self, parent1, parent2, agent_factory: Callable):
        """Create offspring through crossover of two parents."""
        # For now, use simple approach: randomly inherit from one parent or the other
        # More sophisticated crossover could mix neural network weights
        
        if np.random.random() < 0.5:
            offspring = copy.deepcopy(parent1)
            offspring_genotype = parent1.get_genotype()
        else:
            offspring = copy.deepcopy(parent2)
            offspring_genotype = parent2.get_genotype()
        
        # Give new ID
        offspring.agent_id = f"crossover_{uuid.uuid4().hex[:8]}"
        
        # Apply some mutation to the offspring
        offspring = self._mutate(offspring, mutation_strength=self.config.mutation_strength * 0.5)
        
        return offspring
    
    def _mutate(self, agent, mutation_strength: float = None):
        """Create mutated copy of an agent."""
        if mutation_strength is None:
            mutation_strength = self.config.mutation_strength
        
        mutated_agent = copy.deepcopy(agent)
        mutated_agent.agent_id = f"mutant_{uuid.uuid4().hex[:8]}"
        
        # Get current genotype
        genotype = mutated_agent.get_genotype()
        
        # Mutate neural network weights
        if 'network_weights' in genotype:
            # Individual agent
            for i, weights in enumerate(genotype['network_weights']):
                noise = np.random.normal(0, mutation_strength, weights.shape)
                genotype['network_weights'][i] = weights + noise
        
        if 'decision_network_weights' in genotype:
            # Collective agent
            for i, weights in enumerate(genotype['decision_network_weights']):
                noise = np.random.normal(0, mutation_strength, weights.shape)
                genotype['decision_network_weights'][i] = weights + noise
            
            for i, weights in enumerate(genotype['coordination_network_weights']):
                noise = np.random.normal(0, mutation_strength, weights.shape)
                genotype['coordination_network_weights'][i] = weights + noise
            
            # Mutate component states
            for i, state in enumerate(genotype['component_states']):
                noise = np.random.normal(0, mutation_strength, state.shape)
                genotype['component_states'][i] = state + noise
        
        # Set mutated genotype
        mutated_agent.set_genotype(genotype)
        
        return mutated_agent


def create_fitness_function(env_wrapper: MultiAgentEnvironmentWrapper, 
                          config: CollectiveAgencyConfig, 
                          scenario_config: Dict[str, Any]) -> Callable:
    """Create fitness function for evolutionary training."""
    
    fitness_evaluator = CollectiveAgencyFitness(env_wrapper, config, scenario_config)
    # Always use the main config evolution settings (which may have been overridden by command line)
    trainer = CollectiveAgencyEvolution(config.evolution, fitness_evaluator)
    
    return trainer.train