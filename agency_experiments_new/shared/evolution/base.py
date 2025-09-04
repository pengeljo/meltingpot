"""
Base classes for evolutionary training systems.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass

from ..agents.base import BaseAgent


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary training."""
    population_size: int = 20
    elite_size: int = 4
    tournament_size: int = 3
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    crossover_rate: float = 0.8
    generations: int = 50
    fitness_sharing: bool = False
    diversity_bonus: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'tournament_size': self.tournament_size,
            'mutation_rate': self.mutation_rate,
            'mutation_strength': self.mutation_strength,
            'crossover_rate': self.crossover_rate,
            'generations': self.generations,
            'fitness_sharing': self.fitness_sharing,
            'diversity_bonus': self.diversity_bonus
        }


class Population:
    """
    Container for a population of agents with fitness tracking.
    """
    
    def __init__(self, agents: List[BaseAgent]):
        self.agents = agents
        self.fitness_scores = [0.0] * len(agents)
        self.generation = 0
        self.evaluation_history = []
        
    def __len__(self):
        return len(self.agents)
    
    def __iter__(self):
        return iter(self.agents)
    
    def __getitem__(self, index):
        return self.agents[index]
    
    def evaluate_fitness(self, fitness_function: Callable[[List[BaseAgent]], List[float]]):
        """Evaluate fitness of all agents in population."""
        self.fitness_scores = fitness_function(self.agents)
        
        # Update agent fitness
        for agent, fitness in zip(self.agents, self.fitness_scores):
            agent.fitness = fitness
            
        # Sort by fitness (descending)
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        self.agents = [self.agents[i] for i in sorted_indices]
        self.fitness_scores = [self.fitness_scores[i] for i in sorted_indices]
        
        # Record evaluation
        self.evaluation_history.append({
            'generation': self.generation,
            'max_fitness': max(self.fitness_scores),
            'mean_fitness': np.mean(self.fitness_scores),
            'min_fitness': min(self.fitness_scores),
            'std_fitness': np.std(self.fitness_scores)
        })
    
    def get_elite(self, elite_size: int) -> List[BaseAgent]:
        """Get elite agents from population."""
        return self.agents[:elite_size]
    
    def get_diversity_metrics(self) -> Dict[str, float]:
        """Calculate population diversity metrics."""
        if len(self.agents) < 2:
            return {'diversity': 0.0, 'fitness_variance': 0.0}
        
        # Fitness diversity
        fitness_variance = np.var(self.fitness_scores)
        
        # Agent type diversity
        agent_types = [agent.agent_type.value for agent in self.agents]
        type_counts = {}
        for agent_type in agent_types:
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        # Calculate Shannon diversity index
        total = len(self.agents)
        shannon_diversity = 0.0
        for count in type_counts.values():
            if count > 0:
                p = count / total
                shannon_diversity -= p * np.log2(p)
        
        return {
            'fitness_variance': fitness_variance,
            'shannon_diversity': shannon_diversity,
            'num_agent_types': len(type_counts),
            'type_distribution': type_counts
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        stats = {
            'generation': self.generation,
            'population_size': len(self.agents),
            'fitness_stats': {
                'max': max(self.fitness_scores) if self.fitness_scores else 0,
                'mean': np.mean(self.fitness_scores) if self.fitness_scores else 0,
                'min': min(self.fitness_scores) if self.fitness_scores else 0,
                'std': np.std(self.fitness_scores) if self.fitness_scores else 0
            },
            'diversity_metrics': self.get_diversity_metrics(),
            'evaluation_history': self.evaluation_history
        }
        return stats


class EvolutionaryTrainer:
    """
    Main evolutionary training system that can train diverse agent populations.
    Supports different selection, crossover, and mutation strategies.
    """
    
    def __init__(self, config: EvolutionConfig, random_seed: Optional[int] = None):
        self.config = config
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.population = None
        self.training_history = []
        
    def initialize_population(self, agent_factory: Callable[[], BaseAgent]) -> Population:
        """Initialize population using agent factory function."""
        agents = [agent_factory() for _ in range(self.config.population_size)]
        self.population = Population(agents)
        return self.population
    
    def evolve_generation(self, fitness_function: Callable[[List[BaseAgent]], List[float]]) -> Population:
        """Evolve population for one generation."""
        if self.population is None:
            raise ValueError("Population not initialized. Call initialize_population first.")
        
        # Evaluate current population
        self.population.evaluate_fitness(fitness_function)
        
        # Record statistics
        generation_stats = self.population.get_statistics()
        self.training_history.append(generation_stats)
        
        # Create next generation
        new_agents = []
        
        # Elitism - keep best agents
        elite = self.population.get_elite(self.config.elite_size)
        new_agents.extend([self._copy_agent(agent) for agent in elite])
        
        # Generate offspring
        while len(new_agents) < self.config.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if np.random.random() < self.config.crossover_rate:
                offspring = self._crossover(parent1, parent2)
            else:
                offspring = self._copy_agent(parent1)
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                self._mutate(offspring)
            
            new_agents.append(offspring)
        
        # Update population
        self.population.agents = new_agents[:self.config.population_size]
        self.population.generation += 1
        
        return self.population
    
    def train(self, agent_factory: Callable[[], BaseAgent], 
             fitness_function: Callable[[List[BaseAgent]], List[float]],
             generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete evolutionary training process.
        
        Args:
            agent_factory: Function that creates new agents
            fitness_function: Function that evaluates agent fitness
            generations: Number of generations to train (uses config if None)
            
        Returns:
            Training results and statistics
        """
        if generations is None:
            generations = self.config.generations
        
        print(f"Starting evolutionary training for {generations} generations...")
        print(f"Population size: {self.config.population_size}")
        
        # Initialize population
        if self.population is None:
            self.initialize_population(agent_factory)
        
        # Training loop
        for generation in range(generations):
            print(f"\n--- Generation {generation + 1}/{generations} ---")
            
            # Evolve one generation
            self.evolve_generation(fitness_function)
            
            # Print progress
            stats = self.population.get_statistics()
            print(f"Best fitness: {stats['fitness_stats']['max']:.3f}")
            print(f"Mean fitness: {stats['fitness_stats']['mean']:.3f}")
            print(f"Population diversity: {stats['diversity_metrics']['shannon_diversity']:.3f}")
        
        # Compile final results
        final_results = self._compile_training_results()
        
        print(f"\n=== Training Complete ===")
        print(f"Best fitness achieved: {final_results['best_fitness_ever']:.3f}")
        print(f"Final diversity: {final_results['final_diversity']:.3f}")
        
        return final_results
    
    def _tournament_selection(self) -> BaseAgent:
        """Select agent using tournament selection."""
        tournament_indices = np.random.choice(
            len(self.population), 
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        
        best_index = tournament_indices[0]
        best_fitness = self.population.fitness_scores[best_index]
        
        for idx in tournament_indices[1:]:
            if self.population.fitness_scores[idx] > best_fitness:
                best_index = idx
                best_fitness = self.population.fitness_scores[idx]
        
        return self.population.agents[best_index]
    
    def _crossover(self, parent1: BaseAgent, parent2: BaseAgent) -> BaseAgent:
        """Create offspring through crossover."""
        # Simple crossover - create child based on parent1 structure
        offspring = self._copy_agent(parent1)
        
        # Mix some attributes from parent2
        # This is a placeholder - specific crossover logic depends on agent type
        if hasattr(offspring, 'coordination_efficiency') and hasattr(parent2, 'coordination_efficiency'):
            offspring.coordination_efficiency = (parent1.coordination_efficiency + parent2.coordination_efficiency) / 2
        
        return offspring
    
    def _mutate(self, agent: BaseAgent):
        """Apply mutation to agent."""
        # Simple mutation - add noise to energy budget
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config.mutation_strength * 10)
            agent.capabilities.energy_budget = max(50, agent.capabilities.energy_budget + noise)
        
        # Mutate other agent-specific parameters
        if hasattr(agent, 'coordination_efficiency'):
            if np.random.random() < 0.3:
                noise = np.random.normal(0, self.config.mutation_strength)
                agent.coordination_efficiency = max(0.5, agent.coordination_efficiency + noise)
    
    def _copy_agent(self, agent: BaseAgent) -> BaseAgent:
        """Create a deep copy of an agent."""
        # This is a simplified copy - in practice, would need agent-specific copying
        return copy.deepcopy(agent)
    
    def _compile_training_results(self) -> Dict[str, Any]:
        """Compile comprehensive training results."""
        if not self.training_history:
            return {}
        
        # Extract fitness progression
        fitness_progression = [gen['fitness_stats']['max'] for gen in self.training_history]
        diversity_progression = [gen['diversity_metrics']['shannon_diversity'] for gen in self.training_history]
        
        results = {
            'config': self.config.to_dict(),
            'generations_completed': len(self.training_history),
            'best_fitness_ever': max(fitness_progression) if fitness_progression else 0,
            'final_fitness': fitness_progression[-1] if fitness_progression else 0,
            'fitness_improvement': fitness_progression[-1] - fitness_progression[0] if len(fitness_progression) > 1 else 0,
            'final_diversity': diversity_progression[-1] if diversity_progression else 0,
            'fitness_progression': fitness_progression,
            'diversity_progression': diversity_progression,
            'final_population_stats': self.population.get_statistics() if self.population else {},
            'training_history': self.training_history,
            'best_agents': self.population.get_elite(3) if self.population else []
        }
        
        return results
    
    def save_population(self, filepath: str):
        """Save current population to file."""
        if self.population is None:
            raise ValueError("No population to save")
        
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save population metadata (agents themselves need specialized serialization)
        population_data = {
            'generation': self.population.generation,
            'population_size': len(self.population),
            'fitness_scores': self.population.fitness_scores,
            'agent_types': [agent.agent_type.value for agent in self.population.agents],
            'agent_ids': [agent.agent_id for agent in self.population.agents],
            'training_history': self.training_history
        }
        
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(population_data, f, indent=2)
        
        # Save individual agents
        for i, agent in enumerate(self.population.agents):
            agent.save_state(f"{filepath}_agent_{i:03d}.json")