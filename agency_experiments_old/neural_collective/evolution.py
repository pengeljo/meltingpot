"""
Evolutionary Algorithm Framework for Neural Collective Agency

Implements simple evolutionary algorithms to evolve neural networks
for collective decision-making over multiple generations.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Any, Callable
from .networks import NeuralAgentBrain


class Individual:
    """Represents one individual in the evolutionary population."""
    
    def __init__(self, brain: NeuralAgentBrain, agent_id: int = 0):
        self.brain = brain
        self.agent_id = agent_id
        self.fitness = 0.0
        self.collective_score = 0.0
        self.individual_score = 0.0
        self.generation = 0
        
    def evaluate_fitness(self, individual_reward: float, collective_reward: float, 
                        cooperation_bonus: float = 0.0) -> float:
        """Calculate fitness based on individual and collective performance."""
        self.individual_score = individual_reward
        self.collective_score = collective_reward + cooperation_bonus
        
        # Weighted combination of individual and collective performance
        # This weighting can be adjusted to favor different strategies
        self.fitness = 0.6 * self.individual_score + 0.4 * self.collective_score
        return self.fitness
        
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Apply mutations to the neural network."""
        self.brain.mutate(mutation_rate, mutation_strength)
        
    def copy(self) -> 'Individual':
        """Create a deep copy of this individual."""
        new_individual = Individual(self.brain.copy(), self.agent_id)
        new_individual.fitness = self.fitness
        new_individual.collective_score = self.collective_score
        new_individual.individual_score = self.individual_score
        new_individual.generation = self.generation
        return new_individual


class EvolutionaryTrainer:
    """
    Simple evolutionary algorithm for training neural collective agents.
    
    Uses a standard genetic algorithm approach:
    - Selection: Tournament selection 
    - Crossover: Parameter averaging between parents
    - Mutation: Gaussian noise applied to weights
    - Replacement: Generational replacement with elitism
    """
    
    def __init__(self, population_size: int = 10, elite_size: int = 3, 
                 tournament_size: int = 3, mutation_rate: float = 0.1,
                 mutation_strength: float = 0.1, random_seed: int = None):
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size  
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        self.population = []
        self.generation = 0
        self.fitness_history = []
        
    def initialize_population(self, observation_size: int = 64, agent_state_size: int = 16,
                            action_size: int = 7) -> List[Individual]:
        """Initialize random population of neural agents."""
        self.population = []
        
        for i in range(self.population_size):
            brain = NeuralAgentBrain(
                observation_size=observation_size,
                agent_state_size=agent_state_size, 
                action_size=action_size,
                random_seed=random.randint(0, 10000)
            )
            individual = Individual(brain, agent_id=i)
            individual.generation = self.generation
            self.population.append(individual)
            
        return self.population
        
    def evaluate_population(self, fitness_function: Callable[[List[Individual]], List[float]]):
        """Evaluate fitness of entire population using provided fitness function."""
        fitness_scores = fitness_function(self.population)
        
        for individual, fitness in zip(self.population, fitness_scores):
            individual.fitness = fitness
            
        # Sort population by fitness (descending)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Record fitness statistics
        fitness_values = [ind.fitness for ind in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'max_fitness': max(fitness_values),
            'avg_fitness': np.mean(fitness_values),
            'min_fitness': min(fitness_values),
            'std_fitness': np.std(fitness_values)
        })
        
    def tournament_selection(self) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)
        
    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Create offspring by averaging parameters of two parents."""
        # Create new brain
        offspring_brain = parent1.brain.copy()
        
        # Get parameters from both parents
        params1 = parent1.brain.get_parameters()
        params2 = parent2.brain.get_parameters()
        
        # Average parameters
        offspring_params = {}
        for network_name in params1.keys():
            offspring_params[network_name] = []
            for i, (p1, p2) in enumerate(zip(params1[network_name], params2[network_name])):
                # Weighted average (can be adjusted)
                alpha = random.uniform(0.3, 0.7)  # Random mixing ratio
                averaged_params = alpha * p1 + (1 - alpha) * p2
                offspring_params[network_name].append(averaged_params)
                
        offspring_brain.set_parameters(offspring_params)
        
        # Create offspring individual
        offspring = Individual(offspring_brain)
        offspring.generation = self.generation + 1
        return offspring
        
    def evolve_generation(self) -> List[Individual]:
        """Evolve population to next generation."""
        new_population = []
        
        # Keep elite individuals
        elite = self.population[:self.elite_size]
        for individual in elite:
            new_population.append(individual.copy())
            
        # Generate offspring to fill remaining slots
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            offspring = self.crossover(parent1, parent2)
            offspring.mutate(self.mutation_rate, self.mutation_strength)
            
            new_population.append(offspring)
            
        self.population = new_population
        self.generation += 1
        
        return self.population
        
    def get_best_individual(self) -> Individual:
        """Get the best individual from current population."""
        return max(self.population, key=lambda x: x.fitness)
        
    def get_population_diversity(self) -> Dict[str, float]:
        """Calculate diversity metrics for the current population."""
        if len(self.population) < 2:
            return {'parameter_diversity': 0.0, 'fitness_diversity': 0.0}
            
        # Calculate parameter diversity (average pairwise distance)
        param_distances = []
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                distance = self._calculate_parameter_distance(
                    self.population[i].brain, 
                    self.population[j].brain
                )
                param_distances.append(distance)
                
        # Calculate fitness diversity
        fitness_values = [ind.fitness for ind in self.population]
        fitness_diversity = np.std(fitness_values) / (np.mean(fitness_values) + 1e-10)
        
        return {
            'parameter_diversity': np.mean(param_distances),
            'fitness_diversity': fitness_diversity,
            'num_unique_strategies': len(set(fitness_values))  # Rough estimate
        }
        
    def _calculate_parameter_distance(self, brain1: NeuralAgentBrain, 
                                    brain2: NeuralAgentBrain) -> float:
        """Calculate L2 distance between parameters of two neural brains."""
        params1 = brain1.get_parameters()
        params2 = brain2.get_parameters()
        
        total_distance = 0.0
        total_params = 0
        
        for network_name in params1.keys():
            for p1, p2 in zip(params1[network_name], params2[network_name]):
                distance = np.sum((p1 - p2) ** 2)
                total_distance += distance
                total_params += p1.size
                
        return np.sqrt(total_distance / total_params)
        
    def save_population(self, filepath: str):
        """Save current population to file."""
        # For now, just save the fitness history and best individual
        # Full implementation would serialize the neural networks
        import json
        
        data = {
            'generation': self.generation,
            'population_size': self.population_size,
            'fitness_history': self.fitness_history,
            'best_fitness': self.get_best_individual().fitness if self.population else 0.0,
            'diversity_metrics': self.get_population_diversity()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_population(self, filepath: str):
        """Load population from file."""
        # Placeholder - would need to implement neural network serialization
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.generation = data['generation']
        self.fitness_history = data['fitness_history']
        
        print(f"Loaded population data from generation {self.generation}")
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.fitness_history:
            return {}
            
        recent_stats = self.fitness_history[-1] if self.fitness_history else {}
        diversity_stats = self.get_population_diversity()
        
        return {
            'current_generation': self.generation,
            'total_generations': len(self.fitness_history),
            'best_fitness_ever': max([gen['max_fitness'] for gen in self.fitness_history]),
            'current_best_fitness': recent_stats.get('max_fitness', 0),
            'current_avg_fitness': recent_stats.get('avg_fitness', 0),
            'fitness_improvement': self._calculate_improvement_trend(),
            'population_diversity': diversity_stats,
            'convergence_status': self._assess_convergence()
        }
        
    def _calculate_improvement_trend(self) -> float:
        """Calculate fitness improvement trend over recent generations."""
        if len(self.fitness_history) < 5:
            return 0.0
            
        recent_fitness = [gen['max_fitness'] for gen in self.fitness_history[-5:]]
        return (recent_fitness[-1] - recent_fitness[0]) / len(recent_fitness)
        
    def _assess_convergence(self) -> str:
        """Assess whether the population has converged."""
        if len(self.fitness_history) < 10:
            return "early"
            
        recent_improvements = [
            self.fitness_history[i]['max_fitness'] - self.fitness_history[i-1]['max_fitness']
            for i in range(-5, 0)
        ]
        
        avg_improvement = np.mean(recent_improvements)
        
        if avg_improvement < 0.001:
            return "converged"
        elif avg_improvement < 0.01:
            return "slow_progress"  
        else:
            return "improving"