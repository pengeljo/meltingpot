"""
Selection methods for evolutionary algorithms.
"""

import numpy as np
from typing import List, Any
from abc import ABC, abstractmethod


class SelectionMethod(ABC):
    """Base class for selection methods."""
    
    @abstractmethod
    def select(self, population: List, fitness_scores: List[float], num_select: int) -> List:
        """Select individuals from population based on fitness."""
        pass


class TournamentSelection(SelectionMethod):
    """Tournament selection method."""
    
    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size
    
    def select(self, population: List, fitness_scores: List[float], num_select: int) -> List:
        """Select individuals using tournament selection."""
        selected = []
        
        for _ in range(num_select):
            # Select random individuals for tournament
            tournament_indices = np.random.choice(
                len(population), 
                size=min(self.tournament_size, len(population)), 
                replace=False
            )
            
            # Find best individual in tournament
            best_idx = tournament_indices[np.argmax([fitness_scores[i] for i in tournament_indices])]
            selected.append(population[best_idx])
        
        return selected


class FitnessProportionalSelection(SelectionMethod):
    """Fitness proportional (roulette wheel) selection."""
    
    def select(self, population: List, fitness_scores: List[float], num_select: int) -> List:
        """Select individuals using fitness proportional selection."""
        # Handle negative fitness scores
        min_fitness = min(fitness_scores)
        if min_fitness < 0:
            adjusted_scores = [score - min_fitness + 1e-8 for score in fitness_scores]
        else:
            adjusted_scores = [score + 1e-8 for score in fitness_scores]
        
        # Calculate selection probabilities
        total_fitness = sum(adjusted_scores)
        probabilities = [score / total_fitness for score in adjusted_scores]
        
        # Select individuals
        selected_indices = np.random.choice(
            len(population), 
            size=num_select, 
            p=probabilities, 
            replace=True
        )
        
        selected = [population[i] for i in selected_indices]
        return selected


class ElitistSelection(SelectionMethod):
    """Select the best performing individuals."""
    
    def select(self, population: List, fitness_scores: List[float], num_select: int) -> List:
        """Select the top num_select individuals."""
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        elite_indices = sorted_indices[:num_select]
        
        return [population[i] for i in elite_indices]