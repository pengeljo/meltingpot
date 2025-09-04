"""
Mutation methods for evolutionary algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any


class MutationMethod(ABC):
    """Base class for mutation methods."""
    
    @abstractmethod
    def mutate(self, individual: Any, mutation_rate: float) -> Any:
        """Mutate an individual."""
        pass


class GaussianMutation(MutationMethod):
    """Gaussian noise mutation."""
    
    def __init__(self, std_dev: float = 0.1):
        self.std_dev = std_dev
    
    def mutate(self, individual: Any, mutation_rate: float) -> Any:
        """Apply Gaussian mutation."""
        # This is a stub - in practice would mutate individual's parameters
        return individual


class UniformMutation(MutationMethod):
    """Uniform random mutation."""
    
    def __init__(self, mutation_strength: float = 0.1):
        self.mutation_strength = mutation_strength
    
    def mutate(self, individual: Any, mutation_rate: float) -> Any:
        """Apply uniform mutation."""
        # This is a stub - in practice would mutate individual's parameters
        return individual