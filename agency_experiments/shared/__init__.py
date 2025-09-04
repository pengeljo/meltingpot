"""
Shared components for agency experiments.

This module provides reusable components that can be used across different
agency experiments, including agent architectures, evolutionary systems,
neural networks, and analysis tools.
"""

# Version info
__version__ = "1.0.0"
__author__ = "Agency Research Framework"

# Common imports for convenience
from .agents import BaseAgent, IndividualAgent, CollectiveAgent
from .evolution import EvolutionaryTrainer, Population
from .neural import BaseNeuralNetwork, MultiOutputNetwork
from .analysis import ExperimentAnalyzer, plot_fitness_evolution
from .utils import setup_logging, save_experiment_config

__all__ = [
    'BaseAgent', 'IndividualAgent', 'CollectiveAgent',
    'EvolutionaryTrainer', 'Population', 
    'BaseNeuralNetwork', 'MultiOutputNetwork',
    'ExperimentAnalyzer', 'plot_fitness_evolution',
    'setup_logging', 'save_experiment_config'
]