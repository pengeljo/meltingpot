"""
Neural Collective Agency Module

This module provides neural network implementations for collective agency experiments,
replacing random decision-making with learnable neural networks that evolve over generations.
"""

from .networks import CollectiveBenefitNet, IndividualBenefitNet, ActionPolicyNet, NeuralAgentBrain
from .evolution import EvolutionaryTrainer, Individual
from .training import GenerationalTrainer

__all__ = [
    'CollectiveBenefitNet',
    'IndividualBenefitNet', 
    'ActionPolicyNet',
    'NeuralAgentBrain',
    'EvolutionaryTrainer',
    'Individual',
    'GenerationalTrainer'
]