"""
Shared agent architectures for agency experiments.
"""

from .base import BaseAgent, AgentCapabilities
from .individual import IndividualAgent
from .collective import CollectiveAgent, CollectiveCapabilities

__all__ = [
    'BaseAgent', 'AgentCapabilities',
    'IndividualAgent', 
    'CollectiveAgent', 'CollectiveCapabilities'
]