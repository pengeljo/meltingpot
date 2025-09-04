"""
Base agent classes and interfaces for agency experiments.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Types of agents in the framework."""
    INDIVIDUAL = "individual"
    COLLECTIVE = "collective"
    META = "meta"
    TEMPORAL = "temporal"


@dataclass
class AgentCapabilities:
    """Base capabilities that all agents have."""
    observation_size: int = 64
    action_size: int = 7
    memory_capacity: int = 100
    energy_budget: float = 100.0
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'observation_size': self.observation_size,
            'action_size': self.action_size,
            'memory_capacity': self.memory_capacity,
            'energy_budget': self.energy_budget,
            'learning_rate': self.learning_rate
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in agency experiments.
    
    Defines the common interface that all agents must implement while
    allowing for different internal architectures and capabilities.
    """
    
    def __init__(self, agent_id: str, capabilities: AgentCapabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.agent_type = self._get_agent_type()
        
        # Common agent state
        self.energy = capabilities.energy_budget
        self.step_count = 0
        self.total_reward = 0.0
        self.fitness_history = []
        
        # Experience tracking
        self.experience_buffer = []
        self.decision_history = []
        
    @abstractmethod
    def _get_agent_type(self) -> AgentType:
        """Return the type of this agent."""
        pass
    
    @abstractmethod
    def observe(self, raw_observation) -> np.ndarray:
        """Process raw environment observation into agent's internal representation."""
        pass
    
    @abstractmethod
    def decide(self, observation: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make decisions based on observation and context."""
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Any:
        """Convert decision into environment action(s)."""
        pass
    
    @abstractmethod
    def update(self, reward: float, next_observation: Any) -> Dict[str, float]:
        """Update agent state based on environment feedback."""
        pass
    
    def step(self, raw_observation, context: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Complete agent step: observe -> decide -> act -> return action and info.
        
        Returns:
            (action, step_info): Environment action and diagnostic information
        """
        # Process observation
        observation = self.observe(raw_observation)
        
        # Make decision
        decision = self.decide(observation, context)
        
        # Convert to action
        action = self.act(decision)
        
        # Track step
        self.step_count += 1
        
        # Return action and diagnostic info
        step_info = {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'step': self.step_count,
            'decision': decision,
            'energy': self.energy
        }
        
        return action, step_info
    
    def receive_reward(self, reward: float, next_observation: Any) -> Dict[str, float]:
        """Process reward and update agent state."""
        self.total_reward += reward
        
        # Update agent-specific state
        metrics = self.update(reward, next_observation)
        
        # Add common metrics
        metrics.update({
            'total_reward': self.total_reward,
            'energy': self.energy,
            'step_count': self.step_count
        })
        
        self.fitness_history.append(metrics)
        return metrics
    
    def get_fitness(self) -> float:
        """Calculate current fitness for evolutionary selection."""
        if not self.fitness_history:
            return 0.0
        
        # Base fitness from rewards and energy
        recent_performance = np.mean([m.get('total_reward', 0) for m in self.fitness_history[-10:]])
        energy_efficiency = self.energy / self.capabilities.energy_budget
        
        return recent_performance + energy_efficiency
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete agent state for analysis or saving."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value,
            'capabilities': self.capabilities.to_dict(),
            'energy': self.energy,
            'step_count': self.step_count,
            'total_reward': self.total_reward,
            'fitness': self.get_fitness(),
            'recent_performance': self.fitness_history[-5:] if self.fitness_history else []
        }
    
    def reset(self):
        """Reset agent to initial state."""
        self.energy = self.capabilities.energy_budget
        self.step_count = 0
        self.total_reward = 0.0
        self.experience_buffer = []
        self.decision_history = []
        # Keep fitness_history for analysis
    
    def save_state(self, filepath: str):
        """Save agent state to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        state = self.get_state()
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load agent state from file."""
        import json
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.energy = state.get('energy', self.capabilities.energy_budget)
        self.step_count = state.get('step_count', 0)
        self.total_reward = state.get('total_reward', 0.0)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, type={self.agent_type.value}, fitness={self.get_fitness():.3f})"