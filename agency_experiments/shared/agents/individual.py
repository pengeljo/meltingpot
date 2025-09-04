"""
Individual agent implementation for agency experiments.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .base import BaseAgent, AgentCapabilities, AgentType


class IndividualAgent(BaseAgent):
    """
    Standard individual agent - single decision-maker with one body.
    
    This serves as the baseline for comparison with collective agents.
    Individual agents have no internal coordination costs and make
    decisions solely based on their own observations and goals.
    """
    
    def __init__(self, agent_id: str, capabilities: AgentCapabilities):
        super().__init__(agent_id, capabilities)
        
        # Individual agent state
        self.personal_memory = []
        self.individual_goals = self._initialize_individual_goals()
        
    def _get_agent_type(self) -> AgentType:
        return AgentType.INDIVIDUAL
    
    def _initialize_individual_goals(self) -> Dict[str, float]:
        """Initialize individual-specific goals."""
        return {
            'survival': 1.0,
            'resource_accumulation': 0.8,
            'exploration': 0.6,
            'efficiency': 0.7
        }
    
    def observe(self, raw_observation) -> np.ndarray:
        """Process observation for individual decision-making."""
        if isinstance(raw_observation, dict):
            # Handle MeltingPot observation format
            features = self._extract_features_from_dict(raw_observation)
        elif isinstance(raw_observation, (list, tuple)):
            # Handle multiple observations (take first one for individual)
            if len(raw_observation) > 0:
                features = self._extract_features(raw_observation[0])
            else:
                features = np.zeros(self.capabilities.observation_size)
        else:
            features = self._extract_features(raw_observation)
        
        return features
    
    def _extract_features_from_dict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract features from MeltingPot observation dictionary."""
        features = []
        
        # Handle nested observation format
        if 'observation' in obs_dict:
            obs_dict = obs_dict['observation']
        
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                # Image or array data - downsample
                if len(value.shape) == 3:  # RGB image
                    downsampled = value[::8, ::8, :].mean(axis=2).flatten()
                    features.extend(downsampled[:20])  # Max 20 image features
                elif len(value.shape) == 2:  # 2D array
                    downsampled = value[::4, ::4].flatten()
                    features.extend(downsampled[:10])  # Max 10 2D features
                else:  # 1D array
                    features.extend(value.flatten()[:5])  # Max 5 1D features
                    
            elif isinstance(value, (int, float)):
                # Scalar value
                features.append(float(value))
                
            elif isinstance(value, bool):
                # Boolean value
                features.append(float(value))
                
            elif isinstance(value, str):
                # String values - skip or convert known ones
                if value in ['READY_TO_SHOOT', 'NOT_READY']:
                    features.append(1.0 if value == 'READY_TO_SHOOT' else 0.0)
                # Skip other string values
                continue
                
            # Skip other types
        
        # Ensure we have some features
        if not features:
            features = [0.0] * 10  # Default features
            
        # Pad to expected size
        features_array = np.array(features, dtype=np.float32)
        if len(features_array) < self.capabilities.observation_size:
            padded = np.zeros(self.capabilities.observation_size, dtype=np.float32)
            padded[:len(features_array)] = features_array
            return padded
        else:
            return features_array[:self.capabilities.observation_size]
    
    def _extract_features(self, observation) -> np.ndarray:
        """Extract features from generic observation."""
        if isinstance(observation, dict):
            # Fallback to dict processing
            return self._extract_features_from_dict(observation)
        elif hasattr(observation, 'flatten'):
            features = observation.flatten()
        elif isinstance(observation, str):
            # String observations - convert to simple encoding
            return np.array([hash(observation) % 1000 / 1000.0] * min(10, self.capabilities.observation_size), dtype=np.float32)
        else:
            features = np.array([observation] if np.isscalar(observation) else observation)
        
        # Filter out non-numeric values
        numeric_features = []
        for f in features:
            try:
                numeric_features.append(float(f))
            except (ValueError, TypeError):
                numeric_features.append(0.0)  # Replace non-numeric with 0
        
        features = np.array(numeric_features, dtype=np.float32)
        
        if len(features) > self.capabilities.observation_size:
            return features[:self.capabilities.observation_size]
        else:
            padded = np.zeros(self.capabilities.observation_size, dtype=np.float32)
            padded[:len(features)] = features
            return padded
    
    def decide(self, observation: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make individual decision based on observation."""
        # Simple decision-making (can be replaced with neural networks)
        
        # Store experience
        self.personal_memory.append({
            'observation': observation,
            'step': self.step_count,
            'energy': self.energy
        })
        
        # Keep memory within capacity
        if len(self.personal_memory) > self.capabilities.memory_capacity:
            self.personal_memory = self.personal_memory[-self.capabilities.memory_capacity:]
        
        # Simple decision logic (placeholder for neural networks)
        action_preference = np.random.random(self.capabilities.action_size)
        
        # Modify based on energy level
        if self.energy < 50:
            # Conservative actions when low energy
            action_preference[0] = 0.5  # Prefer action 0 (often rest/conservative)
        
        decision = {
            'action_preferences': action_preference,
            'confidence': np.max(action_preference),
            'strategy': 'individual',
            'energy_consideration': self.energy < 50
        }
        
        self.decision_history.append(decision)
        return decision
    
    def act(self, decision: Dict[str, Any]) -> int:
        """Convert decision to environment action."""
        action_probs = decision['action_preferences']
        action_probs = action_probs / np.sum(action_probs)  # Normalize
        
        # Sample action
        action = np.random.choice(self.capabilities.action_size, p=action_probs)
        
        # Apply energy cost for action
        action_cost = 0.5 + (action / self.capabilities.action_size) * 0.5  # Variable cost
        self.energy -= action_cost
        
        return action
    
    def update(self, reward: float, next_observation: Any) -> Dict[str, float]:
        """Update individual agent state."""
        # Update energy with reward
        self.energy += reward * 0.5  # Convert reward to energy
        
        # Ensure energy bounds
        self.energy = max(0.0, min(self.capabilities.energy_budget * 2, self.energy))
        
        # Calculate performance metrics
        recent_rewards = [reward] + [r for r in getattr(self, '_recent_rewards', [])][:4]
        self._recent_rewards = recent_rewards
        
        avg_recent_reward = np.mean(recent_rewards)
        energy_efficiency = self.energy / (self.capabilities.energy_budget + 1e-10)
        
        return {
            'individual_reward': reward,
            'avg_recent_reward': avg_recent_reward,
            'energy_efficiency': energy_efficiency,
            'memory_usage': len(self.personal_memory) / self.capabilities.memory_capacity,
            'decision_confidence': np.mean([d.get('confidence', 0) for d in self.decision_history[-5:]]) if self.decision_history else 0
        }
    
    def get_analysis_data(self) -> Dict[str, Any]:
        """Get data for analyzing individual agent behavior."""
        return {
            'agent_type': 'individual',
            'memory_size': len(self.personal_memory),
            'decision_patterns': {
                'avg_confidence': np.mean([d.get('confidence', 0) for d in self.decision_history]) if self.decision_history else 0,
                'energy_considerations': sum(1 for d in self.decision_history if d.get('energy_consideration', False)),
                'total_decisions': len(self.decision_history)
            },
            'performance_metrics': {
                'total_reward': self.total_reward,
                'energy_remaining': self.energy,
                'steps_survived': self.step_count,
                'efficiency': self.total_reward / max(self.step_count, 1)
            }
        }