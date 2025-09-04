"""
Agent implementations for collective agency experiment.

Implements both individual agents and genuine collective agents with neural learning.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
import uuid

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.agents import IndividualAgent, CollectiveAgent
from shared.neural import BaseNeuralNetwork
from config import CollectiveAgencyConfig


class CollectiveDecisionNetwork(BaseNeuralNetwork):
    """Neural network for collective decision making with minimal conceptual bias."""
    
    def __init__(self, observation_size: int, action_size: int, num_components: int):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.num_components = num_components
        
        # Process environment observation
        self.env_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu')
        ])
        
        # Process component states
        self.component_processor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        # Decision fusion layer
        self.decision_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_size)
        ])
        
    def call(self, environment_obs, component_states):
        """Forward pass for collective decision making."""
        # Process environment observation
        env_features = self.env_processor(environment_obs)
        
        # Process component states
        component_features = []
        for component_state in component_states:
            comp_feat = self.component_processor(component_state)
            component_features.append(comp_feat)
        
        # Combine component features
        combined_components = tf.reduce_mean(tf.stack(component_features), axis=0)
        
        # Fuse environment and component information
        fused_features = tf.concat([env_features, combined_components], axis=-1)
        
        # Generate decision
        decision_logits = self.decision_layer(fused_features)
        
        return decision_logits


class ComponentCoordinationNetwork(BaseNeuralNetwork):
    """Network for managing component coordination and resource allocation."""
    
    def __init__(self, num_components: int, component_state_size: int = 16):
        super().__init__()
        self.num_components = num_components
        self.component_state_size = component_state_size
        
        # Component interaction layer
        self.interaction_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(16, activation='relu')
        ])
        
        # Coordination decision layer
        self.coordination_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_components, activation='softmax')  # Resource allocation
        ])
        
    def call(self, component_states, energy_level):
        """Determine coordination strategy and resource allocation."""
        # Process component interactions
        interaction_features = []
        for state in component_states:
            feat = self.interaction_layer(state)
            interaction_features.append(feat)
        
        # Add energy level information
        energy_input = tf.expand_dims(energy_level, -1)
        combined_input = tf.concat([
            tf.reduce_mean(tf.stack(interaction_features), axis=0),
            energy_input
        ], axis=-1)
        
        # Determine resource allocation
        allocation = self.coordination_layer(combined_input)
        
        return allocation


class EmergentCollectiveAgent(CollectiveAgent):
    """Genuine collective agent that learns emergent strategies through neural networks."""
    
    def __init__(self, agent_id: str, capabilities):
        super().__init__(agent_id, capabilities)
        
        # Initialize neural networks
        self.decision_network = CollectiveDecisionNetwork(
            capabilities.observation_size,
            capabilities.action_size,
            capabilities.num_components
        )
        
        self.coordination_network = ComponentCoordinationNetwork(
            capabilities.num_components
        )
        
        # Component states (learned representations)
        self.component_states = [
            tf.Variable(tf.random.normal([16]), trainable=True)
            for _ in range(capabilities.num_components)
        ]
        
        # Experience buffer for learning
        self.experience_buffer = []
        self.max_buffer_size = 1000
        
    def observe(self, raw_observation) -> np.ndarray:
        """Process raw observation for collective analysis."""
        if isinstance(raw_observation, dict):
            # MeltingPot dictionary format
            obs_arrays = []
            
            # Handle nested observation format
            obs_dict = raw_observation
            if 'observation' in raw_observation:
                obs_dict = raw_observation['observation']
            
            for key, value in obs_dict.items():
                if isinstance(value, np.ndarray):
                    # Downsample large arrays
                    if len(value.shape) == 3:  # RGB image
                        downsampled = value[::8, ::8, :].mean(axis=2).flatten()
                        obs_arrays.append(downsampled[:20])
                    elif len(value.shape) == 2:  # 2D array
                        downsampled = value[::4, ::4].flatten()
                        obs_arrays.append(downsampled[:10])
                    else:  # 1D array
                        obs_arrays.append(value.flatten()[:5])
                elif isinstance(value, (int, float)):
                    obs_arrays.append(np.array([float(value)]))
                elif isinstance(value, bool):
                    obs_arrays.append(np.array([float(value)]))
                elif isinstance(value, str):
                    # Skip string values or convert known ones
                    if value in ['READY_TO_SHOOT', 'NOT_READY']:
                        obs_arrays.append(np.array([1.0 if value == 'READY_TO_SHOOT' else 0.0]))
                    continue
            
            if obs_arrays:
                # Concatenate all arrays
                all_features = []
                for arr in obs_arrays:
                    if isinstance(arr, np.ndarray):
                        all_features.extend(arr.flatten())
                    else:
                        all_features.extend([arr] if np.isscalar(arr) else arr)
                
                # Convert to float, handling non-numeric values
                numeric_features = []
                for f in all_features:
                    try:
                        numeric_features.append(float(f))
                    except (ValueError, TypeError):
                        numeric_features.append(0.0)
                
                processed_obs = np.array(numeric_features, dtype=np.float32)
            else:
                processed_obs = np.zeros(10, dtype=np.float32)  # Default if no valid observations
        else:
            # Already array format
            if hasattr(raw_observation, 'flatten'):
                processed_obs = raw_observation.flatten()
            else:
                processed_obs = np.array([raw_observation] if np.isscalar(raw_observation) else raw_observation)
            
            # Convert to float, handling non-numeric values
            numeric_features = []
            for f in processed_obs:
                try:
                    numeric_features.append(float(f))
                except (ValueError, TypeError):
                    numeric_features.append(0.0)
            
            processed_obs = np.array(numeric_features, dtype=np.float32)
        
        # Pad or truncate to expected size
        target_size = self.capabilities.observation_size
        if len(processed_obs) > target_size:
            processed_obs = processed_obs[:target_size]
        elif len(processed_obs) < target_size:
            padded = np.zeros(target_size, dtype=np.float32)
            padded[:len(processed_obs)] = processed_obs
            processed_obs = padded
        
        return processed_obs
    
    def decide(self, observation: np.ndarray, context=None) -> Dict[str, Any]:
        """Make collective decision using neural networks."""
        # Convert to tensor
        obs_tensor = tf.convert_to_tensor([observation], dtype=tf.float32)
        
        # Get current component states
        current_states = [state.numpy() for state in self.component_states]
        
        # Determine resource allocation
        allocation = self.coordination_network(
            [tf.convert_to_tensor([state], dtype=tf.float32) for state in current_states],
            tf.convert_to_tensor([self.current_energy / self.capabilities.energy_budget], dtype=tf.float32)
        )
        
        # Apply coordination costs
        coordination_cost = np.sum(allocation.numpy()) * self.capabilities.coordination_cost
        self._apply_coordination_costs(coordination_cost)
        
        # Make collective decision
        decision_logits = self.decision_network(
            obs_tensor,
            [tf.convert_to_tensor([state], dtype=tf.float32) for state in current_states]
        )
        
        # Convert to action probabilities
        action_probs = tf.nn.softmax(decision_logits).numpy()[0]
        
        return {
            'action_probabilities': action_probs,
            'resource_allocation': allocation.numpy()[0],
            'coordination_cost': coordination_cost,
            'component_states': current_states,
            'decision_confidence': np.max(action_probs)
        }
    
    def act(self, decision: Dict[str, Any]) -> int:
        """Convert decision to concrete action."""
        action_probs = decision['action_probabilities']
        
        # Sample action from probabilities
        action = np.random.choice(len(action_probs), p=action_probs)
        
        # Store experience for learning
        self.experience_buffer.append({
            'observation': self.last_observation.copy() if hasattr(self, 'last_observation') else None,
            'action': action,
            'decision_info': decision.copy(),
            'timestamp': len(self.experience_buffer)
        })
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        return action
    
    def update(self, reward: float, next_observation: np.ndarray) -> Dict[str, Any]:
        """Update agent based on reward and new observation."""
        metrics = super().update(reward, next_observation)
        
        # Update component states based on reward
        if len(self.experience_buffer) > 0:
            # Simple learning: adjust component states based on reward
            learning_rate = self.capabilities.learning_rate
            reward_signal = tf.convert_to_tensor(reward, dtype=tf.float32)
            
            for i, component_state in enumerate(self.component_states):
                # Adjust component states towards better performance
                if reward > 0:
                    # Reinforce current states
                    noise = tf.random.normal(component_state.shape) * learning_rate * reward_signal
                    component_state.assign_add(noise * 0.1)  # Small adjustment
                else:
                    # Explore different states
                    noise = tf.random.normal(component_state.shape) * learning_rate * abs(reward_signal)
                    component_state.assign_add(noise * 0.2)  # Larger exploration
        
        # Add collective-specific metrics
        metrics.update({
            'coordination_efficiency': self._calculate_coordination_efficiency(),
            'component_diversity': self._calculate_component_diversity(),
            'collective_coherence': self._calculate_collective_coherence()
        })
        
        return metrics
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate how efficiently components are coordinating."""
        if len(self.experience_buffer) < 2:
            return 0.0
        
        # Look at consistency of resource allocation over time
        recent_allocations = [
            exp['decision_info']['resource_allocation']
            for exp in self.experience_buffer[-10:]
            if 'resource_allocation' in exp['decision_info']
        ]
        
        if len(recent_allocations) < 2:
            return 0.0
        
        # Calculate variance in allocations (lower variance = higher efficiency)
        allocations_array = np.array(recent_allocations)
        variance = np.mean(np.var(allocations_array, axis=0))
        
        # Convert to efficiency score (0-1, higher is better)
        efficiency = 1.0 / (1.0 + variance)
        return efficiency
    
    def _calculate_component_diversity(self) -> float:
        """Calculate diversity among component states."""
        if len(self.component_states) < 2:
            return 0.0
        
        states = np.array([state.numpy() for state in self.component_states])
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                dist = np.linalg.norm(states[i] - states[j])
                distances.append(dist)
        
        return np.mean(distances)
    
    def _calculate_collective_coherence(self) -> float:
        """Calculate how coherent the collective's behavior is."""
        if len(self.experience_buffer) < 5:
            return 0.0
        
        # Look at consistency of decision confidence over time
        recent_confidences = [
            exp['decision_info']['decision_confidence']
            for exp in self.experience_buffer[-10:]
            if 'decision_confidence' in exp['decision_info']
        ]
        
        if len(recent_confidences) < 2:
            return 0.0
        
        # Higher mean confidence and lower variance indicates coherence
        mean_confidence = np.mean(recent_confidences)
        confidence_variance = np.var(recent_confidences)
        
        coherence = mean_confidence * (1.0 / (1.0 + confidence_variance))
        return coherence
    
    def get_genotype(self) -> Dict[str, Any]:
        """Get agent's genetic representation for evolution."""
        return {
            'decision_network_weights': [w.numpy() for w in self.decision_network.trainable_variables],
            'coordination_network_weights': [w.numpy() for w in self.coordination_network.trainable_variables],
            'component_states': [state.numpy() for state in self.component_states],
            'agent_type': 'collective'
        }
    
    def set_genotype(self, genotype: Dict[str, Any]):
        """Set agent's genetic representation from evolution."""
        # Set network weights
        for i, weight in enumerate(genotype['decision_network_weights']):
            self.decision_network.trainable_variables[i].assign(weight)
        
        for i, weight in enumerate(genotype['coordination_network_weights']):
            self.coordination_network.trainable_variables[i].assign(weight)
        
        # Set component states
        for i, state in enumerate(genotype['component_states']):
            self.component_states[i].assign(state)


class EnhancedIndividualAgent(IndividualAgent):
    """Individual agent with neural learning for comparison with collective agents."""
    
    def __init__(self, agent_id: str, capabilities):
        super().__init__(agent_id, capabilities)
        
        # Simple neural network for decision making
        self.decision_network = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(capabilities.action_size)
        ])
        
        # Experience buffer
        self.experience_buffer = []
        self.max_buffer_size = 500
    
    def decide(self, observation: np.ndarray, context=None) -> Dict[str, Any]:
        """Make individual decision using neural network."""
        # Convert to tensor
        obs_tensor = tf.convert_to_tensor([observation], dtype=tf.float32)
        
        # Get action logits
        decision_logits = self.decision_network(obs_tensor)
        action_probs = tf.nn.softmax(decision_logits).numpy()[0]
        
        return {
            'action_probabilities': action_probs,
            'decision_confidence': np.max(action_probs),
            'agent_type': 'individual'
        }
    
    def act(self, decision: Dict[str, Any]) -> int:
        """Convert decision to action."""
        action_probs = decision['action_probabilities']
        action = np.random.choice(len(action_probs), p=action_probs)
        
        # Store experience
        self.experience_buffer.append({
            'observation': self.last_observation.copy() if hasattr(self, 'last_observation') else None,
            'action': action,
            'decision_info': decision.copy(),
            'timestamp': len(self.experience_buffer)
        })
        
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
        
        return action
    
    def update(self, reward: float, next_observation: np.ndarray) -> Dict[str, Any]:
        """Update individual agent."""
        metrics = super().update(reward, next_observation)
        
        # Simple weight adjustment based on reward
        if len(self.experience_buffer) > 0:
            learning_rate = self.capabilities.learning_rate
            
            for weight in self.decision_network.trainable_variables:
                if reward > 0:
                    noise = tf.random.normal(weight.shape) * learning_rate * reward * 0.1
                else:
                    noise = tf.random.normal(weight.shape) * learning_rate * abs(reward) * 0.15
                weight.assign_add(noise)
        
        return metrics
    
    def get_genotype(self) -> Dict[str, Any]:
        """Get genetic representation."""
        return {
            'network_weights': [w.numpy() for w in self.decision_network.trainable_variables],
            'agent_type': 'individual'
        }
    
    def set_genotype(self, genotype: Dict[str, Any]):
        """Set genetic representation."""
        for i, weight in enumerate(genotype['network_weights']):
            self.decision_network.trainable_variables[i].assign(weight)


def create_agent_factory(config: CollectiveAgencyConfig, scenario_config: Dict[str, Any]):
    """Create factory function for generating agents based on configuration."""
    
    def agent_factory() -> List:
        """Generate population of agents based on scenario configuration."""
        agents = []
        
        individual_ratio = scenario_config.get('individual_ratio', config.individual_agent_ratio)
        collective_ratio = scenario_config.get('collective_ratio', config.collective_agent_ratio)
        
        total_agents = config.evolution.population_size
        num_individual = int(total_agents * individual_ratio)
        num_collective = int(total_agents * collective_ratio)
        
        # Ensure we have the right total
        if num_individual + num_collective < total_agents:
            num_individual += total_agents - (num_individual + num_collective)
        
        # Create individual agents
        for i in range(num_individual):
            agent_id = f"individual_{i}_{uuid.uuid4().hex[:8]}"
            agent = EnhancedIndividualAgent(agent_id, config.individual_capabilities)
            agents.append(agent)
        
        # Create collective agents
        for i in range(num_collective):
            agent_id = f"collective_{i}_{uuid.uuid4().hex[:8]}"
            agent = EmergentCollectiveAgent(agent_id, config.collective_capabilities)
            agents.append(agent)
        
        return agents
    
    return agent_factory