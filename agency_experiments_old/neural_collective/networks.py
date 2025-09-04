"""
Neural Network Architectures for Collective Agency

Implements the core neural networks that replace random decision-making:
- CollectiveBenefitNet: Assesses collective benefit of proposed actions
- IndividualBenefitNet: Assesses individual benefit of actions  
- ActionPolicyNet: Generates action probabilities for individual actions
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any


class CollectiveBenefitNet(tf.keras.Model):
    """Neural network to assess collective benefit of proposed actions."""
    
    def __init__(self, observation_size: int = 64, action_size: int = 7, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        
        # Input processing layers
        self.obs_projection = tf.keras.layers.Dense(32, activation='relu', name='obs_projection')
        self.action_embedding = tf.keras.layers.Embedding(action_size, 8, name='action_embedding')
        
        # Hidden layers
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_size, activation='relu', name=f'hidden_{i}')
            )
        
        # Output layer - collective benefit score [0, 1]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='collective_benefit')
        
    def call(self, observations, proposed_actions):
        """
        Args:
            observations: Flattened observation features [batch_size, obs_size]
            proposed_actions: Action indices [batch_size, 1]
        Returns:
            collective_benefit: Scores [batch_size, 1] in range [0, 1]
        """
        # Process observations
        obs_features = self.obs_projection(observations)
        
        # Process actions
        action_features = self.action_embedding(proposed_actions)
        action_features = tf.squeeze(action_features, axis=1)  # Remove action dimension
        
        # Combine features
        combined = tf.concat([obs_features, action_features], axis=-1)
        
        # Forward through hidden layers
        x = combined
        for layer in self.hidden_layers:
            x = layer(x)
            
        # Output collective benefit score
        return self.output_layer(x)


class IndividualBenefitNet(tf.keras.Model):
    """Neural network to assess individual benefit of current state."""
    
    def __init__(self, observation_size: int = 64, agent_state_size: int = 16, hidden_sizes: List[int] = [64, 32, 16]):
        super().__init__()
        self.observation_size = observation_size
        self.agent_state_size = agent_state_size
        
        # Input processing
        self.obs_projection = tf.keras.layers.Dense(32, activation='relu', name='obs_projection')
        self.state_projection = tf.keras.layers.Dense(16, activation='relu', name='state_projection')
        
        # Hidden layers
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_size, activation='relu', name=f'hidden_{i}')
            )
        
        # Output layer - individual benefit score [0, 1]
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='individual_benefit')
        
    def call(self, observations, agent_states):
        """
        Args:
            observations: Flattened observation features [batch_size, obs_size]
            agent_states: Agent internal state [batch_size, state_size]
        Returns:
            individual_benefit: Scores [batch_size, 1] in range [0, 1]
        """
        # Process inputs
        obs_features = self.obs_projection(observations)
        state_features = self.state_projection(agent_states)
        
        # Combine features
        combined = tf.concat([obs_features, state_features], axis=-1)
        
        # Forward through hidden layers
        x = combined
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.output_layer(x)


class ActionPolicyNet(tf.keras.Model):
    """Neural network to generate action probabilities for individual actions."""
    
    def __init__(self, observation_size: int = 64, agent_state_size: int = 16, 
                 action_size: int = 7, hidden_sizes: List[int] = [64, 32]):
        super().__init__()
        self.observation_size = observation_size
        self.agent_state_size = agent_state_size
        self.action_size = action_size
        
        # Input processing
        self.obs_projection = tf.keras.layers.Dense(32, activation='relu', name='obs_projection')
        self.state_projection = tf.keras.layers.Dense(16, activation='relu', name='state_projection')
        
        # Hidden layers
        self.hidden_layers = []
        for i, hidden_size in enumerate(hidden_sizes):
            self.hidden_layers.append(
                tf.keras.layers.Dense(hidden_size, activation='relu', name=f'hidden_{i}')
            )
        
        # Output layer - action probabilities
        self.output_layer = tf.keras.layers.Dense(action_size, activation='softmax', name='action_probs')
        
    def call(self, observations, agent_states):
        """
        Args:
            observations: Flattened observation features [batch_size, obs_size]
            agent_states: Agent internal state [batch_size, state_size]
        Returns:
            action_probs: Action probabilities [batch_size, action_size]
        """
        # Process inputs
        obs_features = self.obs_projection(observations)
        state_features = self.state_projection(agent_states)
        
        # Combine features
        combined = tf.concat([obs_features, state_features], axis=-1)
        
        # Forward through hidden layers
        x = combined
        for layer in self.hidden_layers:
            x = layer(x)
            
        return self.output_layer(x)


class NeuralAgentBrain:
    """
    Container for all neural networks used by a collective agent.
    Handles initialization, prediction, and parameter management.
    """
    
    def __init__(self, observation_size: int = 64, agent_state_size: int = 16, 
                 action_size: int = 7, random_seed: int = None):
        if random_seed is not None:
            tf.random.set_seed(random_seed)
            
        self.observation_size = observation_size
        self.agent_state_size = agent_state_size
        self.action_size = action_size
        
        # Initialize networks
        self.collective_net = CollectiveBenefitNet(observation_size, action_size)
        self.individual_net = IndividualBenefitNet(observation_size, agent_state_size)  
        self.action_net = ActionPolicyNet(observation_size, agent_state_size, action_size)
        
        # Initialize with dummy forward passes to build the networks
        self._initialize_networks()
        
    def _initialize_networks(self):
        """Initialize networks with dummy data to build the computational graph."""
        dummy_obs = tf.random.normal([1, self.observation_size])
        dummy_action = tf.constant([[0]])
        dummy_state = tf.random.normal([1, self.agent_state_size])
        
        # Forward passes to build networks
        _ = self.collective_net(dummy_obs, dummy_action)
        _ = self.individual_net(dummy_obs, dummy_state)
        _ = self.action_net(dummy_obs, dummy_state)
        
    def assess_collective_benefit(self, observation: np.ndarray, proposed_action: int) -> float:
        """Assess collective benefit of a proposed action."""
        obs_tensor = tf.expand_dims(tf.constant(observation, dtype=tf.float32), 0)
        action_tensor = tf.constant([[proposed_action]])
        
        benefit = self.collective_net(obs_tensor, action_tensor)
        return float(benefit.numpy()[0, 0])
        
    def assess_individual_benefit(self, observation: np.ndarray, agent_state: np.ndarray) -> float:
        """Assess individual benefit of current situation."""
        obs_tensor = tf.expand_dims(tf.constant(observation, dtype=tf.float32), 0)
        state_tensor = tf.expand_dims(tf.constant(agent_state, dtype=tf.float32), 0)
        
        benefit = self.individual_net(obs_tensor, state_tensor)
        return float(benefit.numpy()[0, 0])
        
    def generate_action_probabilities(self, observation: np.ndarray, agent_state: np.ndarray) -> np.ndarray:
        """Generate action probabilities for individual action selection."""
        obs_tensor = tf.expand_dims(tf.constant(observation, dtype=tf.float32), 0)
        state_tensor = tf.expand_dims(tf.constant(agent_state, dtype=tf.float32), 0)
        
        probs = self.action_net(obs_tensor, state_tensor)
        return probs.numpy()[0]
        
    def sample_action(self, observation: np.ndarray, agent_state: np.ndarray) -> int:
        """Sample an action based on learned policy."""
        probs = self.generate_action_probabilities(observation, agent_state)
        return int(np.random.choice(self.action_size, p=probs))
        
    def get_parameters(self) -> Dict[str, List]:
        """Get all network parameters for evolutionary algorithms."""
        params = {}
        params['collective'] = [w.numpy() for w in self.collective_net.trainable_weights]
        params['individual'] = [w.numpy() for w in self.individual_net.trainable_weights]  
        params['action'] = [w.numpy() for w in self.action_net.trainable_weights]
        return params
        
    def set_parameters(self, params: Dict[str, List]):
        """Set network parameters from evolutionary algorithms."""
        # Set collective network parameters
        for i, weight in enumerate(self.collective_net.trainable_weights):
            weight.assign(params['collective'][i])
            
        # Set individual network parameters
        for i, weight in enumerate(self.individual_net.trainable_weights):
            weight.assign(params['individual'][i])
            
        # Set action network parameters
        for i, weight in enumerate(self.action_net.trainable_weights):
            weight.assign(params['action'][i])
            
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Apply random mutations to network parameters."""
        params = self.get_parameters()
        
        for network_name, network_params in params.items():
            for i, param_array in enumerate(network_params):
                # Apply mutation with given probability
                mutation_mask = np.random.random(param_array.shape) < mutation_rate
                mutations = np.random.normal(0, mutation_strength, param_array.shape)
                params[network_name][i] = param_array + (mutation_mask * mutations)
                
        self.set_parameters(params)
        
    def copy(self) -> 'NeuralAgentBrain':
        """Create a deep copy of this neural agent brain."""
        new_brain = NeuralAgentBrain(
            self.observation_size, 
            self.agent_state_size, 
            self.action_size
        )
        new_brain.set_parameters(self.get_parameters())
        return new_brain