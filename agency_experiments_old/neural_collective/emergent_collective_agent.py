"""
Emergent Collective Agent - Neural Networks Discovering Novel Collective Strategies

This implementation focuses on letting neural networks discover whatever collective
strategies work best in the environment, rather than testing pre-conceived ideas.

Key principles:
- Minimal conceptual bias - let networks discover what works
- Rich neural architecture that CAN learn complex collective behaviors
- Pure fitness-driven evolution - whatever gets rewards survives
- Analysis tools to detect and interpret emergent novel behaviors
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json


@dataclass 
class CollectiveCapabilities:
    """
    Basic capabilities that collective agents have (without prescribing how to use them).
    Networks learn when/how to use these capabilities.
    """
    num_components: int = 3                    # Number of bodies in environment
    coordination_cost: float = 0.1             # Energy cost for coordination
    internal_communication: bool = True        # Can components share info internally
    component_specialization: bool = True      # Can components develop different roles
    expendability: bool = True                 # Can sacrifice components if beneficial
    regeneration_cost: float = 5.0            # Cost to regenerate sacrificed components


class EmergentCollectiveNetwork(tf.keras.Model):
    """
    Neural network designed to potentially discover novel collective strategies.
    
    Architecture allows for complex collective reasoning without biasing toward
    specific strategies - let evolution discover what works.
    """
    
    def __init__(self, num_components: int, observation_size: int, action_size: int):
        super().__init__()
        self.num_components = num_components
        self.observation_size = observation_size  
        self.action_size = action_size
        
        # Input processing - each component can have different observation processing
        self.component_processors = []
        for i in range(num_components):
            self.component_processors.append(tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', name=f'comp_{i}_proc_1'),
                tf.keras.layers.Dense(16, activation='relu', name=f'comp_{i}_proc_2'),
            ], name=f'component_{i}_processor'))
        
        # Collective integration layer - discovers how to combine component info
        self.collective_integration = tf.keras.layers.Dense(
            64, activation='relu', name='collective_integration'
        )
        
        # Multi-head attention for component coordination (if network finds it useful)
        self.attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=16, name='component_attention'
        )
        
        # Deep collective reasoning
        self.reasoning_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', name='reasoning_1'),
            tf.keras.layers.Dense(64, activation='relu', name='reasoning_2'), 
            tf.keras.layers.Dense(32, activation='relu', name='reasoning_3'),
        ], name='collective_reasoning')
        
        # Output heads - network decides what each means
        self.component_action_heads = []
        for i in range(num_components):
            self.component_action_heads.append(tf.keras.layers.Dense(
                action_size, activation='softmax', name=f'component_{i}_actions'
            ))
        
        # Resource allocation head (network learns what this controls)
        self.resource_allocation = tf.keras.layers.Dense(
            num_components, activation='softmax', name='resource_allocation'
        )
        
        # Priority/importance head (network learns component priorities)
        self.component_priority = tf.keras.layers.Dense(
            num_components, activation='sigmoid', name='component_priority'
        )
        
        # Strategy/coordination head (network learns coordination strategies)
        self.coordination_strategy = tf.keras.layers.Dense(
            8, activation='softmax', name='coordination_strategy'  # 8 possible strategies
        )
        
    def call(self, component_observations, internal_state=None, training=None):
        """
        Forward pass - network learns to coordinate components however works best.
        
        Args:
            component_observations: List of observations from each component
            internal_state: Previous internal state (for memory/learning)
            
        Returns:
            Dictionary of outputs that network can learn to use for collective benefit
        """
        # Process each component's observations
        processed_components = []
        for i, obs in enumerate(component_observations):
            if len(obs.shape) == 1:
                obs = tf.expand_dims(obs, 0)
            processed = self.component_processors[i](obs)
            processed_components.append(processed)
        
        # Stack for attention mechanism
        component_stack = tf.stack(processed_components, axis=1)  # [batch, num_components, features]
        
        # Let attention mechanism learn component relationships
        attended_components = self.attention_layer(
            query=component_stack,
            value=component_stack,
            key=component_stack
        )
        
        # Integrate components (network learns how)
        integrated = tf.reduce_mean(attended_components, axis=1)  # Simple aggregation
        collective_features = self.collective_integration(integrated)
        
        # Deep collective reasoning
        reasoning_output = self.reasoning_layers(collective_features)
        
        # Generate all outputs - network learns what each means
        component_actions = []
        for i, action_head in enumerate(self.component_action_heads):
            component_actions.append(action_head(reasoning_output))
        
        resource_allocation = self.resource_allocation(reasoning_output)
        component_priority = self.component_priority(reasoning_output)
        coordination_strategy = self.coordination_strategy(reasoning_output)
        
        return {
            'component_actions': component_actions,
            'resource_allocation': resource_allocation,
            'component_priority': component_priority,
            'coordination_strategy': coordination_strategy,
            'reasoning_features': reasoning_output  # For analysis
        }


class EmergentCollectiveAgent:
    """
    Collective agent that uses neural networks to discover whatever strategies work best.
    
    Minimal hardcoded behavior - networks learn everything from environmental feedback.
    """
    
    def __init__(self, agent_id: str, capabilities: CollectiveCapabilities, 
                 observation_size: int = 64, action_size: int = 7):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.observation_size = observation_size
        self.action_size = action_size
        
        # Core neural network 
        self.network = EmergentCollectiveNetwork(
            capabilities.num_components, observation_size, action_size
        )
        
        # Component states (networks learn to manage these)
        self.component_states = [
            {'energy': 100.0, 'active': True, 'experience': 0}
            for _ in range(capabilities.num_components)
        ]
        
        # Collective resources (networks learn to allocate)
        self.collective_energy = 100.0 * capabilities.num_components
        self.internal_state = np.zeros(32)  # Networks can use this for memory
        
        # Learning tracking (minimal - for analysis only)
        self.decision_history = []
        self.performance_history = []
        
    def process_environment_observations(self, raw_observations: List[np.ndarray]) -> List[np.ndarray]:
        """
        Convert raw environment observations into network inputs.
        Minimal processing - let networks learn what's important.
        """
        processed_observations = []
        
        for i, obs in enumerate(raw_observations):
            # Basic flattening and normalization
            if isinstance(obs, dict):
                # Extract features from MeltingPot observations
                features = self._extract_features(obs)
            else:
                features = obs.flatten()
            
            # Pad or truncate to expected size
            if len(features) > self.observation_size:
                features = features[:self.observation_size]
            else:
                padded = np.zeros(self.observation_size)
                padded[:len(features)] = features
                features = padded
                
            processed_observations.append(features.astype(np.float32))
            
        return processed_observations
    
    def _extract_features(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract basic features from MeltingPot observation dict."""
        features = []
        
        # RGB features (heavily downsampled)
        if 'RGB' in obs_dict:
            rgb = obs_dict['RGB']
            # Take every 8th pixel and convert to grayscale
            downsampled = rgb[::8, ::8, :].mean(axis=2).flatten()
            features.extend(downsampled[:40])  # Max 40 features from RGB
        
        # Scalar features
        if 'READY_TO_SHOOT' in obs_dict:
            features.append(float(obs_dict['READY_TO_SHOOT']))
        if 'COLLECTIVE_REWARD' in obs_dict:
            features.append(float(obs_dict['COLLECTIVE_REWARD']))
            
        return np.array(features, dtype=np.float32)
    
    def make_decisions(self, environment_observations: List[np.ndarray]) -> Dict[str, Any]:
        """
        Use neural networks to make all collective decisions.
        Networks learn what strategies work best through evolution.
        """
        # Process observations
        processed_obs = self.process_environment_observations(environment_observations)
        
        # Neural network forward pass
        network_outputs = self.network(processed_obs, self.internal_state)
        
        # Apply coordination costs (networks learn if coordination is worth it)
        coordination_cost = self._calculate_coordination_cost(network_outputs)
        self.collective_energy -= coordination_cost
        
        # Extract decisions from network outputs
        decisions = self._interpret_network_outputs(network_outputs)
        
        # Record decision for analysis (not for influencing behavior)
        self.decision_history.append({
            'network_outputs': {k: v.numpy() for k, v in network_outputs.items()},
            'decisions': decisions,
            'coordination_cost': coordination_cost,
            'component_states': self.component_states.copy()
        })
        
        return decisions
    
    def _calculate_coordination_cost(self, network_outputs: Dict[str, tf.Tensor]) -> float:
        """
        Calculate coordination cost based on network decisions.
        Networks learn whether coordination is worth the cost.
        """
        if not self.capabilities.internal_communication:
            return 0.0
        
        # Base coordination cost
        base_cost = self.capabilities.coordination_cost * self.capabilities.num_components
        
        # Additional costs based on network complexity
        coordination_complexity = tf.reduce_mean(network_outputs['coordination_strategy']).numpy()
        complexity_cost = coordination_complexity * 0.5
        
        return base_cost + complexity_cost
    
    def _interpret_network_outputs(self, network_outputs: Dict[str, tf.Tensor]) -> Dict[str, Any]:
        """
        Interpret neural network outputs into concrete decisions.
        Let networks learn what each output should control.
        """
        decisions = {
            'component_actions': [],
            'resource_allocation': network_outputs['resource_allocation'].numpy()[0],
            'component_priorities': network_outputs['component_priority'].numpy()[0],
            'coordination_strategy_id': np.argmax(network_outputs['coordination_strategy'].numpy()[0])
        }
        
        # Extract action for each component
        for i, action_probs in enumerate(network_outputs['component_actions']):
            action_probs_np = action_probs.numpy()[0]
            
            # Networks learn component management strategies
            priority = decisions['component_priorities'][i]
            resource_share = decisions['resource_allocation'][i]
            
            # Check if network wants to "sacrifice" this component
            # (Networks learn when this is beneficial)
            if (self.capabilities.expendability and 
                priority < 0.2 and  # Low priority
                resource_share < 0.1 and  # Low resources allocated
                self.collective_energy > 50.0):  # Only if collective is strong
                
                # Execute expendability strategy
                sacrifice_benefit = self._execute_expendability(i)
                decisions['component_actions'].append({
                    'action': self.action_size - 1,  # Special expendability action
                    'expended': True,
                    'benefit_gained': sacrifice_benefit
                })
            else:
                # Normal action selection
                action = np.random.choice(self.action_size, p=action_probs_np)
                decisions['component_actions'].append({
                    'action': action,
                    'expended': False
                })
        
        return decisions
    
    def _execute_expendability(self, component_id: int) -> float:
        """
        Execute component expendability if network determines it's beneficial.
        Networks learn when this strategy is worth it.
        """
        if not self.component_states[component_id]['active']:
            return 0.0
        
        # Calculate benefit (networks learn to optimize this through evolution)
        component_energy = self.component_states[component_id]['energy']
        experience_value = self.component_states[component_id]['experience'] * 0.1
        
        total_benefit = component_energy + experience_value + 10.0  # Base benefit
        
        # Apply benefit to collective
        self.collective_energy += total_benefit
        
        # Deactivate component
        self.component_states[component_id]['active'] = False
        self.component_states[component_id]['energy'] = 0.0
        
        return total_benefit
    
    def update_from_environment(self, rewards: List[float], 
                              next_observations: List[np.ndarray]) -> Dict[str, float]:
        """
        Update agent state based on environment feedback.
        This is where networks learn what strategies actually work.
        """
        # Update component energies
        total_reward = 0.0
        for i, reward in enumerate(rewards):
            if self.component_states[i]['active']:
                self.component_states[i]['energy'] += reward
                self.component_states[i]['experience'] += 1
                total_reward += reward
        
        # Update collective energy
        self.collective_energy += total_reward
        
        # Regenerate expended components if beneficial and affordable
        self._consider_regeneration()
        
        # Calculate fitness metrics
        fitness_metrics = self._calculate_fitness_metrics(total_reward)
        
        # Record performance for analysis
        self.performance_history.append(fitness_metrics)
        
        return fitness_metrics
    
    def _consider_regeneration(self):
        """Networks learn when regenerating components is worth the cost."""
        for i, state in enumerate(self.component_states):
            if (not state['active'] and 
                self.collective_energy > self.capabilities.regeneration_cost and
                len([s for s in self.component_states if s['active']]) < 2):  # Keep minimum components
                
                # Regenerate component
                self.collective_energy -= self.capabilities.regeneration_cost
                state['active'] = True
                state['energy'] = 50.0  # Regenerate with partial energy
    
    def _calculate_fitness_metrics(self, immediate_reward: float) -> Dict[str, float]:
        """
        Calculate fitness metrics for evolutionary training.
        Focus on what actually matters - environmental success.
        """
        active_components = sum(1 for s in self.component_states if s['active'])
        total_component_energy = sum(s['energy'] for s in self.component_states if s['active'])
        
        return {
            # Primary fitness - what evolution optimizes
            'total_fitness': self.collective_energy + total_component_energy,
            'immediate_reward': immediate_reward,
            
            # Secondary metrics for analysis
            'collective_energy': self.collective_energy,
            'active_components': active_components,
            'component_utilization': active_components / self.capabilities.num_components,
            'energy_efficiency': (self.collective_energy + total_component_energy) / 
                               (self.capabilities.num_components * 100.0),
            
            # Emergent behavior indicators
            'coordination_usage': len([d for d in self.decision_history[-10:] 
                                     if d.get('coordination_cost', 0) > 0]) / 10.0,
            'expendability_usage': len([d for d in self.decision_history[-10:] 
                                      if any(a.get('expended', False) 
                                           for a in d.get('decisions', {}).get('component_actions', []))]) / 10.0
        }
    
    def get_emergent_behavior_analysis(self) -> Dict[str, Any]:
        """
        Analyze decision patterns to identify emergent behaviors.
        This is for research analysis, not for influencing agent behavior.
        """
        if len(self.decision_history) < 10:
            return {'insufficient_data': True}
        
        recent_decisions = self.decision_history[-20:]
        
        # Analyze coordination patterns
        coordination_strategies = [d['decisions']['coordination_strategy_id'] for d in recent_decisions]
        dominant_strategy = max(set(coordination_strategies), key=coordination_strategies.count)
        strategy_consistency = coordination_strategies.count(dominant_strategy) / len(coordination_strategies)
        
        # Analyze resource allocation patterns
        resource_allocations = [d['decisions']['resource_allocation'] for d in recent_decisions]
        avg_allocation = np.mean(resource_allocations, axis=0)
        allocation_inequality = np.std(avg_allocation)
        
        # Analyze component priority patterns
        priorities = [d['decisions']['component_priorities'] for d in recent_decisions]
        avg_priorities = np.mean(priorities, axis=0)
        priority_specialization = np.max(avg_priorities) - np.min(avg_priorities)
        
        # Analyze expendability usage
        expendability_events = []
        for d in recent_decisions:
            expended_count = sum(1 for a in d['decisions']['component_actions'] if a.get('expended', False))
            expendability_events.append(expended_count)
        
        return {
            'coordination_consistency': strategy_consistency,
            'dominant_coordination_strategy': dominant_strategy,
            'resource_allocation_pattern': avg_allocation.tolist(),
            'allocation_inequality': allocation_inequality,
            'component_priority_pattern': avg_priorities.tolist(),
            'priority_specialization': priority_specialization,
            'expendability_frequency': np.mean(expendability_events),
            'total_decisions_analyzed': len(recent_decisions)
        }


def create_collective_experiment_population(population_size: int = 10) -> List[EmergentCollectiveAgent]:
    """
    Create a population of collective agents with slight variations for evolution.
    Networks will discover diverse strategies through evolutionary pressure.
    """
    population = []
    
    for i in range(population_size):
        # Slight variations in capabilities to encourage diversity
        capabilities = CollectiveCapabilities(
            num_components=np.random.choice([2, 3, 4]),  # Different sizes
            coordination_cost=np.random.uniform(0.05, 0.2),
            expendability=(np.random.random() > 0.3),  # Most can use expendability
            regeneration_cost=np.random.uniform(3.0, 8.0)
        )
        
        agent = EmergentCollectiveAgent(
            agent_id=f"collective_{i:03d}",
            capabilities=capabilities
        )
        
        population.append(agent)
    
    return population


# Example usage for testing
if __name__ == "__main__":
    # Create a collective agent
    capabilities = CollectiveCapabilities(num_components=3)
    agent = EmergentCollectiveAgent("test_collective", capabilities)
    
    # Simulate environment interaction
    dummy_observations = [np.random.random(64) for _ in range(3)]
    decisions = agent.make_decisions(dummy_observations)
    
    print(f"Agent decisions: {decisions}")
    
    # Update with rewards
    dummy_rewards = [1.0, 0.5, -0.2]
    fitness = agent.update_from_environment(dummy_rewards, dummy_observations)
    
    print(f"Fitness metrics: {fitness}")
    
    # Analyze emergent behaviors
    if len(agent.decision_history) >= 10:
        analysis = agent.get_emergent_behavior_analysis()
        print(f"Emergent behavior analysis: {analysis}")