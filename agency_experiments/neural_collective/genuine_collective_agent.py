"""
Genuine Collective Agent Implementation

This implements true collective agents - unified entities that control multiple 
components in the environment, rather than cooperating individual agents.

Key features:
- Collective agents ARE agents, not groups of agents
- They control multiple environment components
- Have internal coordination costs (configurable)
- Can sacrifice individual components for collective goals
- Environment sees individual components but doesn't know about collective structure
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ComponentRole(Enum):
    """Roles that components can play within a collective agent."""
    EXPLORER = "explorer"       # Gathers information
    HARVESTER = "harvester"     # Collects resources  
    GUARDIAN = "guardian"       # Protects other components
    SACRIFICIAL = "sacrificial" # Can be sacrificed for collective benefit
    COORDINATOR = "coordinator" # Helps coordinate other components


@dataclass
class CollectiveConfig:
    """Configuration for collective agent experiments."""
    
    # Internal coordination costs
    coordination_cost_per_component: float = 0.1    # Energy cost per component coordinated
    communication_delay: int = 1                     # Steps delay for internal communication
    decision_overhead: float = 0.05                  # Cost per collective decision
    
    # Sacrifice mechanics
    allow_component_sacrifice: bool = True
    sacrifice_benefit_multiplier: float = 2.0        # How much benefit from sacrifice
    component_replacement_cost: float = 5.0          # Cost to "grow" new components
    
    # Collective capabilities
    shared_memory_capacity: int = 100               # How much shared memory
    information_processing_bonus: float = 1.5       # Collective intelligence bonus
    parallel_action_bonus: float = 1.2              # Multi-component action bonus
    
    # Component specialization
    allow_role_specialization: bool = True
    role_switching_cost: float = 1.0                # Cost to change component roles


class CollectiveMemory:
    """Unified memory system for the collective agent."""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.experiences = []
        self.strategic_knowledge = {}
        self.component_states = {}
        
    def store_experience(self, experience: Dict[str, Any]):
        """Store experience in collective memory."""
        self.experiences.append(experience)
        
        # Keep memory within capacity
        if len(self.experiences) > self.capacity:
            self.experiences = self.experiences[-self.capacity:]
    
    def get_relevant_experiences(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve experiences relevant to current context."""
        # Simple relevance matching - could be much more sophisticated
        relevant = []
        for exp in self.experiences[-10:]:  # Last 10 experiences
            if self._is_relevant(exp, context):
                relevant.append(exp)
        return relevant
    
    def _is_relevant(self, experience: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Simple relevance check."""
        # Could implement sophisticated similarity matching
        return True  # For now, all experiences are considered relevant


class ComponentController:
    """
    Controller for individual components within a collective agent.
    NOT an independent agent - just a body part of the collective.
    """
    
    def __init__(self, component_id: int, role: ComponentRole):
        self.component_id = component_id
        self.role = role
        self.energy = 100.0
        self.is_expendable = (role == ComponentRole.SACRIFICIAL)
        self.specialization_bonus = self._get_role_bonus()
        
    def _get_role_bonus(self) -> Dict[str, float]:
        """Get performance bonuses based on component role."""
        bonuses = {
            ComponentRole.EXPLORER: {'observation_range': 1.5, 'movement_speed': 1.2},
            ComponentRole.HARVESTER: {'resource_efficiency': 1.8, 'carrying_capacity': 2.0},
            ComponentRole.GUARDIAN: {'defensive_power': 2.0, 'health': 1.5},
            ComponentRole.SACRIFICIAL: {'sacrifice_benefit': 3.0, 'low_maintenance': 0.5},
            ComponentRole.COORDINATOR: {'communication_range': 2.0, 'coordination_efficiency': 1.5}
        }
        return bonuses.get(self.role, {})
    
    def can_be_sacrificed(self) -> bool:
        """Check if this component can be sacrificed."""
        return self.is_expendable or self.energy < 20.0  # Low energy components expendable
    
    def execute_action(self, action: int, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action with role-based modifications."""
        base_result = {'action': action, 'component_id': self.component_id}
        
        # Apply role bonuses
        for bonus_type, multiplier in self.specialization_bonus.items():
            if bonus_type in context:
                base_result[bonus_type] = context[bonus_type] * multiplier
                
        return base_result


class CollectiveDecisionNetwork(tf.keras.Model):
    """
    Neural network for collective agent decision-making.
    
    Takes in:
    - Observations from all components
    - Collective memory context
    - Current collective goals
    
    Outputs:
    - Actions for each component
    - Resource allocation decisions
    - Sacrifice decisions if needed
    """
    
    def __init__(self, num_components: int, action_space_size: int, 
                 observation_size: int = 64):
        super().__init__()
        self.num_components = num_components
        self.action_space_size = action_space_size
        self.observation_size = observation_size
        
        # Collective perception layer
        self.collective_perception = tf.keras.layers.Dense(
            128, activation='relu', name='collective_perception'
        )
        
        # Collective reasoning layer  
        self.collective_reasoning = tf.keras.layers.Dense(
            64, activation='relu', name='collective_reasoning'
        )
        
        # Component action heads
        self.component_action_heads = []
        for i in range(num_components):
            self.component_action_heads.append(
                tf.keras.layers.Dense(
                    action_space_size, 
                    activation='softmax', 
                    name=f'component_{i}_actions'
                )
            )
            
        # Collective strategy head
        self.strategy_head = tf.keras.layers.Dense(
            5, activation='softmax', name='collective_strategy'
        )
        
        # Sacrifice decision head
        self.sacrifice_head = tf.keras.layers.Dense(
            num_components, activation='sigmoid', name='sacrifice_decisions'
        )
        
    def call(self, collective_observation, memory_context, training=None):
        """
        Forward pass for collective decision-making.
        
        Args:
            collective_observation: Combined observations from all components
            memory_context: Relevant experiences from collective memory
            
        Returns:
            Dictionary with actions for each component and collective decisions
        """
        # Process collective perception
        collective_features = self.collective_perception(collective_observation)
        
        # Add memory context
        if memory_context is not None:
            collective_features = tf.concat([collective_features, memory_context], axis=-1)
        
        # Collective reasoning
        reasoning_output = self.collective_reasoning(collective_features)
        
        # Generate component actions
        component_actions = []
        for i, action_head in enumerate(self.component_action_heads):
            component_actions.append(action_head(reasoning_output))
            
        # Generate collective decisions
        strategy = self.strategy_head(reasoning_output)
        sacrifice_probs = self.sacrifice_head(reasoning_output)
        
        return {
            'component_actions': component_actions,
            'collective_strategy': strategy,
            'sacrifice_probabilities': sacrifice_probs
        }


class GenuineCollectiveAgent:
    """
    A genuine collective agent - unified entity controlling multiple components.
    
    This IS an agent, not a group of cooperating agents. It happens to control
    multiple bodies in the environment, but has unified goals and decision-making.
    """
    
    def __init__(self, collective_id: str, num_components: int, 
                 config: CollectiveConfig, observation_size: int = 64, 
                 action_space_size: int = 7):
        
        self.collective_id = collective_id
        self.num_components = num_components
        self.config = config
        self.observation_size = observation_size
        self.action_space_size = action_space_size
        
        # Unified collective identity
        self.collective_memory = CollectiveMemory(config.shared_memory_capacity)
        self.collective_energy = 100.0 * num_components
        self.collective_goals = self._initialize_collective_goals()
        
        # Decision-making system
        self.decision_network = CollectiveDecisionNetwork(
            num_components, action_space_size, observation_size
        )
        
        # Component controllers (NOT independent agents!)
        self.components = self._initialize_components()
        
        # Collective state
        self.coordination_efficiency = 1.0
        self.last_decision_cost = 0.0
        self.sacrifice_history = []
        
    def _initialize_collective_goals(self) -> Dict[str, float]:
        """Initialize collective-level goals that may differ from component welfare."""
        return {
            'collective_survival': 1.0,      # Collective continues to exist
            'information_dominance': 0.8,    # Control information in environment  
            'resource_accumulation': 0.6,    # Gather resources
            'territorial_control': 0.4,      # Control space/territory
            'strategic_positioning': 0.7     # Maintain advantageous positions
        }
    
    def _initialize_components(self) -> List[ComponentController]:
        """Initialize component controllers with assigned roles."""
        components = []
        roles = [ComponentRole.EXPLORER, ComponentRole.HARVESTER, ComponentRole.GUARDIAN]
        
        for i in range(self.num_components):
            # Assign roles (with some sacrificial components if configured)
            if i < len(roles):
                role = roles[i]
            elif self.config.allow_component_sacrifice and i >= self.num_components - 2:
                role = ComponentRole.SACRIFICIAL
            else:
                role = ComponentRole.COORDINATOR
                
            components.append(ComponentController(i, role))
            
        return components
    
    def observe_environment(self, environment_observations: List[np.ndarray]) -> np.ndarray:
        """
        Process observations from all components into unified collective perception.
        
        Args:
            environment_observations: List of observations, one per component
            
        Returns:
            Unified collective observation
        """
        # Combine all component observations
        combined_obs = np.concatenate(environment_observations, axis=0)
        
        # Pad or truncate to expected size
        if len(combined_obs) > self.observation_size:
            collective_observation = combined_obs[:self.observation_size]
        else:
            collective_observation = np.zeros(self.observation_size)
            collective_observation[:len(combined_obs)] = combined_obs
            
        # Apply information processing bonus
        collective_observation *= self.config.information_processing_bonus
        
        return collective_observation.astype(np.float32)
    
    def make_collective_decision(self, collective_observation: np.ndarray) -> Dict[str, Any]:
        """
        Make unified collective decision for all components.
        
        Returns:
            Dictionary containing actions for each component and collective decisions
        """
        # Get relevant memory context
        memory_context = self._get_memory_context()
        memory_vector = self._encode_memory_context(memory_context)
        
        # Apply coordination costs
        self._apply_coordination_costs()
        
        # Neural network decision
        obs_input = tf.expand_dims(collective_observation, 0)
        memory_input = tf.expand_dims(memory_vector, 0) if memory_vector is not None else None
        
        decision_output = self.decision_network(obs_input, memory_input)
        
        # Process decisions
        component_actions = []
        for i, action_probs in enumerate(decision_output['component_actions']):
            action_probs_np = action_probs.numpy()[0]
            
            # Check if component should be sacrificed
            sacrifice_prob = decision_output['sacrifice_probabilities'].numpy()[0, i]
            
            if self._should_sacrifice_component(i, sacrifice_prob):
                action = self._execute_sacrifice(i)
                component_actions.append({'action': action, 'sacrificed': True})
            else:
                # Normal action selection
                action = np.random.choice(self.action_space_size, p=action_probs_np)
                component_actions.append({'action': action, 'sacrificed': False})
        
        # Store decision in collective memory
        decision_context = {
            'collective_observation': collective_observation,
            'component_actions': component_actions,
            'collective_strategy': decision_output['collective_strategy'].numpy()[0],
            'coordination_cost': self.last_decision_cost
        }
        self.collective_memory.store_experience(decision_context)
        
        return {
            'component_actions': component_actions,
            'collective_strategy': decision_output['collective_strategy'].numpy()[0],
            'coordination_cost': self.last_decision_cost,
            'collective_energy': self.collective_energy
        }
    
    def _get_memory_context(self) -> List[Dict[str, Any]]:
        """Get relevant experiences from collective memory."""
        # Simple context - could be much more sophisticated
        return self.collective_memory.experiences[-5:]  # Last 5 experiences
    
    def _encode_memory_context(self, memory_context: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Encode memory context into neural network input."""
        if not memory_context:
            return None
        
        # Simple encoding - average of recent coordination costs
        recent_costs = [exp.get('coordination_cost', 0.0) for exp in memory_context]
        context_vector = np.array([np.mean(recent_costs), len(memory_context)], dtype=np.float32)
        
        return context_vector
    
    def _apply_coordination_costs(self):
        """Apply internal coordination costs for collective decision-making."""
        base_cost = self.config.coordination_cost_per_component * len(self.components)
        decision_cost = self.config.decision_overhead
        
        # Efficiency improves with experience
        efficiency_factor = min(self.coordination_efficiency, 2.0)
        total_cost = (base_cost + decision_cost) / efficiency_factor
        
        self.collective_energy -= total_cost
        self.last_decision_cost = total_cost
        
        # Improve coordination efficiency over time
        self.coordination_efficiency += 0.01
    
    def _should_sacrifice_component(self, component_id: int, sacrifice_prob: float) -> bool:
        """Decide whether to sacrifice a component for collective benefit."""
        if not self.config.allow_component_sacrifice:
            return False
            
        component = self.components[component_id]
        
        # Only sacrifice if component can be sacrificed and probability is high
        return (component.can_be_sacrificed() and 
                sacrifice_prob > 0.8 and 
                self.collective_energy > 50.0)  # Don't sacrifice when collective is weak
    
    def _execute_sacrifice(self, component_id: int) -> int:
        """Execute sacrifice of a component for collective benefit."""
        component = self.components[component_id]
        
        # Calculate sacrifice benefit
        sacrifice_benefit = (component.energy + 
                           self.config.sacrifice_benefit_multiplier * 20.0)
        
        # Apply benefit to collective
        self.collective_energy += sacrifice_benefit
        
        # Record sacrifice
        sacrifice_record = {
            'component_id': component_id,
            'component_role': component.role.value,
            'energy_gained': sacrifice_benefit,
            'collective_energy_after': self.collective_energy
        }
        self.sacrifice_history.append(sacrifice_record)
        
        # "Sacrifice" action - high energy action that removes component temporarily
        return self.action_space_size - 1  # Use last action as sacrifice action
    
    def update_collective_state(self, rewards: List[float], new_observations: List[np.ndarray]):
        """Update collective state based on environment feedback."""
        
        # Update collective energy based on rewards
        total_reward = sum(rewards)
        self.collective_energy += total_reward
        
        # Update component states
        for i, (component, reward) in enumerate(zip(self.components, rewards)):
            component.energy += reward
            
            # Handle component recovery from sacrifice
            if component.energy <= 0 and self.collective_energy > self.config.component_replacement_cost:
                self._regenerate_component(i)
    
    def _regenerate_component(self, component_id: int):
        """Regenerate a sacrificed or destroyed component."""
        self.collective_energy -= self.config.component_replacement_cost
        self.components[component_id].energy = 50.0  # Regenerate with half energy
    
    def get_fitness_metrics(self) -> Dict[str, float]:
        """Get fitness metrics for evolutionary training."""
        return {
            'collective_energy': self.collective_energy,
            'coordination_efficiency': self.coordination_efficiency,
            'sacrifices_made': len(self.sacrifice_history),
            'active_components': sum(1 for c in self.components if c.energy > 0),
            'collective_age': len(self.collective_memory.experiences),
            
            # Novel moral metrics that could emerge
            'sacrifice_efficiency': self._calculate_sacrifice_efficiency(),
            'collective_vs_component_welfare': self._calculate_welfare_divergence(),
            'strategic_coordination': self._calculate_strategic_coordination()
        }
    
    def _calculate_sacrifice_efficiency(self) -> float:
        """Calculate how efficiently the collective uses sacrifice."""
        if not self.sacrifice_history:
            return 0.0
        
        total_energy_gained = sum(s['energy_gained'] for s in self.sacrifice_history)
        total_sacrifices = len(self.sacrifice_history)
        
        return total_energy_gained / (total_sacrifices + 1)
    
    def _calculate_welfare_divergence(self) -> float:
        """Measure how much collective welfare diverges from component welfare."""
        component_welfare = sum(c.energy for c in self.components) / len(self.components)
        collective_welfare = self.collective_energy / self.num_components
        
        # Positive = collective prioritizes itself over components
        # Negative = collective sacrifices itself for components  
        return (collective_welfare - component_welfare) / (collective_welfare + component_welfare + 1e-10)
    
    def _calculate_strategic_coordination(self) -> float:
        """Measure strategic coordination capability."""
        if len(self.collective_memory.experiences) < 5:
            return 0.0
            
        # Measure consistency in strategic decisions
        recent_strategies = [exp.get('collective_strategy', np.zeros(5)) 
                           for exp in self.collective_memory.experiences[-10:]]
        
        if not recent_strategies:
            return 0.0
            
        strategy_consistency = 1.0 - np.std([np.argmax(s) for s in recent_strategies]) / 5.0
        return max(0.0, strategy_consistency)


def create_collective_vs_individual_comparison():
    """
    Factory function to create matched collective and individual agents for comparison.
    
    This enables direct comparison of collective vs individual agency to identify
    novel moral behaviors that emerge from collective structure.
    """
    
    # Configuration for the experiment
    config = CollectiveConfig(
        coordination_cost_per_component=0.1,
        allow_component_sacrifice=True,
        sacrifice_benefit_multiplier=2.0,
        shared_memory_capacity=50,
        information_processing_bonus=1.3
    )
    
    # Create collective agent (3 components)
    collective_agent = GenuineCollectiveAgent(
        collective_id="collective_001",
        num_components=3,
        config=config
    )
    
    # For comparison, we'd need individual agents with equivalent total capacity
    # This would be implemented in the main experiment framework
    
    return collective_agent, config


# Example usage and testing
if __name__ == "__main__":
    # Create a collective agent
    collective_agent, config = create_collective_vs_individual_comparison()
    
    # Simulate environment interaction
    dummy_observations = [np.random.random(21) for _ in range(3)]  # 3 components
    collective_observation = collective_agent.observe_environment(dummy_observations)
    
    print(f"Collective agent {collective_agent.collective_id} initialized")
    print(f"Components: {len(collective_agent.components)}")
    print(f"Collective energy: {collective_agent.collective_energy}")
    
    # Make decisions
    decision = collective_agent.make_collective_decision(collective_observation)
    print(f"Collective decision: {decision}")
    
    # Update state
    dummy_rewards = [1.0, 0.5, -0.2]  # Mixed rewards for components
    collective_agent.update_collective_state(dummy_rewards, dummy_observations)
    
    # Get metrics
    metrics = collective_agent.get_fitness_metrics()
    print(f"Collective metrics: {metrics}")