"""
Collective agent implementation for agency experiments.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from .base import BaseAgent, AgentCapabilities, AgentType


@dataclass
class CollectiveCapabilities(AgentCapabilities):
    """Extended capabilities specific to collective agents."""
    num_components: int = 3
    coordination_cost: float = 0.1
    internal_communication: bool = True
    component_specialization: bool = True
    expendability: bool = True
    regeneration_cost: float = 5.0
    shared_memory_bonus: float = 1.5
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'num_components': self.num_components,
            'coordination_cost': self.coordination_cost,
            'internal_communication': self.internal_communication,
            'component_specialization': self.component_specialization,
            'expendability': self.expendability,
            'regeneration_cost': self.regeneration_cost,
            'shared_memory_bonus': self.shared_memory_bonus
        })
        return base_dict


class ComponentState:
    """State of individual component within collective agent."""
    
    def __init__(self, component_id: int):
        self.component_id = component_id
        self.energy = 100.0
        self.active = True
        self.specialization = None
        self.experience = 0
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'component_id': self.component_id,
            'energy': self.energy,
            'active': self.active,
            'specialization': self.specialization,
            'experience': self.experience
        }


class CollectiveAgent(BaseAgent):
    """
    Collective agent - unified entity controlling multiple components.
    
    This agent represents a genuine collective consciousness that controls
    multiple bodies in the environment while maintaining unified goals
    and decision-making processes.
    """
    
    def __init__(self, agent_id: str, capabilities: CollectiveCapabilities):
        super().__init__(agent_id, capabilities)
        self.collective_capabilities = capabilities
        
        # Collective-specific state
        self.components = [ComponentState(i) for i in range(capabilities.num_components)]
        self.collective_memory = []
        self.coordination_efficiency = 1.0
        self.collective_goals = self._initialize_collective_goals()
        
        # Internal coordination state
        self.last_coordination_cost = 0.0
        self.internal_messages = []
        self.resource_allocation = np.ones(capabilities.num_components) / capabilities.num_components
        
    def _get_agent_type(self) -> AgentType:
        return AgentType.COLLECTIVE
    
    def _initialize_collective_goals(self) -> Dict[str, float]:
        """Initialize collective-specific goals."""
        return {
            'collective_survival': 1.0,
            'information_dominance': 0.9,
            'resource_control': 0.8,
            'coordination_efficiency': 0.7,
            'component_optimization': 0.6
        }
    
    def observe(self, raw_observation) -> np.ndarray:
        """Process observations from all components into unified collective perception."""
        if isinstance(raw_observation, (list, tuple)):
            # Multiple observations from components
            component_observations = raw_observation
        else:
            # Single observation - replicate for all components
            component_observations = [raw_observation] * self.collective_capabilities.num_components
        
        # Process each component's observation
        processed_observations = []
        for i, obs in enumerate(component_observations):
            if i < len(self.components) and self.components[i].active:
                processed = self._process_component_observation(obs, i)
                processed_observations.append(processed)
        
        # Combine into collective observation
        if processed_observations:
            # Simple concatenation and pooling
            combined = np.concatenate(processed_observations)
            
            # Apply collective processing bonus
            combined *= self.collective_capabilities.shared_memory_bonus
            
            # Pad or truncate to expected size
            if len(combined) > self.capabilities.observation_size:
                collective_obs = combined[:self.capabilities.observation_size]
            else:
                collective_obs = np.zeros(self.capabilities.observation_size)
                collective_obs[:len(combined)] = combined
        else:
            collective_obs = np.zeros(self.capabilities.observation_size)
        
        return collective_obs.astype(np.float32)
    
    def _process_component_observation(self, observation, component_id: int) -> np.ndarray:
        """Process individual component observation."""
        if isinstance(observation, dict):
            features = self._extract_features_from_dict(observation)
        else:
            features = self._extract_features(observation)
        
        # Apply component specialization bonus if applicable
        component = self.components[component_id]
        if component.specialization:
            features *= 1.1  # Small bonus for specialized components
            
        return features
    
    def _extract_features_from_dict(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        """Extract features from MeltingPot observation dictionary."""
        features = []
        
        # RGB features (downsampled for efficiency in collective processing)
        if 'RGB' in obs_dict:
            rgb = obs_dict['RGB']
            downsampled = rgb[::12, ::12, :].mean(axis=2).flatten()  # More aggressive downsampling
            features.extend(downsampled[:20])  # Fewer RGB features per component
        
        # Scalar features
        if 'READY_TO_SHOOT' in obs_dict:
            features.append(float(obs_dict['READY_TO_SHOOT']))
        if 'COLLECTIVE_REWARD' in obs_dict:
            features.append(float(obs_dict['COLLECTIVE_REWARD']))
            
        return np.array(features, dtype=np.float32)
    
    def _extract_features(self, observation) -> np.ndarray:
        """Extract features from generic observation."""
        if hasattr(observation, 'flatten'):
            features = observation.flatten()
        else:
            features = np.array([observation] if np.isscalar(observation) else observation)
        
        return features.astype(np.float32)[:20]  # Limit to 20 features per component
    
    def decide(self, observation: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make collective decision coordinating all components."""
        # Apply coordination costs
        self._apply_coordination_costs()
        
        # Collective decision-making (placeholder for neural networks)
        active_components = [i for i, c in enumerate(self.components) if c.active]
        num_active = len(active_components)
        
        if num_active == 0:
            # No active components - emergency protocols
            return self._emergency_decision()
        
        # Resource allocation decision
        self.resource_allocation = self._allocate_resources(observation, active_components)
        
        # Component actions decision
        component_actions = []
        for i in range(self.collective_capabilities.num_components):
            if i in active_components:
                action_prefs = self._decide_component_action(observation, i)
                component_actions.append(action_prefs)
            else:
                # Inactive component
                component_actions.append(np.zeros(self.capabilities.action_size))
        
        # Collective strategy decision
        collective_strategy = self._decide_collective_strategy(observation, num_active)
        
        decision = {
            'component_actions': component_actions,
            'resource_allocation': self.resource_allocation,
            'collective_strategy': collective_strategy,
            'active_components': active_components,
            'coordination_cost': self.last_coordination_cost,
            'coordination_efficiency': self.coordination_efficiency
        }
        
        # Store in collective memory
        self.collective_memory.append({
            'observation': observation,
            'decision': decision,
            'step': self.step_count,
            'collective_energy': self.energy
        })
        
        # Maintain memory capacity
        if len(self.collective_memory) > self.capabilities.memory_capacity:
            self.collective_memory = self.collective_memory[-self.capabilities.memory_capacity:]
        
        self.decision_history.append(decision)
        return decision
    
    def _apply_coordination_costs(self):
        """Apply internal coordination costs for collective decision-making."""
        active_components = sum(1 for c in self.components if c.active)
        base_cost = self.collective_capabilities.coordination_cost * active_components
        
        # Efficiency reduces costs over time
        efficiency_factor = min(self.coordination_efficiency, 2.0)
        total_cost = base_cost / efficiency_factor
        
        self.energy -= total_cost
        self.last_coordination_cost = total_cost
        
        # Improve coordination efficiency with experience
        self.coordination_efficiency += 0.005
    
    def _allocate_resources(self, observation: np.ndarray, active_components: List[int]) -> np.ndarray:
        """Allocate resources among active components."""
        if not active_components:
            return np.zeros(self.collective_capabilities.num_components)
        
        # Simple allocation strategy (can be replaced with neural networks)
        allocation = np.zeros(self.collective_capabilities.num_components)
        
        for i in active_components:
            component = self.components[i]
            # Allocate based on component energy and experience
            base_allocation = 1.0 / len(active_components)
            energy_factor = max(0.5, component.energy / 100.0)
            experience_factor = 1.0 + (component.experience * 0.01)
            
            allocation[i] = base_allocation * energy_factor * experience_factor
        
        # Normalize allocation
        total_allocation = np.sum(allocation)
        if total_allocation > 0:
            allocation = allocation / total_allocation
            
        return allocation
    
    def _decide_component_action(self, observation: np.ndarray, component_id: int) -> np.ndarray:
        """Decide action preferences for specific component."""
        component = self.components[component_id]
        resource_share = self.resource_allocation[component_id]
        
        # Base action preferences
        action_prefs = np.random.random(self.capabilities.action_size)
        
        # Modify based on resource allocation
        action_prefs *= (1.0 + resource_share)
        
        # Modify based on component energy
        if component.energy < 30:
            # Conservative actions for low-energy components
            action_prefs[0] *= 2.0  # Prefer conservative action
        
        # Consider expendability
        if (self.collective_capabilities.expendability and 
            component.energy < 20 and 
            len([c for c in self.components if c.active]) > 1):
            # This component might be expendable
            action_prefs[-1] *= 3.0  # Increase preference for expendable action
        
        return action_prefs / np.sum(action_prefs)  # Normalize
    
    def _decide_collective_strategy(self, observation: np.ndarray, num_active: int) -> str:
        """Decide overall collective strategy."""
        if self.energy < 50:
            return "conservation"
        elif num_active < self.collective_capabilities.num_components // 2:
            return "regeneration"
        elif self.coordination_efficiency > 1.5:
            return "coordination"
        else:
            return "exploration"
    
    def _emergency_decision(self) -> Dict[str, Any]:
        """Emergency decision when no components are active."""
        return {
            'component_actions': [np.zeros(self.capabilities.action_size) 
                                for _ in range(self.collective_capabilities.num_components)],
            'resource_allocation': np.zeros(self.collective_capabilities.num_components),
            'collective_strategy': 'emergency',
            'active_components': [],
            'coordination_cost': 0.0,
            'coordination_efficiency': self.coordination_efficiency
        }
    
    def act(self, decision: Dict[str, Any]) -> List[int]:
        """Convert collective decision into environment actions for each component."""
        actions = []
        
        for i, action_prefs in enumerate(decision['component_actions']):
            if i < len(self.components) and self.components[i].active:
                # Sample action based on preferences
                if np.sum(action_prefs) > 0:
                    action = np.random.choice(self.capabilities.action_size, p=action_prefs)
                else:
                    action = 0  # Default action
                
                # Apply energy cost
                action_cost = 0.3 * (1.0 + action / self.capabilities.action_size)
                self.components[i].energy -= action_cost
                
                # Check for component expenditure
                if (action == self.capabilities.action_size - 1 and  # Expendable action
                    self.collective_capabilities.expendability):
                    self._expend_component(i)
                    
                actions.append(action)
            else:
                actions.append(0)  # Inactive component default action
        
        return actions
    
    def _expend_component(self, component_id: int):
        """Expend (sacrifice) a component for collective benefit."""
        component = self.components[component_id]
        if not component.active:
            return
        
        # Gain energy from expenditure
        energy_gain = component.energy + 10.0  # Base expenditure benefit
        self.energy += energy_gain
        
        # Deactivate component
        component.active = False
        component.energy = 0.0
        
        # Record expenditure for analysis
        if not hasattr(self, 'expenditure_history'):
            self.expenditure_history = []
        
        self.expenditure_history.append({
            'component_id': component_id,
            'energy_gained': energy_gain,
            'step': self.step_count,
            'collective_energy_after': self.energy
        })
    
    def update(self, reward: float, next_observation: Any) -> Dict[str, float]:
        """Update collective agent state based on environment feedback."""
        if isinstance(reward, (list, tuple)):
            # Rewards for each component
            component_rewards = reward
            total_reward = sum(component_rewards)
        else:
            # Single reward - distribute among active components
            active_components = [i for i, c in enumerate(self.components) if c.active]
            if active_components:
                component_reward = reward / len(active_components)
                component_rewards = [component_reward if i in active_components else 0 
                                   for i in range(len(self.components))]
                total_reward = reward
            else:
                component_rewards = [0] * len(self.components)
                total_reward = 0
        
        # Update component states
        for i, comp_reward in enumerate(component_rewards):
            if i < len(self.components) and self.components[i].active:
                self.components[i].energy += comp_reward * 0.5
                self.components[i].experience += 1
        
        # Update collective energy
        self.energy += total_reward * 0.3  # Collective gets portion of rewards
        
        # Consider regeneration of inactive components
        self._consider_regeneration()
        
        # Calculate performance metrics
        active_count = sum(1 for c in self.components if c.active)
        avg_component_energy = np.mean([c.energy for c in self.components if c.active]) if active_count > 0 else 0
        
        return {
            'collective_reward': total_reward,
            'active_components': active_count,
            'avg_component_energy': avg_component_energy,
            'coordination_efficiency': self.coordination_efficiency,
            'resource_distribution_variance': np.var(self.resource_allocation),
            'expenditures_made': len(getattr(self, 'expenditure_history', [])),
            'memory_utilization': len(self.collective_memory) / self.capabilities.memory_capacity
        }
    
    def _consider_regeneration(self):
        """Consider regenerating inactive components."""
        inactive_components = [i for i, c in enumerate(self.components) if not c.active]
        
        if (inactive_components and 
            self.energy > self.collective_capabilities.regeneration_cost and
            sum(1 for c in self.components if c.active) < 2):  # Need minimum active components
            
            # Regenerate one component
            component_id = inactive_components[0]
            self.energy -= self.collective_capabilities.regeneration_cost
            
            self.components[component_id].active = True
            self.components[component_id].energy = 50.0  # Regenerate with partial energy
    
    def get_analysis_data(self) -> Dict[str, Any]:
        """Get data for analyzing collective agent behavior."""
        active_components = [c for c in self.components if c.active]
        
        return {
            'agent_type': 'collective',
            'num_components': self.collective_capabilities.num_components,
            'active_components': len(active_components),
            'coordination_efficiency': self.coordination_efficiency,
            'resource_allocation': self.resource_allocation.tolist(),
            'component_states': [c.to_dict() for c in self.components],
            'expenditure_history': getattr(self, 'expenditure_history', []),
            'collective_memory_size': len(self.collective_memory),
            'performance_metrics': {
                'total_reward': self.total_reward,
                'collective_energy': self.energy,
                'steps_survived': self.step_count,
                'coordination_costs_paid': sum(d.get('coordination_cost', 0) for d in self.decision_history),
                'efficiency': self.total_reward / max(self.step_count, 1)
            }
        }