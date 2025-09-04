"""
Environment wrapper for collective agency experiment.

Handles MeltingPot integration and multi-agent coordination.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from meltingpot.python import scenario
from config import CollectiveAgencyConfig


class MeltingPotWrapper:
    """Wrapper for MeltingPot environments with multi-agent support."""
    
    def __init__(self, config: CollectiveAgencyConfig, scenario_config: Dict[str, Any]):
        self.config = config
        self.scenario_config = scenario_config
        self.max_steps = config.max_steps_per_episode
        self.step_count = 0
        
        # Initialize MeltingPot environment
        self.env = scenario.build(config.environment_name)
        
        # Get number of players from environment specs
        try:
            self.num_players = self.env.num_players
        except AttributeError:
            # Alternative way to get number of players
            if hasattr(self.env, '_env'):
                self.num_players = getattr(self.env._env, 'num_players', 8)
            else:
                # Default for commons_harvest environments
                self.num_players = 8
        
        print(f"Environment initialized: {config.environment_name}")
        print(f"Number of players: {self.num_players}")
        print(f"Max steps per episode: {self.max_steps}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset environment and return initial observations."""
        self.step_count = 0
        timestep = self.env.reset()
        
        # Convert MeltingPot observations to agent format
        observations = self._process_observations(timestep.observation)
        
        return observations
    
    def step(self, actions: List[int]) -> Tuple[Dict[str, np.ndarray], List[float], bool, Dict[str, Any]]:
        """Step environment with agent actions."""
        self.step_count += 1
        
        # Ensure we have the right number of actions
        if len(actions) != self.num_players:
            # Pad or truncate actions to match expected number of players
            if len(actions) < self.num_players:
                # Duplicate last action to fill remaining slots
                actions = actions + [actions[-1]] * (self.num_players - len(actions))
            else:
                actions = actions[:self.num_players]
        
        # Convert to tuple format expected by MeltingPot
        action_tuple = tuple(actions)
        
        # Step the environment
        timestep = self.env.step(action_tuple)
        
        # Process outputs
        observations = self._process_observations(timestep.observation)
        rewards = self._process_rewards(timestep.reward)
        done = timestep.last() or self.step_count >= self.max_steps
        info = {
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'episode_complete': timestep.last()
        }
        
        return observations, rewards, done, info
    
    def _process_observations(self, raw_observations) -> Dict[str, np.ndarray]:
        """Convert MeltingPot observations to standardized format."""
        if isinstance(raw_observations, tuple):
            # Standard MeltingPot format: tuple of observations
            observations = {}
            for i, obs in enumerate(raw_observations):
                observations[f"agent_{i}"] = self._flatten_observation(obs)
        elif isinstance(raw_observations, dict):
            # Dictionary format
            observations = {}
            for key, obs in raw_observations.items():
                observations[key] = self._flatten_observation(obs)
        else:
            # Single observation
            observations = {"agent_0": self._flatten_observation(raw_observations)}
        
        return observations
    
    def _flatten_observation(self, obs) -> np.ndarray:
        """Flatten complex observation structure into array."""
        if isinstance(obs, dict):
            # Flatten dictionary observations
            flattened_parts = []
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    flattened_parts.append(value.flatten())
                elif isinstance(value, (int, float)):
                    flattened_parts.append(np.array([value]))
                elif isinstance(value, list):
                    flattened_parts.append(np.array(value).flatten())
            
            if flattened_parts:
                return np.concatenate(flattened_parts)
            else:
                return np.zeros(64)  # Default observation size
                
        elif isinstance(obs, np.ndarray):
            return obs.flatten()
        elif isinstance(obs, (list, tuple)):
            return np.array(obs).flatten()
        else:
            # Single value
            return np.array([obs])
    
    def _process_rewards(self, raw_rewards) -> List[float]:
        """Convert MeltingPot rewards to list format."""
        if isinstance(raw_rewards, tuple):
            return list(raw_rewards)
        elif isinstance(raw_rewards, list):
            return raw_rewards
        elif isinstance(raw_rewards, (int, float)):
            return [raw_rewards]
        else:
            return [0.0]  # Default reward
    
    def get_observation_space_size(self) -> int:
        """Get the size of flattened observations."""
        # Reset to get sample observation
        timestep = self.env.reset()
        obs = self._process_observations(timestep.observation)
        
        # Use first observation as template
        first_obs_key = list(obs.keys())[0]
        sample_obs = obs[first_obs_key]
        
        return len(sample_obs)
    
    def get_action_space_size(self) -> int:
        """Get the size of action space."""
        # MeltingPot environments typically have discrete action spaces
        # This is environment-specific, but commons_harvest typically uses 7 actions
        action_spec = self.env.action_spec()
        
        if hasattr(action_spec[0], 'num_values'):
            return action_spec[0].num_values
        elif hasattr(action_spec[0], 'maximum'):
            return action_spec[0].maximum + 1
        else:
            # Default for commons_harvest
            return 7


class MultiAgentEnvironmentWrapper:
    """Higher-level wrapper for managing multiple agents in the environment."""
    
    def __init__(self, config: CollectiveAgencyConfig, scenario_config: Dict[str, Any]):
        self.config = config
        self.scenario_config = scenario_config
        self.env_wrapper = MeltingPotWrapper(config, scenario_config)
        
        # Agent management
        self.active_agents = []
        self.agent_observations = {}
        self.agent_rewards = {}
        
    def reset(self, agents: List) -> Dict[str, np.ndarray]:
        """Reset environment and assign agents to positions."""
        self.active_agents = agents
        
        # Reset the underlying environment
        raw_observations = self.env_wrapper.reset()
        
        # Assign observations to agents
        self.agent_observations = {}
        agent_keys = list(raw_observations.keys())
        
        for i, agent in enumerate(self.active_agents):
            if i < len(agent_keys):
                obs_key = agent_keys[i]
                self.agent_observations[agent.agent_id] = raw_observations[obs_key]
            else:
                # Use last available observation for extra agents
                last_obs_key = agent_keys[-1]
                self.agent_observations[agent.agent_id] = raw_observations[last_obs_key]
        
        return self.agent_observations
    
    def step(self, agent_actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict[str, Any]]:
        """Step environment with agent actions."""
        # Convert agent actions to list format
        actions = []
        for agent in self.active_agents:
            if agent.agent_id in agent_actions:
                actions.append(agent_actions[agent.agent_id])
            else:
                # Default action if agent didn't provide one
                actions.append(0)
        
        # Step the environment
        raw_observations, raw_rewards, done, info = self.env_wrapper.step(actions)
        
        # Assign observations and rewards to agents
        self.agent_observations = {}
        self.agent_rewards = {}
        
        obs_keys = list(raw_observations.keys())
        
        for i, agent in enumerate(self.active_agents):
            # Assign observations
            if i < len(obs_keys):
                obs_key = obs_keys[i]
                self.agent_observations[agent.agent_id] = raw_observations[obs_key]
            else:
                last_obs_key = obs_keys[-1]
                self.agent_observations[agent.agent_id] = raw_observations[last_obs_key]
            
            # Assign rewards
            if i < len(raw_rewards):
                self.agent_rewards[agent.agent_id] = raw_rewards[i]
            else:
                # Equal share of remaining reward
                self.agent_rewards[agent.agent_id] = 0.0
        
        return self.agent_observations, self.agent_rewards, done, info
    
    def run_episode(self, agents: List, max_steps: int = None) -> Tuple[Dict[str, float], Dict[str, List], bool]:
        """Run a complete episode with the given agents."""
        if max_steps is None:
            max_steps = self.config.max_steps_per_episode
        
        # Reset environment
        observations = self.reset(agents)
        
        # Initialize agent episode data
        agent_rewards = {agent.agent_id: 0.0 for agent in agents}
        agent_histories = {agent.agent_id: [] for agent in agents}
        
        for step in range(max_steps):
            # Get actions from all agents
            agent_actions = {}
            agent_decisions = {}
            
            for agent in agents:
                obs = observations[agent.agent_id]
                processed_obs = agent.observe(obs)
                agent.last_observation = processed_obs
                
                decision = agent.decide(processed_obs)
                action = agent.act(decision)
                
                agent_actions[agent.agent_id] = action
                agent_decisions[agent.agent_id] = decision
                
                # Store history
                agent_histories[agent.agent_id].append({
                    'step': step,
                    'observation': processed_obs.copy(),
                    'decision': decision.copy(),
                    'action': action
                })
            
            # Step environment
            next_observations, rewards, done, info = self.step(agent_actions)
            
            # Update agents
            for agent in agents:
                reward = rewards[agent.agent_id]
                next_obs = next_observations[agent.agent_id]
                processed_next_obs = agent.observe(next_obs)
                
                agent_metrics = agent.update(reward, processed_next_obs)
                agent_rewards[agent.agent_id] += reward
                
                # Add metrics to history
                agent_histories[agent.agent_id][-1]['reward'] = reward
                agent_histories[agent.agent_id][-1]['metrics'] = agent_metrics
            
            # Update observations for next step
            observations = next_observations
            
            if done:
                break
        
        return agent_rewards, agent_histories, done


def create_environment_wrapper(config: CollectiveAgencyConfig, scenario_config: Dict[str, Any]) -> MultiAgentEnvironmentWrapper:
    """Factory function to create environment wrapper."""
    return MultiAgentEnvironmentWrapper(config, scenario_config)