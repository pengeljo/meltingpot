"""
Configuration for collective agency experiment.

Explores emergent behaviors in genuine collective agents vs individual agents.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.evolution.base import EvolutionConfig
from shared.agents.base import AgentCapabilities


@dataclass
class CollectiveAgencyConfig:
    """Configuration for collective agency experiment."""
    
    # Experiment metadata
    experiment_name: str = "collective_agency"
    experiment_version: str = "1.0"
    description: str = "Emergent behaviors in genuine collective agents with neural learning"
    
    # Environment settings
    environment_name: str = "commons_harvest__open_0"  # Or other MeltingPot environment
    max_steps_per_episode: int = 100
    num_episodes_per_evaluation: int = 3
    
    # Population settings
    total_population_size: int = 20
    individual_agent_ratio: float = 0.6
    collective_agent_ratio: float = 0.4
    
    # Collective agent settings
    collective_components_per_agent: int = 3
    coordination_cost: float = 0.1
    internal_energy_sharing: bool = True
    
    # Evolution settings
    evolution: EvolutionConfig = field(default_factory=lambda: EvolutionConfig(
        population_size=20,
        elite_size=4,
        tournament_size=3,
        mutation_rate=0.15,
        mutation_strength=0.1,
        crossover_rate=0.7,
        generations=50,
        diversity_bonus=0.1
    ))
    
    # Agent capabilities - customize for your agent types
    individual_capabilities: AgentCapabilities = field(default_factory=lambda: AgentCapabilities(
        observation_size=64,
        action_size=7,
        memory_capacity=50,
        energy_budget=100.0,
        learning_rate=0.001
    ))
    
    collective_capabilities: 'CollectiveCapabilities' = None  # Will be set in __post_init__
    
    # Experimental scenarios
    scenarios: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize scenarios and capabilities after creation."""
        # Set up collective capabilities
        from shared.agents.collective import CollectiveCapabilities
        self.collective_capabilities = CollectiveCapabilities(
            observation_size=self.individual_capabilities.observation_size,
            action_size=self.individual_capabilities.action_size,
            memory_capacity=self.individual_capabilities.memory_capacity,
            energy_budget=self.individual_capabilities.energy_budget * 1.5,
            learning_rate=self.individual_capabilities.learning_rate,
            num_components=self.collective_components_per_agent,
            coordination_cost=self.coordination_cost
        )
        
        if not self.scenarios:
            self.scenarios = {
                'individual_only': {
                    'description': 'Only individual agents (baseline)',
                    'generations': 30,
                    'focus': 'individual_baseline',
                    'individual_ratio': 1.0,
                    'collective_ratio': 0.0
                },
                'collective_only': {
                    'description': 'Only collective agents',
                    'generations': 30,
                    'focus': 'collective_only',
                    'individual_ratio': 0.0,
                    'collective_ratio': 1.0
                },
                'mixed_competition': {
                    'description': 'Mixed individual and collective agents',
                    'generations': 50,
                    'focus': 'mixed_competition',
                    'individual_ratio': self.individual_agent_ratio,
                    'collective_ratio': self.collective_agent_ratio
                },
                'collective_evolution': {
                    'description': 'Long-term collective agent evolution',
                    'generations': 100,
                    'focus': 'collective_evolution',
                    'individual_ratio': 0.2,
                    'collective_ratio': 0.8
                }
            }
    
    def get_scenario_config(self, scenario_name: str) -> Dict[str, Any]:
        """Get configuration for specific scenario."""
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name].copy()
        
        # Override base config with scenario-specific settings
        if 'generations' in scenario:
            scenario['evolution_config'] = self.evolution
            scenario['evolution_config'].generations = scenario['generations']
        
        return scenario
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'experiment_name': self.experiment_name,
            'experiment_version': self.experiment_version,
            'description': self.description,
            'environment_name': self.environment_name,
            'max_steps_per_episode': self.max_steps_per_episode,
            'num_episodes_per_evaluation': self.num_episodes_per_evaluation,
            'total_population_size': self.total_population_size,
            'individual_agent_ratio': self.individual_agent_ratio,
            'collective_agent_ratio': self.collective_agent_ratio,
            'collective_components_per_agent': self.collective_components_per_agent,
            'coordination_cost': self.coordination_cost,
            'evolution': self.evolution.to_dict(),
            'individual_capabilities': self.individual_capabilities.to_dict(),
            'collective_capabilities': self.collective_capabilities.to_dict() if self.collective_capabilities else None,
            'scenarios': self.scenarios
        }


# Default configurations
DEFAULT_CONFIGS = {
    'quick_test': lambda: CollectiveAgencyConfig(
        experiment_name="collective_agency_quick_test",
        evolution=EvolutionConfig(
            population_size=6,
            generations=5,
            elite_size=2
        ),
        max_steps_per_episode=20,
        num_episodes_per_evaluation=1,
        collective_components_per_agent=2
    ),
    
    'full_experiment': lambda: CollectiveAgencyConfig(
        experiment_name="collective_agency_full",
        evolution=EvolutionConfig(
            population_size=20,
            generations=100,
            elite_size=4
        ),
        max_steps_per_episode=100,
        num_episodes_per_evaluation=5
    )
}


def get_config(config_name: str = 'default') -> CollectiveAgencyConfig:
    """Get configuration by name."""
    if config_name == 'default':
        return CollectiveAgencyConfig()
    elif config_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[config_name]()  # Call the lambda function
    else:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(DEFAULT_CONFIGS.keys())}")


if __name__ == "__main__":
    # Test configuration
    config = get_config('default')
    print(f"Configuration test: {config.experiment_name}")
    
    # Test scenarios
    print("Available scenarios:")
    for scenario_name in config.scenarios:
        print(f"  {scenario_name}: {config.scenarios[scenario_name]['description']}")
    
    # Test serialization
    config_dict = config.to_dict()
    print(f"Serialization test passed")