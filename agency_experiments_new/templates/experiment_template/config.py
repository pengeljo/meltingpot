"""
Configuration template for agency experiments.

Copy this file and modify for your specific experiment.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from shared.evolution.base import EvolutionConfig
from shared.agents.base import AgentCapabilities


@dataclass
class ExperimentConfig:
    """Configuration for [experiment name]."""
    
    # Experiment metadata
    experiment_name: str = "[experiment_name]"
    experiment_version: str = "1.0"
    description: str = "[Brief experiment description]"
    
    # Environment settings
    environment_name: str = "commons_harvest__open_0"  # Or other MeltingPot environment
    max_steps_per_episode: int = 100
    num_episodes_per_evaluation: int = 3
    
    # Population settings
    total_population_size: int = 20
    # Add agent type ratios as needed
    
    # Evolution settings
    evolution: EvolutionConfig = EvolutionConfig(
        population_size=20,
        elite_size=4,
        tournament_size=3,
        mutation_rate=0.15,
        mutation_strength=0.1,
        crossover_rate=0.7,
        generations=50,
        diversity_bonus=0.1
    )
    
    # Agent capabilities - customize for your agent types
    agent_capabilities: AgentCapabilities = AgentCapabilities(
        observation_size=64,
        action_size=7,
        memory_capacity=50,
        energy_budget=100.0,
        learning_rate=0.001
    )
    
    # Experimental scenarios
    scenarios: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize scenarios after creation."""
        if self.scenarios is None:
            self.scenarios = {
                'baseline': {
                    'description': 'Baseline scenario',
                    'generations': 25,
                    'focus': 'baseline'
                },
                'main_experiment': {
                    'description': 'Main experimental condition',
                    'generations': 50,
                    'focus': 'main'
                },
                # Add more scenarios as needed
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
            'evolution': self.evolution.to_dict(),
            'agent_capabilities': self.agent_capabilities.to_dict(),
            'scenarios': self.scenarios
        }


# Default configurations
DEFAULT_CONFIGS = {
    'quick_test': ExperimentConfig(
        experiment_name="[experiment_name]_quick_test",
        evolution=EvolutionConfig(
            population_size=6,
            generations=5,
            elite_size=2
        ),
        max_steps_per_episode=20,
        num_episodes_per_evaluation=1
    ),
    
    'full_experiment': ExperimentConfig(
        experiment_name="[experiment_name]_full",
        evolution=EvolutionConfig(
            population_size=20,
            generations=100,
            elite_size=4
        ),
        max_steps_per_episode=100,
        num_episodes_per_evaluation=5
    )
}


def get_config(config_name: str = 'default') -> ExperimentConfig:
    """Get configuration by name."""
    if config_name == 'default':
        return ExperimentConfig()
    elif config_name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[config_name]
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