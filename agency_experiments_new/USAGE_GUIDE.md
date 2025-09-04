# Agency Experiments Framework Usage Guide

This guide shows how to use the systematic agency experiments framework for conducting research on novel forms of agency.

## Quick Start

### 1. Create a New Experiment

```bash
# Copy the template
cp -r templates/experiment_template experiments/my_new_experiment
cd experiments/my_new_experiment

# Customize the template files
# Edit: README.md, config.py, agents.py, environment.py, training.py, analysis.py
```

### 2. Run a Quick Test

```bash
# Test your experiment with minimal settings
python run_experiment.py --config quick_test --scenario baseline
```

### 3. Run Full Experiment

```bash
# Run with full settings
python run_experiment.py --config full_experiment --scenario main_experiment
```

### 4. Analyze Results

```bash
# Analyze the latest results
python analysis.py --results_dir results/main_experiment_20231204_143022/
```

## Framework Components

### Shared Components (`shared/`)

These are reusable across all experiments:

#### Agents (`shared/agents/`)
```python
from shared.agents import BaseAgent, IndividualAgent, CollectiveAgent

# Base class for all agents
class MyCustomAgent(BaseAgent):
    def _get_agent_type(self):
        return AgentType.INDIVIDUAL
    
    def observe(self, raw_observation):
        # Process observation
        return processed_obs
    
    def decide(self, observation, context=None):
        # Make decision
        return decision_dict
    
    def act(self, decision):
        # Convert decision to action
        return action
    
    def update(self, reward, next_observation):
        # Update agent state
        return metrics_dict
```

#### Evolution (`shared/evolution/`)
```python
from shared.evolution import EvolutionaryTrainer, Population, EvolutionConfig

# Configure evolution
config = EvolutionConfig(
    population_size=20,
    generations=50,
    mutation_rate=0.1
)

# Create trainer
trainer = EvolutionaryTrainer(config)

# Run evolution
results = trainer.train(agent_factory, fitness_function)
```

#### Neural Networks (`shared/neural/`)
```python
from shared.neural import BaseNeuralNetwork, MultiOutputNetwork

# Create custom neural network
class MyNetwork(BaseNeuralNetwork):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(output_size)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.output_layer(x)
```

### Experiment Structure

Each experiment follows this standard structure:

```
my_experiment/
├── README.md              # Research questions, methodology, results
├── config.py              # All experimental configurations
├── agents.py              # Experiment-specific agent implementations  
├── environment.py         # Environment setup and wrappers
├── training.py            # Training loops and fitness functions
├── analysis.py            # Analysis and visualization tools
├── run_experiment.py      # Main experiment runner
├── results/               # All experimental results
│   ├── raw_data/         # Raw training data  
│   ├── analysis/         # Processed analysis
│   ├── figures/          # Generated plots
│   └── models/           # Saved trained agents
└── tests/                # Unit tests
```

## Creating Agents

### Individual Agents

```python
# agents.py
from shared.agents import IndividualAgent, AgentCapabilities

class MyIndividualAgent(IndividualAgent):
    def __init__(self, agent_id: str):
        capabilities = AgentCapabilities(
            observation_size=64,
            action_size=7,
            energy_budget=100.0
        )
        super().__init__(agent_id, capabilities)
    
    def decide(self, observation, context=None):
        # Your decision-making logic
        action_preferences = self.neural_network(observation)
        
        return {
            'action_preferences': action_preferences,
            'confidence': np.max(action_preferences),
            'strategy': 'individual'
        }
```

### Collective Agents

```python
# agents.py
from shared.agents import CollectiveAgent, CollectiveCapabilities

class MyCollectiveAgent(CollectiveAgent):
    def __init__(self, agent_id: str):
        capabilities = CollectiveCapabilities(
            observation_size=64,
            action_size=7,
            energy_budget=150.0,
            num_components=3,
            coordination_cost=0.1
        )
        super().__init__(agent_id, capabilities)
    
    def decide(self, observation, context=None):
        # Collective decision-making with coordination costs
        self._apply_coordination_costs()
        
        # Your collective logic here
        component_actions = self.collective_network(observation)
        
        return {
            'component_actions': component_actions,
            'coordination_strategy': strategy,
            'resource_allocation': allocation
        }
```

## Configuration System

### Basic Configuration

```python
# config.py
from dataclasses import dataclass
from shared.evolution.base import EvolutionConfig

@dataclass
class MyExperimentConfig:
    experiment_name: str = "my_experiment"
    environment_name: str = "commons_harvest__open_0"
    
    evolution: EvolutionConfig = EvolutionConfig(
        population_size=20,
        generations=50,
        mutation_rate=0.15
    )
    
    scenarios: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        self.scenarios = {
            'baseline': {
                'description': 'Baseline condition',
                'generations': 25,
                'population_ratio': {'individual': 0.8, 'collective': 0.2}
            },
            'competition': {
                'description': 'Direct competition',
                'generations': 50,
                'population_ratio': {'individual': 0.5, 'collective': 0.5}
            }
        }
```

### Multiple Configurations

```python
# config.py
DEFAULT_CONFIGS = {
    'quick_test': MyExperimentConfig(
        evolution=EvolutionConfig(population_size=6, generations=5),
        max_steps_per_episode=20
    ),
    'full_experiment': MyExperimentConfig(
        evolution=EvolutionConfig(population_size=20, generations=100),
        max_steps_per_episode=100
    )
}

def get_config(name='default'):
    if name in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[name]
    return MyExperimentConfig()
```

## Training System

### Basic Training Loop

```python
# training.py
from shared.evolution import EvolutionaryTrainer

def create_fitness_function(env_wrapper, config):
    def fitness_function(agents):
        fitness_scores = []
        
        for agent in agents:
            # Run agent in environment
            total_reward = 0
            for episode in range(config.num_episodes_per_evaluation):
                episode_reward = run_single_episode(agent, env_wrapper)
                total_reward += episode_reward
            
            fitness = total_reward / config.num_episodes_per_evaluation
            fitness_scores.append(fitness)
            
        return fitness_scores
    
    return fitness_function

def run_single_episode(agent, env_wrapper):
    observation = env_wrapper.reset()
    total_reward = 0
    
    for step in range(env_wrapper.max_steps):
        action, _ = agent.step(observation)
        observation, reward, done, _ = env_wrapper.step(action)
        total_reward += reward
        
        if done:
            break
    
    return total_reward
```

### Multi-Agent Training

```python
# training.py
def create_multi_agent_fitness_function(env_wrapper, config):
    def fitness_function(agents):
        # Group agents for multi-agent episodes
        episodes_per_group = config.num_episodes_per_evaluation
        fitness_scores = [0.0] * len(agents)
        
        for episode in range(episodes_per_group):
            # Run all agents in same environment
            observations = env_wrapper.reset()
            
            for step in range(env_wrapper.max_steps):
                actions = []
                for agent in agents:
                    action, _ = agent.step(observations[agent.agent_id])
                    actions.append(action)
                
                observations, rewards, done, _ = env_wrapper.step(actions)
                
                # Update fitness scores
                for i, reward in enumerate(rewards):
                    fitness_scores[i] += reward
                
                if done:
                    break
        
        # Average across episodes
        return [score / episodes_per_group for score in fitness_scores]
    
    return fitness_function
```

## Analysis System

### Basic Analysis

```python
# analysis.py
from shared.analysis import ExperimentAnalyzer
import matplotlib.pyplot as plt

class MyExperimentAnalyzer(ExperimentAnalyzer):
    def analyze_training_results(self, results):
        analysis = super().analyze_training_results(results)
        
        # Add experiment-specific analysis
        analysis['novel_behaviors'] = self.detect_novel_behaviors(results)
        analysis['agent_interactions'] = self.analyze_interactions(results)
        
        return analysis
    
    def detect_novel_behaviors(self, results):
        # Your novel behavior detection logic
        behaviors = []
        
        # Example: Detect emergent cooperation
        if self.detect_cooperation_patterns(results):
            behaviors.append('emergent_cooperation')
        
        return behaviors
    
    def generate_custom_plots(self, results, output_dir):
        # Generate experiment-specific visualizations
        self.plot_behavior_evolution(results, f"{output_dir}/behavior_evolution.png")
        self.plot_agent_interactions(results, f"{output_dir}/interactions.png")
```

### Behavior Detection

```python
# analysis.py
def detect_emergent_behaviors(self, agent_histories):
    """Detect novel behaviors that emerge during training."""
    behaviors = {}
    
    for agent_id, history in agent_histories.items():
        agent_behaviors = []
        
        # Look for pattern changes
        early_decisions = history[:len(history)//4]  # First quarter
        late_decisions = history[len(history)*3//4:]  # Last quarter
        
        early_pattern = self.extract_decision_pattern(early_decisions)
        late_pattern = self.extract_decision_pattern(late_decisions)
        
        if self.patterns_significantly_different(early_pattern, late_pattern):
            agent_behaviors.append('strategy_evolution')
        
        # Look for coordination behaviors (if collective agent)
        if hasattr(agent, 'components'):
            coordination_frequency = self.measure_coordination(history)
            if coordination_frequency > 0.7:
                agent_behaviors.append('high_coordination')
            elif coordination_frequency < 0.3:
                agent_behaviors.append('minimal_coordination')
        
        behaviors[agent_id] = agent_behaviors
    
    return behaviors
```

## Environment Integration

### MeltingPot Integration

```python
# environment.py
from meltingpot.python import scenario

class MeltingPotWrapper:
    def __init__(self, scenario_name, max_steps=100):
        self.env = scenario.build(scenario_name)
        self.max_steps = max_steps
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        timestep = self.env.reset()
        return self.process_observations(timestep.observation)
    
    def step(self, actions):
        self.step_count += 1
        
        # Convert action format if needed
        if isinstance(actions, dict):
            actions = list(actions.values())
        
        timestep = self.env.step(tuple(actions))
        
        observations = self.process_observations(timestep.observation)
        rewards = self.process_rewards(timestep.reward)
        done = timestep.last() or self.step_count >= self.max_steps
        
        return observations, rewards, done, {}
    
    def process_observations(self, raw_observations):
        # Convert MeltingPot observations to agent format
        if isinstance(raw_observations, tuple):
            return {i: obs for i, obs in enumerate(raw_observations)}
        return raw_observations
    
    def process_rewards(self, raw_rewards):
        # Convert MeltingPot rewards to agent format
        if isinstance(raw_rewards, tuple):
            return list(raw_rewards)
        return raw_rewards
```

## Running Experiments

### Command Line Usage

```bash
# Basic usage
python run_experiment.py

# Specify configuration
python run_experiment.py --config full_experiment

# Run specific scenario
python run_experiment.py --scenario competition --generations 75

# Override settings
python run_experiment.py --population_size 30 --mutation_rate 0.2

# Set random seed for reproducibility
python run_experiment.py --seed 12345
```

### Programmatic Usage

```python
# Custom experiment runner
from config import get_config
from run_experiment import run_scenario

# Load configuration
config = get_config('full_experiment')

# Modify as needed
config.evolution.population_size = 25
config.evolution.generations = 60

# Run experiment
results = run_scenario(config, 'competition', 'results/custom_run')

# Analyze results
from analysis import MyExperimentAnalyzer
analyzer = MyExperimentAnalyzer(config)
analysis = analyzer.analyze_training_results(results)

print("Novel behaviors detected:", analysis['novel_behaviors'])
```

## Best Practices

### 1. Configuration Management
- Use dataclasses for type safety
- Provide multiple configuration presets
- Save configuration with every experiment run
- Use descriptive parameter names

### 2. Result Organization
- Use timestamped directories for runs
- Separate raw data from analysis
- Save both data and visualizations
- Include experiment metadata

### 3. Code Organization
- Inherit from shared base classes when possible
- Keep experiment-specific logic in experiment directory
- Use clear, descriptive naming
- Document research questions and findings

### 4. Analysis and Visualization
- Generate plots automatically
- Look for unexpected patterns
- Compare across different runs
- Document novel behaviors discovered

### 5. Reproducibility
- Set random seeds
- Save complete configuration
- Document environment versions
- Include clear usage instructions

This framework is designed to accelerate your agency research while maintaining scientific rigor and reproducibility!