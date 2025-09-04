# Migration Guide: Old to New Agency Experiments Structure

This guide explains how to migrate from the old `agency_experiments/` structure to the new systematic framework.

## Overview of Changes

### Old Structure (Problematic)
```
agency_experiments/
├── collective_agency_experiment.py        # Monolithic file
├── identity_fusion_experiment.py          # Separate, inconsistent
├── cross_temporal_agency_experiment.py    # No shared components
├── neural_collective/                     # Experiment-specific only
├── meta-agency/                           # Inconsistent naming
└── test_*.py                              # Ad-hoc testing
```

### New Structure (Systematic)
```
agency_experiments/
├── README.md                              # Framework documentation
├── shared/                                # Reusable components
│   ├── agents/                           # Base agent classes
│   ├── environments/                     # Environment wrappers  
│   ├── evolution/                        # Evolutionary systems
│   ├── neural/                           # Neural network components
│   ├── analysis/                         # Analysis tools
│   └── utils/                            # Utility functions
├── experiments/                           # Individual experiments
│   ├── collective_agency/                # Structured experiment
│   ├── meta_agency/                      # Structured experiment
│   └── [future experiments]/
└── templates/                             # Templates for new experiments
```

## Migration Steps

### Step 1: Understand the New Framework

Read the main `README.md` to understand:
- How shared components work
- Standard experiment structure
- Usage patterns and conventions

### Step 2: Migrate Existing Experiments

For each existing experiment:

1. **Create experiment directory**:
   ```bash
   mkdir -p experiments/[experiment_name]/{results/{raw_data,analysis,figures,models},tests}
   ```

2. **Break down monolithic files**:
   - Extract configuration → `config.py`
   - Extract agent classes → `agents.py`
   - Extract environment setup → `environment.py`
   - Extract training logic → `training.py`
   - Extract analysis → `analysis.py`
   - Create main runner → `run_experiment.py`

3. **Use shared components**:
   ```python
   # Instead of custom implementations
   from shared.agents import BaseAgent, IndividualAgent, CollectiveAgent
   from shared.evolution import EvolutionaryTrainer
   from shared.analysis import ExperimentAnalyzer
   ```

4. **Follow naming conventions**:
   - Use snake_case for directories: `collective_agency`, `meta_agency`
   - Use descriptive file names: `config.py`, `agents.py`, etc.
   - Use clear class names: `CollectiveAgent`, `MetaAgent`

### Step 3: Update Import Statements

#### Old Imports (Don't work anymore)
```python
from neural_collective import NeuralAgentBrain
from collective_agency_experiment import CollectiveAgent
```

#### New Imports
```python
from shared.agents import CollectiveAgent
from shared.neural import BaseNeuralNetwork
from experiments.collective_agency.agents import EmergentCollectiveAgent
```

### Step 4: Migrate Specific Experiments

#### Collective Agency Experiment

**Old**: `collective_agency_experiment.py` (480 lines, monolithic)

**New**: `experiments/collective_agency/` with:
- `config.py` - All experimental configurations
- `agents.py` - Agent implementations using shared base classes
- `environment.py` - MeltingPot environment wrapper
- `training.py` - Evolutionary training system  
- `analysis.py` - Analysis and visualization
- `run_experiment.py` - Main experiment runner

**Migration commands**:
```bash
# Create the new structure
cp -r templates/experiment_template experiments/collective_agency

# Update the template files with collective agency specifics
# (You'll need to customize config.py, agents.py, etc.)
```

#### Neural Collective Components

**Old**: `neural_collective/` (experiment-specific)

**New**: Split between:
- `shared/neural/` - Reusable neural network components
- `shared/agents/collective.py` - Collective agent base class
- `experiments/collective_agency/agents.py` - Experiment-specific agents

## Key Benefits of New Structure

### 1. Modularity
- Shared components reduce code duplication
- Each experiment is self-contained
- Easy to reuse successful patterns

### 2. Consistency
- All experiments follow same structure
- Standard configuration system
- Consistent analysis and visualization

### 3. Maintainability  
- Clear separation of concerns
- Well-organized code is easier to debug
- Standardized testing approach

### 4. Extensibility
- Templates make new experiments easy
- Shared components accelerate development
- Clear patterns for adding functionality

### 5. Reproducibility
- Standardized configuration saving
- Consistent result storage
- Clear experiment documentation

## Common Migration Issues

### Issue 1: Import Errors
**Problem**: Old imports don't work
**Solution**: Update to new import structure, use shared components

### Issue 2: Configuration Complexity
**Problem**: Old hardcoded parameters scattered throughout code
**Solution**: Centralize in `config.py` with dataclass structure

### Issue 3: Results Storage
**Problem**: Inconsistent result saving
**Solution**: Use standardized `results/` structure with subdirectories

### Issue 4: Analysis Inconsistency
**Problem**: Each experiment had different analysis approaches
**Solution**: Use shared analysis tools, extend as needed

## Example: Migrating Collective Agency

### Old Code (collective_agency_experiment.py)
```python
class CollectiveAgencyExperiment:
    def __init__(self, scenario_name="commons_harvest__open_0"):
        self.scenario_name = scenario_name
        # ... lots of hardcoded setup
        
    def run_multiple_episodes(self, num_episodes=5):
        # ... monolithic experiment logic
```

### New Code Structure

**config.py**:
```python
@dataclass
class CollectiveAgencyConfig:
    environment_name: str = "commons_harvest__open_0"
    num_episodes: int = 5
    # ... all configuration centralized
```

**agents.py**:
```python
from shared.agents import CollectiveAgent

class EmergentCollectiveAgent(CollectiveAgent):
    # ... experiment-specific agent logic
```

**run_experiment.py**:
```python
from config import get_config
from agents import create_agent_factory
from training import create_fitness_function

def main():
    config = get_config()
    # ... clean, structured experiment runner
```

## Migration Checklist

- [ ] Create new experiment directory structure
- [ ] Extract configuration to `config.py`
- [ ] Move agent classes to `agents.py`, inherit from shared base classes
- [ ] Create environment wrapper in `environment.py`
- [ ] Implement training system in `training.py`
- [ ] Set up analysis in `analysis.py`
- [ ] Create main runner in `run_experiment.py`
- [ ] Write experiment README with research questions and findings
- [ ] Update imports to use shared components
- [ ] Test experiment with `python run_experiment.py --config quick_test`
- [ ] Run full experiment and verify results
- [ ] Document any novel shared components created

## Getting Help

1. **Look at templates**: `templates/experiment_template/` shows the standard structure
2. **Check existing experiments**: `experiments/collective_agency/` for a complete example
3. **Review shared components**: `shared/` for available reusable code
4. **Read documentation**: Each module has docstrings explaining usage

The new structure is designed to make your research more systematic, reproducible, and extensible. While migration takes some initial effort, it will significantly accelerate future experiment development!