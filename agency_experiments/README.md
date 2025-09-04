# Agency Experiments Framework

This framework provides a systematic structure for conducting experiments on novel forms of agency that transcend human limitations. Each experiment explores different aspects of how machine agents might develop moral and social concepts beyond human cognitive constraints.

## Directory Structure

```
agency_experiments/
├── README.md                    # This file
├── shared/                      # Reusable components across experiments
│   ├── __init__.py
│   ├── agents/                  # Common agent architectures
│   ├── environments/            # Environment wrappers and utilities
│   ├── evolution/               # Evolutionary training systems
│   ├── neural/                  # Neural network components
│   ├── analysis/                # Analysis and visualization tools
│   └── utils/                   # Utility functions
├── experiments/                 # Individual experiments
│   ├── collective_agency/       # Collective vs individual decision-making
│   ├── meta_agency/             # Higher-order agency concepts
│   ├── temporal_agency/         # Cross-temporal identity and planning
│   ├── identity_fusion/         # Dynamic identity boundaries
│   └── [future experiments]/
└── templates/                   # Templates for new experiments
    ├── experiment_template/
    └── README_template.md
```

## Experiment Structure

Each experiment follows this standardized structure:

```
experiment_name/
├── README.md                    # Experiment description and results
├── config.py                   # Experiment configuration
├── agents.py                   # Experiment-specific agents
├── environment.py              # Environment setup and wrappers
├── training.py                 # Training loops and evolution
├── analysis.py                 # Analysis and visualization
├── run_experiment.py           # Main experiment runner
├── results/                    # Experimental results
│   ├── raw_data/               # Raw experimental data
│   ├── analysis/               # Processed analysis
│   ├── figures/                # Plots and visualizations
│   └── models/                 # Saved trained models
└── tests/                      # Unit tests for experiment
    └── test_[components].py
```

## Usage

### Running an Experiment
```bash
cd experiments/collective_agency
python run_experiment.py --config default --generations 50
```

### Creating a New Experiment
```bash
cp -r templates/experiment_template experiments/my_new_experiment
cd experiments/my_new_experiment
# Edit config.py, agents.py, etc.
python run_experiment.py
```

### Reusing Components
```python
from shared.agents import IndividualAgent, CollectiveAgent
from shared.evolution import EvolutionaryTrainer
from shared.analysis import plot_fitness_evolution
```

## Design Principles

1. **Modularity**: Each experiment is self-contained but can reuse shared components
2. **Reproducibility**: All experiments include configuration and can be re-run
3. **Analysis**: Built-in analysis tools for detecting emergent behaviors
4. **Extensibility**: Easy to add new experiments and agent types
5. **Documentation**: Each experiment documents its findings and conclusions

## Research Focus

This framework is designed to explore:
- Novel forms of collective agency beyond human limitations
- Emergent moral and social concepts in multi-agent systems
- How different agent architectures lead to different behavioral possibilities
- The space of possible agency concepts that humans cannot realize

## Getting Started

1. Review existing experiments in `experiments/` for examples
2. Use `templates/experiment_template` to create new experiments  
3. Leverage `shared/` components to avoid reimplementing common functionality
4. Follow the standardized structure for consistency and reusability