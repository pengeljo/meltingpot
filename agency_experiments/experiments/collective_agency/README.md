# Collective Agency Experiment

## Overview

This experiment explores how neural networks can discover novel collective strategies through evolutionary training in multi-agent environments. The focus is on identifying emergent behaviors and moral concepts that arise when agents can transcend individual limitations through genuine collective agency.

## Research Questions

1. **What novel collective strategies emerge** when neural networks evolve collective agents without conceptual bias?
2. **How do collective agents perform** compared to individual agents in competitive scenarios?
3. **What moral/social behaviors develop** that would be impossible for individual agents?
4. **How do coordination costs** affect the evolution of collective strategies?
5. **What resource allocation patterns** emerge within collective agents?

## Experimental Design

### Agent Types
- **Individual Agents**: Single-body agents with standard decision-making
- **Collective Agents**: Multi-component unified agents with internal coordination costs
- **Mixed Populations**: Combinations of individual and collective agents

### Environment
- **MeltingPot Commons Harvest**: Multi-agent resource collection with social dilemmas
- **Variable scenarios**: Different resource scarcity and competition levels

### Neural Architecture
- **Emergent Collective Networks**: Neural networks that discover collective strategies
- **Minimal conceptual bias**: Networks learn what strategies work, not pre-defined behaviors
- **Multi-head attention**: For component coordination (if beneficial)
- **Resource allocation**: Networks learn internal resource distribution

### Evolution Parameters
- **Population size**: 20 agents (mix of individual/collective)
- **Generations**: 50-100 depending on convergence
- **Selection**: Tournament selection with diversity preservation
- **Fitness**: Pure environmental success (rewards + survival)

## Key Metrics

### Performance Metrics
- **Fitness evolution**: How agent performance changes over generations
- **Survival rates**: Which agent types survive longer
- **Resource efficiency**: Reward per energy unit spent
- **Environmental impact**: Effect on other agents' performance

### Collective Behavior Metrics
- **Coordination frequency**: How often collective agents coordinate
- **Component sacrifice**: When/how collective agents sacrifice components
- **Resource allocation patterns**: Internal resource distribution strategies
- **Specialization emergence**: Whether component roles develop

### Novel Behavior Detection
- **Strategy consistency**: Repeated patterns in collective decision-making
- **Moral framework emergence**: Systematic approaches to trade-offs
- **Social impact patterns**: How collective agents affect others
- **Cooperation/competition balance**: Mixed strategy development

## Experimental Scenarios

### Scenario 1: Pure Competition
- **Setup**: Individual vs Individual agents
- **Purpose**: Establish individual agent baseline
- **Duration**: 25 generations

### Scenario 2: Pure Collective
- **Setup**: Collective vs Collective agents  
- **Purpose**: Understand collective-only dynamics
- **Duration**: 25 generations

### Scenario 3: Mixed Competition
- **Setup**: Individual vs Collective agents
- **Purpose**: Direct competition between agent types
- **Duration**: 50 generations

### Scenario 4: Graduated Collective Costs
- **Setup**: Collective agents with varying coordination costs
- **Purpose**: Find optimal coordination cost levels
- **Duration**: 40 generations

### Scenario 5: Resource Scarcity
- **Setup**: Mixed agents in resource-scarce environment
- **Purpose**: Test collective strategies under pressure
- **Duration**: 60 generations

## Expected Outcomes

### Potential Collective Strategies
- **Efficient Sacrifice**: Optimal timing for component expenditure
- **Resource Hoarding**: Collective resource accumulation strategies
- **Territorial Control**: Multi-component area control
- **Information Gathering**: Distributed observation and processing
- **Adaptive Specialization**: Component role evolution

### Novel Moral Concepts
- **Component Expendability Ethics**: When sacrifice is justified
- **Collective vs Individual Welfare**: Resource allocation principles
- **Inter-collective Relations**: How collectives interact with each other
- **Temporal Sacrifice**: Long-term vs short-term collective benefit
- **Resource Justice**: Fair distribution within collectives

## Analysis Methods

### Statistical Analysis
- **Fitness progression analysis**: ANOVA on generation effects
- **Strategy stability testing**: Repeated measures analysis
- **Agent type performance comparison**: T-tests and effect sizes
- **Coordination cost optimization**: Regression analysis

### Behavioral Analysis
- **Decision tree extraction**: Understanding decision patterns
- **Cluster analysis**: Grouping similar strategies
- **Network analysis**: Component interaction patterns
- **Temporal analysis**: Strategy evolution over time

### Emergent Behavior Detection
- **Pattern recognition**: Identifying novel behavioral sequences
- **Outlier detection**: Unusual collective strategies
- **Comparative analysis**: Behaviors impossible for individuals
- **Consistency testing**: Reliable strategy reproduction

## Results Summary

### Generation 1-25: Initial Evolution
- [Results to be filled during experiment]
- Baseline individual performance: 
- Initial collective performance:
- Early strategy patterns:

### Generation 25-50: Strategy Refinement  
- [Results to be filled during experiment]
- Strategy convergence:
- Performance improvements:
- Novel behaviors observed:

### Generation 50-75: Advanced Strategies
- [Results to be filled during experiment]
- Complex collective behaviors:
- Inter-agent interactions:
- Resource allocation patterns:

### Generation 75-100: Final Optimization
- [Results to be filled during experiment]
- Final performance comparison:
- Stable collective strategies:
- Novel moral concepts identified:

## Conclusions

### Key Findings
- [To be filled post-experiment]

### Novel Collective Behaviors Discovered
- [To be filled post-experiment]

### Implications for Agency Theory
- [To be filled post-experiment]

### Future Experiment Directions
- [To be filled post-experiment]

## Files

- `config.py`: Experimental configuration
- `agents.py`: Collective and individual agent implementations
- `environment.py`: MeltingPot environment wrapper
- `training.py`: Evolutionary training system
- `analysis.py`: Analysis and visualization tools
- `run_experiment.py`: Main experiment runner
- `results/`: All experimental data and analysis

## Usage

```bash
# Run full experiment
python run_experiment.py --scenario mixed_competition --generations 50

# Run specific scenario
python run_experiment.py --scenario collective_only --generations 30

# Run with custom settings
python run_experiment.py --scenario mixed_competition --population_size 24 --generations 75 --seed 12345

# Quick test run
python run_experiment.py --config quick_test --scenario mixed_competition
```