"""
Main experiment runner for collective agency experiment.

Implements neural collective agents vs individual agents in MeltingPot environments.
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any

# Set random seeds for reproducibility
import random
import sys
import os

# Add path for shared components
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import shared components
from shared.utils import setup_logging, save_experiment_config

# Import experiment-specific components
from config import get_config, CollectiveAgencyConfig
from agents import create_agent_factory
from environment import create_environment_wrapper
from training import create_fitness_function
from analysis import create_analyzer


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_experiment_directory(config: CollectiveAgencyConfig, scenario_name: str) -> str:
    """Create directory for experiment results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/{scenario_name}_{timestamp}"
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/raw_data", exist_ok=True)
    os.makedirs(f"{experiment_dir}/analysis", exist_ok=True)
    os.makedirs(f"{experiment_dir}/figures", exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    
    return experiment_dir


def run_scenario(config: CollectiveAgencyConfig, scenario_name: str, 
                experiment_dir: str) -> Dict[str, Any]:
    """Run a specific experimental scenario."""
    print(f"\n=== Running Scenario: {scenario_name} ===")
    
    # Get scenario-specific configuration
    scenario_config = config.get_scenario_config(scenario_name)
    print(f"Scenario focus: {scenario_config['focus']}")
    print(f"Generations: {scenario_config.get('generations', config.evolution.generations)}")
    print(f"Individual ratio: {scenario_config.get('individual_ratio', 0.5):.2f}")
    print(f"Collective ratio: {scenario_config.get('collective_ratio', 0.5):.2f}")
    
    # Create environment wrapper
    env_wrapper = create_environment_wrapper(config, scenario_config)
    
    # Create agent factory
    agent_factory = create_agent_factory(config, scenario_config)
    
    # Create and run training
    training_function = create_fitness_function(env_wrapper, config, scenario_config)
    
    print("Starting evolutionary training...")
    results = training_function(
        agent_factory=agent_factory,
        generations=scenario_config.get('generations', config.evolution.generations)
    )
    
    # Save results
    results_file = f"{experiment_dir}/raw_data/training_results.json"
    with open(results_file, 'w') as f:
        # Convert non-serializable objects for JSON
        serializable_results = {}
        for key, value in results.items():
            if key not in ['best_agent_ever', 'best_agents_history', 'final_population']:
                if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    return results


def analyze_results(results: Dict[str, Any], experiment_dir: str, 
                   config: CollectiveAgencyConfig):
    """Analyze and visualize experimental results."""
    print("\n=== Analyzing Results ===")
    
    analyzer = create_analyzer(config)
    
    # Generate analysis
    analysis = analyzer.analyze_training_results(results)
    
    # Save analysis (handle non-serializable objects)
    analysis_file = f"{experiment_dir}/analysis/training_analysis.json"
    with open(analysis_file, 'w') as f:
        serializable_analysis = {}
        for key, value in analysis.items():
            if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
                serializable_analysis[key] = value
            else:
                serializable_analysis[key] = str(value)
        
        json.dump(serializable_analysis, f, indent=2)
    
    # Generate figures
    figures_dir = f"{experiment_dir}/figures"
    try:
        analyzer.generate_fitness_plots(results, figures_dir)
        print(f"Generated fitness plots")
    except Exception as e:
        print(f"Warning: Could not generate fitness plots: {e}")
    
    try:
        if hasattr(analyzer, 'generate_diversity_plots'):
            analyzer.generate_diversity_plots(results, figures_dir)
            print(f"Generated diversity plots")
    except Exception as e:
        print(f"Warning: Could not generate diversity plots: {e}")
    
    try:
        if hasattr(analyzer, 'generate_behavior_analysis'):
            analyzer.generate_behavior_analysis(results, figures_dir)
            print(f"Generated behavior analysis")
    except Exception as e:
        print(f"Warning: Could not generate behavior analysis: {e}")
    
    print(f"Analysis saved to {experiment_dir}/analysis/")
    print(f"Figures saved to {experiment_dir}/figures/")
    
    return analysis


def main():
    """Main experiment execution."""
    parser = argparse.ArgumentParser(description="Run agency experiment")
    parser.add_argument('--config', default='default', 
                       help='Configuration name')
    parser.add_argument('--scenario', default='mixed_competition',
                       help='Scenario to run')
    parser.add_argument('--generations', type=int,
                       help='Override number of generations')
    parser.add_argument('--population_size', type=int,
                       help='Override population size')
    parser.add_argument('--output_dir', default='results',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    set_random_seeds(args.seed)
    
    # Load configuration
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.generations:
        config.evolution.generations = args.generations
    if args.population_size:
        config.evolution.population_size = args.population_size
    
    # Set up logging
    setup_logging(level='INFO')
    
    print(f"=== Collective Agency Experiment ===")
    print(f"Configuration: {args.config}")
    print(f"Scenario: {args.scenario}")
    print(f"Random seed: {args.seed}")
    print(f"Population size: {config.evolution.population_size}")
    print(f"Generations: {config.evolution.generations}")
    print(f"Environment: {config.environment_name}")
    
    # Create experiment directory
    experiment_dir = create_experiment_directory(config, args.scenario)
    print(f"Experiment directory: {experiment_dir}")
    
    # Save configuration
    save_experiment_config(config, f"{experiment_dir}/config.json")
    
    # Save command line arguments
    with open(f"{experiment_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    try:
        # Run experiment
        results = run_scenario(config, args.scenario, experiment_dir)
        
        # Analyze results
        analysis = analyze_results(results, experiment_dir, config)
        
        # Print summary
        print(f"\n=== Experiment Complete ===")
        print(f"Results saved to: {experiment_dir}")
        print(f"Best fitness achieved: {results.get('best_fitness_ever', 'N/A'):.3f}")
        print(f"Generations completed: {results.get('generations_completed', 'N/A')}")
        
        # Print key findings
        if 'key_findings' in analysis:
            print("\nKey Findings:")
            for finding in analysis['key_findings']:
                print(f"  - {finding}")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error information
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        
        with open(f"{experiment_dir}/error_log.json", 'w') as f:
            json.dump(error_info, f, indent=2)
        
        raise


if __name__ == "__main__":
    main()