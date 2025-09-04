"""
Utility functions for neural collective agency experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import json
import os


def plot_training_progress(training_results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot training progress including fitness and diversity evolution."""
    fitness_progression = training_results.get('fitness_progression', [])
    diversity_progression = training_results.get('diversity_progression', [])
    
    if not fitness_progression:
        print("No fitness data to plot")
        return
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot fitness evolution
    generations = range(len(fitness_progression))
    ax1.plot(generations, fitness_progression, 'b-', linewidth=2, label='Best Fitness')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution Across Generations')
    ax1.grid(True)
    ax1.legend()
    
    # Plot diversity evolution
    if diversity_progression:
        ax2.plot(generations, diversity_progression, 'r-', linewidth=2, label='Population Diversity')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Parameter Diversity')
        ax2.set_title('Population Diversity Evolution')
        ax2.grid(True)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No diversity data available', transform=ax2.transAxes, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def plot_collective_behavior_analysis(training_results: Dict[str, Any], save_path: Optional[str] = None):
    """Plot analysis of collective vs individual behavior evolution."""
    collective_analysis = training_results.get('collective_behavior_analysis', {})
    
    collective_scores = collective_analysis.get('collective_score_trend', [])
    individual_scores = collective_analysis.get('individual_score_trend', [])
    
    if not collective_scores or not individual_scores:
        print("No collective behavior data to plot")
        return
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(collective_scores))
    ax.plot(generations, collective_scores, 'g-', linewidth=2, label='Collective Score')
    ax.plot(generations, individual_scores, 'b-', linewidth=2, label='Individual Score')
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('Score')
    ax.set_title('Evolution of Collective vs Individual Behavior')
    ax.grid(True)
    ax.legend()
    
    # Add final strategy annotation
    final_strategy = collective_analysis.get('strategy_evolution', 'unknown')
    ax.text(0.02, 0.98, f'Final Strategy: {final_strategy.capitalize()}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Collective behavior plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()


def save_training_results(training_results: Dict[str, Any], filepath: str):
    """Save training results to JSON file (serializable parts only)."""
    # Create a serializable version of results
    serializable_results = {
        'total_training_time': training_results.get('total_training_time', 0),
        'generations_completed': training_results.get('generations_completed', 0),
        'best_fitness_ever': training_results.get('best_fitness_ever', 0),
        'final_fitness': training_results.get('final_fitness', 0),
        'fitness_improvement': training_results.get('fitness_improvement', 0),
        'fitness_progression': training_results.get('fitness_progression', []),
        'diversity_progression': training_results.get('diversity_progression', []),
        'convergence_analysis': training_results.get('convergence_analysis', {}),
        'collective_behavior_analysis': training_results.get('collective_behavior_analysis', {})
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Training results saved to {filepath}")


def load_training_results(filepath: str) -> Dict[str, Any]:
    """Load training results from JSON file."""
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"Training results loaded from {filepath}")
    return results


def analyze_strategy_diversity(population: List) -> Dict[str, Any]:
    """Analyze the diversity of strategies in a population."""
    if not population:
        return {'error': 'empty_population'}
    
    # Collect fitness and behavior statistics
    fitness_values = [ind.fitness for ind in population]
    collective_scores = [ind.collective_score for ind in population]
    individual_scores = [ind.individual_score for ind in population]
    
    # Calculate diversity metrics
    fitness_std = np.std(fitness_values)
    collective_std = np.std(collective_scores)
    individual_std = np.std(individual_scores)
    
    # Classify strategies
    collective_agents = sum(1 for ind in population if ind.collective_score > ind.individual_score)
    individual_agents = len(population) - collective_agents
    
    return {
        'population_size': len(population),
        'fitness_diversity': {
            'mean': np.mean(fitness_values),
            'std': fitness_std,
            'min': min(fitness_values),
            'max': max(fitness_values),
            'range': max(fitness_values) - min(fitness_values)
        },
        'strategy_distribution': {
            'collective_agents': collective_agents,
            'individual_agents': individual_agents,
            'collective_ratio': collective_agents / len(population)
        },
        'behavior_diversity': {
            'collective_score_std': collective_std,
            'individual_score_std': individual_std,
            'score_correlation': np.corrcoef(collective_scores, individual_scores)[0, 1]
        }
    }


def create_experiment_summary(training_results: Dict[str, Any]) -> str:
    """Create a human-readable summary of experiment results."""
    summary_lines = []
    
    # Basic training info
    summary_lines.append("=== NEURAL COLLECTIVE AGENCY EXPERIMENT SUMMARY ===")
    summary_lines.append("")
    
    total_time = training_results.get('total_training_time', 0)
    generations = training_results.get('generations_completed', 0)
    summary_lines.append(f"Training Duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    summary_lines.append(f"Generations Completed: {generations}")
    summary_lines.append("")
    
    # Fitness results
    best_fitness = training_results.get('best_fitness_ever', 0)
    final_fitness = training_results.get('final_fitness', 0)
    improvement = training_results.get('fitness_improvement', 0)
    
    summary_lines.append("FITNESS EVOLUTION:")
    summary_lines.append(f"  Best Fitness Achieved: {best_fitness:.3f}")
    summary_lines.append(f"  Final Generation Fitness: {final_fitness:.3f}")
    summary_lines.append(f"  Total Improvement: {improvement:.3f}")
    summary_lines.append("")
    
    # Convergence analysis
    convergence = training_results.get('convergence_analysis', {})
    if convergence.get('status') == 'analyzed':
        summary_lines.append("CONVERGENCE ANALYSIS:")
        conv_gen = convergence.get('convergence_generation', 'unknown')
        improvement_rate = convergence.get('improvement_rate', 0)
        summary_lines.append(f"  Convergence Generation: {conv_gen}")
        summary_lines.append(f"  Average Improvement per Generation: {improvement_rate:.4f}")
        summary_lines.append("")
    
    # Collective behavior analysis
    collective_analysis = training_results.get('collective_behavior_analysis', {})
    if collective_analysis:
        summary_lines.append("COLLECTIVE BEHAVIOR ANALYSIS:")
        final_strategy = collective_analysis.get('strategy_evolution', 'unknown')
        collective_ratio = collective_analysis.get('final_collective_ratio', 0)
        collective_improvement = collective_analysis.get('collective_improvement', 0)
        
        summary_lines.append(f"  Final Dominant Strategy: {final_strategy.upper()}")
        summary_lines.append(f"  Collective vs Individual Ratio: {collective_ratio:.3f}")
        summary_lines.append(f"  Collective Score Improvement: {collective_improvement:.3f}")
        summary_lines.append("")
    
    # Research implications
    summary_lines.append("RESEARCH IMPLICATIONS:")
    if collective_analysis.get('strategy_evolution') == 'collective':
        summary_lines.append("  ✓ Evolution favored collective decision-making strategies")
        summary_lines.append("  ✓ Agents learned to transcend individual limitations through cooperation")
    else:
        summary_lines.append("  • Evolution favored individual decision-making strategies")
        summary_lines.append("  • Collective benefits were not sufficient to overcome individual incentives")
    
    if improvement > 0.1:
        summary_lines.append("  ✓ Significant learning occurred - neural networks adapted successfully")
    else:
        summary_lines.append("  • Limited learning observed - may need longer training or parameter tuning")
        
    summary_lines.append("")
    summary_lines.append("This experiment tested the hypothesis that machine agents can explore")
    summary_lines.append("novel forms of collective agency beyond human cognitive limitations.")
    
    return "\n".join(summary_lines)


def compare_random_vs_neural(random_results: Dict[str, Any], neural_results: Dict[str, Any]) -> str:
    """Compare results between random baseline and neural network approaches."""
    comparison_lines = []
    
    comparison_lines.append("=== RANDOM vs NEURAL COLLECTIVE AGENCY COMPARISON ===")
    comparison_lines.append("")
    
    # Performance comparison
    random_collective_ratio = random_results.get('collective_ratio', 0)
    neural_collective_ratio = neural_results.get('collective_behavior_analysis', {}).get('final_collective_ratio', 0)
    
    comparison_lines.append("COLLECTIVE BEHAVIOR:")
    comparison_lines.append(f"  Random Baseline:  {random_collective_ratio:.3f} collective ratio")
    comparison_lines.append(f"  Neural Networks: {neural_collective_ratio:.3f} collective ratio")
    
    if neural_collective_ratio > random_collective_ratio:
        improvement = neural_collective_ratio - random_collective_ratio
        comparison_lines.append(f"  ✓ Neural networks improved collective behavior by {improvement:.3f}")
    else:
        decline = random_collective_ratio - neural_collective_ratio
        comparison_lines.append(f"  • Neural networks reduced collective behavior by {decline:.3f}")
    
    comparison_lines.append("")
    comparison_lines.append("LEARNING CAPABILITIES:")
    neural_improvement = neural_results.get('fitness_improvement', 0)
    if neural_improvement > 0:
        comparison_lines.append(f"  ✓ Neural networks demonstrated learning (fitness improved by {neural_improvement:.3f})")
        comparison_lines.append("  ✓ Agents adapted their strategies over generations")
    else:
        comparison_lines.append("  • Limited learning observed in neural networks")
        comparison_lines.append("  • Consider adjusting hyperparameters or training duration")
    
    return "\n".join(comparison_lines)