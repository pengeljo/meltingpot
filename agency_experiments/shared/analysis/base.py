"""
Base analysis classes for agency experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import os


class ExperimentAnalyzer:
    """Base analyzer for agency experiments."""
    
    def __init__(self, config: Any):
        self.config = config
    
    def analyze_training_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze training results and return analysis dictionary."""
        analysis = {
            'experiment_name': getattr(self.config, 'experiment_name', 'unknown'),
            'total_generations': results.get('generations_completed', 0),
            'best_fitness': results.get('best_fitness_ever', 0.0),
            'convergence_analysis': self._analyze_convergence(results),
            'performance_statistics': self._calculate_performance_stats(results)
        }
        
        return analysis
    
    def _analyze_convergence(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        fitness_values = [stat.get('mean_fitness', 0) for stat in generation_stats]
        
        # Simple convergence metrics
        if len(fitness_values) > 1:
            improvement = fitness_values[-1] - fitness_values[0]
            final_stability = np.std(fitness_values[-5:]) if len(fitness_values) >= 5 else 0
            
            return {
                'total_improvement': improvement,
                'final_stability': final_stability,
                'converged': final_stability < 0.01  # Simple threshold
            }
        
        return {}
    
    def _calculate_performance_stats(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic performance statistics."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        mean_fitness = [stat.get('mean_fitness', 0) for stat in generation_stats]
        max_fitness = [stat.get('max_fitness', 0) for stat in generation_stats]
        
        return {
            'mean_fitness_final': mean_fitness[-1] if mean_fitness else 0,
            'max_fitness_final': max_fitness[-1] if max_fitness else 0,
            'mean_fitness_peak': max(mean_fitness) if mean_fitness else 0,
            'max_fitness_peak': max(max_fitness) if max_fitness else 0
        }
    
    def generate_fitness_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate basic fitness evolution plots."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return
        
        generations = [stat['generation'] for stat in generation_stats]
        mean_fitness = [stat.get('mean_fitness', 0) for stat in generation_stats]
        max_fitness = [stat.get('max_fitness', 0) for stat in generation_stats]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, mean_fitness, 'b-', label='Mean Fitness', linewidth=2)
        plt.plot(generations, max_fitness, 'r-', label='Max Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/fitness_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()