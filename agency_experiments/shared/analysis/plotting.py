"""
Plotting utilities for agency experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any


def plot_fitness_evolution(generations: List[int], 
                          fitness_data: Dict[str, List[float]], 
                          title: str = "Fitness Evolution",
                          output_path: str = None):
    """
    Plot fitness evolution over generations.
    
    Args:
        generations: List of generation numbers
        fitness_data: Dict mapping series names to fitness values
        title: Plot title
        output_path: Path to save plot (if provided)
    """
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (series_name, fitness_values) in enumerate(fitness_data.items()):
        color = colors[i % len(colors)]
        plt.plot(generations, fitness_values, color=color, label=series_name, linewidth=2)
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()