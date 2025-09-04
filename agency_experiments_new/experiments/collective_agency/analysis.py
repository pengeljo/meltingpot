"""
Analysis module for collective agency experiment.

Analyzes emergence of novel behaviors in collective vs individual agents.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import json
import os
import sys

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from shared.analysis import ExperimentAnalyzer
from config import CollectiveAgencyConfig


class CollectiveAgencyAnalyzer(ExperimentAnalyzer):
    """Specialized analyzer for collective agency experiments."""
    
    def __init__(self, config: CollectiveAgencyConfig):
        super().__init__(config)
        self.config = config
        
    def analyze_training_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of training results."""
        analysis = super().analyze_training_results(results)
        
        # Add collective agency specific analysis
        analysis.update({
            'agent_type_comparison': self._analyze_agent_type_performance(results),
            'emergence_patterns': self._detect_emergence_patterns(results),
            'collective_behaviors': self._analyze_collective_behaviors(results),
            'cooperation_metrics': self._analyze_cooperation(results),
            'novel_strategies': self._detect_novel_strategies(results),
            'collective_coherence_evolution': self._analyze_coherence_evolution(results),
            'resource_allocation_patterns': self._analyze_resource_allocation(results)
        })
        
        # Generate key findings
        analysis['key_findings'] = self._generate_key_findings(analysis)
        
        return analysis
    
    def _analyze_agent_type_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance between individual and collective agents."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        individual_fitness = []
        collective_fitness = []
        generations = []
        
        for gen_stat in generation_stats:
            generations.append(gen_stat['generation'])
            
            if 'individual_mean_fitness' in gen_stat:
                individual_fitness.append(gen_stat['individual_mean_fitness'])
            else:
                individual_fitness.append(np.nan)
            
            if 'collective_mean_fitness' in gen_stat:
                collective_fitness.append(gen_stat['collective_mean_fitness'])
            else:
                collective_fitness.append(np.nan)
        
        # Remove NaN values for comparison
        valid_individual = [f for f in individual_fitness if not np.isnan(f)]
        valid_collective = [f for f in collective_fitness if not np.isnan(f)]
        
        comparison = {
            'individual_fitness_trend': individual_fitness,
            'collective_fitness_trend': collective_fitness,
            'generations': generations
        }
        
        if valid_individual and valid_collective:
            comparison.update({
                'individual_mean': np.mean(valid_individual),
                'collective_mean': np.mean(valid_collective),
                'individual_final': valid_individual[-1] if valid_individual else 0,
                'collective_final': valid_collective[-1] if valid_collective else 0,
                'performance_advantage': 'collective' if np.mean(valid_collective) > np.mean(valid_individual) else 'individual'
            })
            
            # Statistical test
            if HAS_SCIPY and len(valid_individual) > 1 and len(valid_collective) > 1:
                statistic, p_value = stats.ttest_ind(valid_collective, valid_individual)
                comparison.update({
                    'statistical_significance': p_value < 0.05,
                    'p_value': p_value,
                    't_statistic': statistic
                })
        
        return comparison
    
    def _detect_emergence_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emergence of novel behaviors over generations."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        emergence_indicators = {
            'fitness_acceleration': [],
            'diversity_changes': [],
            'cooperation_emergence': [],
            'strategy_novelty': []
        }
        
        # Track changes over time
        for i, gen_stat in enumerate(generation_stats):
            if i == 0:
                continue
            
            prev_stat = generation_stats[i-1]
            
            # Fitness acceleration (sudden improvements)
            fitness_change = gen_stat.get('mean_fitness', 0) - prev_stat.get('mean_fitness', 0)
            emergence_indicators['fitness_acceleration'].append(fitness_change)
            
            # Diversity changes
            diversity_change = gen_stat.get('fitness_std', 0) - prev_stat.get('fitness_std', 0)
            emergence_indicators['diversity_changes'].append(diversity_change)
        
        # Detect emergence events
        emergence_events = []
        
        # Fitness acceleration events (sudden jumps > 2 std deviations)
        if emergence_indicators['fitness_acceleration']:
            fitness_changes = emergence_indicators['fitness_acceleration']
            threshold = np.mean(fitness_changes) + 2 * np.std(fitness_changes)
            
            for i, change in enumerate(fitness_changes):
                if change > threshold:
                    emergence_events.append({
                        'type': 'fitness_acceleration',
                        'generation': i + 1,
                        'magnitude': change,
                        'description': f'Sudden fitness improvement of {change:.4f}'
                    })
        
        return {
            'emergence_indicators': emergence_indicators,
            'emergence_events': emergence_events,
            'total_emergence_events': len(emergence_events)
        }
    
    def _analyze_collective_behaviors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze specific collective behaviors and patterns."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        collective_metrics = {
            'coordination_efficiency': [],
            'component_diversity': [],
            'collective_coherence': [],
            'generations': []
        }
        
        for gen_stat in generation_stats:
            collective_metrics['generations'].append(gen_stat['generation'])
            collective_metrics['coordination_efficiency'].append(
                gen_stat.get('mean_coordination_efficiency', 0)
            )
            collective_metrics['component_diversity'].append(
                gen_stat.get('mean_component_diversity', 0)
            )
            collective_metrics['collective_coherence'].append(
                gen_stat.get('mean_collective_coherence', 0)
            )
        
        # Analyze trends
        behaviors = {}
        
        if collective_metrics['coordination_efficiency']:
            coord_efficiency = collective_metrics['coordination_efficiency']
            behaviors['coordination_trend'] = 'improving' if coord_efficiency[-1] > coord_efficiency[0] else 'declining'
            behaviors['peak_coordination'] = max(coord_efficiency)
            behaviors['final_coordination'] = coord_efficiency[-1]
        
        if collective_metrics['component_diversity']:
            diversity = collective_metrics['component_diversity']
            behaviors['diversity_trend'] = 'increasing' if diversity[-1] > diversity[0] else 'decreasing'
            behaviors['peak_diversity'] = max(diversity)
            behaviors['final_diversity'] = diversity[-1]
        
        if collective_metrics['collective_coherence']:
            coherence = collective_metrics['collective_coherence']
            behaviors['coherence_trend'] = 'increasing' if coherence[-1] > coherence[0] else 'decreasing'
            behaviors['peak_coherence'] = max(coherence)
            behaviors['final_coherence'] = coherence[-1]
        
        return {
            'metrics_over_time': collective_metrics,
            'behavioral_trends': behaviors
        }
    
    def _analyze_cooperation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cooperation patterns between agents."""
        # This would require episode-level data, which might not be available in current results
        # For now, we'll analyze based on fitness differences and collective performance
        
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return {}
        
        cooperation_metrics = {}
        
        # Look for signs of cooperative vs competitive behavior
        for gen_stat in generation_stats:
            individual_count = gen_stat.get('individual_agents', 0)
            collective_count = gen_stat.get('collective_agents', 0)
            
            if individual_count > 0 and collective_count > 0:
                # Mixed population - analyze interaction
                individual_fitness = gen_stat.get('individual_mean_fitness', 0)
                collective_fitness = gen_stat.get('collective_mean_fitness', 0)
                
                # If both types maintain good fitness, suggests cooperation
                # If one dominates, suggests competition
                fitness_ratio = collective_fitness / individual_fitness if individual_fitness > 0 else 0
                
                cooperation_metrics[f"generation_{gen_stat['generation']}_cooperation"] = {
                    'fitness_ratio': fitness_ratio,
                    'interaction_type': 'cooperative' if 0.8 <= fitness_ratio <= 1.2 else 'competitive'
                }
        
        return cooperation_metrics
    
    def _detect_novel_strategies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potentially novel strategies that emerged."""
        novel_strategies = []
        
        best_agents_history = results.get('best_agents_history', [])
        
        if not best_agents_history:
            return novel_strategies
        
        # Look for agents that significantly outperformed others
        generation_stats = results.get('generation_stats', [])
        
        for i, gen_stat in enumerate(generation_stats):
            mean_fitness = gen_stat.get('mean_fitness', 0)
            max_fitness = gen_stat.get('max_fitness', 0)
            
            # If best agent significantly outperforms average
            if max_fitness > mean_fitness * 1.5 and max_fitness > 0.1:
                novel_strategies.append({
                    'generation': gen_stat['generation'],
                    'fitness_advantage': max_fitness - mean_fitness,
                    'relative_advantage': (max_fitness - mean_fitness) / mean_fitness if mean_fitness > 0 else 0,
                    'description': f'High-performing strategy emerged at generation {gen_stat["generation"]}'
                })
        
        return novel_strategies
    
    def _analyze_coherence_evolution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how collective coherence evolved over time."""
        generation_stats = results.get('generation_stats', [])
        
        coherence_data = []
        for gen_stat in generation_stats:
            if 'mean_collective_coherence' in gen_stat:
                coherence_data.append(gen_stat['mean_collective_coherence'])
        
        if not coherence_data:
            return {}
        
        # Analyze coherence patterns
        coherence_evolution = {
            'initial_coherence': coherence_data[0],
            'final_coherence': coherence_data[-1],
            'peak_coherence': max(coherence_data),
            'coherence_improvement': coherence_data[-1] - coherence_data[0],
            'coherence_stability': 1.0 / (1.0 + np.std(coherence_data))  # Higher is more stable
        }
        
        # Detect coherence phases
        phases = []
        if len(coherence_data) > 10:
            # Simple phase detection: look for periods of low/high coherence
            threshold = np.mean(coherence_data)
            current_phase = 'high' if coherence_data[0] > threshold else 'low'
            phase_start = 0
            
            for i, coherence in enumerate(coherence_data[1:], 1):
                new_phase = 'high' if coherence > threshold else 'low'
                if new_phase != current_phase:
                    phases.append({
                        'phase': current_phase,
                        'start_generation': phase_start,
                        'end_generation': i-1,
                        'duration': i - phase_start
                    })
                    current_phase = new_phase
                    phase_start = i
            
            # Add final phase
            phases.append({
                'phase': current_phase,
                'start_generation': phase_start,
                'end_generation': len(coherence_data)-1,
                'duration': len(coherence_data) - phase_start
            })
        
        coherence_evolution['coherence_phases'] = phases
        
        return coherence_evolution
    
    def _analyze_resource_allocation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze resource allocation patterns in collective agents."""
        # This would require access to agent decision histories
        # For now, return placeholder analysis
        return {
            'allocation_strategy': 'dynamic',  # Could be 'static', 'dynamic', 'adaptive'
            'resource_efficiency': 0.75,  # Placeholder
            'allocation_fairness': 0.82   # Placeholder
        }
    
    def _generate_key_findings(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate list of key findings from analysis."""
        findings = []
        
        # Agent type performance comparison
        agent_comparison = analysis.get('agent_type_comparison', {})
        if 'performance_advantage' in agent_comparison:
            advantage = agent_comparison['performance_advantage']
            findings.append(f"{advantage.capitalize()} agents showed superior performance overall")
            
            if agent_comparison.get('statistical_significance', False):
                findings.append("Performance difference between agent types was statistically significant")
        
        # Emergence patterns
        emergence = analysis.get('emergence_patterns', {})
        if emergence.get('total_emergence_events', 0) > 0:
            findings.append(f"Detected {emergence['total_emergence_events']} emergence events during evolution")
        
        # Novel strategies
        novel_strategies = analysis.get('novel_strategies', [])
        if novel_strategies:
            findings.append(f"Identified {len(novel_strategies)} potentially novel strategies")
        
        # Collective behavior trends
        collective_behaviors = analysis.get('collective_behaviors', {})
        behavioral_trends = collective_behaviors.get('behavioral_trends', {})
        
        if 'coordination_trend' in behavioral_trends:
            trend = behavioral_trends['coordination_trend']
            findings.append(f"Collective coordination efficiency showed {trend} trend")
        
        if 'coherence_trend' in behavioral_trends:
            trend = behavioral_trends['coherence_trend']
            findings.append(f"Collective coherence was {trend} over time")
        
        return findings
    
    def generate_fitness_plots(self, results: Dict[str, Any], output_dir: str):
        """Generate fitness evolution plots."""
        super().generate_fitness_plots(results, output_dir)
        
        # Additional collective agency specific plots
        self._plot_agent_type_comparison(results, output_dir)
        self._plot_collective_metrics(results, output_dir)
        self._plot_emergence_events(results, output_dir)
    
    def _plot_agent_type_comparison(self, results: Dict[str, Any], output_dir: str):
        """Plot comparison between individual and collective agent performance."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return
        
        generations = [stat['generation'] for stat in generation_stats]
        individual_fitness = [stat.get('individual_mean_fitness', np.nan) for stat in generation_stats]
        collective_fitness = [stat.get('collective_mean_fitness', np.nan) for stat in generation_stats]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        if not all(np.isnan(individual_fitness)):
            plt.plot(generations, individual_fitness, 'b-', label='Individual Agents', linewidth=2)
        if not all(np.isnan(collective_fitness)):
            plt.plot(generations, collective_fitness, 'r-', label='Collective Agents', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Mean Fitness')
        plt.title('Agent Type Performance Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Fitness distribution comparison (final generation)
        plt.subplot(2, 2, 2)
        final_individual = [f for f in individual_fitness if not np.isnan(f)][-5:] if individual_fitness else []
        final_collective = [f for f in collective_fitness if not np.isnan(f)][-5:] if collective_fitness else []
        
        if final_individual and final_collective:
            plt.boxplot([final_individual, final_collective], labels=['Individual', 'Collective'])
            plt.ylabel('Fitness')
            plt.title('Final Performance Distribution')
        
        # Agent count over time
        plt.subplot(2, 2, 3)
        individual_counts = [stat.get('individual_agents', 0) for stat in generation_stats]
        collective_counts = [stat.get('collective_agents', 0) for stat in generation_stats]
        
        plt.stackplot(generations, individual_counts, collective_counts, 
                     labels=['Individual', 'Collective'], alpha=0.7)
        plt.xlabel('Generation')
        plt.ylabel('Agent Count')
        plt.title('Population Composition')
        plt.legend()
        
        # Performance advantage over time
        plt.subplot(2, 2, 4)
        advantage = []
        for i, c in zip(individual_fitness, collective_fitness):
            if not np.isnan(i) and not np.isnan(c) and i > 0:
                advantage.append(c / i)
            else:
                advantage.append(np.nan)
        
        valid_advantage = [a for a in advantage if not np.isnan(a)]
        valid_generations = [g for g, a in zip(generations, advantage) if not np.isnan(a)]
        
        if valid_advantage:
            plt.plot(valid_generations, valid_advantage, 'g-', linewidth=2)
            plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Equal Performance')
            plt.xlabel('Generation')
            plt.ylabel('Collective/Individual Fitness Ratio')
            plt.title('Performance Advantage')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/agent_type_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_collective_metrics(self, results: Dict[str, Any], output_dir: str):
        """Plot collective-specific metrics over time."""
        generation_stats = results.get('generation_stats', [])
        
        if not generation_stats:
            return
        
        generations = [stat['generation'] for stat in generation_stats]
        coordination = [stat.get('mean_coordination_efficiency', 0) for stat in generation_stats]
        diversity = [stat.get('mean_component_diversity', 0) for stat in generation_stats]
        coherence = [stat.get('mean_collective_coherence', 0) for stat in generation_stats]
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(generations, coordination, 'b-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Coordination Efficiency')
        plt.title('Collective Coordination Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(generations, diversity, 'g-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Component Diversity')
        plt.title('Component Diversity Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(generations, coherence, 'r-', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Collective Coherence')
        plt.title('Collective Coherence Evolution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/collective_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emergence_events(self, results: Dict[str, Any], output_dir: str):
        """Plot emergence events over time."""
        emergence = results.get('emergence_patterns', {})
        emergence_events = emergence.get('emergence_events', [])
        
        if not emergence_events:
            return
        
        generation_stats = results.get('generation_stats', [])
        generations = [stat['generation'] for stat in generation_stats]
        fitness = [stat.get('mean_fitness', 0) for stat in generation_stats]
        
        plt.figure(figsize=(12, 6))
        
        # Plot fitness over time
        plt.plot(generations, fitness, 'b-', linewidth=2, label='Mean Fitness')
        
        # Mark emergence events
        for event in emergence_events:
            plt.axvline(x=event['generation'], color='r', linestyle='--', alpha=0.7)
            plt.text(event['generation'], max(fitness) * 0.9, 
                    event['type'], rotation=90, ha='center', va='bottom')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Emergence Events During Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/emergence_events.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_analyzer(config: CollectiveAgencyConfig) -> CollectiveAgencyAnalyzer:
    """Factory function to create analyzer."""
    return CollectiveAgencyAnalyzer(config)