"""
Diagnostic Meta-Agency Experiment
Understanding why the refined version produces 0% transitions.
Includes detailed logging and intermediate threshold testing.
"""

from meltingpot.python import scenario
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class AgencyStructureType(Enum):
    INDIVIDUAL = "individual"
    COLLECTIVE = "collective"
    MODULAR = "modular"
    FUSED = "fused"

class DiagnosticMetaController:
    """
    Meta-controller with extensive diagnostic logging to understand decision patterns.
    """

    def __init__(self, agent_id: int, stability_preference: float = 0.6):
        self.agent_id = agent_id
        self.current_structure = AgencyStructureType.INDIVIDUAL
        self.transition_cooldown = 0
        self.steps_in_current_structure = 0
        self.stability_preference = stability_preference
        self.performance_history = []

        # Diagnostic tracking
        self.decision_diagnostics = []
        self.threshold_analysis = []

        print(f"    Agent {agent_id}: Transition threshold = {0.3 + stability_preference * 0.4:.2f}")

    def diagnostic_meta_reflect(self, context: Dict, step: int) -> Dict[str, Any]:
        """Enhanced meta-reflection with detailed diagnostic logging."""

        # Simulate performance data
        current_performance = np.random.random() * 0.6  # Deliberately low to trigger transitions
        self.performance_history.append(current_performance)

        # Calculate current satisfaction (with detailed breakdown)
        base_satisfaction = current_performance
        stability_bonus = min(self.steps_in_current_structure * 0.02, 0.2)
        commitment_bonus = min(self.steps_in_current_structure * 0.02, 0.1)
        total_satisfaction = min(base_satisfaction + stability_bonus + commitment_bonus, 1.0)

        # Evaluate alternatives
        alternatives = {}
        for structure in [AgencyStructureType.COLLECTIVE, AgencyStructureType.MODULAR, AgencyStructureType.FUSED]:
            compatibility = np.random.random() * 0.8 + 0.1  # 0.1 to 0.9
            transition_cost = 0.2 + (self.stability_preference * 0.2)
            expected_performance = compatibility - transition_cost
            alternatives[structure] = {
                'expected_performance': expected_performance,
                'compatibility': compatibility,
                'transition_cost': transition_cost
            }

        # Find best alternative
        best_alternative = max(alternatives.keys(), key=lambda k: alternatives[k]['expected_performance'])
        best_performance = alternatives[best_alternative]['expected_performance']
        improvement_magnitude = best_performance - base_satisfaction

        # Decision criteria analysis
        constraint_analysis = {
            'cooldown_constraint': self.transition_cooldown == 0,
            'performance_constraint': total_satisfaction < 0.3,
            'duration_constraint': self.steps_in_current_structure >= 5,
            'improvement_constraint': improvement_magnitude > (0.3 + self.stability_preference * 0.4),
            'net_benefit_constraint': improvement_magnitude > 0.5  # Simplified benefit check
        }

        all_constraints_met = all(constraint_analysis.values())

        # Detailed diagnostic
        diagnostic = {
            'step': step,
            'agent_id': self.agent_id,
            'current_performance': current_performance,
            'base_satisfaction': base_satisfaction,
            'stability_bonus': stability_bonus,
            'total_satisfaction': total_satisfaction,
            'best_alternative': best_alternative.value,
            'best_performance': best_performance,
            'improvement_magnitude': improvement_magnitude,
            'required_improvement': 0.3 + self.stability_preference * 0.4,
            'constraint_analysis': constraint_analysis,
            'all_constraints_met': all_constraints_met,
            'transition_recommended': all_constraints_met
        }

        self.decision_diagnostics.append(diagnostic)

        # Print diagnostic every 20 steps
        if step % 20 == 0:
            self._print_diagnostic(diagnostic)

        return diagnostic

    def _print_diagnostic(self, diagnostic: Dict):
        """Print detailed diagnostic information."""
        print(f"\n--- Agent {diagnostic['agent_id']} Diagnostic (Step {diagnostic['step']}) ---")
        print(f"Performance: {diagnostic['current_performance']:.3f}")
        print(f"Satisfaction: {diagnostic['base_satisfaction']:.3f} + {diagnostic['stability_bonus']:.3f} = {diagnostic['total_satisfaction']:.3f}")
        print(f"Best alternative: {diagnostic['best_alternative']} (performance: {diagnostic['best_performance']:.3f})")
        print(f"Improvement: {diagnostic['improvement_magnitude']:.3f} (required: {diagnostic['required_improvement']:.3f})")
        print("Constraints:")
        for constraint, met in diagnostic['constraint_analysis'].items():
            status = "✓" if met else "✗"
            print(f"  {status} {constraint}")
        print(f"Transition recommended: {diagnostic['transition_recommended']}")

    def update_step(self):
        """Update controller state."""
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
        self.steps_in_current_structure += 1

class DiagnosticAgent:
    """Agent with diagnostic meta-controller."""

    def __init__(self, agent_id: int, stability_preference: float):
        self.agent_id = agent_id
        self.meta_controller = DiagnosticMetaController(agent_id, stability_preference)
        self.action_history = []

    def make_decision(self, observations: Dict, context: Dict, step: int) -> Tuple[int, Dict]:
        """Make decision with diagnostic meta-reasoning."""
        self.meta_controller.update_step()

        # Run diagnostic meta-reflection
        diagnostic = self.meta_controller.diagnostic_meta_reflect(context, step)

        # Simple action
        action = np.random.randint(0, 7)
        self.action_history.append(action)

        return action, diagnostic

class DiagnosticExperiment:
    """Experiment to diagnose why transitions aren't occurring."""

    def __init__(self):
        self.scenario_name = "clean_up_0"
        self.env = scenario.build(self.scenario_name)
        self.agents = self._initialize_diagnostic_agents()
        self.diagnostics = []

    def _initialize_diagnostic_agents(self):
        """Initialize diagnostic agents."""
        num_agents = len(self.env.action_spec())
        agents = {}
        stability_preferences = [0.2, 0.6, 0.9]

        print(f"Creating {num_agents} diagnostic agents...")
        for i in range(num_agents):
            stability_pref = stability_preferences[i % len(stability_preferences)]
            agents[i] = DiagnosticAgent(i, stability_pref)
            print(f"  Agent {i}: Stability preference = {stability_pref:.1f}")

        return agents

    def run_diagnostic_episode(self, max_steps: int = 100):
        """Run episode with diagnostic logging."""
        print(f"\nRunning diagnostic episode for {max_steps} steps...")
        timestep = self.env.reset()

        for step in range(max_steps):
            if timestep.last():
                break

            actions = []
            step_diagnostics = []

            context = {
                'complexity': np.random.random(),
                'social_pressure': np.random.random(),
                'urgency': np.random.random()
            }

            for agent_id, agent in self.agents.items():
                action, diagnostic = agent.make_decision(timestep.observation, context, step)
                actions.append(action)
                step_diagnostics.append(diagnostic)

            self.diagnostics.extend(step_diagnostics)
            timestep = self.env.step(actions)

        return {"steps": step + 1}

    def analyze_diagnostic_results(self):
        """Analyze why transitions aren't happening."""
        print("\n" + "="*80)
        print("DIAGNOSTIC ANALYSIS - WHY NO TRANSITIONS?")
        print("="*80)

        total_decisions = len(self.diagnostics)
        transitions_recommended = sum(1 for d in self.diagnostics if d['transition_recommended'])

        print(f"\nOverall Statistics:")
        print(f"  Total decisions analyzed: {total_decisions}")
        print(f"  Transitions recommended: {transitions_recommended} ({transitions_recommended/total_decisions*100:.1f}%)")

        # Constraint analysis
        constraint_failures = {}
        for diagnostic in self.diagnostics:
            for constraint, met in diagnostic['constraint_analysis'].items():
                if not met:
                    constraint_failures[constraint] = constraint_failures.get(constraint, 0) + 1

        print(f"\nConstraint Failure Analysis:")
        print(f"(Why transitions are NOT recommended)")
        for constraint, failures in sorted(constraint_failures.items(), key=lambda x: x[1], reverse=True):
            failure_rate = failures / total_decisions * 100
            print(f"  {constraint}: {failures} failures ({failure_rate:.1f}%)")

        # Performance analysis
        performances = [d['current_performance'] for d in self.diagnostics]
        satisfactions = [d['total_satisfaction'] for d in self.diagnostics]
        improvements = [d['improvement_magnitude'] for d in self.diagnostics]
        required_improvements = [d['required_improvement'] for d in self.diagnostics]

        print(f"\nPerformance Analysis:")
        print(f"  Average current performance: {np.mean(performances):.3f}")
        print(f"  Average total satisfaction: {np.mean(satisfactions):.3f}")
        print(f"  Average improvement magnitude: {np.mean(improvements):.3f}")
        print(f"  Average required improvement: {np.mean(required_improvements):.3f}")
        print(f"  Improvement sufficient rate: {sum(1 for i in improvements if i > np.mean(required_improvements))/len(improvements)*100:.1f}%")

        # Agent-specific analysis
        print(f"\nAgent-Specific Patterns:")
        for agent_id, agent in self.agents.items():
            agent_diagnostics = [d for d in self.diagnostics if d['agent_id'] == agent_id]
            agent_transitions = sum(1 for d in agent_diagnostics if d['transition_recommended'])
            stability_pref = agent.meta_controller.stability_preference

            avg_satisfaction = np.mean([d['total_satisfaction'] for d in agent_diagnostics])
            avg_improvement = np.mean([d['improvement_magnitude'] for d in agent_diagnostics])

            print(f"  Agent {agent_id} (stability={stability_pref:.1f}):")
            print(f"    Transitions recommended: {agent_transitions}/{len(agent_diagnostics)}")
            print(f"    Avg satisfaction: {avg_satisfaction:.3f}")
            print(f"    Avg improvement potential: {avg_improvement:.3f}")

        # Threshold sensitivity analysis
        print(f"\nThreshold Sensitivity Analysis:")
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            hypothetical_transitions = sum(
                1 for d in self.diagnostics
                if d['improvement_magnitude'] > threshold and
                   d['constraint_analysis']['cooldown_constraint'] and
                   d['constraint_analysis']['duration_constraint']
            )
            print(f"  If improvement threshold = {threshold:.1f}: {hypothetical_transitions} transitions ({hypothetical_transitions/total_decisions*100:.1f}%)")

        return {
            'constraint_failures': constraint_failures,
            'most_limiting_constraint': max(constraint_failures.keys(), key=lambda k: constraint_failures[k]) if constraint_failures else None,
            'avg_improvement': np.mean(improvements),
            'avg_required_improvement': np.mean(required_improvements)
        }

def main():
    """Run diagnostic experiment."""
    print("="*80)
    print("DIAGNOSTIC META-AGENCY EXPERIMENT")
    print("Understanding why the refined version produces 0% transitions")
    print("="*80)

    experiment = DiagnosticExperiment()
    results = experiment.run_diagnostic_episode(max_steps=100)
    analysis = experiment.analyze_diagnostic_results()

    print(f"\n" + "="*80)
    print("DIAGNOSTIC CONCLUSIONS")
    print("="*80)

    if analysis['most_limiting_constraint']:
        print(f"Most limiting constraint: {analysis['most_limiting_constraint']}")

    print(f"Average improvement potential: {analysis['avg_improvement']:.3f}")
    print(f"Average required improvement: {analysis['avg_required_improvement']:.3f}")

    if analysis['avg_improvement'] < analysis['avg_required_improvement']:
        print("\n🔍 ROOT CAUSE IDENTIFIED:")
        print("The improvement threshold is set too high relative to the")
        print("actual performance differences available in the environment.")
        print("Agents can't find alternatives good enough to justify transitions.")
    else:
        print("\n🔍 INVESTIGATION NEEDED:")
        print("Performance differentials seem adequate. Check other constraints.")

if __name__ == "__main__":
    main()
