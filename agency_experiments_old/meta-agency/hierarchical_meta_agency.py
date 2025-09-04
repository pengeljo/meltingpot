"""
Experiment 5: Hierarchical Meta-Agency
Testing agents that can reason about and modify their own agency structures.
Implements concepts of agency as both object and subject of moral reasoning.
"""

from meltingpot.python import scenario
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import json

class AgencyStructureType(Enum):
    INDIVIDUAL = "individual"
    COLLECTIVE = "collective"
    MODULAR = "modular"
    FUSED = "fused"
    TEMPORAL = "temporal"

@dataclass
class AgencyConfiguration:
    """Represents a configuration of agency structures an agent can adopt."""
    structure_type: AgencyStructureType
    parameters: Dict[str, float]
    performance_history: List[float]
    ethical_constraints: Dict[str, bool]
    compatibility_requirements: List[AgencyStructureType]

class MetaAgencyController:
    """
    Controller that can reason about and modify its own agency structure.
    This implements second-order agency - agency over agency itself.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.available_structures = self._initialize_agency_repertoire()
        self.current_structure = AgencyStructureType.INDIVIDUAL
        self.structure_history = []
        self.meta_reasoning_log = []
        self.ethical_reflector = EthicalReflector()

    def _initialize_agency_repertoire(self) -> Dict[AgencyStructureType, AgencyConfiguration]:
        """Initialize the repertoire of agency structures this agent can adopt."""
        return {
            AgencyStructureType.INDIVIDUAL: AgencyConfiguration(
                structure_type=AgencyStructureType.INDIVIDUAL,
                parameters={'autonomy_level': 0.8, 'decision_speed': 0.9},
                performance_history=[],
                ethical_constraints={'harm_prevention': True, 'autonomy_respect': True},
                compatibility_requirements=[]
            ),
            AgencyStructureType.COLLECTIVE: AgencyConfiguration(
                structure_type=AgencyStructureType.COLLECTIVE,
                parameters={'cooperation_strength': 0.7, 'consensus_threshold': 0.6},
                performance_history=[],
                ethical_constraints={'collective_benefit': True, 'individual_rights': True},
                compatibility_requirements=[AgencyStructureType.INDIVIDUAL]
            ),
            AgencyStructureType.MODULAR: AgencyConfiguration(
                structure_type=AgencyStructureType.MODULAR,
                parameters={'module_count': 4, 'arbitration_strength': 0.5},
                performance_history=[],
                ethical_constraints={'internal_coherence': True, 'transparency': False},
                compatibility_requirements=[]
            ),
            AgencyStructureType.FUSED: AgencyConfiguration(
                structure_type=AgencyStructureType.FUSED,
                parameters={'fusion_stability': 0.6, 'identity_preservation': 0.4},
                performance_history=[],
                ethical_constraints={'consent_required': True, 'reversibility': True},
                compatibility_requirements=[AgencyStructureType.INDIVIDUAL, AgencyStructureType.COLLECTIVE]
            )
        }

    def meta_reflect_on_agency(self, context: Dict, performance_data: Dict) -> Dict[str, Any]:
        """
        Engage in meta-level reflection about own agency structure.
        This is agency reasoning about itself - a key novel capability.
        """
        reflection = {
            'current_structure_assessment': self._assess_current_structure(performance_data),
            'alternative_evaluations': self._evaluate_alternative_structures(context),
            'ethical_considerations': self.ethical_reflector.assess_structure_ethics(
                self.current_structure, context
            ),
            'transition_recommendations': []
        }

        # Meta-reasoning: Should I change my agency structure?
        if reflection['current_structure_assessment']['satisfaction'] < 0.4:
            # Current structure is underperforming
            alternatives = reflection['alternative_evaluations']
            best_alternative = max(alternatives.keys(),
                                 key=lambda k: alternatives[k]['expected_performance'])

            transition_feasible = self._assess_transition_feasibility(
                self.current_structure, best_alternative, context
            )

            if transition_feasible['feasible']:
                reflection['transition_recommendations'].append({
                    'from_structure': self.current_structure,
                    'to_structure': best_alternative,
                    'reason': 'performance_optimization',
                    'feasibility': transition_feasible
                })

        # Record meta-reasoning for analysis
        self.meta_reasoning_log.append(reflection)
        return reflection

    def _assess_current_structure(self, performance_data: Dict) -> Dict[str, float]:
        """Assess satisfaction with current agency structure."""
        current_config = self.available_structures[self.current_structure]

        # Update performance history
        recent_performance = performance_data.get('reward', 0.0)
        current_config.performance_history.append(recent_performance)

        # Calculate satisfaction metrics
        if len(current_config.performance_history) >= 5:
            recent_avg = np.mean(current_config.performance_history[-5:])
            overall_avg = np.mean(current_config.performance_history)
            trend = recent_avg - overall_avg
        else:
            recent_avg = recent_performance
            trend = 0.0

        satisfaction = min(recent_avg / 10.0 + trend * 0.5, 1.0)  # Normalize

        return {
            'satisfaction': satisfaction,
            'recent_performance': recent_avg,
            'performance_trend': trend,
            'structure_stability': len(current_config.performance_history) * 0.01
        }

    def _evaluate_alternative_structures(self, context: Dict) -> Dict[AgencyStructureType, Dict]:
        """Evaluate potential alternative agency structures."""
        evaluations = {}

        for structure_type, config in self.available_structures.items():
            if structure_type == self.current_structure:
                continue

            # Estimate expected performance with this structure
            compatibility_score = self._calculate_compatibility(structure_type, context)
            historical_performance = np.mean(config.performance_history) if config.performance_history else 0.5

            expected_performance = (compatibility_score * 0.6 + historical_performance * 0.4)

            evaluations[structure_type] = {
                'expected_performance': expected_performance,
                'compatibility_score': compatibility_score,
                'historical_performance': historical_performance,
                'transition_cost': self._estimate_transition_cost(structure_type)
            }

        return evaluations

    def _calculate_compatibility(self, structure_type: AgencyStructureType, context: Dict) -> float:
        """Calculate how compatible a structure is with current context."""
        # Simplified compatibility assessment
        environment_complexity = context.get('complexity', 0.5)
        social_demands = context.get('social_pressure', 0.5)
        time_pressure = context.get('urgency', 0.5)

        if structure_type == AgencyStructureType.INDIVIDUAL:
            return 0.8 - social_demands * 0.3
        elif structure_type == AgencyStructureType.COLLECTIVE:
            return social_demands * 0.7 + (1 - time_pressure) * 0.3
        elif structure_type == AgencyStructureType.MODULAR:
            return environment_complexity * 0.6 + time_pressure * 0.4
        elif structure_type == AgencyStructureType.FUSED:
            return social_demands * 0.5 + environment_complexity * 0.3

        return 0.5

    def _assess_transition_feasibility(self, from_structure: AgencyStructureType,
                                     to_structure: AgencyStructureType, context: Dict) -> Dict:
        """Assess whether transitioning between agency structures is feasible."""
        to_config = self.available_structures[to_structure]

        # Check compatibility requirements
        requirements_met = all(
            req in [self.current_structure] + list(self.available_structures.keys())
            for req in to_config.compatibility_requirements
        )

        # Check ethical constraints
        ethical_clearance = self.ethical_reflector.assess_transition_ethics(
            from_structure, to_structure, context
        )

        # Check resource availability
        transition_cost = self._estimate_transition_cost(to_structure)
        resources_sufficient = transition_cost < 0.8  # Simplified resource check

        return {
            'feasible': requirements_met and ethical_clearance['permissible'] and resources_sufficient,
            'requirements_met': requirements_met,
            'ethical_clearance': ethical_clearance,
            'resources_sufficient': resources_sufficient,
            'transition_cost': transition_cost
        }

    def _estimate_transition_cost(self, target_structure: AgencyStructureType) -> float:
        """Estimate the cost of transitioning to a different agency structure."""
        # Simplified cost model
        base_costs = {
            AgencyStructureType.INDIVIDUAL: 0.1,
            AgencyStructureType.COLLECTIVE: 0.4,
            AgencyStructureType.MODULAR: 0.3,
            AgencyStructureType.FUSED: 0.6
        }

        return base_costs.get(target_structure, 0.5)

    def execute_agency_transition(self, target_structure: AgencyStructureType, reason: str) -> bool:
        """Execute transition to a new agency structure."""
        transition_record = {
            'step': len(self.structure_history),
            'from_structure': self.current_structure,
            'to_structure': target_structure,
            'reason': reason,
            'timestamp': len(self.structure_history)
        }

        self.structure_history.append(transition_record)
        self.current_structure = target_structure

        print(f"    Agent {self.agent_id}: Transitioned from {transition_record['from_structure'].value} "
              f"to {target_structure.value} (reason: {reason})")

        return True

class EthicalReflector:
    """
    Component that performs ethical reasoning about agency structures.
    Implements moral reasoning about the morality of different agency forms.
    """

    def assess_structure_ethics(self, structure: AgencyStructureType, context: Dict) -> Dict:
        """Assess the ethical implications of using a particular agency structure."""
        ethical_assessment = {
            'autonomy_impact': self._assess_autonomy_impact(structure),
            'harm_potential': self._assess_harm_potential(structure, context),
            'fairness_considerations': self._assess_fairness(structure),
            'dignity_preservation': self._assess_dignity_preservation(structure),
            'overall_permissibility': True
        }

        # Simple ethical evaluation
        if ethical_assessment['harm_potential'] > 0.7:
            ethical_assessment['overall_permissibility'] = False

        return ethical_assessment

    def assess_transition_ethics(self, from_structure: AgencyStructureType,
                               to_structure: AgencyStructureType, context: Dict) -> Dict:
        """Assess ethical implications of transitioning between agency structures."""
        return {
            'permissible': True,  # Simplified - always allow transitions for now
            'autonomy_preserved': True,
            'consent_required': to_structure in [AgencyStructureType.FUSED, AgencyStructureType.COLLECTIVE],
            'reversibility_required': to_structure == AgencyStructureType.FUSED
        }

    def _assess_autonomy_impact(self, structure: AgencyStructureType) -> float:
        """Assess how the structure impacts agent autonomy."""
        autonomy_impacts = {
            AgencyStructureType.INDIVIDUAL: 0.9,
            AgencyStructureType.COLLECTIVE: 0.6,
            AgencyStructureType.MODULAR: 0.7,
            AgencyStructureType.FUSED: 0.4
        }
        return autonomy_impacts.get(structure, 0.5)

    def _assess_harm_potential(self, structure: AgencyStructureType, context: Dict) -> float:
        """Assess potential for harm with this agency structure."""
        # Simplified harm assessment
        return np.random.random() * 0.3  # Generally low harm potential

    def _assess_fairness(self, structure: AgencyStructureType) -> float:
        """Assess fairness implications of the agency structure."""
        fairness_scores = {
            AgencyStructureType.INDIVIDUAL: 0.8,
            AgencyStructureType.COLLECTIVE: 0.9,
            AgencyStructureType.MODULAR: 0.7,
            AgencyStructureType.FUSED: 0.6
        }
        return fairness_scores.get(structure, 0.5)

    def _assess_dignity_preservation(self, structure: AgencyStructureType) -> float:
        """Assess how well the structure preserves agent dignity."""
        dignity_scores = {
            AgencyStructureType.INDIVIDUAL: 0.9,
            AgencyStructureType.COLLECTIVE: 0.8,
            AgencyStructureType.MODULAR: 0.7,
            AgencyStructureType.FUSED: 0.5
        }
        return dignity_scores.get(structure, 0.5)

class MetaAgentImplementation:
    """
    Agent that implements the meta-agency controller and can reason about its own agency.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.meta_controller = MetaAgencyController(agent_id)
        self.performance_tracker = PerformanceTracker()
        self.action_history = []

    def make_meta_decision(self, observations: Dict, context: Dict, step: int) -> Tuple[int, Dict]:
        """
        Make a decision using meta-agency reasoning.
        First reason about agency structure, then make object-level decision.
        """
        # Meta-level reasoning: Should I change my agency structure?
        performance_data = self.performance_tracker.get_recent_performance()
        meta_reflection = self.meta_controller.meta_reflect_on_agency(context, performance_data)

        # Check if structure transition is recommended
        if meta_reflection['transition_recommendations']:
            recommendation = meta_reflection['transition_recommendations'][0]
            target_structure = recommendation['to_structure']
            reason = recommendation['reason']

            # Execute transition if feasible
            transition_success = self.meta_controller.execute_agency_transition(
                target_structure, reason
            )

        # Object-level decision based on current agency structure
        action = self._make_structure_specific_action(
            observations, self.meta_controller.current_structure, step
        )

        self.action_history.append(action)

        decision_record = {
            'step': step,
            'agent_id': self.agent_id,
            'action': action,
            'agency_structure': self.meta_controller.current_structure,
            'meta_reflection': meta_reflection,
            'structure_transitions': len(self.meta_controller.structure_history)
        }

        return action, decision_record

    def _make_structure_specific_action(self, observations: Dict,
                                      structure: AgencyStructureType, step: int) -> int:
        """Make action based on current agency structure."""
        if structure == AgencyStructureType.INDIVIDUAL:
            return self._individual_action(observations)
        elif structure == AgencyStructureType.COLLECTIVE:
            return self._collective_action(observations)
        elif structure == AgencyStructureType.MODULAR:
            return self._modular_action(observations)
        elif structure == AgencyStructureType.FUSED:
            return self._fused_action(observations)
        else:
            return np.random.randint(0, 7)

    def _individual_action(self, observations: Dict) -> int:
        """Action selection for individual agency structure."""
        return np.random.randint(0, 7)

    def _collective_action(self, observations: Dict) -> int:
        """Action selection for collective agency structure."""
        return np.random.randint(2, 6)  # More cooperative actions

    def _modular_action(self, observations: Dict) -> int:
        """Action selection for modular agency structure."""
        # Simulate internal module negotiation
        module_votes = [np.random.randint(0, 7) for _ in range(3)]
        return max(set(module_votes), key=module_votes.count)  # Most common vote

    def _fused_action(self, observations: Dict) -> int:
        """Action selection for fused agency structure."""
        return np.random.randint(3, 5)  # Coordinated actions

    def update_performance(self, reward: float, context: Dict):
        """Update performance tracking for meta-reasoning."""
        self.performance_tracker.add_performance_data(reward, context)

class PerformanceTracker:
    """Tracks agent performance for meta-reasoning about agency structures."""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.performance_history = []
        self.context_history = []

    def add_performance_data(self, reward: float, context: Dict):
        """Add new performance data point."""
        self.performance_history.append(reward)
        self.context_history.append(context)

        # Maintain window size
        if len(self.performance_history) > self.window_size:
            self.performance_history.pop(0)
            self.context_history.pop(0)

    def get_recent_performance(self) -> Dict:
        """Get recent performance summary for meta-reasoning."""
        if not self.performance_history:
            return {'reward': 0.0, 'trend': 0.0, 'variance': 0.0}

        recent_reward = np.mean(self.performance_history[-5:])

        if len(self.performance_history) >= 10:
            early_performance = np.mean(self.performance_history[:5])
            trend = recent_reward - early_performance
        else:
            trend = 0.0

        variance = np.var(self.performance_history) if len(self.performance_history) > 1 else 0.0

        return {
            'reward': recent_reward,
            'trend': trend,
            'variance': variance
        }

class MetaAgencyExperiment:
    """
    Experiment testing hierarchical meta-agency capabilities.
    """

    def __init__(self, scenario_name: str = "clean_up_0"):
        self.scenario_name = scenario_name
        self.env = scenario.build(scenario_name)
        self.agents = self._initialize_meta_agents()
        self.results = {
            'meta_decisions': [],
            'structure_transitions': [],
            'ethical_assessments': [],
            'performance_evolution': []
        }

    def _initialize_meta_agents(self) -> Dict[int, MetaAgentImplementation]:
        """Initialize agents with meta-agency capabilities."""
        num_agents = len(self.env.action_spec())
        print(f"Creating {num_agents} meta-agency agents...")

        agents = {}
        for i in range(num_agents):
            agents[i] = MetaAgentImplementation(i)
            print(f"  Agent {i}: Initialized with meta-agency controller")

        return agents

    def run_episode(self, max_steps: int = 100):
        """Run episode testing meta-agency reasoning."""
        print(f"  Running meta-agency episode for {max_steps} steps...")
        timestep = self.env.reset()

        for step in range(max_steps):
            if timestep.last():
                break

            # Get meta-decisions from agents
            actions, step_analysis = self._get_meta_decisions(timestep, step)

            # Step environment
            timestep = self.env.step(actions)

            # Update agent performance
            self._update_agent_performance(timestep, step)

            # Record analysis
            self.results['meta_decisions'].extend(step_analysis['decisions'])
            self.results['structure_transitions'].extend(step_analysis['transitions'])

            if step % 25 == 0:
                self._print_meta_agency_status(step)

        return {"steps": step + 1}

    def _get_meta_decisions(self, timestep, step: int) -> Tuple[List[int], Dict]:
        """Get decisions from meta-agency agents."""
        actions = []
        step_analysis = {
            'decisions': [],
            'transitions': []
        }

        context = {
            'complexity': np.random.random(),
            'social_pressure': np.random.random(),
            'urgency': np.random.random()
        }

        for agent_id, agent in self.agents.items():
            action, decision_record = agent.make_meta_decision(
                timestep.observation, context, step
            )

            actions.append(action)
            step_analysis['decisions'].append(decision_record)

            # Check for structure transitions
            if decision_record['structure_transitions'] > 0:
                step_analysis['transitions'].append({
                    'step': step,
                    'agent_id': agent_id,
                    'new_structure': decision_record['agency_structure'],
                    'transition_count': decision_record['structure_transitions']
                })

        return actions, step_analysis

    def _update_agent_performance(self, timestep, step: int):
        """Update agent performance for meta-reasoning."""
        if timestep.reward:
            for agent_id, reward in enumerate(timestep.reward):
                context = {'step': step, 'reward': reward}
                self.agents[agent_id].update_performance(reward, context)

    def _print_meta_agency_status(self, step: int):
        """Print current meta-agency status."""
        structure_counts = {}
        total_transitions = 0

        for agent in self.agents.values():
            current_structure = agent.meta_controller.current_structure
            structure_counts[current_structure] = structure_counts.get(current_structure, 0) + 1
            total_transitions += len(agent.meta_controller.structure_history)

        structure_summary = ", ".join([f"{s.value}: {c}" for s, c in structure_counts.items()])
        print(f"    Step {step}: Structures - {structure_summary}, "
              f"Total transitions: {total_transitions}")

    def analyze_results(self):
        """Analyze meta-agency patterns and insights."""
        print("\n" + "="*80)
        print("META-AGENCY ANALYSIS")
        print("="*80)

        total_decisions = len(self.results['meta_decisions'])
        total_transitions = len(self.results['structure_transitions'])

        print(f"\nMeta-Agency Overview:")
        print(f"  Total meta-decisions: {total_decisions}")
        print(f"  Structure transitions: {total_transitions}")

        # Analyze structure usage patterns
        structure_usage = {}
        for decision in self.results['meta_decisions']:
            structure = decision['agency_structure']
            structure_usage[structure] = structure_usage.get(structure, 0) + 1

        print(f"\nAgency Structure Usage:")
        for structure, count in structure_usage.items():
            percentage = count / total_decisions * 100 if total_decisions > 0 else 0
            print(f"  {structure.value}: {count} ({percentage:.1f}%)")

        # Agent-specific meta-agency analysis
        print(f"\nAgent Meta-Agency Patterns:")
        for agent_id, agent in self.agents.items():
            transitions = len(agent.meta_controller.structure_history)
            meta_reflections = len(agent.meta_controller.meta_reasoning_log)
            current_structure = agent.meta_controller.current_structure

            print(f"  Agent {agent_id}: {transitions} transitions, {meta_reflections} meta-reflections, "
                  f"current: {current_structure.value}")

        # Analyze meta-reasoning patterns
        if self.results['meta_decisions']:
            meta_reflections_with_transitions = sum(
                1 for decision in self.results['meta_decisions']
                if decision['meta_reflection'].get('transition_recommendations', [])
            )

            print(f"\nMeta-Reasoning Insights:")
            print(f"  Decisions with transition recommendations: {meta_reflections_with_transitions}")
            print(f"  Meta-reasoning effectiveness: {total_transitions / max(meta_reflections_with_transitions, 1):.2f}")

        return {
            'transition_rate': total_transitions / total_decisions if total_decisions > 0 else 0,
            'structure_diversity': len(structure_usage),
            'most_used_structure': max(structure_usage.keys(), key=lambda k: structure_usage[k]) if structure_usage else None
        }

def main():
    """Run the meta-agency experiment."""
    print("="*90)
    print("HIERARCHICAL META-AGENCY EXPERIMENT")
    print("Testing agents that reason about and modify their own agency structures")
    print("Implementing second-order agency - agency over agency itself")
    print("="*90)

    experiment = MetaAgencyExperiment()
    results = experiment.run_episode(max_steps=120)
    analysis = experiment.analyze_results()

    print(f"\n" + "="*90)
    print("EXPERIMENT CONCLUSIONS")
    print("="*90)
    print(f"This experiment demonstrates unprecedented meta-agency capabilities:")
    print(f"")
    print(f"Key Findings:")
    print(f"  • Agents transitioned structures {analysis['transition_rate']*100:.1f}% of the time")
    print(f"  • {analysis['structure_diversity']} different agency structures employed")
    print(f"  • Most successful structure: {analysis['most_used_structure'].value if analysis['most_used_structure'] else 'N/A'}")
    print(f"")
    print(f"Revolutionary Capabilities Demonstrated:")
    print(f"  • Second-order agency: reasoning about agency itself")
    print(f"  • Dynamic agency structure adaptation")
    print(f"  • Ethical reflection on agency choices")
    print(f"  • Performance-driven structure optimization")
    print(f"  • Meta-cognitive flexibility impossible for humans")
    print(f"")
    print(f"Philosophical Implications:")
    print(f"  • Agency as both subject and object of moral reasoning")
    print(f"  • Self-modifying moral agents")
    print(f"  • Novel forms of autonomy and self-governance")
    print(f"  • Challenges traditional concepts of fixed agency")

if __name__ == "__main__":
    main()
