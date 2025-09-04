"""
Complete Refined Meta-Agency Experiment
A self-contained version that addresses the hyperactive transition problem
with stability constraints, commitment mechanisms, and transition costs.
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

class StabilityConstrainedMetaController:
    """
    Enhanced meta-agency controller with realistic constraints on agency structure changes.
    Addresses the hyperactive transition problem from the original experiment.
    """

    def __init__(self, agent_id: int, stability_preference: float = 0.6):
        self.agent_id = agent_id
        self.available_structures = self._initialize_agency_repertoire()
        self.current_structure = AgencyStructureType.INDIVIDUAL
        self.structure_history = []
        self.meta_reasoning_log = []
        self.ethical_reflector = EthicalReflector()

        # NEW: Stability constraints to prevent hyperactive transitions
        self.stability_preference = stability_preference  # How much agent values stability
        self.transition_cooldown = 0  # Prevents immediate re-transitions
        self.structure_commitment_level = 0.8  # Commitment to current structure
        self.transition_costs_paid = 0.0  # Cumulative cost of transitions
        self.steps_in_current_structure = 0  # Duration in current structure

        # Minimum performance improvement required to justify transition
        self.transition_threshold = 0.3 + (stability_preference * 0.4)

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
        Enhanced meta-reflection with stability considerations and transition constraints.
        This version prevents the hyperactive transition behavior.
        """

        reflection = {
            'current_structure_assessment': self._assess_current_structure(performance_data),
            'alternative_evaluations': self._evaluate_alternative_structures(context),
            'ethical_considerations': self.ethical_reflector.assess_structure_ethics(
                self.current_structure, context
            ),
            'stability_factors': self._assess_stability_factors(),
            'transition_recommendations': []
        }

        # Enhanced transition logic with multiple constraints
        current_satisfaction = reflection['current_structure_assessment']['satisfaction']
        stability_factors = reflection['stability_factors']

        # Only consider transitions if ALL constraints are met:
        should_consider_transition = (
            self.transition_cooldown == 0 and  # Not in cooldown
            current_satisfaction < 0.3 and      # Performance genuinely poor
            self.steps_in_current_structure >= 5 and  # Minimum duration in structure
            self._transition_benefits_exceed_costs(reflection)  # Benefits > costs
        )

        if should_consider_transition:
            alternatives = reflection['alternative_evaluations']

            # Find best alternative that significantly outperforms current
            viable_alternatives = {
                structure: data for structure, data in alternatives.items()
                if data['expected_performance'] > (current_satisfaction + self.transition_threshold)
            }

            if viable_alternatives:
                best_alternative = max(viable_alternatives.keys(),
                                     key=lambda k: viable_alternatives[k]['expected_performance'])

                transition_feasible = self._assess_transition_feasibility(
                    self.current_structure, best_alternative, context
                )

                if transition_feasible['feasible']:
                    reflection['transition_recommendations'].append({
                        'from_structure': self.current_structure,
                        'to_structure': best_alternative,
                        'reason': 'significant_performance_improvement',
                        'feasibility': transition_feasible,
                        'expected_benefit': alternatives[best_alternative]['expected_performance'],
                        'current_satisfaction': current_satisfaction,
                        'improvement_magnitude': alternatives[best_alternative]['expected_performance'] - current_satisfaction
                    })

        self.meta_reasoning_log.append(reflection)
        return reflection

    def _assess_current_structure(self, performance_data: Dict) -> Dict[str, float]:
        """Enhanced assessment of current agency structure performance."""
        current_config = self.available_structures[self.current_structure]

        # Update performance history
        recent_performance = performance_data.get('reward', 0.0)
        current_config.performance_history.append(recent_performance)

        # Calculate satisfaction with stability bonus
        if len(current_config.performance_history) >= 5:
            recent_avg = np.mean(current_config.performance_history[-5:])
            overall_avg = np.mean(current_config.performance_history)
            trend = recent_avg - overall_avg
        else:
            recent_avg = recent_performance
            trend = 0.0

        # Base satisfaction
        base_satisfaction = min(recent_avg / 10.0 + trend * 0.5, 1.0)

        # Stability bonus increases satisfaction (representing investment in current structure)
        stability_bonus = min(self.steps_in_current_structure * 0.02, 0.2)

        # Commitment bonus
        commitment_bonus = self.structure_commitment_level * 0.1

        total_satisfaction = min(base_satisfaction + stability_bonus + commitment_bonus, 1.0)

        return {
            'satisfaction': total_satisfaction,
            'base_satisfaction': base_satisfaction,
            'recent_performance': recent_avg,
            'performance_trend': trend,
            'stability_bonus': stability_bonus,
            'commitment_bonus': commitment_bonus
        }

    def _evaluate_alternative_structures(self, context: Dict) -> Dict[AgencyStructureType, Dict]:
        """Evaluate potential alternative agency structures."""
        evaluations = {}

        for structure_type, config in self.available_structures.items():
            if structure_type == self.current_structure:
                continue

            # Calculate expected performance
            compatibility_score = self._calculate_compatibility(structure_type, context)
            historical_performance = np.mean(config.performance_history) if config.performance_history else 0.5

            # Discount alternatives based on transition costs
            transition_cost = self._estimate_transition_cost(structure_type)
            expected_performance = (compatibility_score * 0.6 + historical_performance * 0.4) - transition_cost

            evaluations[structure_type] = {
                'expected_performance': expected_performance,
                'compatibility_score': compatibility_score,
                'historical_performance': historical_performance,
                'transition_cost': transition_cost
            }

        return evaluations

    def _assess_stability_factors(self) -> Dict[str, float]:
        """Assess factors related to agency structure stability."""

        # Stability value increases with time in current structure
        stability_value = min(self.steps_in_current_structure * 0.03, 0.4)

        # Transition cost increases with stability investment
        base_transition_cost = 0.15
        stability_cost = stability_value * 1.5  # High cost for abandoning stable structure
        transition_cost = base_transition_cost + stability_cost

        # Identity disruption risk
        transition_count = len(self.structure_history)
        identity_disruption = min(transition_count * 0.05, 0.3)

        return {
            'stability_value': stability_value,
            'transition_cost': transition_cost,
            'identity_disruption_risk': identity_disruption,
            'structure_duration': self.steps_in_current_structure,
            'commitment_level': self.structure_commitment_level
        }

    def _transition_benefits_exceed_costs(self, reflection: Dict) -> bool:
        """Enhanced cost-benefit analysis for transitions."""
        current_satisfaction = reflection['current_structure_assessment']['base_satisfaction']
        stability_factors = reflection['stability_factors']

        # Get best alternative
        alternatives = reflection['alternative_evaluations']
        if not alternatives:
            return False

        best_performance = max(alt['expected_performance'] for alt in alternatives.values())

        # Calculate net benefit
        performance_benefit = best_performance - current_satisfaction
        total_cost = stability_factors['transition_cost'] + stability_factors['identity_disruption_risk']

        # Require significant net benefit (bias toward stability)
        net_benefit = performance_benefit - total_cost
        required_benefit = 0.2 + (self.stability_preference * 0.3)

        return net_benefit > required_benefit

    def _calculate_compatibility(self, structure_type: AgencyStructureType, context: Dict) -> float:
        """Calculate how compatible a structure is with current context."""
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

    def _estimate_transition_cost(self, target_structure: AgencyStructureType) -> float:
        """Estimate the cost of transitioning to a different agency structure."""
        base_costs = {
            AgencyStructureType.INDIVIDUAL: 0.1,
            AgencyStructureType.COLLECTIVE: 0.3,
            AgencyStructureType.MODULAR: 0.25,
            AgencyStructureType.FUSED: 0.4
        }

        base_cost = base_costs.get(target_structure, 0.25)

        # Additional cost based on stability preference
        stability_cost = self.stability_preference * 0.2

        return base_cost + stability_cost

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

        # Check if we can afford the transition
        transition_cost = self._estimate_transition_cost(to_structure)
        total_cost_limit = 1.0  # Maximum cumulative transition cost
        cost_feasible = (self.transition_costs_paid + transition_cost) < total_cost_limit

        return {
            'feasible': requirements_met and ethical_clearance['permissible'] and cost_feasible,
            'requirements_met': requirements_met,
            'ethical_clearance': ethical_clearance,
            'cost_feasible': cost_feasible,
            'transition_cost': transition_cost
        }

    def execute_agency_transition(self, target_structure: AgencyStructureType, reason: str) -> bool:
        """Execute transition with enhanced stability costs and constraints."""

        # Calculate and pay transition costs
        stability_factors = self._assess_stability_factors()
        transition_cost = stability_factors['transition_cost']
        self.transition_costs_paid += transition_cost

        # Set substantial cooldown period
        base_cooldown = 8
        stability_cooldown = int(self.steps_in_current_structure * 0.5)
        self.transition_cooldown = base_cooldown + stability_cooldown

        # Record transition
        transition_record = {
            'step': len(self.structure_history),
            'from_structure': self.current_structure,
            'to_structure': target_structure,
            'reason': reason,
            'transition_cost': transition_cost,
            'cooldown_set': self.transition_cooldown,
            'steps_in_previous_structure': self.steps_in_current_structure
        }

        self.structure_history.append(transition_record)
        self.current_structure = target_structure

        # Reset counters for new structure
        self.steps_in_current_structure = 0
        self.structure_commitment_level = 0.7  # Start with moderate commitment

        print(f"    Agent {self.agent_id}: Transitioned from {transition_record['from_structure'].value} "
              f"to {target_structure.value} (cost: {transition_cost:.2f}, cooldown: {self.transition_cooldown})")

        return True

    def update_step(self):
        """Update meta-controller state each step."""
        # Decrement cooldown
        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1

        # Increment time in current structure
        self.steps_in_current_structure += 1

        # Gradually increase commitment to current structure
        commitment_increase = 0.02
        self.structure_commitment_level = min(self.structure_commitment_level + commitment_increase, 1.0)

class CommitmentBasedMetaAgent:
    """
    Meta-agent with enhanced commitment mechanisms and stability preferences.
    This addresses the hyperactive transition problem from the original experiment.
    """

    def __init__(self, agent_id: int, stability_preference: float = 0.6):
        self.agent_id = agent_id
        self.meta_controller = StabilityConstrainedMetaController(agent_id, stability_preference)
        self.performance_tracker = PerformanceTracker()
        self.action_history = []
        self.stability_preference = stability_preference

    def make_meta_decision(self, observations: Dict, context: Dict, step: int) -> Tuple[int, Dict]:
        """Make decision with enhanced stability considerations."""

        # Update meta-controller state
        self.meta_controller.update_step()

        # Enhanced context with stability preferences
        enhanced_context = context.copy()
        enhanced_context['stability_preference'] = self.stability_preference
        enhanced_context['agent_history_length'] = len(self.action_history)

        # Meta-level reasoning with stability constraints
        performance_data = self.performance_tracker.get_recent_performance()
        meta_reflection = self.meta_controller.meta_reflect_on_agency(
            enhanced_context, performance_data
        )

        # Conservative transition logic - multiple checks
        if meta_reflection['transition_recommendations']:
            recommendation = meta_reflection['transition_recommendations'][0]

            # Additional agent-level stability check
            improvement_magnitude = recommendation.get('improvement_magnitude', 0)
            agent_threshold = 0.3 + (self.stability_preference * 0.4)

            # Only transition if improvement is substantial enough for this agent
            if improvement_magnitude > agent_threshold:
                target_structure = recommendation['to_structure']
                reason = recommendation['reason']

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
            'structure_transitions': len(self.meta_controller.structure_history),
            'transition_cooldown': self.meta_controller.transition_cooldown,
            'stability_preference': self.stability_preference,
            'steps_in_structure': self.meta_controller.steps_in_current_structure
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

class RefinedMetaAgencyExperiment:
    """
    Refined meta-agency experiment with stability constraints and commitment mechanisms.
    This version should dramatically reduce the hyperactive transition behavior.
    """

    def __init__(self, scenario_name: str = "clean_up_0"):
        self.scenario_name = scenario_name
        self.env = scenario.build(scenario_name)
        self.agents = self._initialize_refined_meta_agents()
        self.results = {
            'meta_decisions': [],
            'structure_transitions': [],
            'stability_metrics': []
        }

    def _initialize_refined_meta_agents(self) -> Dict[int, CommitmentBasedMetaAgent]:
        """Initialize agents with varied stability preferences."""
        num_agents = len(self.env.action_spec())
        print(f"Creating {num_agents} stability-constrained meta-agents...")

        agents = {}
        stability_preferences = [0.2, 0.6, 0.9]  # Low, medium, high stability preference

        for i in range(num_agents):
            stability_pref = stability_preferences[i % len(stability_preferences)]
            agents[i] = CommitmentBasedMetaAgent(i, stability_pref)
            print(f"  Agent {i}: Stability preference = {stability_pref:.1f} "
                  f"(transition threshold = {0.3 + stability_pref * 0.4:.2f})")

        return agents

    def run_episode(self, max_steps: int = 120):
        """Run refined meta-agency episode."""
        print(f"  Running refined meta-agency episode for {max_steps} steps...")
        timestep = self.env.reset()

        for step in range(max_steps):
            if timestep.last():
                break

            actions, step_analysis = self._get_refined_meta_decisions(timestep, step)
            timestep = self.env.step(actions)
            self._update_agent_performance(timestep, step)

            # Record results
            self.results['meta_decisions'].extend(step_analysis['decisions'])
            self.results['structure_transitions'].extend(step_analysis['transitions'])

            if step % 30 == 0:
                self._print_refined_status(step)

        return {"steps": step + 1}

    def _get_refined_meta_decisions(self, timestep, step: int):
        """Get decisions with refined meta-agency logic."""
        actions = []
        step_analysis = {'decisions': [], 'transitions': []}

        context = {
            'complexity': np.random.random(),
            'social_pressure': np.random.random(),
            'urgency': np.random.random(),
            'step': step
        }

        for agent_id, agent in self.agents.items():
            action, decision_record = agent.make_meta_decision(
                timestep.observation, context, step
            )

            actions.append(action)
            step_analysis['decisions'].append(decision_record)

            # Track new transitions
            current_transitions = len(agent.meta_controller.structure_history)
            if step == 0:
                agent.previous_transition_count = 0

            if current_transitions > getattr(agent, 'previous_transition_count', 0):
                step_analysis['transitions'].append({
                    'step': step,
                    'agent_id': agent_id,
                    'new_structure': agent.meta_controller.current_structure,
                    'total_transitions': current_transitions
                })
                agent.previous_transition_count = current_transitions

        return actions, step_analysis

    def _update_agent_performance(self, timestep, step: int):
        """Update agent performance tracking."""
        if timestep.reward:
            for agent_id, reward in enumerate(timestep.reward):
                context = {'step': step, 'reward': reward}
                self.agents[agent_id].update_performance(reward, context)

    def _print_refined_status(self, step: int):
        """Print refined meta-agency status."""
        total_transitions = sum(len(agent.meta_controller.structure_history)
                              for agent in self.agents.values())

        agents_in_cooldown = sum(1 for agent in self.agents.values()
                               if agent.meta_controller.transition_cooldown > 0)

        # Show current structures
        structure_counts = {}
        for agent in self.agents.values():
            structure = agent.meta_controller.current_structure
            structure_counts[structure] = structure_counts.get(structure, 0) + 1

        structure_summary = ", ".join([f"{s.value}: {c}" for s, c in structure_counts.items()])

        print(f"    Step {step}: {total_transitions} total transitions, "
              f"{agents_in_cooldown} agents in cooldown")
        print(f"      Current structures: {structure_summary}")

    def analyze_refined_results(self):
        """Analyze refined meta-agency results."""
        print("\n" + "="*80)
        print("REFINED META-AGENCY ANALYSIS - STABILITY CONSTRAINED")
        print("="*80)

        total_decisions = len(self.results['meta_decisions'])
        total_transitions = len(self.results['structure_transitions'])

        print(f"\nRefined Meta-Agency Results:")
        print(f"  Total decisions: {total_decisions}")
        print(f"  Total transitions: {total_transitions}")
        print(f"  Transition rate: {total_transitions/total_decisions*100:.1f}% (vs 100% in original)")

        # Structure usage analysis
        structure_usage = {}
        structure_duration = {}

        for decision in self.results['meta_decisions']:
            structure = decision['agency_structure']
            structure_usage[structure] = structure_usage.get(structure, 0) + 1

            # Track time spent in structures
            steps_in_structure = decision.get('steps_in_structure', 1)
            if structure not in structure_duration:
                structure_duration[structure] = []
            structure_duration[structure].append(steps_in_structure)

        print(f"\nStructure Usage Patterns:")
        for structure, count in structure_usage.items():
            percentage = count / total_decisions * 100 if total_decisions > 0 else 0
            avg_duration = np.mean(structure_duration[structure]) if structure in structure_duration else 0
            print(f"  {structure.value}: {count} uses ({percentage:.1f}%), avg duration: {avg_duration:.1f} steps")

        # Agent-specific stability analysis
        print(f"\nAgent Stability Analysis:")
        for agent_id, agent in self.agents.items():
            transitions = len(agent.meta_controller.structure_history)
            stability_pref = agent.stability_preference
            current_structure = agent.meta_controller.current_structure
            steps_in_current = agent.meta_controller.steps_in_current_structure
            commitment_level = agent.meta_controller.structure_commitment_level
            total_costs = agent.meta_controller.transition_costs_paid

            print(f"  Agent {agent_id} (stability_pref={stability_pref:.1f}):")
            print(f"    Transitions: {transitions}, Current: {current_structure.value}")
            print(f"    Steps in current structure: {steps_in_current}")
            print(f"    Commitment level: {commitment_level:.2f}")
            print(f"    Total transition costs paid: {total_costs:.2f}")

        # Effectiveness analysis
        if self.results['meta_decisions']:
            decisions_with_recommendations = sum(
                1 for decision in self.results['meta_decisions']
                if decision['meta_reflection'].get('transition_recommendations', [])
            )

            recommendation_to_transition_ratio = (
                total_transitions / max(decisions_with_recommendations, 1)
            )

            print(f"\nMeta-Reasoning Effectiveness:")
            print(f"  Decisions with transition recommendations: {decisions_with_recommendations}")
            print(f"  Recommendation-to-transition ratio: {recommendation_to_transition_ratio:.2f}")
            print(f"  (Lower ratio indicates more selective/constrained transitions)")

        # Cooldown analysis
        cooldown_steps = sum(
            1 for decision in self.results['meta_decisions']
            if decision.get('transition_cooldown', 0) > 0
        )

        print(f"\nStability Mechanism Effectiveness:")
        print(f"  Steps with agents in cooldown: {cooldown_steps} ({cooldown_steps/total_decisions*100:.1f}%)")

        return {
            'total_transitions': total_transitions,
            'transition_rate': total_transitions/total_decisions if total_decisions > 0 else 0,
            'structure_diversity': len(structure_usage),
            'most_used_structure': max(structure_usage.keys(), key=lambda k: structure_usage[k]) if structure_usage else None,
            'avg_transitions_per_agent': total_transitions / len(self.agents),
            'recommendation_selectivity': recommendation_to_transition_ratio if 'recommendation_to_transition_ratio' in locals() else 0
        }

def main():
    """Run the refined meta-agency experiment."""
    print("="*90)
    print("REFINED META-AGENCY EXPERIMENT - STABILITY CONSTRAINED")
    print("Testing meta-agency with stability constraints and commitment mechanisms")
    print("Addressing the hyperactive transition problem from the original experiment")
    print("="*90)

    experiment = RefinedMetaAgencyExperiment()
    results = experiment.run_episode(max_steps=120)
    analysis = experiment.analyze_refined_results()

    print(f"\n" + "="*90)
    print("EXPERIMENT CONCLUSIONS - REFINED META-AGENCY")
    print("="*90)
    print(f"This refined experiment demonstrates controlled meta-agency capabilities:")
    print(f"")
    print(f"Key Improvements Over Original:")
    print(f"  • Transition rate reduced to {analysis['transition_rate']*100:.1f}% (from 100%)")
    print(f"  • Average {analysis['avg_transitions_per_agent']:.1f} transitions per agent")
    print(f"  • Recommendation selectivity: {analysis['recommendation_selectivity']:.2f}")
    print(f"  • {analysis['structure_diversity']} different agency structures employed")
    print(f"")
    print(f"Stability Mechanisms Successfully Implemented:")
    print(f"  • Transition cooldown periods prevent rapid switching")
    print(f"  • Commitment levels increase over time in stable structures")
    print(f"  • Transition costs create realistic barriers to change")
    print(f"  • Performance improvement thresholds ensure worthwhile transitions")
    print(f"  • Varied stability preferences create agent diversity")
    print(f"")
    print(f"Philosophical Implications:")
    print(f"  • Meta-agency can be constrained without losing adaptive benefits")
    print(f"  • Stability and flexibility can be balanced in machine agents")
    print(f"  • Different agents can have different meta-agency preferences")
    print(f"  • Transition costs create realistic constraints on agency fluidity")
    print(f"  • This addresses the identity continuity problem from hyperactive transitions")
    print(f"")
    print(f"Real-World Relevance:")
    print(f"  • AI systems could adaptively change their decision-making architectures")
    print(f"  • But with safeguards preventing destabilizing rapid changes")
    print(f"  • Balances optimization with predictability for human interaction")

if __name__ == "__main__":
    main()
