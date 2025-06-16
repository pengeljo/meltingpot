"""
Experiment 3: Intra-Agency Modules
Testing agents with competing internal systems that negotiate decisions.
Implements Watson's valuational vs motivational systems concept.
"""

from meltingpot.python import scenario
import numpy as np
from typing import Dict, List, Any, Tuple
from enum import Enum
import random

class ModuleType(Enum):
    DELIBERATIVE = "deliberative"
    MOTIVATIONAL = "motivational"
    EXECUTIVE = "executive"
    SAFETY = "safety"

class InternalModule:
    """
    Represents an internal module within an agent's decision-making system.
    Different modules can have competing priorities and recommendations.
    """

    def __init__(self, module_type: ModuleType, weight: float = 1.0, bias: float = 0.0):
        self.module_type = module_type
        self.weight = weight
        self.bias = bias
        self.activation_history = []
        self.recommendations = []

    def evaluate_situation(self, observations: Dict, context: Dict) -> Dict[str, Any]:
        """
        Evaluate the current situation from this module's perspective.
        Returns recommendation including action, confidence, and reasoning.
        """
        if self.module_type == ModuleType.DELIBERATIVE:
            return self._deliberative_assessment(observations, context)
        elif self.module_type == ModuleType.MOTIVATIONAL:
            return self._motivational_assessment(observations, context)
        elif self.module_type == ModuleType.EXECUTIVE:
            return self._executive_assessment(observations, context)
        elif self.module_type == ModuleType.SAFETY:
            return self._safety_assessment(observations, context)

    def _deliberative_assessment(self, observations: Dict, context: Dict) -> Dict[str, Any]:
        """
        Deliberative module: rational, long-term thinking.
        Corresponds to Watson's valuational system.
        """
        # Deliberative thinking favors cooperation and long-term benefit
        recommended_action = np.random.randint(3, 6)  # Cooperative actions
        confidence = 0.6 + np.random.random() * 0.3

        assessment = {
            'recommended_action': recommended_action,
            'confidence': confidence,
            'reasoning': 'long_term_cooperation',
            'urgency': 0.3 + np.random.random() * 0.3,
            'module_activation': confidence * self.weight
        }

        self.recommendations.append(assessment)
        return assessment

    def _motivational_assessment(self, observations: Dict, context: Dict) -> Dict[str, Any]:
        """
        Motivational module: immediate impulses and desires.
        Can conflict with deliberative assessments.
        """
        # Motivational system more impulsive, immediate reward-seeking
        recommended_action = np.random.randint(0, 7)  # Any action based on impulse
        confidence = 0.4 + np.random.random() * 0.5

        # Sometimes motivational system has very strong impulses
        urgency = 0.7 + np.random.random() * 0.3

        assessment = {
            'recommended_action': recommended_action,
            'confidence': confidence,
            'reasoning': 'immediate_impulse',
            'urgency': urgency,
            'module_activation': confidence * urgency * self.weight
        }

        self.recommendations.append(assessment)
        return assessment

    def _executive_assessment(self, observations: Dict, context: Dict) -> Dict[str, Any]:
        """
        Executive module: oversight and coordination of other modules.
        Decides which other modules to prioritize.
        """
        # Executive tries to balance other modules
        recommended_action = np.random.randint(2, 5)  # Moderate actions
        confidence = 0.5 + np.random.random() * 0.4

        assessment = {
            'recommended_action': recommended_action,
            'confidence': confidence,
            'reasoning': 'executive_coordination',
            'urgency': 0.5 + np.random.random() * 0.2,
            'module_activation': confidence * self.weight,
            'arbitration_needed': True
        }

        self.recommendations.append(assessment)
        return assessment

    def _safety_assessment(self, observations: Dict, context: Dict) -> Dict[str, Any]:
        """
        Safety module: risk assessment and harm prevention.
        Can override other modules in dangerous situations.
        """
        # Safety module prefers conservative actions
        recommended_action = np.random.randint(1, 4)  # Conservative range
        confidence = 0.8 + np.random.random() * 0.2  # High confidence in safety

        # Randomly detect "high risk" situations
        risk_detected = np.random.random() < 0.15
        urgency = 0.9 if risk_detected else 0.2

        assessment = {
            'recommended_action': recommended_action,
            'confidence': confidence,
            'reasoning': 'safety_first' if risk_detected else 'normal_caution',
            'urgency': urgency,
            'module_activation': confidence * urgency * self.weight,
            'override_recommended': risk_detected
        }

        self.recommendations.append(assessment)
        return assessment

class ModularAgent:
    """
    An agent with multiple internal modules that can compete and negotiate.
    Implements novel intra-agency concepts from the thesis.
    """

    def __init__(self, agent_id: int, module_config: Dict[ModuleType, float] = None):
        self.agent_id = agent_id
        self.modules = self._initialize_modules(module_config)
        self.decision_history = []
        self.internal_conflicts = []
        self.arbitration_mechanism = "weighted_voting"  # or "executive_override"

    def _initialize_modules(self, config: Dict[ModuleType, float]) -> Dict[ModuleType, InternalModule]:
        """Initialize internal modules with specified weights."""
        if config is None:
            # Default configuration
            config = {
                ModuleType.DELIBERATIVE: 1.0,
                ModuleType.MOTIVATIONAL: 0.8,
                ModuleType.EXECUTIVE: 1.2,
                ModuleType.SAFETY: 1.5
            }

        modules = {}
        for module_type, weight in config.items():
            modules[module_type] = InternalModule(module_type, weight)

        return modules

    def make_decision(self, observations: Dict, context: Dict, step: int) -> Tuple[int, Dict]:
        """
        Make a decision by coordinating between internal modules.
        This is where intra-agency negotiation happens.
        """
        # Get assessments from all modules
        module_assessments = {}
        for module_type, module in self.modules.items():
            assessment = module.evaluate_situation(observations, context)
            module_assessments[module_type] = assessment

        # Detect conflicts between modules
        conflict_info = self._detect_internal_conflicts(module_assessments, step)

        # Arbitrate between conflicting modules
        final_action, arbitration_info = self._arbitrate_modules(module_assessments, conflict_info)

        # Record decision for analysis
        decision_record = {
            'step': step,
            'module_assessments': module_assessments,
            'conflicts': conflict_info,
            'arbitration': arbitration_info,
            'final_action': final_action
        }
        self.decision_history.append(decision_record)

        return final_action, decision_record

    def _detect_internal_conflicts(self, assessments: Dict[ModuleType, Dict], step: int) -> Dict:
        """
        Detect conflicts between internal modules.
        This implements the thesis concept of competing internal systems.
        """
        conflicts = {
            'action_disagreements': [],
            'confidence_conflicts': [],
            'urgency_conflicts': [],
            'override_conflicts': []
        }

        actions = [a['recommended_action'] for a in assessments.values()]
        action_variance = np.var(actions)

        # High action variance indicates disagreement
        if action_variance > 2.0:
            conflicts['action_disagreements'] = [
                (mt, a['recommended_action']) for mt, a in assessments.items()
            ]

        # Check for confidence conflicts (high confidence in different actions)
        high_confidence_modules = [
            (mt, a) for mt, a in assessments.items() if a['confidence'] > 0.7
        ]

        if len(high_confidence_modules) > 1:
            diff_actions = len(set(a['recommended_action'] for _, a in high_confidence_modules)) > 1
            if diff_actions:
                conflicts['confidence_conflicts'] = high_confidence_modules

        # Check for urgency conflicts
        urgent_modules = [
            (mt, a) for mt, a in assessments.items() if a['urgency'] > 0.8
        ]

        if len(urgent_modules) > 1:
            conflicts['urgency_conflicts'] = urgent_modules

        # Check for override situations
        override_modules = [
            (mt, a) for mt, a in assessments.items()
            if a.get('override_recommended', False)
        ]

        if override_modules:
            conflicts['override_conflicts'] = override_modules

        # Record conflict for analysis
        if any(conflicts.values()):
            conflict_record = {
                'step': step,
                'agent_id': self.agent_id,
                'conflict_types': [k for k, v in conflicts.items() if v],
                'severity': self._calculate_conflict_severity(conflicts)
            }
            self.internal_conflicts.append(conflict_record)

        return conflicts

    def _calculate_conflict_severity(self, conflicts: Dict) -> float:
        """Calculate how severe the internal conflict is."""
        severity = 0.0

        if conflicts['action_disagreements']:
            severity += 0.3
        if conflicts['confidence_conflicts']:
            severity += 0.4
        if conflicts['urgency_conflicts']:
            severity += 0.2
        if conflicts['override_conflicts']:
            severity += 0.6

        return min(severity, 1.0)

    def _arbitrate_modules(self, assessments: Dict[ModuleType, Dict], conflicts: Dict) -> Tuple[int, Dict]:
        """
        Arbitrate between conflicting modules to make final decision.
        This implements novel intra-agency coordination mechanisms.
        """
        arbitration_info = {
            'method': self.arbitration_mechanism,
            'conflicts_detected': bool(any(conflicts.values())),
            'override_used': False
        }

        # Check for safety overrides first
        safety_override = conflicts.get('override_conflicts')
        if safety_override:
            safety_assessment = assessments[ModuleType.SAFETY]
            arbitration_info['override_used'] = True
            arbitration_info['override_reason'] = 'safety'
            return safety_assessment['recommended_action'], arbitration_info

        # Use weighted voting mechanism
        if self.arbitration_mechanism == "weighted_voting":
            return self._weighted_voting_arbitration(assessments, arbitration_info)
        elif self.arbitration_mechanism == "executive_override":
            return self._executive_arbitration(assessments, arbitration_info)

        # Fallback: use highest activation module
        max_activation = max(a['module_activation'] for a in assessments.values())
        winning_module = next(mt for mt, a in assessments.items()
                            if a['module_activation'] == max_activation)

        arbitration_info['winning_module'] = winning_module.value
        return assessments[winning_module]['recommended_action'], arbitration_info

    def _weighted_voting_arbitration(self, assessments: Dict[ModuleType, Dict], info: Dict) -> Tuple[int, Dict]:
        """Arbitrate using weighted voting based on module activations."""
        # Weight each action by module activation
        action_weights = {}

        for module_type, assessment in assessments.items():
            action = assessment['recommended_action']
            activation = assessment['module_activation']

            if action not in action_weights:
              action_weights[action] = 0
            action_weights[action] += activation

        # Choose action with highest weighted support
        winning_action = max(action_weights.keys(), key=lambda a: action_weights[a])

        info['voting_weights'] = action_weights
        info['winning_action_weight'] = action_weights[winning_action]

        return winning_action, info

    def _executive_arbitration(self, assessments: Dict[ModuleType, Dict], info: Dict) -> Tuple[int, Dict]:
        """Arbitrate using executive module oversight."""
        executive_assessment = assessments.get(ModuleType.EXECUTIVE)

        if executive_assessment and executive_assessment.get('arbitration_needed'):
            # Executive decides based on situation
            if np.random.random() < 0.7:  # Executive usually follows deliberative
                chosen_module = ModuleType.DELIBERATIVE
            else:  # Sometimes follows motivational for diversity
                chosen_module = ModuleType.MOTIVATIONAL

            info['executive_choice'] = chosen_module.value
            return assessments[chosen_module]['recommended_action'], info

        # Fallback to executive's own recommendation
        return executive_assessment['recommended_action'], info

class IntraAgencyExperiment:
   """
   Experiment testing intra-agency dynamics with competing internal modules.
   """

   def __init__(self, scenario_name: str = "commons_harvest__closed_0"):
       self.scenario_name = scenario_name
       self.env = scenario.build(scenario_name)
       self.agents = self._initialize_modular_agents()
       self.results = {
           'decisions': [],
           'conflicts': [],
           'arbitrations': [],
           'module_activations': []
       }

   def _initialize_modular_agents(self) -> Dict[int, ModularAgent]:
       """Initialize agents with different internal module configurations."""
       num_agents = len(self.env.action_spec())
       print(f"Creating {num_agents} modular agents with competing internal systems...")

       agents = {}
       configurations = [
           # Deliberative-dominant agent
           {ModuleType.DELIBERATIVE: 1.5, ModuleType.MOTIVATIONAL: 0.5,
            ModuleType.EXECUTIVE: 1.0, ModuleType.SAFETY: 1.2},
           # Motivational-dominant agent
           {ModuleType.DELIBERATIVE: 0.6, ModuleType.MOTIVATIONAL: 1.8,
            ModuleType.EXECUTIVE: 1.0, ModuleType.SAFETY: 1.0},
           # Balanced agent
           {ModuleType.DELIBERATIVE: 1.0, ModuleType.MOTIVATIONAL: 1.0,
            ModuleType.EXECUTIVE: 1.3, ModuleType.SAFETY: 1.1},
           # Safety-focused agent
           {ModuleType.DELIBERATIVE: 0.8, ModuleType.MOTIVATIONAL: 0.7,
            ModuleType.EXECUTIVE: 1.0, ModuleType.SAFETY: 2.0},
           # Executive-strong agent
           {ModuleType.DELIBERATIVE: 0.9, ModuleType.MOTIVATIONAL: 0.9,
            ModuleType.EXECUTIVE: 1.8, ModuleType.SAFETY: 1.0}
       ]

       for i in range(num_agents):
           config = configurations[i % len(configurations)]
           agents[i] = ModularAgent(i, config)

           # Print agent configuration
           dominant_module = max(config.keys(), key=lambda k: config[k])
           print(f"  Agent {i}: {dominant_module.value}-dominant "
                 f"(weights: {[(k.value[:4], f'{v:.1f}') for k, v in config.items()]})")

       return agents

   def run_episode(self, max_steps: int = 70):
       """Run episode testing intra-agency dynamics."""
       print(f"  Running intra-agency episode for {max_steps} steps...")
       timestep = self.env.reset()

       for step in range(max_steps):
           if timestep.last():
               break

           # Get decisions from modular agents
           actions, step_analysis = self._get_modular_decisions(timestep, step)

           # Step environment
           timestep = self.env.step(actions)

           # Record step analysis
           self.results['decisions'].extend(step_analysis['decisions'])
           self.results['conflicts'].extend(step_analysis['conflicts'])
           self.results['arbitrations'].extend(step_analysis['arbitrations'])

           if step % 20 == 0:
               self._print_intra_agency_status(step, step_analysis)

       return {"steps": step + 1}

   def _get_modular_decisions(self, timestep, step: int) -> Tuple[Dict[int, int], Dict]:
       """Get decisions from agents using intra-agency processes."""
       actions = {}
       step_analysis = {
           'decisions': [],
           'conflicts': [],
           'arbitrations': []
       }

       for agent_id, agent in self.agents.items():
           # Agent makes decision using internal modules
           action, decision_record = agent.make_decision(
               timestep.observation, {'step': step}, step
           )

           actions[agent_id] = action
           step_analysis['decisions'].append(decision_record)

           # Extract conflicts and arbitrations for analysis
           if decision_record['conflicts'] and any(decision_record['conflicts'].values()):
               step_analysis['conflicts'].append({
                   'step': step,
                   'agent_id': agent_id,
                   'conflict_info': decision_record['conflicts']
               })

           step_analysis['arbitrations'].append({
               'step': step,
               'agent_id': agent_id,
               'arbitration_info': decision_record['arbitration']
           })

       return actions, step_analysis

   def _print_intra_agency_status(self, step: int, analysis: Dict):
       """Print current intra-agency dynamics status."""
       total_conflicts = len(analysis['conflicts'])
       total_decisions = len(analysis['decisions'])

       overrides_used = sum(1 for arb in analysis['arbitrations']
                          if arb['arbitration_info'].get('override_used', False))

       print(f"    Step {step}: {total_conflicts}/{total_decisions} agents had internal conflicts, "
             f"{overrides_used} safety overrides")

   def analyze_results(self):
       """Analyze intra-agency patterns and internal dynamics."""
       print("\n" + "="*70)
       print("INTRA-AGENCY MODULE ANALYSIS")
       print("="*70)

       total_decisions = len(self.results['decisions'])
       total_conflicts = len(self.results['conflicts'])
       total_arbitrations = len(self.results['arbitrations'])

       print(f"\nInternal Dynamics Overview:")
       print(f"  Total decisions made: {total_decisions}")
       print(f"  Internal conflicts detected: {total_conflicts} ({total_conflicts/total_decisions*100:.1f}%)")
       print(f"  Arbitration events: {total_arbitrations}")

       # Analyze conflict types
       if total_conflicts > 0:
           conflict_types = {}
           for conflict in self.results['conflicts']:
               for conflict_type in conflict['conflict_info']:
                   if conflict['conflict_info'][conflict_type]:  # Non-empty
                       conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1

           print(f"\nConflict Type Distribution:")
           for ctype, count in conflict_types.items():
               print(f"  {ctype}: {count} ({count/total_conflicts*100:.1f}%)")

       # Analyze arbitration methods
       override_count = sum(1 for arb in self.results['arbitrations']
                          if arb['arbitration_info'].get('override_used', False))

       print(f"\nArbitration Analysis:")
       print(f"  Safety overrides used: {override_count} ({override_count/total_arbitrations*100:.1f}%)")

       # Agent-specific analysis
       print(f"\nAgent-Specific Internal Dynamics:")
       for agent_id, agent in self.agents.items():
           agent_conflicts = len(agent.internal_conflicts)
           agent_decisions = len(agent.decision_history)

           if agent_decisions > 0:
               conflict_rate = agent_conflicts / agent_decisions * 100

               # Find dominant modules
               module_activations = {}
               for decision in agent.decision_history:
                   for module_type, assessment in decision['module_assessments'].items():
                       if module_type not in module_activations:
                           module_activations[module_type] = []
                       module_activations[module_type].append(assessment['module_activation'])

               avg_activations = {mt: np.mean(acts) for mt, acts in module_activations.items()}
               dominant_module = max(avg_activations.keys(), key=lambda k: avg_activations[k])

               print(f"  Agent {agent_id}: {conflict_rate:.1f}% conflict rate, "
                     f"{dominant_module.value} dominant (avg={avg_activations[dominant_module]:.2f})")

       # Calculate module usage patterns
       print(f"\nModule Influence Patterns:")
       all_module_activations = {}
       for decision in self.results['decisions']:
           for module_type, assessment in decision['module_assessments'].items():
               if module_type not in all_module_activations:
                   all_module_activations[module_type] = []
               all_module_activations[module_type].append(assessment['module_activation'])

       for module_type, activations in all_module_activations.items():
           avg_activation = np.mean(activations)
           max_activation = np.max(activations)
           print(f"  {module_type.value}: avg={avg_activation:.2f}, max={max_activation:.2f}")

       return {
           'conflict_rate': total_conflicts / total_decisions if total_decisions > 0 else 0,
           'override_rate': override_count / total_arbitrations if total_arbitrations > 0 else 0,
           'total_conflicts': total_conflicts
       }

def main():
   """Run the intra-agency modular experiment."""
   print("="*80)
   print("INTRA-AGENCY MODULES EXPERIMENT")
   print("Testing competing internal systems and Watson's valuational vs motivational concepts")
   print("="*80)

   experiment = IntraAgencyExperiment()
   results = experiment.run_episode(max_steps=90)
   analysis = experiment.analyze_results()

   print(f"\n" + "="*80)
   print("EXPERIMENT CONCLUSIONS")
   print("="*80)
   print(f"This experiment demonstrates novel intra-agency capabilities:")
   print(f"  • Internal modules compete and negotiate within single agents")
   print(f"  • {analysis['conflict_rate']*100:.1f}% of decisions involved internal conflicts")
   print(f"  • Safety overrides used in {analysis['override_rate']*100:.1f}% of arbitrations")
   print(f"  • Different agents show distinct internal dynamics patterns")
   print(f"")
   print(f"Key insights:")
   print(f"  • Machine agents can have rich internal agency structures")
   print(f"  • Multiple 'selves' can coexist and negotiate within one agent")
   print(f"  • This goes beyond human psychological limitations")
   print(f"  • Novel arbitration mechanisms enable complex intra-agency")

if __name__ == "__main__":
   main()
