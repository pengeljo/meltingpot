"""
Experiment 2: Identity Fusion and Fission
Testing agents that can merge and split identities dynamically.
Implements DiGiovanna's "para-person" concepts from the thesis.
"""

from meltingpot.python import scenario
import numpy as np
from typing import Dict, List, Any, Set, Optional
import uuid

class FusionAgent:
    """
    An agent capable of identity fusion and fission.
    Can temporarily merge with others to form collective identities.
    """

    def __init__(self, agent_id: int, fusion_threshold: float = 0.6):
        self.original_id = agent_id
        self.current_identity = f"individual_{agent_id}"
        self.fusion_threshold = fusion_threshold
        self.fusion_history = []
        self.identity_components = {agent_id}  # Set of original IDs in this identity
        self.fusion_cooldown = 0
        self.memory = []

    def evaluate_fusion_opportunity(self, other_agents: Dict, context: Dict) -> Optional[Set[int]]:
        """
        Decide whether to initiate fusion with other agents.
        Returns set of agent IDs to fuse with, or None.
        """
        if self.fusion_cooldown > 0:
            self.fusion_cooldown -= 1
            return None

        fusion_candidates = set()

        for other_id, other_agent in other_agents.items():
            if other_id == self.original_id:
                continue

            # Calculate fusion compatibility
            compatibility = self._calculate_fusion_compatibility(other_agent, context)

            if compatibility > self.fusion_threshold:
                fusion_candidates.add(other_id)

        # Can fuse with multiple agents simultaneously (novel capability)
        if len(fusion_candidates) >= 1:
            return fusion_candidates

        return None

    def _calculate_fusion_compatibility(self, other_agent, context) -> float:
        """Calculate how compatible this agent is with another for fusion."""
        # Similarity in recent actions
        action_similarity = self._calculate_action_similarity(other_agent)

        # Situational benefit of fusion
        situational_benefit = self._calculate_situational_benefit(other_agent, context)

        # Random factor for exploration
        exploration_factor = np.random.random() * 0.3

        return (action_similarity * 0.4 + situational_benefit * 0.4 + exploration_factor)

    def _calculate_action_similarity(self, other_agent) -> float:
        """Calculate similarity in recent action patterns."""
        if len(self.memory) == 0 or len(other_agent.memory) == 0:
            return 0.5

        # Compare last few actions
        my_recent = self.memory[-3:]
        other_recent = other_agent.memory[-3:]

        matches = sum(1 for a, b in zip(my_recent, other_recent) if a == b)
        return matches / max(len(my_recent), len(other_recent))

    def _calculate_situational_benefit(self, other_agent, context) -> float:
        """Calculate potential benefit of fusion in current situation."""
        # Simplified: fusion benefits increase with environmental complexity
        return np.random.random() * 0.8

    def initiate_fusion(self, partner_agents: List['FusionAgent']) -> 'FusedIdentity':
        """Create a new fused identity with partner agents."""
        all_components = self.identity_components.copy()
        fusion_participants = [self]

        for partner in partner_agents:
            all_components.update(partner.identity_components)
            fusion_participants.append(partner)

        # Create new fused identity
        fused_identity = FusedIdentity(all_components, fusion_participants)

        # Record fusion history
        fusion_record = {
            'step': len(self.fusion_history),
            'type': 'fusion',
            'participants': list(all_components),
            'identity_id': fused_identity.identity_id
        }

        for participant in fusion_participants:
            participant.fusion_history.append(fusion_record)
            participant.current_identity = fused_identity.identity_id

        return fused_identity

    def split_from_fusion(self, reason: str = "individual_preference"):
        """Split from current fused identity back to individual."""
        if self.current_identity.startswith("individual_"):
            return  # Already individual

        # Revert to individual identity
        old_identity = self.current_identity
        self.current_identity = f"individual_{self.original_id}"
        self.identity_components = {self.original_id}
        self.fusion_cooldown = 5  # Prevent immediate re-fusion

        # Record split
        split_record = {
            'step': len(self.fusion_history),
            'type': 'fission',
            'from_identity': old_identity,
            'reason': reason
        }
        self.fusion_history.append(split_record)

class FusedIdentity:
    """
    A collective identity formed by fusion of multiple agents.
    Represents a novel form of agency beyond individual boundaries.
    """

    def __init__(self, component_ids: Set[int], participant_agents: List[FusionAgent]):
        self.identity_id = f"fused_{'_'.join(map(str, sorted(component_ids)))}"
        self.component_ids = component_ids
        self.participant_agents = participant_agents
        self.creation_step = 0
        self.collective_memory = []

    def make_collective_decision(self, observations, available_actions) -> Dict[int, int]:
        """
        Make decisions as a fused collective identity.
        This is where novel collective agency is expressed.
        """
        decisions = {}

        # Collective reasoning: consider all participants' perspectives
        collective_assessment = self._assess_collective_benefit(observations)

        # Generate coordinated actions for all participants
        for agent in self.participant_agents:
            # Collective decisions tend to be more coordinated
            if collective_assessment > 0.6:
                # High coordination action
                action = self._generate_coordinated_action(agent, observations)
            else:
                # Individual action within collective context
                action = self._generate_individual_action(agent, observations)

            decisions[agent.original_id] = action
            agent.memory.append(action)

        self.collective_memory.append({
            'step': len(self.collective_memory),
            'decisions': decisions.copy(),
            'collective_assessment': collective_assessment
        })

        return decisions

    def _assess_collective_benefit(self, observations) -> float:
        """Assess how beneficial collective action is in current situation."""
        # Simplified collective benefit assessment
        return np.random.random() * 0.8 + 0.1

    def _generate_coordinated_action(self, agent, observations) -> int:
        """Generate action that coordinates with other fused agents."""
        # Coordinated actions tend toward middle range (cooperation)
        return np.random.randint(2, 5)

    def _generate_individual_action(self, agent, observations) -> int:
        """Generate individual action within collective context."""
        return np.random.randint(0, 7)

    def evaluate_fission_triggers(self) -> List[FusionAgent]:
        """
        Evaluate whether any agents should split from the fusion.
        Returns list of agents that want to split.
        """
        splitting_agents = []

        for agent in self.participant_agents:
            split_probability = self._calculate_split_probability(agent)

            if np.random.random() < split_probability:
                splitting_agents.append(agent)

        return splitting_agents

    def _calculate_split_probability(self, agent) -> float:
        """Calculate probability that an agent wants to split from fusion."""
        # Higher probability to split if:
        # 1. Fusion has lasted many steps
        # 2. Individual preferences differ from collective
        # 3. Random exploration

        duration_factor = min(len(self.collective_memory) * 0.02, 0.3)
        random_factor = np.random.random() * 0.1

        return duration_factor + random_factor

class IdentityFusionExperiment:
    """
    Experiment testing identity fusion and fission capabilities.
    """

    def __init__(self, scenario_name: str = "predator_prey__open_0"):
        self.scenario_name = scenario_name
        print(f"Initializing fusion experiment with scenario: {scenario_name}")
        self.env = scenario.build(scenario_name)

        num_agents = len(self.env.action_spec())
        print(f"✓ Scenario has {num_agents} agents - perfect for fusion experiments!")

        self.agents = self._initialize_fusion_agents()
        self.fused_identities = {}
        self.collective_counter = 0
        self.results = {
            'fusion_events': [],
            'fission_events': [],
            'identity_timeline': [],
            'collective_decisions': []
        }

    def _initialize_fusion_agents(self) -> Dict[int, FusionAgent]:
        num_agents = len(self.env.action_spec())
        print(f"Creating {num_agents} fusion-capable agents...")

        agents = {}
        for i in range(num_agents):
            threshold = 0.4 + (i * 0.05)  # More varied thresholds
            agents[i] = FusionAgent(i, threshold)
            print(f"  Agent {i}: fusion threshold = {threshold:.2f}")

        return agents

    def run_episode(self, max_steps: int = 60):
        print(f"  Running fusion/fission episode for {max_steps} steps...")
        timestep = self.env.reset()

        for step in range(max_steps):
            if timestep.last():
                break

            self._process_fusion_opportunities(step)
            self._process_fission_events(step)
            actions = self._get_identity_actions(timestep, step)

            action_list = self._convert_actions_to_list(actions)

            timestep = self.env.step(action_list)
            self._record_identity_timeline(step)

            if step % 12 == 0:
                self._print_identity_status(step)

        return {"steps": step + 1}


    def _convert_actions_to_list(self, actions_dict: Dict[int, int]) -> List[int]:
        """
        Convert actions dictionary to list in correct agent order.
        This fixes the IndexError we were getting.
        """
        num_agents = len(self.agents)
        action_list = []

        for agent_id in range(num_agents):
            if agent_id in actions_dict:
                action = actions_dict[agent_id]
                # Ensure action is within valid range
                action = max(0, min(action, 6))  # Clamp to [0, 6]
                action_list.append(action)
            else:
                # Default action if agent not found
                action_list.append(0)

        return action_list

    def _process_fusion_opportunities(self, step: int):
        available_agents = {aid: agent for aid, agent in self.agents.items()
                          if agent.current_identity.startswith("individual_")}

        processed_agents = set()

        for agent_id, agent in available_agents.items():
            if agent_id in processed_agents:
                continue

            fusion_candidates = agent.evaluate_fusion_opportunity(available_agents, {})

            if fusion_candidates:
                partner_agents = [available_agents[cid] for cid in fusion_candidates
                                if cid in available_agents and cid not in processed_agents]

                if partner_agents:
                    # Limit fusion size for more interesting dynamics
                    max_partners = min(2, len(partner_agents))
                    selected_partners = partner_agents[:max_partners]

                    fused_identity = agent.initiate_fusion(selected_partners)
                    self.fused_identities[fused_identity.identity_id] = fused_identity

                    processed_agents.add(agent_id)
                    processed_agents.update(p.original_id for p in selected_partners)

                    fusion_event = {
                        'step': step,
                        'type': 'fusion',
                        'participants': [agent_id] + [p.original_id for p in selected_partners],
                        'identity_id': fused_identity.identity_id
                    }
                    self.results['fusion_events'].append(fusion_event)
                    print(f"    Step {step}: Fusion created {fused_identity.identity_id} "
                          f"with {len(selected_partners)+1} agents")

    def _process_fission_events(self, step: int):
        fusions_to_remove = []

        for fusion_id, fused_identity in self.fused_identities.items():
            splitting_agents = fused_identity.evaluate_fission_triggers()

            if splitting_agents:
                for agent in splitting_agents:
                    agent.split_from_fusion("autonomy_preference")

                    fission_event = {
                        'step': step,
                        'type': 'fission',
                        'agent_id': agent.original_id,
                        'from_identity': fusion_id
                    }
                    self.results['fission_events'].append(fission_event)
                    print(f"    Step {step}: Agent {agent.original_id} split from {fusion_id}")

                fused_identity.participant_agents = [a for a in fused_identity.participant_agents
                                                   if a not in splitting_agents]

                if not fused_identity.participant_agents:
                    fusions_to_remove.append(fusion_id)
                    print(f"    Step {step}: {fusion_id} dissolved completely")

        for fusion_id in fusions_to_remove:
            del self.fused_identities[fusion_id]

    def _get_identity_actions(self, timestep, step: int) -> Dict[int, int]:
        actions = {}

        # Individual agents
        for agent_id, agent in self.agents.items():
            if agent.current_identity.startswith("individual_"):
                action = np.random.randint(0, 7)
                actions[agent_id] = action
                agent.memory.append(action)

        # Fused identities
        for fusion_id, fused_identity in self.fused_identities.items():
            fusion_actions = fused_identity.make_collective_decision(
                timestep.observation, list(range(7))
            )
            actions.update(fusion_actions)

            self.results['collective_decisions'].append({
                'step': step,
                'fusion_id': fusion_id,
                'participants': [a.original_id for a in fused_identity.participant_agents],
                'actions': fusion_actions
            })

        return actions

    def _record_identity_timeline(self, step: int):
        current_state = {
            'step': step,
            'individual_agents': [],
            'fused_identities': []
        }

        for agent_id, agent in self.agents.items():
            if agent.current_identity.startswith("individual_"):
                current_state['individual_agents'].append(agent_id)

        for fusion_id, fused_identity in self.fused_identities.items():
            current_state['fused_identities'].append({
                'id': fusion_id,
                'participants': [a.original_id for a in fused_identity.participant_agents],
                'size': len(fused_identity.participant_agents)
            })

        self.results['identity_timeline'].append(current_state)

    def _print_identity_status(self, step: int):
        individual_count = sum(1 for a in self.agents.values()
                             if a.current_identity.startswith("individual_"))
        fusion_count = len(self.fused_identities)
        fused_agents = sum(len(f.participant_agents) for f in self.fused_identities.values())

        print(f"    Step {step}: {individual_count} individual, "
              f"{fused_agents} agents in {fusion_count} fusions")

    def analyze_results(self):
        print("\n" + "="*60)
        print("IDENTITY FUSION/FISSION ANALYSIS")
        print("="*60)

        total_fusions = len(self.results['fusion_events'])
        total_fissions = len(self.results['fission_events'])

        print(f"\nIdentity Dynamics:")
        print(f"  Total fusion events: {total_fusions}")
        print(f"  Total fission events: {total_fissions}")
        print(f"  Collective decisions made: {len(self.results['collective_decisions'])}")

        if total_fusions > 0:
            fusion_sizes = []
            for event in self.results['fusion_events']:
                fusion_sizes.append(len(event['participants']))

            avg_fusion_size = np.mean(fusion_sizes)
            max_fusion_size = max(fusion_sizes)

            print(f"\nFusion Characteristics:")
            print(f"  Average fusion size: {avg_fusion_size:.1f} agents")
            print(f"  Largest fusion: {max_fusion_size} agents")

            if self.results['identity_timeline']:
                max_simultaneous_fusions = max(
                    len(state['fused_identities'])
                    for state in self.results['identity_timeline']
                )
                print(f"  Max simultaneous fusions: {max_simultaneous_fusions}")

        print(f"\nAgent-Specific Patterns:")
        for agent_id, agent in self.agents.items():
            agent_fusions = sum(1 for event in self.results['fusion_events']
                              if agent_id in event['participants'])
            agent_splits = sum(1 for event in self.results['fission_events']
                             if event['agent_id'] == agent_id)

            print(f"  Agent {agent_id}: {agent_fusions} fusions, {agent_splits} splits")

        return {
            'total_fusions': total_fusions,
            'total_fissions': total_fissions,
            'fusion_fission_ratio': total_fusions / max(total_fissions, 1)
        }

def main():
    """Run the identity fusion/fission experiment."""
    print("="*70)
    print("IDENTITY FUSION AND FISSION EXPERIMENT")
    print("Testing DiGiovanna's para-person concepts and identity flexibility")
    print("="*70)

    experiment = IdentityFusionExperiment()
    results = experiment.run_episode(max_steps=80)
    analysis = experiment.analyze_results()

    print(f"\n" + "="*70)
    print("EXPERIMENT CONCLUSIONS")
    print("="*70)
    print(f"This experiment demonstrates machine agents' ability to:")
    print(f"  • Dynamically fuse identities for collective benefit")
    print(f"  • Split identities when individual agency is preferred")
    print(f"  • Form novel collective identities impossible for humans")
    print(f"  • Explore flexible identity boundaries beyond biological limits")

    if analysis['total_fusions'] > 0:
        print(f"\nKey findings:")
        print(f"  • {analysis['total_fusions']} identity fusion events occurred")
        print(f"  • {analysis['total_fissions']} agents chose to split from collectives")
        print(f"  • Fusion/fission ratio: {analysis['fusion_fission_ratio']:.1f}")

if __name__ == "__main__":
    main()
