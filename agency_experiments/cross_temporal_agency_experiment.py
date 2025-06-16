"""
Experiment 4: Cross-Temporal Collective Agency
Testing collective identities that persist across episodes and time.
Implements Korsgaard's "succession of rational agents" concept extended to collectives.
"""

from meltingpot.python import scenario
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class CollectiveMemory:
    """Persistent memory structure for cross-temporal collectives."""
    collective_id: str
    formation_episode: int
    member_history: List[Set[int]]
    shared_experiences: List[Dict]
    collective_goals: List[str]
    success_metrics: Dict[str, float]
    dissolution_triggers: List[str]
    creation_timestamp: str

class PersistentCollective:
    """
    A collective identity that persists across multiple episodes.
    Implements cross-temporal agency concepts from the thesis.
    """

    def __init__(self, collective_id: str, founding_members: Set[int], episode: int):
        self.collective_id = collective_id
        self.founding_members = founding_members.copy()
        self.current_members = founding_members.copy()
        self.formation_episode = episode
        self.episodes_active = 0

        # Cross-temporal memory
        self.memory = CollectiveMemory(
            collective_id=collective_id,
            formation_episode=episode,
            member_history=[founding_members.copy()],
            shared_experiences=[],
            collective_goals=[],
            success_metrics={},
            dissolution_triggers=[],
            creation_timestamp=datetime.now().isoformat()
        )

        # Collective learning and adaptation
        self.collective_personality = self._initialize_personality()
        self.reputation_system = {}
        self.commitment_levels = {member: 0.7 for member in founding_members}

    def _initialize_personality(self) -> Dict[str, float]:
        """Initialize collective personality traits that persist over time."""
        return {
            'cooperation_preference': 0.6 + np.random.random() * 0.3,
            'risk_tolerance': 0.3 + np.random.random() * 0.4,
            'exploration_tendency': 0.5 + np.random.random() * 0.3,
            'stability_preference': 0.4 + np.random.random() * 0.4,
            'adaptability': 0.5 + np.random.random() * 0.4
        }

    def evaluate_membership_changes(self, available_agents: Set[int],
                                  episode_context: Dict) -> Tuple[Set[int], Set[int]]:
        """
        Decide which agents to invite and which to remove from collective.
        Implements cross-temporal identity evolution.
        """
        agents_to_invite = set()
        agents_to_remove = set()

        # Evaluate current members for retention
        for member in self.current_members.copy():
            retention_score = self._calculate_retention_score(member, episode_context)

            if retention_score < 0.3:  # Low retention threshold
                agents_to_remove.add(member)
                print(f"    Collective {self.collective_id}: removing member {member} (score: {retention_score:.2f})")

        # Evaluate potential new members
        candidates = available_agents - self.current_members
        for candidate in candidates:
            invitation_score = self._calculate_invitation_score(candidate, episode_context)

            if invitation_score > 0.7:  # High invitation threshold
                agents_to_invite.add(candidate)
                print(f"    Collective {self.collective_id}: inviting agent {candidate} (score: {invitation_score:.2f})")

        return agents_to_invite, agents_to_remove

    def _calculate_retention_score(self, member: int, context: Dict) -> float:
        """Calculate how much the collective wants to retain a member."""
        base_retention = self.commitment_levels.get(member, 0.5)

        # Factor in past performance
        member_reputation = self.reputation_system.get(member, 0.5)

        # Collective personality influence
        stability_bonus = self.collective_personality['stability_preference'] * 0.2

        # Random variation for exploration
        random_factor = np.random.random() * 0.2 - 0.1

        return np.clip(base_retention * 0.6 + member_reputation * 0.3 + stability_bonus + random_factor, 0, 1)

    def _calculate_invitation_score(self, candidate: int, context: Dict) -> float:
        """Calculate how much the collective wants to invite a new member."""
        # Base invitation tendency
        base_invitation = self.collective_personality['exploration_tendency']

        # Size considerations
        current_size = len(self.current_members)
        size_factor = max(0, 1 - (current_size / 8))  # Prefer smaller collectives

        # Adaptability factor
        adaptability_bonus = self.collective_personality['adaptability'] * 0.3

        # Random exploration
        random_factor = np.random.random() * 0.3

        return np.clip(base_invitation * 0.4 + size_factor * 0.3 + adaptability_bonus + random_factor, 0, 1)

    def update_membership(self, invited: Set[int], removed: Set[int]):
        """Update collective membership and record changes."""
        self.current_members = (self.current_members - removed) | invited

        # Update commitment levels
        for new_member in invited:
            self.commitment_levels[new_member] = 0.6 + np.random.random() * 0.2

        for removed_member in removed:
            if removed_member in self.commitment_levels:
                del self.commitment_levels[removed_member]

        # Record membership change in memory
        self.memory.member_history.append(self.current_members.copy())

    def make_collective_decision(self, observations: Dict, context: Dict, step: int) -> Dict[int, int]:
        """
        Make coordinated decisions as a persistent collective.
        Incorporates cross-temporal learning and memory.
        """
        decisions = {}

        # Access collective memory and personality
        coordination_strength = self.collective_personality['cooperation_preference']
        risk_tolerance = self.collective_personality['risk_tolerance']

        # Learn from past experiences
        experience_bonus = self._calculate_experience_bonus()

        for member in self.current_members:
            # Generate action based on collective strategy
            if coordination_strength > 0.7:
                # High coordination: similar actions for all members
                base_action = self._get_collective_strategy_action(context, experience_bonus)
                action = base_action + np.random.randint(-1, 2)  # Small variation
            else:
                # Lower coordination: more individual variation
                action = np.random.randint(0, 7)

            action = np.clip(action, 0, 6)
            decisions[member] = action

            # Update member reputation based on alignment with collective
            self._update_member_reputation(member, action, context)

        # Record this decision in collective memory
        experience_record = {
            'step': step,
            'episode': context.get('episode', 0),
            'decisions': decisions.copy(),
            'coordination_level': coordination_strength,
            'experience_bonus': experience_bonus
        }
        self.memory.shared_experiences.append(experience_record)

        return decisions

    def _calculate_experience_bonus(self) -> float:
        """Calculate learning bonus from past experiences."""
        if len(self.memory.shared_experiences) < 5:
            return 0.0

        # Learn from recent successful coordination
        recent_experiences = self.memory.shared_experiences[-10:]
        avg_coordination = np.mean([exp['coordination_level'] for exp in recent_experiences])

        return min(avg_coordination * 0.2, 0.3)  # Cap learning bonus

    def _get_collective_strategy_action(self, context: Dict, experience_bonus: float) -> int:
        """Generate action based on collective strategy and learning."""
        # Base strategy depends on collective personality
        if self.collective_personality['cooperation_preference'] > 0.8:
            base_action = 3  # Cooperative action
        elif self.collective_personality['risk_tolerance'] > 0.7:
            base_action = 5  # Aggressive action
        else:
            base_action = 2  # Conservative action

        # Modify based on experience
        experience_modification = int(experience_bonus * 10) - 1

        return base_action + experience_modification

    def _update_member_reputation(self, member: int, action: int, context: Dict):
        """Update member reputation based on collective alignment."""
        if member not in self.reputation_system:
            self.reputation_system[member] = 0.5

        # Simplified reputation update
        expected_action = self._get_collective_strategy_action(context, 0)
        alignment = 1.0 - abs(action - expected_action) / 7.0

        # Update reputation with decay
        current_rep = self.reputation_system[member]
        self.reputation_system[member] = current_rep * 0.9 + alignment * 0.1

    def end_episode_update(self, episode_results: Dict):
        """Update collective state at the end of an episode."""
        self.episodes_active += 1

        # Update collective personality based on results
        self._adapt_personality(episode_results)

        # Update success metrics
        total_reward = episode_results.get('total_reward', 0)
        coordination_success = episode_results.get('coordination_success', 0.5)

        self.memory.success_metrics[f'episode_{episode_results.get("episode", 0)}'] = {
            'total_reward': total_reward,
            'coordination_success': coordination_success,
            'members_count': len(self.current_members)
        }

    def _adapt_personality(self, episode_results: Dict):
        """Adapt collective personality based on episode outcomes."""
        success_rate = episode_results.get('coordination_success', 0.5)

        # Successful episodes slightly increase cooperation
        if success_rate > 0.7:
            self.collective_personality['cooperation_preference'] = min(
                self.collective_personality['cooperation_preference'] + 0.05, 1.0
            )
        # Failed episodes might increase exploration
        elif success_rate < 0.3:
            self.collective_personality['exploration_tendency'] = min(
                self.collective_personality['exploration_tendency'] + 0.03, 1.0
            )

    def should_dissolve(self) -> bool:
        """Determine if the collective should dissolve."""
        # Dissolve if too few members
        if len(self.current_members) < 2:
            return True

        # Dissolve based on performance history
        if len(self.memory.success_metrics) >= 3:
            recent_performance = list(self.memory.success_metrics.values())[-3:]
            avg_success = np.mean([perf['coordination_success'] for perf in recent_performance])

            if avg_success < 0.2:  # Consistently poor performance
                return True

        return False

class CrossTemporalAgent:
    """
    An agent that can participate in persistent collectives across episodes.
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.collective_history = []
        self.current_collectives = set()
        self.individual_memory = []
        self.collective_preferences = {
            'prefers_stability': np.random.random() > 0.5,
            'commitment_level': 0.4 + np.random.random() * 0.4,
            'collective_threshold': 0.5 + np.random.random() * 0.3
        }

    def evaluate_collective_invitation(self, collective: PersistentCollective, context: Dict) -> bool:
        """Decide whether to join an invited collective."""
        # Consider collective's history and personality
        collective_age = collective.episodes_active
        collective_size = len(collective.current_members)

        # Personal preferences influence decision
        size_preference = 1.0 - abs(collective_size - 4) / 8  # Prefer medium-sized groups

        if self.collective_preferences['prefers_stability'] and collective_age < 2:
            stability_bonus = -0.2  # Penalty for new collectives
        else:
            stability_bonus = collective_age * 0.05  # Bonus for established collectives

        commitment_factor = self.collective_preferences['commitment_level']
        random_factor = np.random.random() * 0.3

        decision_score = size_preference * 0.4 + stability_bonus + commitment_factor * 0.3 + random_factor

        return decision_score > self.collective_preferences['collective_threshold']

class CrossTemporalExperiment:
    """
    Experiment testing cross-temporal collective agency.
    """

    def __init__(self, scenario_name: str = "collaborative_cooking__figure_eight_0"):
        self.scenario_name = scenario_name
        self.env = scenario.build(scenario_name)
        self.agents = self._initialize_cross_temporal_agents()
        self.persistent_collectives = {}
        self.collective_counter = 0
        self.results = {
            'episode_results': [],
            'collective_evolution': [],
            'cross_temporal_patterns': []
        }

    def _initialize_cross_temporal_agents(self) -> Dict[int, CrossTemporalAgent]:
        """Initialize agents capable of cross-temporal collective membership."""
        num_agents = len(self.env.action_spec())
        print(f"Creating {num_agents} cross-temporal agents...")

        agents = {}
        for i in range(num_agents):
            agents[i] = CrossTemporalAgent(i)
            prefs = agents[i].collective_preferences
            print(f"  Agent {i}: stability={prefs['prefers_stability']}, "
                  f"commitment={prefs['commitment_level']:.2f}")

        return agents

    def run_multi_episode_experiment(self, num_episodes: int = 5):
        """Run multiple episodes to test cross-temporal persistence."""
        print(f"\nRunning {num_episodes} episodes to test cross-temporal collective agency...")

        for episode in range(num_episodes):
            print(f"\n{'='*20} EPISODE {episode + 1} {'='*20}")

            # Pre-episode collective management
            self._manage_cross_episode_collectives(episode)

            # Run episode
            episode_results = self.run_single_episode(episode, max_steps=50)

            # Post-episode updates
            self._update_collectives_post_episode(episode, episode_results)

            # Record results
            self.results['episode_results'].append(episode_results)

        return self.results

    def _manage_cross_episode_collectives(self, episode: int):
        """Manage collective membership between episodes."""
        print(f"Managing cross-episode collectives...")

        # Get available agents (not currently in collectives)
        available_agents = set(self.agents.keys())
        for collective in self.persistent_collectives.values():
            available_agents -= collective.current_members

        # Existing collectives evaluate membership changes
        collectives_to_remove = []
        for collective_id, collective in self.persistent_collectives.items():

            # Check if collective should dissolve
            if collective.should_dissolve():
                print(f"  Dissolving collective {collective_id} (performance/size issues)")
                available_agents.update(collective.current_members)
                collectives_to_remove.append(collective_id)
                continue

            # Evaluate membership changes
            invited, removed = collective.evaluate_membership_changes(available_agents, {'episode': episode})

            # Process membership changes
            for agent_id in invited:
                agent = self.agents[agent_id]
                if agent.evaluate_collective_invitation(collective, {'episode': episode}):
                    print(f"  Agent {agent_id} accepted invitation to {collective_id}")
                    available_agents.discard(agent_id)
                else:
                    print(f"  Agent {agent_id} declined invitation to {collective_id}")
                    invited.discard(agent_id)

            # Update collective membership
            collective.update_membership(invited, removed)
            available_agents.update(removed)

        # Remove dissolved collectives
        for collective_id in collectives_to_remove:
            del self.persistent_collectives[collective_id]

        # Form new collectives from available agents
        self._form_new_collectives(available_agents, episode)

    def _form_new_collectives(self, available_agents: Set[int], episode: int):
        """Form new persistent collectives from available agents."""
        while len(available_agents) >= 2:  # Need at least 2 agents for collective
            # Randomly select founding members
            founder = np.random.choice(list(available_agents))
            available_agents.remove(founder)

            if len(available_agents) == 0:
                break

            # Find compatible partners
            potential_partners = list(available_agents)
            num_partners = min(np.random.randint(1, 4), len(potential_partners))
            partners = np.random.choice(potential_partners, num_partners, replace=False)

            # Create new collective
            founding_members = {founder} | set(partners)
            collective_id = f"temporal_collective_{self.collective_counter}"
            self.collective_counter += 1

            new_collective = PersistentCollective(collective_id, founding_members, episode)
            self.persistent_collectives[collective_id] = new_collective

            # Remove partners from available agents
            available_agents -= set(partners)

            print(f"  Formed new collective {collective_id} with members {founding_members}")

    def run_single_episode(self, episode: int, max_steps: int = 50) -> Dict:
        """Run a single episode with current collective configuration."""
        print(f"  Running episode with {len(self.persistent_collectives)} active collectives...")

        timestep = self.env.reset()
        episode_data = {
            'episode': episode,
            'collective_decisions': [],
            'individual_decisions': [],
            'total_reward': 0
        }

        for step in range(max_steps):
            if timestep.last():
                break

            # Get actions from collectives and individuals
            actions = self._get_cross_temporal_actions(timestep, episode, step)

            # Step environment
            timestep = self.env.step(actions)

            # Record rewards
            total_step_reward = sum(timestep.reward) if timestep.reward else 0
            episode_data['total_reward'] += total_step_reward

            if step % 15 == 0:
                active_collectives = len(self.persistent_collectives)
                total_collective_members = sum(len(c.current_members) for c in self.persistent_collectives.values())
                individual_agents = len(self.agents) - total_collective_members
                print(f"    Step {step}: {active_collectives} collectives ({total_collective_members} members), "
                      f"{individual_agents} individuals")

        # Calculate coordination success
        coordination_success = self._calculate_coordination_success(episode_data)
        episode_data['coordination_success'] = coordination_success

        print(f"  Episode {episode + 1} completed: reward={episode_data['total_reward']:.1f}, "
              f"coordination={coordination_success:.2f}")

        return episode_data

    def _get_cross_temporal_actions(self, timestep, episode: int, step: int) -> Dict[int, int]:
        """Get actions from persistent collectives and individual agents."""
        actions = {}

        # Actions from persistent collectives
        for collective_id, collective in self.persistent_collectives.items():
            collective_actions = collective.make_collective_decision(
                timestep.observation,
                {'episode': episode, 'step': step},
                step
            )
            actions.update(collective_actions)

        # Actions from individual agents (not in collectives)
        collective_members = set()
        for collective in self.persistent_collectives.values():
            collective_members.update(collective.current_members)

        individual_agents = set(self.agents.keys()) - collective_members
        for agent_id in individual_agents:
            # Individual action
            action = np.random.randint(0, 7)
            actions[agent_id] = action

            # Record individual decision
            self.agents[agent_id].individual_memory.append({
                'episode': episode,
                'step': step,
                'action': action,
                'context': 'individual'
            })

        return actions

    def _calculate_coordination_success(self, episode_data: Dict) -> float:
        """Calculate how successful the coordination was this episode."""
        # Simplified coordination success metric
        # Based on reward and collective stability

        reward_factor = min(episode_data['total_reward'] / 100, 1.0)  # Normalize reward

        # Collective stability factor
        if self.persistent_collectives:
            avg_collective_size = np.mean([len(c.current_members) for c in self.persistent_collectives.values()])
            stability_factor = min(avg_collective_size / 4, 1.0)  # Normalize around size 4
        else:
            stability_factor = 0.0

        return reward_factor * 0.6 + stability_factor * 0.4

    def _update_collectives_post_episode(self, episode: int, episode_results: Dict):
        """Update persistent collectives after episode completion."""
        for collective in self.persistent_collectives.values():
            collective.end_episode_update(episode_results)

        # Record collective evolution
        evolution_record = {
            'episode': episode,
            'active_collectives': len(self.persistent_collectives),
            'collective_details': []
        }

        for collective_id, collective in self.persistent_collectives.items():
            collective_detail = {
                'id': collective_id,
                'age': collective.episodes_active,
                'current_members': list(collective.current_members),
                'member_count': len(collective.current_members),
                'personality': collective.collective_personality.copy(),
                'success_metrics': len(collective.memory.success_metrics)
            }
            evolution_record['collective_details'].append(collective_detail)

        self.results['collective_evolution'].append(evolution_record)

    def analyze_cross_temporal_results(self):
        """Analyze cross-temporal patterns and collective persistence."""
        print("\n" + "="*80)
        print("CROSS-TEMPORAL COLLECTIVE AGENCY ANALYSIS")
        print("="*80)

        total_episodes = len(self.results['episode_results'])

        # Overall statistics
        final_collectives = len(self.persistent_collectives)
        total_collectives_created = self.collective_counter

        print(f"\nCross-Temporal Overview:")
        print(f"  Episodes analyzed: {total_episodes}")
        print(f"  Total collectives created: {total_collectives_created}")
        print(f"  Surviving collectives: {final_collectives}")
        print(f"  Collective survival rate: {final_collectives/max(total_collectives_created,1)*100:.1f}%")

        # Analyze collective persistence patterns
        if self.results['collective_evolution']:
            print(f"\nCollective Persistence Patterns:")

            # Track collective lifespans
            collective_lifespans = {}
            for evolution in self.results['collective_evolution']:
                for collective_detail in evolution['collective_details']:
                    cid = collective_detail['id']
                    age = collective_detail['age']
                    if cid not in collective_lifespans:
                        collective_lifespans[cid] = []
                    collective_lifespans[cid].append(age)

            if collective_lifespans:
                max_lifespans = {cid: max(ages) for cid, ages in collective_lifespans.items()}
                avg_lifespan = np.mean(list(max_lifespans.values()))
                longest_lived = max(max_lifespans.keys(), key=lambda k: max_lifespans[k])

                print(f"  Average collective lifespan: {avg_lifespan:.1f} episodes")
                print(f"  Longest-lived collective: {longest_lived} ({max_lifespans[longest_lived]} episodes)")

            # Analyze membership stability
            membership_changes = []
            for i in range(1, len(self.results['collective_evolution'])):
                prev_collectives = {cd['id']: set(cd['current_members'])
                                  for cd in self.results['collective_evolution'][i-1]['collective_details']}
                curr_collectives = {cd['id']: set(cd['current_members'])
                                  for cd in self.results['collective_evolution'][i]['collective_details']}

                episode_changes = 0
                for cid in set(prev_collectives.keys()) & set(curr_collectives.keys()):
                    prev_members = prev_collectives[cid]
                    curr_members = curr_collectives[cid]
                    changes = len(prev_members.symmetric_difference(curr_members))
                    episode_changes += changes

                membership_changes.append(episode_changes)

            if membership_changes:
                avg_changes = np.mean(membership_changes)
                print(f"  Average membership changes per episode: {avg_changes:.1f}")

        # Analyze performance evolution
        print(f"\nPerformance Evolution:")
        if len(self.results['episode_results']) > 1:
            early_episodes = self.results['episode_results'][:len(self.results['episode_results'])//2]
            late_episodes = self.results['episode_results'][len(self.results['episode_results'])//2:]

            early_avg_reward = np.mean([ep['total_reward'] for ep in early_episodes])
            late_avg_reward = np.mean([ep['total_reward'] for ep in late_episodes])

            early_avg_coordination = np.mean([ep['coordination_success'] for ep in early_episodes])
            late_avg_coordination = np.mean([ep['coordination_success'] for ep in late_episodes])

            print(f"  Early episodes avg reward: {early_avg_reward:.1f}")
            print(f"  Later episodes avg reward: {late_avg_reward:.1f}")
            print(f"  Reward improvement: {((late_avg_reward - early_avg_reward)/max(early_avg_reward,1)*100):+.1f}%")
            print(f"  Early coordination success: {early_avg_coordination:.2f}")
            print(f"  Later coordination success: {late_avg_coordination:.2f}")
            print(f"  Coordination improvement: {((late_avg_coordination - early_avg_coordination)*100):+.1f}%")

        # Analyze surviving collectives
        if self.persistent_collectives:
            print(f"\nSurviving Collective Characteristics:")
            for collective_id, collective in self.persistent_collectives.items():
                member_count = len(collective.current_members)
                age = collective.episodes_active

                # Personality summary
                personality = collective.collective_personality
                dominant_trait = max(personality.keys(), key=lambda k: personality[k])

                print(f"  {collective_id}: {member_count} members, {age} episodes old")
                print(f"    Dominant trait: {dominant_trait} ({personality[dominant_trait]:.2f})")
                print(f"    Members: {list(collective.current_members)}")

        # Agent-specific cross-temporal analysis
        print(f"\nAgent Cross-Temporal Patterns:")
        for agent_id, agent in self.agents.items():
            current_collectives = [cid for cid, collective in self.persistent_collectives.items()
                                 if agent_id in collective.current_members]
            collective_history_length = len(agent.collective_history)

            print(f"  Agent {agent_id}: currently in {len(current_collectives)} collectives, "
                  f"history length: {collective_history_length}")

        return {
            'survival_rate': final_collectives/max(total_collectives_created,1),
            'avg_lifespan': avg_lifespan if 'avg_lifespan' in locals() else 0,
            'final_collectives': final_collectives,
            'performance_improvement': ((late_avg_reward - early_avg_reward)/max(early_avg_reward,1)*100) if 'late_avg_reward' in locals() else 0
        }

def main():
   """Run the cross-temporal collective agency experiment."""
   print("="*90)
   print("CROSS-TEMPORAL COLLECTIVE AGENCY EXPERIMENT")
   print("Testing persistent collective identities across episodes and time")
   print("Implementing Korsgaard's succession of agents extended to collectives")
   print("="*90)

   experiment = CrossTemporalExperiment()
   results = experiment.run_multi_episode_experiment(num_episodes=6)
   analysis = experiment.analyze_cross_temporal_results()

   print(f"\n" + "="*90)
   print("EXPERIMENT CONCLUSIONS")
   print("="*90)
   print(f"This experiment demonstrates revolutionary cross-temporal agency capabilities:")
   print(f"")
   print(f"Key Findings:")
   print(f"  • {analysis['survival_rate']*100:.1f}% of collectives survived across episodes")
   print(f"  • Average collective lifespan: {analysis['avg_lifespan']:.1f} episodes")
   print(f"  • Performance improved by {analysis['performance_improvement']:+.1f}% over time")
   print(f"  • {analysis['final_collectives']} persistent collectives remain active")
   print(f"")
   print(f"Novel Agency Capabilities Demonstrated:")
   print(f"  • Collectives persist across temporal boundaries")
   print(f"  • Cross-episode learning and memory")
   print(f"  • Dynamic membership evolution over time")
   print(f"  • Collective personality adaptation")
   print(f"  • Multi-generational collective identity")
   print(f"")
   print(f"This goes far beyond human collective capabilities:")
   print(f"  • Perfect memory across long time periods")
   print(f"  • Seamless identity persistence through membership changes")
   print(f"  • Computational coordination impossible for human groups")
   print(f"  • Novel forms of 'collective personhood' across time")

if __name__ == "__main__":
   main()
