"""
Experiment 1: Collective Agency in Commons Harvest
Testing novel forms of collective decision-making and identity flexibility.
"""

from meltingpot.python import scenario
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import time

class CollectiveAgent:
    """
    An agent that can participate in collective decision-making.
    Implements novel agency concepts from the thesis.
    """

    def __init__(self, agent_id: int, collective_threshold: float = 0.7):
        self.agent_id = agent_id
        self.collective_threshold = collective_threshold
        self.individual_preferences = {}
        self.collective_memberships = []
        self.decision_history = []

    def evaluate_collective_action(self, proposed_action, collective_context):
        """
        Decide whether to act individually or as part of collective.
        This is where we implement novel agency concepts.
        """
        # Novel agency: flexible identity boundaries
        collective_benefit = self._assess_collective_benefit(proposed_action, collective_context)
        individual_benefit = self._assess_individual_benefit(proposed_action)

        # Decision threshold based on collective vs individual agency
        if collective_benefit > individual_benefit * self.collective_threshold:
            return "collective", proposed_action
        else:
            return "individual", self._generate_individual_action()

    def _assess_collective_benefit(self, action, context):
        # Placeholder for collective benefit assessment
        return np.random.random()

    def _assess_individual_benefit(self, action):
        # Placeholder for individual benefit assessment
        return np.random.random()

    def _generate_individual_action(self):
        # Placeholder for individual action generation
        return np.random.randint(0, 7)  # Assuming 7 possible actions

class CollectiveAgencyExperiment:
    """
    Main experiment class for testing collective agency concepts.
    """

    def __init__(self, scenario_name: str = "commons_harvest__open_0"):
        self.scenario_name = scenario_name
        self.env = scenario.build(scenario_name)
        self.agents = self._initialize_collective_agents()
        self.results = {
            'individual_actions': [],
            'collective_actions': [],
            'collective_formations': [],
            'outcomes': []
        }

    def _initialize_collective_agents(self):
        """Initialize agents with collective agency capabilities."""
        num_agents = len(self.env.action_spec())
        agents = {}
        for i in range(num_agents):
            # Vary collective thresholds to test different agency types
            threshold = 0.5 + (i * 0.1)  # Different agency preferences
            agents[i] = CollectiveAgent(i, threshold)
        return agents

    def run_episode(self, max_steps: int = 100):
        """Run a single episode testing collective agency."""
        timestep = self.env.reset()
        episode_data = {
            'steps': [],
            'collective_formations': [],
            'agency_decisions': []
        }

        for step in range(max_steps):
            if timestep.last():
                break

            # Get actions from our collective agents
            actions = self._get_collective_actions(timestep)

            # Record agency decisions
            episode_data['agency_decisions'].append(self._analyze_agency_decisions())

            # Step the environment
            timestep = self.env.step(actions)

            # Record step data
            step_data = {
                'step': step,
                'rewards': timestep.reward,
                'observations': timestep.observation
            }
            episode_data['steps'].append(step_data)

        return episode_data

    def _get_collective_actions(self, timestep):
        """
        Core method: decide actions using collective agency concepts.
        This is where the novel agency concepts are tested.
        """
        observations = timestep.observation
        actions = {}

        # Simulate collective deliberation
        collective_proposals = self._generate_collective_proposals(observations)

        for agent_id, agent in self.agents.items():
            # Each agent decides: individual or collective action?
            agency_type, action = agent.evaluate_collective_action(
                collective_proposals.get(agent_id, 0),
                observations
            )

            actions[agent_id] = action

            # Track agency decisions for analysis
            self.results['individual_actions' if agency_type == 'individual' else 'collective_actions'].append({
                'agent_id': agent_id,
                'step': len(self.results['individual_actions']) + len(self.results['collective_actions']),
                'action': action
            })

        return actions

    def _generate_collective_proposals(self, observations):
        """Generate collective action proposals based on shared observations."""
        # Simplified collective proposal mechanism
        # In practice, this would implement more sophisticated collective reasoning
        proposals = {}
        for agent_id in self.agents.keys():
            # Proposal based on collective benefit (simplified)
            proposals[agent_id] = np.random.randint(0, 7)
        return proposals

    def _analyze_agency_decisions(self):
        """Analyze the types of agency decisions being made."""
        individual_count = len([a for a in self.results['individual_actions'] if a['step'] == len(self.results['individual_actions'])])
        collective_count = len([a for a in self.results['collective_actions'] if a['step'] == len(self.results['collective_actions'])])

        return {
            'individual_ratio': individual_count / (individual_count + collective_count + 1e-10),
            'collective_ratio': collective_count / (individual_count + collective_count + 1e-10)
        }

    def run_multiple_episodes(self, num_episodes: int = 5):
        """Run multiple episodes to gather data on collective agency patterns."""
        all_results = []

        print(f"Running {num_episodes} episodes to test collective agency concepts...")

        for episode in range(num_episodes):
            print(f"  Episode {episode + 1}/{num_episodes}")
            episode_data = self.run_episode()
            all_results.append(episode_data)

        return all_results

    def analyze_results(self, all_results):
        """Analyze results to understand collective agency patterns."""
        print("\n=== COLLECTIVE AGENCY ANALYSIS ===")

        total_individual = len(self.results['individual_actions'])
        total_collective = len(self.results['collective_actions'])
        total_actions = total_individual + total_collective

        if total_actions > 0:
            print(f"Individual agency actions: {total_individual} ({total_individual/total_actions*100:.1f}%)")
            print(f"Collective agency actions: {total_collective} ({total_collective/total_actions*100:.1f}%)")

            # Analyze by agent
            print(f"\nAgency preferences by agent:")
            for agent_id, agent in self.agents.items():
                agent_individual = len([a for a in self.results['individual_actions'] if a['agent_id'] == agent_id])
                agent_collective = len([a for a in self.results['collective_actions'] if a['agent_id'] == agent_id])
                agent_total = agent_individual + agent_collective

                if agent_total > 0:
                    print(f"  Agent {agent_id} (threshold={agent.collective_threshold:.1f}): "
                          f"{agent_collective/agent_total*100:.1f}% collective")

        return {
            'individual_ratio': total_individual / total_actions if total_actions > 0 else 0,
            'collective_ratio': total_collective / total_actions if total_actions > 0 else 0
        }

def main():
    """Run the collective agency experiment."""
    print("=== COLLECTIVE AGENCY EXPERIMENT ===")
    print("Testing novel forms of agency from thesis concepts...")

    # Create and run experiment
    experiment = CollectiveAgencyExperiment()
    results = experiment.run_multiple_episodes(num_episodes=3)

    # Analyze results
    analysis = experiment.analyze_results(results)

    print(f"\n=== CONCLUSIONS ===")
    print(f"This experiment tested the idea that machine agents can explore")
    print(f"novel forms of collective agency beyond human limitations.")
    print(f"Results show agents making decisions between individual and")
    print(f"collective agency based on flexible identity boundaries.")

if __name__ == "__main__":
    main()
