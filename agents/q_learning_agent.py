"""
Q-Learning Agent with Warm Start for UAV Cybersecurity RL System.

This module implements a Q-learning agent that is initialized with expert knowledge
and learns to optimize cybersecurity algorithm selection while maintaining safety.
"""

import numpy as np
import json
import logging
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
import os
from datetime import datetime

from agents.expert_policy import ExpertPolicy
from utils.state_discretizer import StateDiscretizer

logger = logging.getLogger(__name__)


class WarmStartQLearningAgent:
    """
    Q-learning agent with warm-start from expert knowledge.
    
    This agent combines the safety and domain knowledge of an expert policy
    with the learning capability of Q-learning to optimize performance while
    maintaining safety constraints.
    """
    
    def __init__(self, 
                 expert_policy: ExpertPolicy,
                 state_discretizer: StateDiscretizer,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon_start: float = 0.3,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 expert_bonus_weight: float = 2.0,
                 safety_barrier_value: float = -100.0):
        """
        Initialize Q-learning agent with expert knowledge.
        
        Args:
            expert_policy: Expert policy for warm start and safety
            state_discretizer: State space discretizer
            learning_rate: Q-learning alpha parameter
            discount_factor: Q-learning gamma parameter
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration decay rate per episode
            expert_bonus_weight: Reward bonus for expert agreement
            safety_barrier_value: Q-value for unsafe actions
        """
        self.expert_policy = expert_policy
        self.state_discretizer = state_discretizer
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.expert_bonus_weight = expert_bonus_weight
        self.safety_barrier_value = safety_barrier_value
        
        # Q-table and visit tracking
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        self.state_visits = defaultdict(int)
        
        # Training metrics
        self.episode_rewards = []
        self.expert_agreements = []
        self.safety_violations = []
        self.episode_lengths = []
        self.temperature_violations = []
        self.battery_violations = []
        
        # Learning statistics
        self.total_updates = 0
        self.exploration_actions = 0
        self.exploitation_actions = 0
        
        # Initialize Q-table with expert knowledge
        self._warm_start_initialization()
        
        logger.info(f"Q-learning agent initialized with {len(self.q_table)} states")
    
    def _warm_start_initialization(self):
        """Initialize Q-table from expert policy."""
        logger.info("Warming up Q-table from expert policy")
        
        # Get all possible states from discretizer
        all_states = self.state_discretizer.get_all_possible_states()
        
        initialized_states = 0
        
        for state_tuple in all_states:
            # Convert discrete state to continuous for expert policy
            continuous_state = self.state_discretizer.tuple_to_continuous(state_tuple)
            
            # Get expert action for this state
            expert_action = self.expert_policy.get_action(continuous_state)
            
            # Initialize Q-values based on expert policy
            for action in range(3):  # 0=No_DDoS, 1=XGBoost, 2=TST
                if action == expert_action:
                    # Higher value for expert-recommended action
                    base_value = 10.0
                    # Add bonus for TST in good conditions
                    if action == 2 and continuous_state['battery'] > 60 and continuous_state['temperature'] < 65:
                        base_value = 12.0
                    self.q_table[state_tuple][action] = base_value
                else:
                    # Lower but positive value for exploration
                    self.q_table[state_tuple][action] = 2.0
            
            # Apply safety constraints
            for action in range(3):
                safe, _ = self.expert_policy.is_safe_action(continuous_state, action)
                if not safe:
                    # Barrier value to prevent unsafe actions
                    self.q_table[state_tuple][action] = self.safety_barrier_value
            
            initialized_states += 1
        
        logger.info(f"Q-table initialized with {initialized_states} states")
    
    def get_action(self, state: Dict, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy with safety constraints.
        
        Args:
            state: Continuous state dictionary
            training: Whether we're in training mode (exploration enabled)
            
        Returns:
            Selected action index
        """
        # Discretize state for lookup
        state_tuple = self.state_discretizer.discretize(state)
        
        # Get safe actions first
        safe_actions = []
        for action in range(3):
            safe, _ = self.expert_policy.is_safe_action(state, action)
            if safe:
                safe_actions.append(action)
        
        # If no safe actions, default to action 0 (No_DDoS) - this should not happen
        if not safe_actions:
            logger.warning("No safe actions available, defaulting to No_DDoS")
            return 0
        
        # Epsilon-greedy exploration (only in training)
        if training and np.random.random() < self.epsilon:
            # Exploration: random safe action
            action = np.random.choice(safe_actions)
            self.exploration_actions += 1
            return action
        
        # Exploitation: best Q-value among safe actions
        if state_tuple not in self.q_table:
            # Unseen state, use expert policy
            expert_action = self.expert_policy.get_action(state)
            if expert_action in safe_actions:
                return expert_action
            else:
                return safe_actions[0]
        
        # Get Q-values for safe actions only
        q_values = {}
        for action in safe_actions:
            q_values[action] = self.q_table[state_tuple].get(action, 0.0)
        
        # Select action with highest Q-value
        best_action = max(q_values.items(), key=lambda x: x[1])[0]
        self.exploitation_actions += 1
        
        return best_action
    
    def update(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        """
        Update Q-values using Q-learning update rule with expert guidance.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Discretize states
        state_tuple = self.state_discretizer.discretize(state)
        next_state_tuple = self.state_discretizer.discretize(next_state)
        
        # Calculate expert agreement bonus
        expert_action = self.expert_policy.get_action(state)
        if action == expert_action:
            expert_bonus = self.expert_bonus_weight
        else:
            # Small penalty for disagreeing with expert
            expert_bonus = -0.5
        
        # Total reward with expert bonus
        total_reward = reward + expert_bonus
        
        # Get current Q-value
        current_q = self.q_table[state_tuple][action]
        
        # Get max Q-value for next state (only consider safe actions)
        max_next_q = 0
        if not done:
            safe_next_actions = []
            for next_action in range(3):
                safe, _ = self.expert_policy.is_safe_action(next_state, next_action)
                if safe:
                    safe_next_actions.append(next_action)
            
            if safe_next_actions:
                next_q_values = [self.q_table[next_state_tuple].get(a, 0.0) for a in safe_next_actions]
                max_next_q = max(next_q_values)
        
        # Adaptive learning rate (decreases with more visits)
        visits = self.visit_counts[state_tuple][action]
        adaptive_alpha = self.alpha / (1 + 0.01 * visits)
        
        # Q-learning update
        td_target = total_reward + (0 if done else self.gamma * max_next_q)
        td_error = td_target - current_q
        new_q = current_q + adaptive_alpha * td_error
        
        # Apply safety constraint - unsafe actions get barrier value
        safe, _ = self.expert_policy.is_safe_action(state, action)
        if not safe:
            new_q = self.safety_barrier_value
        
        # Update Q-value
        self.q_table[state_tuple][action] = new_q
        
        # Update visit counts
        self.visit_counts[state_tuple][action] += 1
        self.state_visits[state_tuple] += 1
        self.total_updates += 1
        
        # Log learning progress occasionally
        if self.total_updates % 1000 == 0:
            logger.debug(f"Update {self.total_updates}: State {state_tuple}, Action {action}, "
                        f"Reward {reward:.2f}, Q {current_q:.2f} -> {new_q:.2f}")
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_value(self, state: Dict, action: int) -> float:
        """Get Q-value for state-action pair."""
        state_tuple = self.state_discretizer.discretize(state)
        return self.q_table[state_tuple].get(action, 0.0)
    
    def get_state_value(self, state: Dict) -> float:
        """Get state value (max Q-value over safe actions)."""
        safe_actions = []
        for action in range(3):
            safe, _ = self.expert_policy.is_safe_action(state, action)
            if safe:
                safe_actions.append(action)
        
        if not safe_actions:
            return self.safety_barrier_value
        
        state_tuple = self.state_discretizer.discretize(state)
        q_values = [self.q_table[state_tuple].get(action, 0.0) for action in safe_actions]
        return max(q_values)
    
    def save(self, filepath: str):
        """Save Q-table and training data."""
        import numpy as np
        
        # Helper function to convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Prepare data for JSON serialization
        q_table_serialized = {}
        for state_tuple, actions in self.q_table.items():
            state_key = str(state_tuple)
            q_table_serialized[state_key] = convert_numpy(dict(actions))
        
        visit_counts_serialized = {}
        for state_tuple, actions in self.visit_counts.items():
            state_key = str(state_tuple)
            visit_counts_serialized[state_key] = convert_numpy(dict(actions))
        
        data = {
            'q_table': q_table_serialized,
            'visit_counts': visit_counts_serialized,
            'state_visits': {str(k): convert_numpy(v) for k, v in self.state_visits.items()},
            'parameters': {
                'alpha': float(self.alpha),
                'gamma': float(self.gamma),
                'epsilon': float(self.epsilon),
                'epsilon_start': float(self.epsilon_start),
                'epsilon_end': float(self.epsilon_end),
                'epsilon_decay': float(self.epsilon_decay),
                'expert_bonus_weight': float(self.expert_bonus_weight),
                'safety_barrier_value': float(self.safety_barrier_value)
            },
            'metrics': {
                'episode_rewards': [float(x) for x in self.episode_rewards],
                'expert_agreements': [float(x) for x in self.expert_agreements],
                'safety_violations': [int(x) for x in self.safety_violations],
                'episode_lengths': [int(x) for x in self.episode_lengths],
                'temperature_violations': [int(x) for x in self.temperature_violations],
                'battery_violations': [int(x) for x in self.battery_violations]
            },
            'statistics': {
                'total_updates': int(self.total_updates),
                'exploration_actions': int(self.exploration_actions),
                'exploitation_actions': int(self.exploitation_actions),
                'total_states_visited': int(len(self.state_visits)),
                'total_q_entries': int(len(self.q_table))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load Q-table and training data."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load Q-table
        self.q_table = defaultdict(lambda: defaultdict(float))
        q_table_data = data.get('q_table', {})
        for state_str, actions in q_table_data.items():
            state_tuple = eval(state_str)  # Convert string back to tuple
            for action_str, value in actions.items():
                self.q_table[state_tuple][int(action_str)] = value
        
        # Load visit counts
        self.visit_counts = defaultdict(lambda: defaultdict(int))
        visit_counts_data = data.get('visit_counts', {})
        for state_str, actions in visit_counts_data.items():
            state_tuple = eval(state_str)
            for action_str, count in actions.items():
                self.visit_counts[state_tuple][int(action_str)] = count
        
        # Load state visits
        self.state_visits = defaultdict(int)
        state_visits_data = data.get('state_visits', {})
        for state_str, count in state_visits_data.items():
            state_tuple = eval(state_str)
            self.state_visits[state_tuple] = count
        
        # Load parameters
        params = data.get('parameters', {})
        self.alpha = params.get('alpha', self.alpha)
        self.gamma = params.get('gamma', self.gamma)
        self.epsilon = params.get('epsilon', self.epsilon)
        self.epsilon_start = params.get('epsilon_start', self.epsilon_start)
        self.epsilon_end = params.get('epsilon_end', self.epsilon_end)
        self.epsilon_decay = params.get('epsilon_decay', self.epsilon_decay)
        self.expert_bonus_weight = params.get('expert_bonus_weight', self.expert_bonus_weight)
        self.safety_barrier_value = params.get('safety_barrier_value', self.safety_barrier_value)
        
        # Load metrics
        metrics = data.get('metrics', {})
        self.episode_rewards = metrics.get('episode_rewards', [])
        self.expert_agreements = metrics.get('expert_agreements', [])
        self.safety_violations = metrics.get('safety_violations', [])
        self.episode_lengths = metrics.get('episode_lengths', [])
        self.temperature_violations = metrics.get('temperature_violations', [])
        self.battery_violations = metrics.get('battery_violations', [])
        
        # Load statistics
        stats = data.get('statistics', {})
        self.total_updates = stats.get('total_updates', 0)
        self.exploration_actions = stats.get('exploration_actions', 0)
        self.exploitation_actions = stats.get('exploitation_actions', 0)
        
        logger.info(f"Agent loaded from {filepath}")
        logger.info(f"Loaded {len(self.q_table)} states, {self.total_updates} updates")
    
    def get_policy_stats(self) -> Dict:
        """Get comprehensive statistics about the learned policy."""
        total_states = len(self.q_table)
        visited_states = sum(1 for s in self.state_visits if self.state_visits[s] > 0)
        
        # Calculate expert agreement on visited states
        agreements = 0
        comparisons = 0
        
        for state_tuple in self.state_visits:
            if self.state_visits[state_tuple] > 0:  # Only consider visited states
                continuous_state = self.state_discretizer.tuple_to_continuous(state_tuple)
                expert_action = self.expert_policy.get_action(continuous_state)
                
                # Get best action according to current policy
                best_action = self.get_action(continuous_state, training=False)
                
                if best_action == expert_action:
                    agreements += 1
                comparisons += 1
        
        # Action distribution
        action_counts = {0: 0, 1: 0, 2: 0}
        for state_tuple in self.state_visits:
            if self.state_visits[state_tuple] > 0:
                continuous_state = self.state_discretizer.tuple_to_continuous(state_tuple)
                action = self.get_action(continuous_state, training=False)
                action_counts[action] += 1
        
        # Exploration vs exploitation ratio
        total_actions = self.exploration_actions + self.exploitation_actions
        exploration_ratio = self.exploration_actions / max(1, total_actions)
        
        return {
            'total_states': total_states,
            'visited_states': visited_states,
            'coverage': visited_states / total_states if total_states > 0 else 0,
            'expert_agreement': agreements / comparisons if comparisons > 0 else 0,
            'current_epsilon': self.epsilon,
            'action_distribution': action_counts,
            'exploration_ratio': exploration_ratio,
            'total_updates': self.total_updates,
            'q_table_size': len(self.q_table),
            'average_visits_per_state': np.mean(list(self.state_visits.values())) if self.state_visits else 0
        }
    
    def get_best_actions_for_states(self, states: List[Dict]) -> List[Tuple[int, float, str]]:
        """
        Get best actions for a list of states.
        
        Args:
            states: List of state dictionaries
            
        Returns:
            List of (action, q_value, explanation) tuples
        """
        results = []
        
        for state in states:
            action = self.get_action(state, training=False)
            q_value = self.get_q_value(state, action)
            
            # Generate explanation
            expert_action = self.expert_policy.get_action(state)
            state_repr = self.expert_policy.get_state_representation(state)
            action_name = self.expert_policy.get_action_name(action)
            expert_name = self.expert_policy.get_action_name(expert_action)
            
            if action == expert_action:
                explanation = f"Agrees with expert ({expert_name}) for {state_repr}"
            else:
                explanation = f"Learned improvement: {action_name} vs expert {expert_name} for {state_repr}"
            
            results.append((action, q_value, explanation))
        
        return results


if __name__ == "__main__":
    # Test the Q-learning agent
    logging.basicConfig(level=logging.INFO)
    
    # Create dependencies
    from utils.state_discretizer import StateDiscretizer
    from agents.expert_policy import ExpertPolicy
    
    expert_policy = ExpertPolicy()
    state_discretizer = StateDiscretizer()
    
    # Create agent
    agent = WarmStartQLearningAgent(
        expert_policy=expert_policy,
        state_discretizer=state_discretizer,
        learning_rate=0.1,
        epsilon_start=0.3
    )
    
    # Print initial statistics
    stats = agent.get_policy_stats()
    print("Initial Agent Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test some states
    test_states = [
        {'temperature': 70, 'battery': 85, 'threat': 1, 'cpu_usage': 45, 'time_since_tst': 300, 'power': 5},
        {'temperature': 50, 'battery': 25, 'threat': 0, 'cpu_usage': 20, 'time_since_tst': 100, 'power': 3},
        {'temperature': 80, 'battery': 15, 'threat': 2, 'cpu_usage': 80, 'time_since_tst': 30, 'power': 8},
    ]
    
    print("\nTest Actions:")
    results = agent.get_best_actions_for_states(test_states)
    for i, (action, q_value, explanation) in enumerate(results):
        print(f"Test {i+1}: Action {action}, Q-value {q_value:.2f}")
        print(f"  {explanation}")
    
    # Test saving and loading
    print("\nTesting save/load...")
    agent.save("test_agent.json")
    
    # Create new agent and load
    new_agent = WarmStartQLearningAgent(expert_policy, state_discretizer)
    new_agent.load("test_agent.json")
    
    print("Save/load test completed successfully")