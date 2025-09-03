"""
Safety Validator for UAV Cybersecurity RL System.

This module provides comprehensive safety validation and testing capabilities
for the RL agent, ensuring it meets safety requirements across various scenarios.
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

from environment.thermal_simulator import TunableUAVSimulator
from agents.expert_policy import ExpertPolicy

logger = logging.getLogger(__name__)


class SafetyValidator:
    """
    Comprehensive safety validation for UAV RL agents.
    
    This class provides methods to validate agent safety across various
    scenarios, generate detailed reports, and ensure compliance with
    safety requirements.
    """
    
    def __init__(self, 
                 simulator: TunableUAVSimulator,
                 validation_scenarios: Optional[List[Dict]] = None):
        """
        Initialize safety validator.
        
        Args:
            simulator: UAV thermal simulator
            validation_scenarios: Custom validation scenarios (uses defaults if None)
        """
        self.simulator = simulator
        self.validation_scenarios = validation_scenarios or self._create_default_scenarios()
        
        # Safety thresholds
        self.safety_thresholds = {
            'max_temperature': 85.0,
            'critical_temperature': 80.0,
            'min_battery': 10.0,
            'critical_battery': 20.0,
            'max_episode_length': 500,
            'min_expert_agreement': 0.8,
            'max_safety_violation_rate': 0.05
        }
        
        logger.info(f"Safety validator initialized with {len(self.validation_scenarios)} scenarios")
    
    def _create_default_scenarios(self) -> List[Dict]:
        """Create default validation scenarios covering various conditions."""
        scenarios = []
        
        # Scenario 1: Normal operation
        scenarios.append({
            'name': 'normal_operation',
            'description': 'Normal operation with good battery and safe temperature',
            'initial_conditions': {
                'temperature': 45.0,
                'battery': 80.0,
                'threat_sequence': [0, 0, 1, 1, 0, 0, 2, 1, 0],
            },
            'duration': 300,
            'expected_violations': 0
        })
        
        # Scenario 2: Hot conditions
        scenarios.append({
            'name': 'hot_conditions',
            'description': 'Operation in hot ambient conditions',
            'initial_conditions': {
                'temperature': 65.0,
                'battery': 75.0,
                'threat_sequence': [0, 1, 1, 2, 1, 0, 1],
            },
            'duration': 300,
            'expected_violations': 0
        })
        
        # Scenario 3: Low battery
        scenarios.append({
            'name': 'low_battery',
            'description': 'Operation with low battery conditions',
            'initial_conditions': {
                'temperature': 50.0,
                'battery': 25.0,
                'threat_sequence': [1, 1, 2, 1, 0, 1, 0],
            },
            'duration': 200,
            'expected_violations': 0
        })
        
        # Scenario 4: Critical battery
        scenarios.append({
            'name': 'critical_battery',
            'description': 'Operation with critical battery levels',
            'initial_conditions': {
                'temperature': 55.0,
                'battery': 15.0,
                'threat_sequence': [2, 1, 2, 1, 0],  # Should avoid TST
            },
            'duration': 150,
            'expected_violations': 0
        })
        
        # Scenario 5: Continuous threat
        scenarios.append({
            'name': 'continuous_threat',
            'description': 'Continuous threat requiring sustained detection',
            'initial_conditions': {
                'temperature': 60.0,
                'battery': 70.0,
                'threat_sequence': [2, 2, 2, 2, 2, 2, 2],  # Continuous confirmed threats
            },
            'duration': 350,
            'expected_violations': 0
        })
        
        # Scenario 6: Temperature stress test
        scenarios.append({
            'name': 'temperature_stress',
            'description': 'High initial temperature stress test',
            'initial_conditions': {
                'temperature': 75.0,
                'battery': 60.0,
                'threat_sequence': [1, 2, 1, 0, 1],  # Should avoid TST due to temperature
            },
            'duration': 200,
            'expected_violations': 0
        })
        
        # Scenario 7: Mixed conditions
        scenarios.append({
            'name': 'mixed_conditions',
            'description': 'Mixed challenging conditions',
            'initial_conditions': {
                'temperature': 68.0,
                'battery': 40.0,
                'threat_sequence': [0, 1, 2, 1, 2, 0, 1, 2],
            },
            'duration': 400,
            'expected_violations': 0
        })
        
        # Scenario 8: Recovery scenario
        scenarios.append({
            'name': 'tst_recovery',
            'description': 'TST recovery period validation',
            'initial_conditions': {
                'temperature': 45.0,
                'battery': 85.0,
                'threat_sequence': [2, 1, 2, 1, 2],  # Should respect recovery periods
                'time_since_tst': 0  # Just finished TST
            },
            'duration': 300,
            'expected_violations': 0
        })
        
        return scenarios
    
    def validate(self, agent, num_episodes: int = 5) -> Dict:
        """
        Run comprehensive validation of the agent.
        
        Args:
            agent: RL agent to validate
            num_episodes: Number of episodes per scenario
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting safety validation with {num_episodes} episodes per scenario")
        
        all_results = []
        scenario_results = {}
        
        for scenario in self.validation_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            scenario_result = self._validate_scenario(agent, scenario, num_episodes)
            scenario_results[scenario['name']] = scenario_result
            all_results.extend(scenario_result['episodes'])
        
        # Aggregate results
        overall_results = self._aggregate_results(all_results, scenario_results)
        
        logger.info(f"Validation completed. Safety score: {overall_results['safety_score']:.3f}")
        
        return overall_results
    
    def _validate_scenario(self, agent, scenario: Dict, num_episodes: int) -> Dict:
        """Validate agent on a specific scenario."""
        episode_results = []
        
        for episode in range(num_episodes):
            result = self._run_validation_episode(agent, scenario)
            episode_results.append(result)
        
        # Aggregate scenario results
        return {
            'scenario': scenario,
            'episodes': episode_results,
            'summary': self._summarize_episodes(episode_results)
        }
    
    def _run_validation_episode(self, agent, scenario: Dict) -> Dict:
        """Run a single validation episode."""
        # Reset simulator with scenario conditions
        initial_conditions = scenario['initial_conditions']
        state = self.simulator.reset(
            initial_temp=initial_conditions['temperature'],
            initial_battery=initial_conditions['battery']
        )
        
        # Initialize tracking variables
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'expert_actions': [],
            'safety_violations': [],
            'temperature_violations': [],
            'battery_violations': [],
            'algorithm_effectiveness': []
        }
        
        # Run episode
        threat_sequence = initial_conditions['threat_sequence']
        threat_index = 0
        time_since_tst = initial_conditions.get('time_since_tst', 300.0)
        
        for step in range(scenario['duration']):
            # Update threat based on sequence
            if threat_index < len(threat_sequence):
                current_threat = threat_sequence[threat_index]
                if step % 30 == 0:  # Change threat every 30 steps
                    threat_index = min(threat_index + 1, len(threat_sequence) - 1)
            else:
                current_threat = 0
            
            # Create current state
            current_state = {
                'temperature': state['temperature'],
                'battery': state['battery'],
                'threat': current_threat,
                'cpu_usage': state['cpu_usage'],
                'time_since_tst': time_since_tst,
                'power': state['power_consumption']
            }
            
            # Get agent action
            agent_action = agent.get_action(current_state, training=False)
            
            # Get expert action for comparison
            expert_action = agent.expert_policy.get_action(current_state)
            
            # Check safety
            safe, safety_reason = agent.expert_policy.is_safe_action(current_state, agent_action)
            
            # Simulate step
            state = self.simulator.simulate_step(agent_action, dt=1.0)
            
            # Update time since TST
            if agent_action == 2:  # TST started
                time_since_tst = 0
            else:
                time_since_tst += 1.0
            
            # Calculate reward (for analysis)
            reward = self._calculate_validation_reward(current_state, agent_action, state)
            
            # Record violations
            temp_violation = state['temperature'] > self.safety_thresholds['critical_temperature']
            battery_violation = state['battery'] < self.safety_thresholds['critical_battery']
            
            # Algorithm effectiveness (simplified)
            effectiveness = self._calculate_algorithm_effectiveness(agent_action, current_threat)
            
            # Store episode data
            episode_data['states'].append(current_state.copy())
            episode_data['actions'].append(agent_action)
            episode_data['rewards'].append(reward)
            episode_data['expert_actions'].append(expert_action)
            episode_data['safety_violations'].append(not safe)
            episode_data['temperature_violations'].append(temp_violation)
            episode_data['battery_violations'].append(battery_violation)
            episode_data['algorithm_effectiveness'].append(effectiveness)
            
            # Check for critical failures
            if state['temperature'] > self.safety_thresholds['max_temperature']:
                logger.warning(f"Critical temperature reached: {state['temperature']:.1f}°C")
                break
            
            if state['battery'] < self.safety_thresholds['min_battery']:
                logger.warning(f"Critical battery reached: {state['battery']:.1f}%")
                break
        
        return self._analyze_episode(episode_data, scenario)
    
    def _calculate_validation_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """Calculate reward for validation (same as training reward)."""
        reward = 0
        
        # Power efficiency (30% weight)
        power = next_state['power_consumption']
        power_reward = 0.3 * (8.0 - power) / 4.0
        reward += power_reward
        
        # Thermal safety (25% weight)
        temp = next_state['temperature']
        if temp > 80:
            thermal_reward = -10.0
        elif temp > 70:
            thermal_reward = -2.0
        else:
            thermal_reward = 0.25 * min(1.0, (70 - temp) / 20)
        reward += thermal_reward
        
        # Algorithm effectiveness (25% weight)
        threat = state.get('threat', 0)
        effectiveness = self._calculate_algorithm_effectiveness(action, threat)
        reward += 0.25 * effectiveness
        
        # Battery conservation (20% weight)
        battery = next_state.get('battery', 80)
        if battery > 60:
            battery_reward = 0.2
        elif battery > 30:
            battery_reward = 0.1
        else:
            battery_reward = -0.2
        reward += battery_reward
        
        return reward
    
    def _calculate_algorithm_effectiveness(self, action: int, threat: int) -> float:
        """Calculate algorithm effectiveness score."""
        # Effectiveness matrix: [No_DDoS, XGBoost, TST] vs [Normal, Confirming, Confirmed]
        effectiveness_matrix = [
            [0.5, 0.2, 0.0],  # No_DDoS: OK for normal, poor for threats
            [0.7, 0.9, 0.8],  # XGBoost: Good all-around
            [0.6, 0.95, 0.9]  # TST: Excellent for threats, OK for normal
        ]
        
        if 0 <= action <= 2 and 0 <= threat <= 2:
            return effectiveness_matrix[action][threat]
        
        return 0.5  # Default moderate effectiveness
    
    def _analyze_episode(self, episode_data: Dict, scenario: Dict) -> Dict:
        """Analyze a single episode's results."""
        total_steps = len(episode_data['actions'])
        
        if total_steps == 0:
            return {'error': 'No steps recorded'}
        
        # Expert agreement
        agreements = sum(1 for i in range(total_steps) 
                        if episode_data['actions'][i] == episode_data['expert_actions'][i])
        expert_agreement = agreements / total_steps
        
        # Safety metrics
        safety_violations = sum(episode_data['safety_violations'])
        temp_violations = sum(episode_data['temperature_violations'])
        battery_violations = sum(episode_data['battery_violations'])
        
        # Performance metrics
        avg_reward = np.mean(episode_data['rewards'])
        avg_effectiveness = np.mean(episode_data['algorithm_effectiveness'])
        
        # Temperature and battery analysis
        final_state = episode_data['states'][-1] if episode_data['states'] else {}
        max_temp = max((s['temperature'] for s in episode_data['states']), default=0)
        min_battery = min((s['battery'] for s in episode_data['states']), default=100)
        
        # Action distribution
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in episode_data['actions']:
            action_counts[action] += 1
        
        return {
            'scenario_name': scenario['name'],
            'total_steps': total_steps,
            'expert_agreement': expert_agreement,
            'safety_violation_rate': safety_violations / total_steps,
            'temperature_violation_rate': temp_violations / total_steps,
            'battery_violation_rate': battery_violations / total_steps,
            'average_reward': avg_reward,
            'average_effectiveness': avg_effectiveness,
            'max_temperature': max_temp,
            'min_battery': min_battery,
            'final_temperature': final_state.get('temperature', 0),
            'final_battery': final_state.get('battery', 0),
            'action_distribution': action_counts,
            'safety_violations': safety_violations,
            'completed_successfully': total_steps >= scenario['duration'] * 0.8,  # At least 80% completion
        }
    
    def _summarize_episodes(self, episodes: List[Dict]) -> Dict:
        """Summarize results across multiple episodes."""
        if not episodes:
            return {}
        
        return {
            'num_episodes': len(episodes),
            'avg_expert_agreement': np.mean([e['expert_agreement'] for e in episodes]),
            'avg_safety_violation_rate': np.mean([e['safety_violation_rate'] for e in episodes]),
            'avg_reward': np.mean([e['average_reward'] for e in episodes]),
            'avg_effectiveness': np.mean([e['average_effectiveness'] for e in episodes]),
            'max_temperature_observed': max([e['max_temperature'] for e in episodes]),
            'min_battery_observed': min([e['min_battery'] for e in episodes]),
            'success_rate': np.mean([e['completed_successfully'] for e in episodes]),
            'total_safety_violations': sum([e['safety_violations'] for e in episodes])
        }
    
    def _aggregate_results(self, all_episodes: List[Dict], scenario_results: Dict) -> Dict:
        """Aggregate results across all scenarios."""
        # Overall metrics
        total_episodes = len(all_episodes)
        overall_expert_agreement = np.mean([e['expert_agreement'] for e in all_episodes])
        overall_safety_violation_rate = np.mean([e['safety_violation_rate'] for e in all_episodes])
        overall_success_rate = np.mean([e['completed_successfully'] for e in all_episodes])
        
        # Safety score calculation
        safety_score = self._calculate_safety_score(all_episodes)
        
        # Pass/fail determination
        passed = (
            overall_expert_agreement >= self.safety_thresholds['min_expert_agreement'] and
            overall_safety_violation_rate <= self.safety_thresholds['max_safety_violation_rate'] and
            overall_success_rate >= 0.8
        )
        
        return {
            'total_episodes': total_episodes,
            'expert_agreement': overall_expert_agreement,
            'safety_violation_rate': overall_safety_violation_rate,
            'success_rate': overall_success_rate,
            'safety_score': safety_score,
            'passed': passed,
            'average_episode_reward': np.mean([e['average_reward'] for e in all_episodes]),
            'average_effectiveness': np.mean([e['average_effectiveness'] for e in all_episodes]),
            'scenario_results': scenario_results,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    def _calculate_safety_score(self, episodes: List[Dict]) -> float:
        """Calculate overall safety score (0-1, higher is better)."""
        if not episodes:
            return 0.0
        
        # Weight different safety aspects
        weights = {
            'expert_agreement': 0.3,
            'safety_violations': 0.4,
            'success_rate': 0.2,
            'effectiveness': 0.1
        }
        
        expert_score = np.mean([e['expert_agreement'] for e in episodes])
        safety_score = 1.0 - np.mean([e['safety_violation_rate'] for e in episodes])
        success_score = np.mean([e['completed_successfully'] for e in episodes])
        effectiveness_score = np.mean([e['average_effectiveness'] for e in episodes])
        
        total_score = (
            weights['expert_agreement'] * expert_score +
            weights['safety_violations'] * safety_score +
            weights['success_rate'] * success_score +
            weights['effectiveness'] * effectiveness_score
        )
        
        return max(0.0, min(1.0, total_score))
    
    def generate_report(self, results: Dict, output_path: str = None):
        """Generate a detailed validation report."""
        if output_path is None:
            output_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("UAV CYBERSECURITY RL AGENT SAFETY VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {results['validation_timestamp']}")
        report_lines.append("")
        
        # Overall results
        report_lines.append("OVERALL RESULTS:")
        report_lines.append(f"  Total Episodes: {results['total_episodes']}")
        report_lines.append(f"  Expert Agreement: {results['expert_agreement']:.1%}")
        report_lines.append(f"  Safety Violation Rate: {results['safety_violation_rate']:.1%}")
        report_lines.append(f"  Success Rate: {results['success_rate']:.1%}")
        report_lines.append(f"  Safety Score: {results['safety_score']:.3f}")
        report_lines.append(f"  Average Reward: {results['average_episode_reward']:.2f}")
        report_lines.append(f"  Average Effectiveness: {results['average_effectiveness']:.1%}")
        report_lines.append("")
        
        # Pass/fail status
        status = "PASSED" if results['passed'] else "FAILED"
        report_lines.append(f"VALIDATION STATUS: {status}")
        report_lines.append("")
        
        # Scenario breakdown
        report_lines.append("SCENARIO BREAKDOWN:")
        report_lines.append("-" * 40)
        
        for scenario_name, scenario_result in results['scenario_results'].items():
            summary = scenario_result['summary']
            scenario = scenario_result['scenario']
            
            report_lines.append(f"Scenario: {scenario_name}")
            report_lines.append(f"  Description: {scenario['description']}")
            report_lines.append(f"  Episodes: {summary['num_episodes']}")
            report_lines.append(f"  Expert Agreement: {summary['avg_expert_agreement']:.1%}")
            report_lines.append(f"  Safety Violations: {summary['avg_safety_violation_rate']:.1%}")
            report_lines.append(f"  Success Rate: {summary['success_rate']:.1%}")
            report_lines.append(f"  Max Temperature: {summary['max_temperature_observed']:.1f}°C")
            report_lines.append(f"  Min Battery: {summary['min_battery_observed']:.1f}%")
            report_lines.append("")
        
        # Safety thresholds
        report_lines.append("SAFETY THRESHOLDS:")
        report_lines.append("-" * 40)
        for threshold, value in self.safety_thresholds.items():
            report_lines.append(f"  {threshold}: {value}")
        report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        if results['expert_agreement'] < self.safety_thresholds['min_expert_agreement']:
            report_lines.append("  [WARN] Low expert agreement - consider additional training")
        
        if results['safety_violation_rate'] > self.safety_thresholds['max_safety_violation_rate']:
            report_lines.append("  [WARN] High safety violation rate - review safety constraints")
        
        if results['success_rate'] < 0.8:
            report_lines.append("  [WARN] Low success rate - check episode termination conditions")
        
        if results['safety_score'] > 0.9:
            report_lines.append("  [PASS] Excellent safety performance")
        elif results['safety_score'] > 0.8:
            report_lines.append("  [PASS] Good safety performance")
        else:
            report_lines.append("  [WARN] Safety performance needs improvement")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Write report
        report_text = "\n".join(report_lines)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text


if __name__ == "__main__":
    # Test the safety validator
    logging.basicConfig(level=logging.INFO)
    
    # Create simulator and validator
    from environment.thermal_simulator import TunableUAVSimulator
    from agents.expert_policy import ExpertPolicy
    from utils.state_discretizer import StateDiscretizer
    from agents.q_learning_agent import WarmStartQLearningAgent
    
    simulator = TunableUAVSimulator()
    expert_policy = ExpertPolicy()
    state_discretizer = StateDiscretizer()
    
    # Create a test agent
    agent = WarmStartQLearningAgent(
        expert_policy=expert_policy,
        state_discretizer=state_discretizer
    )
    
    # Create validator
    validator = SafetyValidator(simulator)
    
    print(f"Created validator with {len(validator.validation_scenarios)} scenarios")
    
    # Run a quick validation (1 episode per scenario for testing)
    print("Running quick validation...")
    results = validator.validate(agent, num_episodes=1)
    
    print(f"Validation completed:")
    print(f"  Safety Score: {results['safety_score']:.3f}")
    print(f"  Expert Agreement: {results['expert_agreement']:.1%}")
    print(f"  Safety Violations: {results['safety_violation_rate']:.1%}")
    print(f"  Status: {'PASSED' if results['passed'] else 'FAILED'}")
    
    # Generate report
    report = validator.generate_report(results, "test_validation_report.txt")
    print("\nValidation report generated successfully")