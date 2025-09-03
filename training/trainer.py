"""
RL Trainer for UAV Cybersecurity RL System.

This module provides comprehensive training capabilities for the Q-learning agent
with monitoring, evaluation, and visualization features.
"""

import numpy as np
import logging
import time
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless operation
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from agents.q_learning_agent import WarmStartQLearningAgent
from environment.thermal_simulator import TunableUAVSimulator
from validation.safety_validator import SafetyValidator

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Trainer for Q-learning agent with comprehensive monitoring.
    
    This class manages the complete training pipeline including episode execution,
    performance monitoring, periodic evaluation, and result visualization.
    """
    
    def __init__(self,
                 agent: WarmStartQLearningAgent,
                 simulator: TunableUAVSimulator,
                 validator: SafetyValidator = None,
                 config: Dict = None):
        """
        Initialize the trainer.
        
        Args:
            agent: Q-learning agent to train
            simulator: UAV thermal simulator
            validator: Safety validator (optional)
            config: Training configuration
        """
        self.agent = agent
        self.simulator = simulator
        self.validator = validator
        
        # Training configuration
        self.config = config or {
            'num_episodes': 1000,
            'max_steps_per_episode': 300,
            'eval_interval': 100,
            'save_interval': 200,
            'log_interval': 50,
            'output_dir': 'results/' + datetime.now().strftime("%Y%m%d_%H%M%S"),
            'experiment_name': 'uav_ddos_rl'
        }
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'expert_agreements': [],
            'safety_violations': [],
            'temperature_violations': [],
            'battery_violations': [],
            'max_temperatures': [],
            'min_batteries': [],
            'avg_powers': [],
            'epsilons': [],
            'actions_taken': {0: 0, 1: 0, 2: 0},
            'algorithm_effectiveness': [],
            'thermal_zones': {'Safe': 0, 'Warning': 0, 'Critical': 0},
            'battery_zones': {'0-20%': 0, '21-40%': 0, '41-60%': 0, '61-80%': 0, '81-100%': 0}
        }
        
        # Threat patterns for training variety
        self.threat_patterns = [
            [0, 0, 0, 1, 1, 0, 0],          # Occasional threats
            [0, 1, 2, 1, 0, 1, 0],          # Mixed threats
            [2, 2, 1, 2, 2, 1, 0],          # Heavy threat load
            [0, 0, 1, 0, 0, 1, 0],          # Light monitoring
            [1, 2, 2, 2, 1, 0, 0],          # Burst detection
            [0, 0, 0, 0, 0, 0, 0],          # Normal operation
            [1, 1, 1, 1, 1, 1, 1],          # Continuous monitoring
        ]
        
        logger.info(f"RL Trainer initialized with config: {self.config['experiment_name']}")
    
    def train(self) -> Dict:
        """Run the complete training loop."""
        logger.info(f"Starting training for {self.config['num_episodes']} episodes")
        
        start_time = time.time()
        
        # Save initial configuration
        self._save_config()
        
        try:
            for episode in range(1, self.config['num_episodes'] + 1):
                # Run episode
                episode_metrics = self._run_episode(episode)
                
                # Update metrics
                self._update_metrics(episode_metrics)
                
                # Decay exploration
                self.agent.decay_epsilon()
                self.metrics['epsilons'].append(self.agent.epsilon)
                
                # Periodic logging
                if episode % self.config['log_interval'] == 0:
                    self._log_progress(episode)
                
                # Periodic evaluation
                if self.validator and episode % self.config['eval_interval'] == 0:
                    self._evaluate(episode)
                
                # Periodic saving
                if episode % self.config['save_interval'] == 0:
                    self._save_checkpoint(episode)
                
                # Early stopping check
                if self._check_early_stopping(episode):
                    logger.info(f"Early stopping triggered at episode {episode}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
        
        # Final evaluation and saving
        if self.validator:
            self._evaluate(self.config['num_episodes'], final=True)
        
        self._save_checkpoint(self.config['num_episodes'], final=True)
        
        # Generate visualizations
        self._visualize_training()
        
        # Calculate training time
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Prepare final results
        final_results = self._prepare_final_results(total_time)
        
        return final_results
    
    def _run_episode(self, episode: int) -> Dict:
        """Run a single training episode."""
        # Reset simulator with varied initial conditions
        initial_temp = np.random.uniform(40, 65)  # Varied starting temperature
        initial_battery = np.random.uniform(60, 95)  # Good battery levels
        
        state_dict = self.simulator.reset(initial_temp, initial_battery)
        
        # Select threat pattern for this episode
        threat_pattern = self.threat_patterns[episode % len(self.threat_patterns)]
        threat_index = 0
        
        # Episode tracking
        total_reward = 0
        episode_length = 0
        expert_agreements = 0
        safety_violations = 0
        temp_violations = 0
        battery_violations = 0
        max_temp = initial_temp
        min_battery = initial_battery
        power_sum = 0
        actions_taken = {0: 0, 1: 0, 2: 0}
        effectiveness_scores = []
        
        # Episode loop
        for step in range(self.config['max_steps_per_episode']):
            # Update threat based on pattern
            current_threat = threat_pattern[threat_index]
            if step > 0 and step % 30 == 0:  # Change threat every 30 steps
                threat_index = min(threat_index + 1, len(threat_pattern) - 1)
            
            # Create state for agent
            current_state = {
                'temperature': state_dict['temperature'],
                'battery': state_dict['battery'],
                'threat': current_threat,
                'cpu_usage': state_dict['cpu_usage'],
                'time_since_tst': state_dict['time_since_tst'],
                'power': state_dict['power_consumption']
            }
            
            # Get action from agent
            action = self.agent.get_action(current_state, training=True)
            
            # Track expert agreement
            expert_action = self.agent.expert_policy.get_action(current_state)
            expert_agreements += int(action == expert_action)
            
            # Check safety
            safe, _ = self.agent.expert_policy.is_safe_action(current_state, action)
            safety_violations += int(not safe)
            
            # Track action taken
            actions_taken[action] += 1
            
            # Simulate step
            next_state_dict = self.simulator.simulate_step(action, dt=1.0)
            
            # Create next state for agent
            next_state = {
                'temperature': next_state_dict['temperature'],
                'battery': next_state_dict['battery'],
                'threat': current_threat,
                'cpu_usage': next_state_dict['cpu_usage'],
                'time_since_tst': next_state_dict['time_since_tst'],
                'power': next_state_dict['power_consumption']
            }
            
            # Calculate reward
            reward = self._calculate_reward(current_state, action, next_state)
            total_reward += reward
            
            # Calculate algorithm effectiveness
            effectiveness = self._calculate_algorithm_effectiveness(action, current_threat)
            effectiveness_scores.append(effectiveness)
            
            # Update agent
            done = self._is_episode_done(next_state, step)
            self.agent.update(current_state, action, reward, next_state, done)
            
            # Track metrics
            max_temp = max(max_temp, next_state['temperature'])
            min_battery = min(min_battery, next_state['battery'])
            power_sum += next_state['power']
            
            # Track violations
            temp_violations += int(next_state['temperature'] > 80)
            battery_violations += int(next_state['battery'] < 20)
            
            # Update state
            state_dict = next_state_dict
            episode_length += 1
            
            # Check termination
            if done:
                break
        
        # Calculate averages
        avg_power = power_sum / max(1, episode_length)
        expert_agreement_rate = expert_agreements / max(1, episode_length)
        avg_effectiveness = np.mean(effectiveness_scores) if effectiveness_scores else 0.5
        
        # Track thermal and battery zones
        thermal_zone = self.simulator.get_thermal_zone(max_temp)
        battery_zone = self.simulator.get_battery_zone(min_battery)
        
        return {
            'episode_rewards': total_reward,
            'episode_lengths': episode_length,
            'expert_agreements': expert_agreement_rate,
            'safety_violations': safety_violations,
            'temperature_violations': temp_violations,
            'battery_violations': battery_violations,
            'max_temperatures': max_temp,
            'min_batteries': min_battery,
            'avg_powers': avg_power,
            'actions_taken': actions_taken,
            'algorithm_effectiveness': avg_effectiveness,
            'thermal_zone': thermal_zone,
            'battery_zone': battery_zone
        }
    
    def _calculate_reward(self, state: Dict, action: int, next_state: Dict) -> float:
        """Calculate reward for training."""
        reward = 0
        
        # Power efficiency (30% weight)
        power = next_state['power']
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
        algo_reward = 0.25 * effectiveness * 4.0  # Scale to 0-1 range
        reward += algo_reward
        
        # Battery conservation (20% weight)
        battery = next_state['battery']
        if battery > 60:
            battery_reward = 0.2
        elif battery > 30:
            battery_reward = 0.1
        else:
            battery_reward = -0.2
        reward += battery_reward
        
        # Safety violations (hard penalty)
        safe, _ = self.agent.expert_policy.is_safe_action(state, action)
        if not safe:
            reward -= 20.0
        
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
    
    def _is_episode_done(self, state: Dict, step: int) -> bool:
        """Check if episode should terminate."""
        # Terminal conditions
        if state['temperature'] >= 85.0:  # Critical temperature
            return True
        
        if state['battery'] <= 5.0:  # Critical battery
            return True
        
        if step >= self.config['max_steps_per_episode'] - 1:  # Max steps
            return True
        
        return False
    
    def _update_metrics(self, episode_metrics: Dict):
        """Update training metrics with episode results."""
        # Basic metrics
        self.metrics['episode_rewards'].append(episode_metrics['episode_rewards'])
        self.metrics['episode_lengths'].append(episode_metrics['episode_lengths'])
        self.metrics['expert_agreements'].append(episode_metrics['expert_agreements'])
        self.metrics['safety_violations'].append(episode_metrics['safety_violations'])
        self.metrics['temperature_violations'].append(episode_metrics['temperature_violations'])
        self.metrics['battery_violations'].append(episode_metrics['battery_violations'])
        self.metrics['max_temperatures'].append(episode_metrics['max_temperatures'])
        self.metrics['min_batteries'].append(episode_metrics['min_batteries'])
        self.metrics['avg_powers'].append(episode_metrics['avg_powers'])
        self.metrics['algorithm_effectiveness'].append(episode_metrics['algorithm_effectiveness'])
        
        # Action counts
        for action, count in episode_metrics['actions_taken'].items():
            self.metrics['actions_taken'][action] += count
        
        # Zone tracking
        thermal_zone = episode_metrics['thermal_zone']
        battery_zone = episode_metrics['battery_zone']
        
        if thermal_zone in self.metrics['thermal_zones']:
            self.metrics['thermal_zones'][thermal_zone] += 1
        
        if battery_zone in self.metrics['battery_zones']:
            self.metrics['battery_zones'][battery_zone] += 1
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_window = min(self.config['log_interval'], len(self.metrics['episode_rewards']))
        
        recent_rewards = self.metrics['episode_rewards'][-recent_window:]
        recent_lengths = self.metrics['episode_lengths'][-recent_window:]
        recent_agreements = self.metrics['expert_agreements'][-recent_window:]
        recent_temps = self.metrics['max_temperatures'][-recent_window:]
        recent_effectiveness = self.metrics['algorithm_effectiveness'][-recent_window:]
        recent_safety = self.metrics['safety_violations'][-recent_window:]
        
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_agreement = np.mean(recent_agreements)
        avg_temp = np.mean(recent_temps)
        avg_effectiveness = np.mean(recent_effectiveness)
        avg_safety_violations = np.mean(recent_safety)
        
        logger.info(
            f"Episode {episode}/{self.config['num_episodes']}: "
            f"Reward={avg_reward:.2f}, "
            f"Length={avg_length:.1f}, "
            f"Expert={avg_agreement:.1%}, "
            f"MaxTemp={avg_temp:.1f}°C, "
            f"Effectiveness={avg_effectiveness:.1%}, "
            f"Safety={avg_safety_violations:.1f}, "
            f"Epsilon={self.agent.epsilon:.3f}"
        )
    
    def _evaluate(self, episode: int, final: bool = False):
        """Evaluate current policy."""
        if self.validator is None:
            return
        
        logger.info(f"Evaluating policy at episode {episode}...")
        
        try:
            # Run validation with fewer episodes during training
            num_eval_episodes = 3 if not final else 5
            metrics = self.validator.validate(self.agent, num_episodes=num_eval_episodes)
            
            logger.info(
                f"Evaluation: "
                f"Expert Agreement={metrics['expert_agreement']:.1%}, "
                f"Safety={1-metrics['safety_violation_rate']:.1%}, "
                f"Reward={metrics['average_episode_reward']:.2f}, "
                f"Score={metrics['safety_score']:.3f}"
            )
            
            # Save evaluation report
            if final:
                report_path = os.path.join(self.config['output_dir'], 'final_evaluation.txt')
                self.validator.generate_report(metrics, report_path)
                
                # Save evaluation metrics
                eval_path = os.path.join(self.config['output_dir'], 'evaluation_metrics.json')
                with open(eval_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save agent checkpoint and training metrics."""
        checkpoint_dir = os.path.join(self.config['output_dir'], 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save agent
        if final:
            agent_path = os.path.join(checkpoint_dir, 'final_model.json')
        else:
            agent_path = os.path.join(checkpoint_dir, f'model_ep{episode}.json')
        
        self.agent.save(agent_path)
        
        # Save training metrics
        metrics_path = os.path.join(self.config['output_dir'], 'training_metrics.json')
        
        # Prepare metrics for JSON serialization
        metrics_json = {}
        for k, v in self.metrics.items():
            if isinstance(v, (list, dict)):
                metrics_json[k] = v
            else:
                metrics_json[k] = str(v)
        
        # Add agent statistics
        agent_stats = self.agent.get_policy_stats()
        metrics_json['agent_stats'] = agent_stats
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"Checkpoint saved: episode {episode}")
    
    def _save_config(self):
        """Save training configuration."""
        config_path = os.path.join(self.config['output_dir'], 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _check_early_stopping(self, episode: int) -> bool:
        """Check if early stopping criteria are met."""
        # Only check after minimum number of episodes
        if episode < 200:
            return False
        
        # Check convergence of recent rewards
        recent_window = 100
        if len(self.metrics['episode_rewards']) >= recent_window:
            recent_rewards = self.metrics['episode_rewards'][-recent_window:]
            reward_std = np.std(recent_rewards)
            
            # If reward standard deviation is very low, we might have converged
            if reward_std < 0.5:
                logger.info(f"Potential convergence detected (reward std: {reward_std:.3f})")
        
        # Check if epsilon has reached minimum and performance is stable
        if (self.agent.epsilon <= self.agent.epsilon_end * 1.1 and 
            len(self.metrics['expert_agreements']) >= 50):
            
            recent_agreements = self.metrics['expert_agreements'][-50:]
            avg_agreement = np.mean(recent_agreements)
            
            if avg_agreement > 0.85:  # High agreement with expert
                logger.info(f"High expert agreement achieved: {avg_agreement:.1%}")
                return True
        
        return False
    
    def _visualize_training(self):
        """Generate comprehensive training visualizations."""
        logger.info("Generating training visualizations...")
        
        # Create figures directory
        fig_dir = os.path.join(self.config['output_dir'], 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        try:
            # Set up plotting style
            plt.style.use('default')
            
            # 1. Training progress overview (2x2 subplots)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('UAV RL Training Progress', fontsize=16)
            
            # Episode rewards
            axes[0, 0].plot(self.metrics['episode_rewards'], alpha=0.7)
            axes[0, 0].plot(self._smooth_curve(self.metrics['episode_rewards'], window=50), 
                           color='red', linewidth=2, label='Smoothed')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Total Reward')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            
            # Expert agreement
            axes[0, 1].plot(self.metrics['expert_agreements'], alpha=0.7)
            axes[0, 1].plot(self._smooth_curve(self.metrics['expert_agreements'], window=50),
                           color='red', linewidth=2, label='Smoothed')
            axes[0, 1].set_title('Expert Agreement Rate')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Agreement Rate')
            axes[0, 1].set_ylim([0, 1])
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            
            # Temperature control
            axes[1, 0].plot(self.metrics['max_temperatures'], alpha=0.7)
            axes[1, 0].axhline(y=70, color='orange', linestyle='--', label='Warning (70°C)')
            axes[1, 0].axhline(y=80, color='red', linestyle='--', label='Critical (80°C)')
            axes[1, 0].set_title('Maximum Temperature per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Exploration rate
            axes[1, 1].plot(self.metrics['epsilons'])
            axes[1, 1].set_title('Exploration Rate (Epsilon)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Epsilon')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Action distribution (pie chart)
            plt.figure(figsize=(10, 8))
            actions = self.metrics['actions_taken']
            labels = ['No_DDoS', 'XGBoost', 'TST']
            sizes = [actions[0], actions[1], actions[2]]
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            
            # Main pie chart
            plt.subplot(2, 2, 1)
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            plt.title('Overall Action Distribution')
            
            # Thermal zones distribution
            plt.subplot(2, 2, 2)
            thermal_data = self.metrics['thermal_zones']
            thermal_labels = list(thermal_data.keys())
            thermal_sizes = list(thermal_data.values())
            plt.pie(thermal_sizes, labels=thermal_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Thermal Zone Distribution')
            
            # Battery zones distribution
            plt.subplot(2, 2, 3)
            battery_data = self.metrics['battery_zones']
            battery_labels = list(battery_data.keys())
            battery_sizes = list(battery_data.values())
            plt.pie(battery_sizes, labels=battery_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Battery Zone Distribution')
            
            # Algorithm effectiveness over time
            plt.subplot(2, 2, 4)
            effectiveness_smooth = self._smooth_curve(self.metrics['algorithm_effectiveness'], window=50)
            plt.plot(effectiveness_smooth)
            plt.title('Algorithm Effectiveness')
            plt.xlabel('Episode')
            plt.ylabel('Effectiveness Score')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'distributions.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Safety metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Safety Metrics', fontsize=16)
            
            # Safety violations
            axes[0, 0].plot(self.metrics['safety_violations'])
            axes[0, 0].set_title('Safety Violations per Episode')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Violations')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Temperature violations
            axes[0, 1].plot(self.metrics['temperature_violations'])
            axes[0, 1].set_title('Temperature Violations per Episode')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Violations')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Battery analysis
            axes[1, 0].plot(self.metrics['min_batteries'])
            axes[1, 0].axhline(y=20, color='red', linestyle='--', label='Critical (20%)')
            axes[1, 0].axhline(y=40, color='orange', linestyle='--', label='Low (40%)')
            axes[1, 0].set_title('Minimum Battery per Episode')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Battery (%)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            
            # Power consumption
            axes[1, 1].plot(self.metrics['avg_powers'])
            axes[1, 1].set_title('Average Power Consumption')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Power (W)')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'safety_metrics.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to {fig_dir}")
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
    
    def _smooth_curve(self, data: List[float], window: int = 50) -> List[float]:
        """Apply moving average smoothing to data."""
        if len(data) < window:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(data), i + window // 2 + 1)
            smoothed.append(np.mean(data[start_idx:end_idx]))
        
        return smoothed
    
    def _prepare_final_results(self, training_time: float) -> Dict:
        """Prepare final training results summary."""
        total_episodes = len(self.metrics['episode_rewards'])
        
        if total_episodes == 0:
            return {'error': 'No episodes completed'}
        
        # Calculate final performance metrics
        final_window = min(100, total_episodes)
        recent_rewards = self.metrics['episode_rewards'][-final_window:]
        recent_agreements = self.metrics['expert_agreements'][-final_window:]
        recent_effectiveness = self.metrics['algorithm_effectiveness'][-final_window:]
        
        # Agent statistics
        agent_stats = self.agent.get_policy_stats()
        
        # Safety summary
        total_safety_violations = sum(self.metrics['safety_violations'])
        total_temp_violations = sum(self.metrics['temperature_violations'])
        total_battery_violations = sum(self.metrics['battery_violations'])
        
        results = {
            'training_summary': {
                'total_episodes': total_episodes,
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60,
                'episodes_per_minute': total_episodes / (training_time / 60)
            },
            'performance': {
                'final_avg_reward': np.mean(recent_rewards),
                'final_expert_agreement': np.mean(recent_agreements),
                'final_effectiveness': np.mean(recent_effectiveness),
                'reward_improvement': np.mean(recent_rewards) - np.mean(self.metrics['episode_rewards'][:final_window]) if total_episodes > final_window else 0,
            },
            'safety': {
                'total_safety_violations': total_safety_violations,
                'total_temperature_violations': total_temp_violations,
                'total_battery_violations': total_battery_violations,
                'safety_violation_rate': total_safety_violations / total_episodes,
                'max_temperature_observed': max(self.metrics['max_temperatures']),
                'min_battery_observed': min(self.metrics['min_batteries'])
            },
            'agent_stats': agent_stats,
            'action_distribution': self.metrics['actions_taken'],
            'config': self.config,
            'output_directory': self.config['output_dir']
        }
        
        # Save final results
        results_path = os.path.join(self.config['output_dir'], 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


if __name__ == "__main__":
    # Test the trainer
    logging.basicConfig(level=logging.INFO)
    
    # Create all required components
    from environment.thermal_simulator import TunableUAVSimulator
    from agents.expert_policy import ExpertPolicy
    from utils.state_discretizer import StateDiscretizer
    from agents.q_learning_agent import WarmStartQLearningAgent
    from validation.safety_validator import SafetyValidator
    
    # Initialize components
    simulator = TunableUAVSimulator()
    expert_policy = ExpertPolicy()
    state_discretizer = StateDiscretizer()
    
    agent = WarmStartQLearningAgent(
        expert_policy=expert_policy,
        state_discretizer=state_discretizer,
        learning_rate=0.1,
        epsilon_start=0.3
    )
    
    validator = SafetyValidator(simulator)
    
    # Create trainer with short test config
    test_config = {
        'num_episodes': 10,
        'max_steps_per_episode': 50,
        'eval_interval': 5,
        'save_interval': 5,
        'log_interval': 2,
        'output_dir': 'test_results',
        'experiment_name': 'test_training'
    }
    
    trainer = RLTrainer(agent, simulator, validator, test_config)
    
    print("Running test training...")
    results = trainer.train()
    
    print(f"Test training completed!")
    print(f"Episodes: {results['training_summary']['total_episodes']}")
    print(f"Final reward: {results['performance']['final_avg_reward']:.2f}")
    print(f"Expert agreement: {results['performance']['final_expert_agreement']:.1%}")
    print(f"Output directory: {results['output_directory']}")