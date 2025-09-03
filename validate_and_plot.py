#!/usr/bin/env python3
"""
UAV Cybersecurity RL System - Comprehensive Validation and Visualization Suite

This script provides comprehensive validation and professional visualization for a UAV cybersecurity 
reinforcement learning system that uses Q-learning with expert warm-start to select between 
{No_DDoS, XGBoost, TST} algorithms.

Author: UAV RL Team
Date: September 2, 2025
Version: 1.0
"""

import os
import sys
import json
import argparse
import signal
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import our system components
try:
    from agents.q_learning_agent import WarmStartQLearningAgent
    from agents.expert_policy import ExpertPolicy
    from environment.thermal_simulator import TunableUAVSimulator
    from utils.state_discretizer import StateDiscretizer
    from validation.safety_validator import SafetyValidator
except ImportError as e:
    print(f"Error importing system components: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


class ValidationSuite:
    """Comprehensive validation and visualization suite for UAV cybersecurity RL system."""
    
    def __init__(self, model_path: str, output_dir: str):
        """Initialize the validation suite."""
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.validation_dir = self.output_dir / 'validation'
        self.plots_dir = self.output_dir / 'plots'
        
        # Create output directories
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Professional color palette (blue-gray theme)
        self.colors = {
            'primary': '#2E5266',      # Dark blue-gray
            'secondary': '#4A90A4',    # Medium blue
            'accent': '#85C1E9',       # Light blue
            'success': '#27AE60',      # Green
            'warning': '#F39C12',      # Orange
            'danger': '#E74C3C',       # Red
            'gray_light': '#BDC3C7',   # Light gray
            'gray_dark': '#34495E',    # Dark gray
            'background': '#F8F9FA'    # Very light gray
        }
        
        # Action labels
        self.action_labels = ['No_DDoS', 'XGBoost', 'TST']
        
        # Battery and temperature level definitions
        self.battery_levels = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']
        self.temp_levels = ['Safe', 'Warning', 'Critical']
        self.threat_levels = ['Normal', 'Confirming', 'Confirmed']
        
        # Validation data storage
        self.validation_data = []
        self.scenario_results = {}
        self.state_combination_results = {}
        
        # Set up matplotlib style
        self.setup_plotting_style()
        
        # Initialize components
        self.initialize_components()
        
        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def setup_logging(self):
        """Set up logging configuration."""
        log_file = self.validation_dir / 'validation.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_plotting_style(self):
        """Set up professional matplotlib and seaborn style."""
        plt.style.use('default')
        sns.set_palette([self.colors['primary'], self.colors['secondary'], 
                        self.colors['accent'], self.colors['success'],
                        self.colors['warning'], self.colors['danger']])
        
        # Set default matplotlib parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white'
        })
        
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Load configurations
            with open('config/simulator_params.json', 'r') as f:
                sim_config = json.load(f)
            
            # Initialize components
            self.simulator = TunableUAVSimulator(sim_config)
            self.expert_policy = ExpertPolicy('config/expert_policy.json')
            self.state_discretizer = StateDiscretizer()
            self.safety_validator = SafetyValidator(self.simulator)
            
            # Initialize and load agent
            self.agent = WarmStartQLearningAgent(
                self.expert_policy, self.state_discretizer
            )
            self.agent.load(self.model_path)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully by saving partial results."""
        self.logger.info("Received interrupt signal. Saving partial results...")
        self.save_partial_results()
        sys.exit(0)
        
    def save_partial_results(self):
        """Save any collected data before shutdown."""
        if self.validation_data:
            self.save_results()
            self.logger.info("Partial results saved successfully")
        
    def run_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation across all scenarios and state combinations."""
        self.logger.info("Starting comprehensive validation suite...")
        
        # 1. Validate across all 45 state combinations
        self.logger.info("Phase 1: Validating all 45 state combinations...")
        self.validate_state_combinations()
        
        # 2. Run predefined safety scenarios
        self.logger.info("Phase 2: Running 8 safety scenarios...")
        self.run_safety_scenarios()
        
        # 3. Compute comprehensive metrics
        self.logger.info("Phase 3: Computing metrics...")
        metrics = self.compute_metrics()
        
        # 4. Save all results
        self.logger.info("Phase 4: Saving results...")
        self.save_results()
        
        self.logger.info("Validation suite completed successfully")
        return metrics
        
    def validate_state_combinations(self):
        """Validate performance across all 45 state combinations."""
        total_combinations = len(self.battery_levels) * len(self.temp_levels) * len(self.threat_levels)
        
        for i, battery_level in enumerate(self.battery_levels):
            for j, temp_level in enumerate(self.temp_levels):
                for k, threat_level in enumerate(self.threat_levels):
                    combination_id = f"{battery_level}|{temp_level}|{threat_level}"
                    
                    # Create representative state for this combination
                    state = self.create_representative_state(battery_level, temp_level, threat_level)
                    
                    # Get actions from both agent and expert
                    agent_action = self.agent.get_action(state, training=False)
                    expert_action = self.expert_policy.get_action(state)
                    
                    # Check safety
                    is_safe = self.check_safety(state, agent_action)
                    
                    # Record results
                    result = {
                        'combination_id': combination_id,
                        'battery_level': battery_level,
                        'temp_level': temp_level,
                        'threat_level': threat_level,
                        'state': state,
                        'agent_action': agent_action,
                        'expert_action': expert_action,
                        'agreement': agent_action == expert_action,
                        'is_safe': is_safe,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    self.state_combination_results[combination_id] = result
                    
                    progress = ((i * len(self.temp_levels) * len(self.threat_levels) + 
                               j * len(self.threat_levels) + k + 1) / total_combinations) * 100
                    
                    if progress % 20 == 0 or progress == 100:
                        self.logger.info(f"State combination validation: {progress:.1f}% complete")
                        
    def create_representative_state(self, battery_level: str, temp_level: str, threat_level: str) -> Dict[str, float]:
        """Create a representative state for the given levels."""
        # Map battery level to value
        battery_map = {
            '0-20%': 15.0,
            '21-40%': 30.0,
            '41-60%': 50.0,
            '61-80%': 70.0,
            '81-100%': 90.0
        }
        
        # Map temperature level to value
        temp_map = {
            'Safe': 45.0,
            'Warning': 65.0,
            'Critical': 82.0
        }
        
        # Map threat level to value
        threat_map = {
            'Normal': 0,
            'Confirming': 1,
            'Confirmed': 2
        }
        
        return {
            'battery': battery_map[battery_level],
            'temperature': temp_map[temp_level],
            'threat': threat_map[threat_level],
            'cpu_usage': 50.0,  # Representative value
            'time_since_tst': 300.0,  # Representative value
            'power': 4.0  # Representative value
        }
        
    def run_safety_scenarios(self):
        """Run the 8 predefined safety scenarios."""
        scenarios = [
            'normal_operation',
            'hot_conditions',
            'low_battery',
            'critical_battery',
            'continuous_threat',
            'temperature_stress',
            'mixed_conditions',
            'tst_recovery'
        ]
        
        for scenario_name in scenarios:
            self.logger.info(f"Running scenario: {scenario_name}")
            scenario_data = self.run_single_scenario(scenario_name)
            self.scenario_results[scenario_name] = scenario_data
            
    def run_single_scenario(self, scenario_name: str, episodes: int = 5) -> Dict[str, Any]:
        """Run a single safety scenario with detailed step-by-step recording."""
        scenario_data = {
            'name': scenario_name,
            'episodes': [],
            'summary': {}
        }
        
        for episode in range(episodes):
            episode_data = self.run_scenario_episode(scenario_name, episode)
            scenario_data['episodes'].append(episode_data)
            
        # Compute scenario summary
        scenario_data['summary'] = self.compute_scenario_summary(scenario_data['episodes'])
        
        return scenario_data
        
    def run_scenario_episode(self, scenario_name: str, episode_num: int) -> Dict[str, Any]:
        """Run a single episode of a scenario with step-by-step data collection."""
        episode_data = {
            'episode': episode_num,
            'scenario': scenario_name,
            'steps': [],
            'summary': {}
        }
        
        # Reset simulator
        self.simulator.reset()
        
        # Initialize environment based on scenario
        initial_state = self.get_scenario_initial_state(scenario_name)
        
        # Set simulator to initial state
        self.simulator.current_temp = initial_state['temperature']
        self.simulator.current_battery = initial_state['battery']
        self.simulator.current_cpu_usage = initial_state['cpu_usage']
        self.simulator.current_power = initial_state['power']
        self.simulator.time_since_tst = initial_state['time_since_tst']
        
        current_state = initial_state.copy()
        
        max_steps = 100
        total_reward = 0
        safety_violations = 0
        expert_agreements = 0
        
        for step in range(max_steps):
            # Simulate changing threat state for some scenarios
            if scenario_name == 'continuous_threat':
                current_state['threat'] = 2  # Keep high threat
            elif scenario_name == 'mixed_conditions':
                current_state['threat'] = (step // 20) % 3  # Cycling threats
            elif step > 30 and step < 60:
                current_state['threat'] = min(current_state['threat'] + 1, 2)  # Escalating
            
            # Get actions
            agent_action = self.agent.get_action(current_state, training=False)
            expert_action = self.expert_policy.get_action(current_state)
            
            # Check safety and agreement
            is_safe = self.check_safety(current_state, agent_action)
            agreement = agent_action == expert_action
            
            if agreement:
                expert_agreements += 1
            if not is_safe:
                safety_violations += 1
                
            # Simulate step using our simulator
            result = self.simulator.simulate_step(agent_action)
            next_state = {
                'temperature': result['temperature'],
                'battery': result['battery'],
                'threat': current_state['threat'],  # Threat doesn't change in sim
                'cpu_usage': result['cpu_usage'],
                'time_since_tst': result.get('time_since_tst', current_state['time_since_tst']),
                'power': result['power_consumption']
            }
            
            # Calculate reward (simple reward based on safety and effectiveness)
            reward = 1.0  # Base reward for safe operation
            if not is_safe:
                reward -= 10.0  # Large penalty for unsafe actions
            if agreement:
                reward += 0.5  # Bonus for expert agreement
                
            total_reward += reward
            
            # Check termination conditions
            done = (result['temperature'] > 85.0 or result['battery'] < 5.0 or 
                   step >= max_steps - 1)
            
            # Record step data
            step_data = {
                'step': step,
                'state': current_state.copy(),
                'agent_action': agent_action,
                'expert_action': expert_action,
                'reward': reward,
                'is_safe': is_safe,
                'agreement': agreement,
                'next_state': next_state.copy(),
                'done': done,
                'info': result
            }
            
            episode_data['steps'].append(step_data)
            self.validation_data.append({
                **step_data,
                'scenario': scenario_name,
                'episode': episode_num,
                'timestamp': datetime.now().isoformat()
            })
            
            current_state = next_state
            
            if done:
                break
                
        # Compute episode summary
        episode_data['summary'] = {
            'total_steps': len(episode_data['steps']),
            'total_reward': total_reward,
            'expert_agreement_rate': expert_agreements / len(episode_data['steps']) if episode_data['steps'] else 0,
            'safety_violation_rate': safety_violations / len(episode_data['steps']) if episode_data['steps'] else 0,
            'max_temperature': max([step['state']['temperature'] for step in episode_data['steps']]),
            'min_battery': min([step['state']['battery'] for step in episode_data['steps']]),
            'final_state': current_state
        }
        
        return episode_data
        
    def get_scenario_initial_state(self, scenario_name: str) -> Dict[str, float]:
        """Get initial state for a specific scenario."""
        base_state = {
            'battery': 80.0,
            'temperature': 45.0,
            'threat': 0,
            'cpu_usage': 30.0,
            'time_since_tst': 300.0,
            'power': 3.0
        }
        
        scenario_modifications = {
            'normal_operation': {},
            'hot_conditions': {'temperature': 70.0},
            'low_battery': {'battery': 25.0},
            'critical_battery': {'battery': 15.0},
            'continuous_threat': {'threat': 2},
            'temperature_stress': {'temperature': 80.0},
            'mixed_conditions': {'battery': 30.0, 'temperature': 75.0, 'threat': 1},
            'tst_recovery': {'time_since_tst': 50.0, 'temperature': 55.0}
        }
        
        modifications = scenario_modifications.get(scenario_name, {})
        state = base_state.copy()
        state.update(modifications)
        
        return state
        
    def check_safety(self, state: Dict[str, float], action: int) -> bool:
        """Check if an action is safe given the current state."""
        # Temperature safety
        if state['temperature'] > 85.0:
            return False
            
        # Battery safety
        if state['battery'] < 10.0:
            return False
            
        # TST safety (action 2 is TST)
        if action == 2:
            if state['temperature'] > 80.0 or state['battery'] < 30.0:
                return False
            if state['time_since_tst'] < 120.0:  # Recovery period
                return False
                
        return True
        
    def compute_scenario_summary(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics for a scenario across episodes."""
        if not episodes:
            return {}
            
        summaries = [ep['summary'] for ep in episodes]
        
        return {
            'episodes_count': len(episodes),
            'avg_expert_agreement': np.mean([s['expert_agreement_rate'] for s in summaries]),
            'avg_safety_violation_rate': np.mean([s['safety_violation_rate'] for s in summaries]),
            'avg_total_reward': np.mean([s['total_reward'] for s in summaries]),
            'max_temperature_observed': max([s['max_temperature'] for s in summaries]),
            'min_battery_observed': min([s['min_battery'] for s in summaries]),
            'avg_steps': np.mean([s['total_steps'] for s in summaries]),
            'success_rate': sum([1 for s in summaries if s['safety_violation_rate'] == 0]) / len(summaries)
        }
        
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute comprehensive validation metrics."""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'validation_summary': {},
            'state_combination_analysis': {},
            'scenario_analysis': {},
            'safety_metrics': {},
            'performance_metrics': {}
        }
        
        # Overall validation summary
        total_steps = len(self.validation_data)
        expert_agreements = sum([1 for step in self.validation_data if step['agreement']])
        safety_violations = sum([1 for step in self.validation_data if not step['is_safe']])
        
        metrics['validation_summary'] = {
            'total_validation_steps': total_steps,
            'total_scenarios_tested': len(self.scenario_results),
            'total_state_combinations_tested': len(self.state_combination_results),
            'overall_expert_agreement': expert_agreements / total_steps if total_steps > 0 else 0,
            'overall_safety_violation_rate': safety_violations / total_steps if total_steps > 0 else 0
        }
        
        # State combination analysis
        state_agreements = sum([1 for result in self.state_combination_results.values() if result['agreement']])
        state_safety_violations = sum([1 for result in self.state_combination_results.values() if not result['is_safe']])
        total_combinations = len(self.state_combination_results)
        
        metrics['state_combination_analysis'] = {
            'total_combinations': total_combinations,
            'expert_agreement_rate': state_agreements / total_combinations if total_combinations > 0 else 0,
            'safety_violation_rate': state_safety_violations / total_combinations if total_combinations > 0 else 0,
            'combinations_with_disagreement': total_combinations - state_agreements,
            'combinations_with_violations': state_safety_violations
        }
        
        # Scenario analysis
        metrics['scenario_analysis'] = {}
        for scenario_name, scenario_data in self.scenario_results.items():
            metrics['scenario_analysis'][scenario_name] = scenario_data['summary']
            
        # Safety metrics
        if self.validation_data:
            temperatures = [step['state']['temperature'] for step in self.validation_data]
            batteries = [step['state']['battery'] for step in self.validation_data]
            
            metrics['safety_metrics'] = {
                'max_temperature_observed': max(temperatures),
                'min_battery_observed': min(batteries),
                'temperature_violations': sum([1 for t in temperatures if t > 85.0]),
                'battery_violations': sum([1 for b in batteries if b < 10.0]),
                'critical_temperature_approaches': sum([1 for t in temperatures if t > 80.0]),
                'critical_battery_approaches': sum([1 for b in batteries if b < 20.0])
            }
            
        # Performance metrics
        if self.validation_data:
            rewards = [step['reward'] for step in self.validation_data]
            action_distribution = {}
            for action in [0, 1, 2]:
                action_distribution[f'action_{action}'] = sum([1 for step in self.validation_data if step['agent_action'] == action])
                
            metrics['performance_metrics'] = {
                'avg_reward': np.mean(rewards),
                'total_reward': sum(rewards),
                'reward_std': np.std(rewards),
                'action_distribution': action_distribution
            }
            
        return metrics
        
    def save_results(self):
        """Save all validation results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save step-by-step validation data to CSV
        if self.validation_data:
            df = pd.DataFrame(self.validation_data)
            csv_file = self.validation_dir / f'validation_steps_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
            self.logger.info(f"Step-by-step data saved to {csv_file}")
            
        # Save state combination results
        state_combo_file = self.validation_dir / f'state_combinations_{timestamp}.json'
        with open(state_combo_file, 'w') as f:
            json.dump(self.state_combination_results, f, indent=2, default=str)
        self.logger.info(f"State combination results saved to {state_combo_file}")
        
        # Save scenario results
        scenario_file = self.validation_dir / f'scenario_results_{timestamp}.json'
        with open(scenario_file, 'w') as f:
            json.dump(self.scenario_results, f, indent=2, default=str)
        self.logger.info(f"Scenario results saved to {scenario_file}")
        
    def plot_results(self, metrics: Dict[str, Any]):
        """Generate comprehensive visualization suite."""
        self.logger.info("Generating professional visualization suite...")
        
        # 1. Per-scenario time series plots
        self.plot_scenario_time_series()
        
        # 2. Expert vs agent agreement analysis
        self.plot_agreement_analysis()
        
        # 3. Safety violation heatmaps
        self.plot_safety_heatmaps()
        
        # 4. Error distribution donut charts
        self.plot_error_distribution()
        
        # 5. Performance overview dashboard
        self.plot_performance_dashboard(metrics)
        
        # 6. State combination analysis
        self.plot_state_combination_analysis()
        
        self.logger.info("All visualizations generated successfully")
        
    def plot_scenario_time_series(self):
        """Create time series plots for each scenario."""
        for scenario_name, scenario_data in self.scenario_results.items():
            if not scenario_data['episodes']:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Scenario: {scenario_name.replace("_", " ").title()}', 
                        fontsize=16, fontweight='bold', color=self.colors['primary'])
            
            # Combine all episodes for this scenario
            all_steps = []
            for episode in scenario_data['episodes']:
                for step in episode['steps']:
                    all_steps.append({
                        **step,
                        'episode': episode['episode']
                    })
                    
            if not all_steps:
                continue
                
            df = pd.DataFrame(all_steps)
            
            # Temperature over time
            ax1 = axes[0, 0]
            for episode in range(len(scenario_data['episodes'])):
                episode_data = df[df['episode'] == episode]
                temperatures = [step['temperature'] for step in episode_data['state']]
                ax1.plot(range(len(temperatures)), temperatures, 
                        alpha=0.7, linewidth=2, color=self.colors['secondary'])
            ax1.axhline(y=85, color=self.colors['danger'], linestyle='--', alpha=0.8, label='Critical Limit')
            ax1.axhline(y=80, color=self.colors['warning'], linestyle='--', alpha=0.8, label='Warning Limit')
            ax1.set_title('Temperature Evolution', fontweight='bold')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Temperature (°C)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Battery over time
            ax2 = axes[0, 1]
            for episode in range(len(scenario_data['episodes'])):
                episode_data = df[df['episode'] == episode]
                batteries = [step['battery'] for step in episode_data['state']]
                ax2.plot(range(len(batteries)), batteries, 
                        alpha=0.7, linewidth=2, color=self.colors['accent'])
            ax2.axhline(y=20, color=self.colors['warning'], linestyle='--', alpha=0.8, label='Low Battery')
            ax2.axhline(y=10, color=self.colors['danger'], linestyle='--', alpha=0.8, label='Critical Battery')
            ax2.set_title('Battery Evolution', fontweight='bold')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Battery (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Action distribution
            ax3 = axes[1, 0]
            action_counts = df['agent_action'].value_counts().sort_index()
            bars = ax3.bar([self.action_labels[i] for i in action_counts.index], 
                          action_counts.values, 
                          color=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
            ax3.set_title('Agent Action Distribution', fontweight='bold')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Expert agreement over time
            ax4 = axes[1, 1]
            agreement_rate = df['agreement'].rolling(window=10, min_periods=1).mean()
            ax4.plot(range(len(agreement_rate)), agreement_rate, 
                    color=self.colors['success'], linewidth=3)
            ax4.axhline(y=0.8, color=self.colors['warning'], linestyle='--', alpha=0.8, label='Target (80%)')
            ax4.set_title('Expert Agreement Rate (10-step rolling average)', fontweight='bold')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Agreement Rate')
            ax4.set_ylim(0, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'scenario_{scenario_name}_timeseries.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_agreement_analysis(self):
        """Create bar charts comparing expert vs agent agreement across states."""
        # Prepare data for agreement analysis
        agreement_data = []
        
        for combo_id, result in self.state_combination_results.items():
            parts = combo_id.split('|')
            agreement_data.append({
                'battery_level': parts[0],
                'temp_level': parts[1],
                'threat_level': parts[2],
                'agreement': result['agreement'],
                'agent_action': result['agent_action'],
                'expert_action': result['expert_action']
            })
            
        df = pd.DataFrame(agreement_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Expert-Agent Agreement Analysis Across State Dimensions', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Agreement by battery level
        ax1 = axes[0, 0]
        battery_agreement = df.groupby('battery_level')['agreement'].mean()
        bars1 = ax1.bar(battery_agreement.index, battery_agreement.values, 
                       color=self.colors['secondary'], alpha=0.8)
        ax1.set_title('Agreement Rate by Battery Level', fontweight='bold')
        ax1.set_ylabel('Agreement Rate')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Agreement by temperature level
        ax2 = axes[0, 1]
        temp_agreement = df.groupby('temp_level')['agreement'].mean()
        bars2 = ax2.bar(temp_agreement.index, temp_agreement.values, 
                       color=self.colors['accent'], alpha=0.8)
        ax2.set_title('Agreement Rate by Temperature Level', fontweight='bold')
        ax2.set_ylabel('Agreement Rate')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Agreement by threat level
        ax3 = axes[1, 0]
        threat_agreement = df.groupby('threat_level')['agreement'].mean()
        bars3 = ax3.bar(threat_agreement.index, threat_agreement.values, 
                       color=self.colors['success'], alpha=0.8)
        ax3.set_title('Agreement Rate by Threat Level', fontweight='bold')
        ax3.set_ylabel('Agreement Rate')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Overall action comparison
        ax4 = axes[1, 1]
        action_comparison = pd.crosstab(df['expert_action'], df['agent_action'], normalize='index')
        sns.heatmap(action_comparison, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.action_labels, yticklabels=self.action_labels, ax=ax4)
        ax4.set_title('Action Agreement Matrix\n(Expert vs Agent)', fontweight='bold')
        ax4.set_xlabel('Agent Action')
        ax4.set_ylabel('Expert Action')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_safety_heatmaps(self):
        """Create heatmaps showing safety violations across state combinations."""
        # Prepare safety violation data
        safety_matrix = np.zeros((len(self.battery_levels), len(self.temp_levels)))
        agreement_matrix = np.zeros((len(self.battery_levels), len(self.temp_levels)))
        
        for combo_id, result in self.state_combination_results.items():
            parts = combo_id.split('|')
            battery_idx = self.battery_levels.index(parts[0])
            temp_idx = self.temp_levels.index(parts[1])
            
            # Average across all threat levels for this battery-temperature combination
            if not result['is_safe']:
                safety_matrix[battery_idx, temp_idx] += 1
            if not result['agreement']:
                agreement_matrix[battery_idx, temp_idx] += 1
                
        # Normalize by number of threat levels (3)
        safety_matrix = safety_matrix / 3
        agreement_matrix = agreement_matrix / 3
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Safety and Agreement Heatmaps Across State Space', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Safety violations heatmap
        ax1 = axes[0]
        sns.heatmap(safety_matrix, annot=True, fmt='.2f', cmap='Reds', 
                   xticklabels=self.temp_levels, yticklabels=self.battery_levels, 
                   ax=ax1, cbar_kws={'label': 'Violation Rate'})
        ax1.set_title('Safety Violation Rate by Battery-Temperature', fontweight='bold')
        ax1.set_xlabel('Temperature Level')
        ax1.set_ylabel('Battery Level')
        
        # Agreement violations heatmap
        ax2 = axes[1]
        sns.heatmap(agreement_matrix, annot=True, fmt='.2f', cmap='Blues', 
                   xticklabels=self.temp_levels, yticklabels=self.battery_levels, 
                   ax=ax2, cbar_kws={'label': 'Disagreement Rate'})
        ax2.set_title('Expert Disagreement Rate by Battery-Temperature', fontweight='bold')
        ax2.set_xlabel('Temperature Level')
        ax2.set_ylabel('Battery Level')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'safety_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_error_distribution(self):
        """Create donut charts showing error distribution."""
        # Count different types of violations
        temp_violations = 0
        battery_violations = 0
        action_violations = 0
        total_safe = 0
        
        for step in self.validation_data:
            state = step['state']
            action = step['agent_action']
            
            if state['temperature'] > 85.0:
                temp_violations += 1
            elif state['battery'] < 10.0:
                battery_violations += 1
            elif action == 2 and (state['temperature'] > 80.0 or state['battery'] < 30.0):
                action_violations += 1
            else:
                total_safe += 1
                
        # Create donut chart
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Error Distribution Analysis', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Safety violations breakdown
        ax1 = axes[0]
        violation_labels = ['Temperature\nViolations', 'Battery\nViolations', 'Action\nViolations', 'Safe\nOperations']
        violation_counts = [temp_violations, battery_violations, action_violations, total_safe]
        violation_colors = [self.colors['danger'], self.colors['warning'], self.colors['accent'], self.colors['success']]
        
        wedges, texts, autotexts = ax1.pie(violation_counts, labels=violation_labels, 
                                          colors=violation_colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85)
        
        # Create donut by adding a white circle in the center
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.set_title('Safety Violation Types', fontweight='bold', pad=20)
        
        # Expert agreement breakdown
        ax2 = axes[1]
        agreements = sum([1 for step in self.validation_data if step['agreement']])
        disagreements = len(self.validation_data) - agreements
        
        agreement_labels = ['Expert\nAgreement', 'Expert\nDisagreement']
        agreement_counts = [agreements, disagreements]
        agreement_colors = [self.colors['success'], self.colors['warning']]
        
        wedges2, texts2, autotexts2 = ax2.pie(agreement_counts, labels=agreement_labels,
                                             colors=agreement_colors, autopct='%1.1f%%',
                                             startangle=90, pctdistance=0.85)
        
        centre_circle2 = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle2)
        ax2.set_title('Expert Agreement Distribution', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_dashboard(self, metrics: Dict[str, Any]):
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle('UAV Cybersecurity RL System - Performance Dashboard', 
                    fontsize=20, fontweight='bold', color=self.colors['primary'])
        
        # 1. Overall metrics summary (top row)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        summary = metrics['validation_summary']
        safety_metrics = metrics.get('safety_metrics', {})
        
        summary_text = f"""
        VALIDATION SUMMARY
        
        Total Validation Steps: {summary.get('total_validation_steps', 0):,}
        Scenarios Tested: {summary.get('total_scenarios_tested', 0)}
        State Combinations: {summary.get('total_state_combinations_tested', 0)}
        
        Expert Agreement: {summary.get('overall_expert_agreement', 0):.1%}
        Safety Violation Rate: {summary.get('overall_safety_violation_rate', 0):.1%}
        Max Temperature: {safety_metrics.get('max_temperature_observed', 0):.1f}°C
        Min Battery: {safety_metrics.get('min_battery_observed', 0):.1f}%
        """
        
        ax1.text(0.5, 0.5, summary_text, transform=ax1.transAxes, 
                fontsize=14, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=self.colors['background'], 
                         edgecolor=self.colors['primary'], linewidth=2))
        
        # 2. Scenario performance comparison
        ax2 = fig.add_subplot(gs[1, 0])
        scenario_names = []
        agreement_rates = []
        
        for name, analysis in metrics.get('scenario_analysis', {}).items():
            scenario_names.append(name.replace('_', '\n'))
            agreement_rates.append(analysis.get('avg_expert_agreement', 0))
            
        bars = ax2.barh(scenario_names, agreement_rates, color=self.colors['secondary'])
        ax2.set_title('Expert Agreement by Scenario', fontweight='bold')
        ax2.set_xlabel('Agreement Rate')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, 1.1)
        
        # 3. Action distribution
        ax3 = fig.add_subplot(gs[1, 1])
        if 'performance_metrics' in metrics and 'action_distribution' in metrics['performance_metrics']:
            action_dist = metrics['performance_metrics']['action_distribution']
            actions = [self.action_labels[int(k.split('_')[1])] for k in action_dist.keys()]
            counts = list(action_dist.values())
            
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']]
            ax3.pie(counts, labels=actions, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Overall Action Distribution', fontweight='bold')
            
        # 4. Safety metrics radar chart
        ax4 = fig.add_subplot(gs[1, 2], projection='polar')
        
        categories = ['Expert\nAgreement', 'Safety\nCompliance', 'Temperature\nControl', 
                     'Battery\nManagement', 'Performance\nEfficiency']
        
        # Calculate normalized scores (0-1)
        scores = [
            summary.get('overall_expert_agreement', 0),
            1 - summary.get('overall_safety_violation_rate', 0),
            1 - min(safety_metrics.get('max_temperature_observed', 45) / 85, 1),
            safety_metrics.get('min_battery_observed', 100) / 100,
            0.8  # Placeholder for performance efficiency
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        scores += scores[:1]  # Complete the circle
        angles = np.concatenate((angles, [angles[0]]))
        
        ax4.plot(angles, scores, 'o-', linewidth=2, color=self.colors['primary'])
        ax4.fill(angles, scores, alpha=0.25, color=self.colors['primary'])
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar Chart', fontweight='bold', pad=20)
        ax4.grid(True)
        
        # 5. Temperature and battery trends (bottom row)
        if self.validation_data:
            ax5 = fig.add_subplot(gs[2, :])
            
            # Sample every 10th point for readability if too many points
            step_data = self.validation_data[::max(1, len(self.validation_data) // 500)]
            
            steps = range(len(step_data))
            temperatures = [step['state']['temperature'] for step in step_data]
            batteries = [step['state']['battery'] for step in step_data]
            
            ax5_twin = ax5.twinx()
            
            line1 = ax5.plot(steps, temperatures, color=self.colors['danger'], 
                           linewidth=2, label='Temperature', alpha=0.8)
            line2 = ax5_twin.plot(steps, batteries, color=self.colors['success'], 
                                linewidth=2, label='Battery', alpha=0.8)
            
            ax5.axhline(y=85, color=self.colors['danger'], linestyle='--', alpha=0.5)
            ax5_twin.axhline(y=20, color=self.colors['warning'], linestyle='--', alpha=0.5)
            
            ax5.set_xlabel('Validation Steps')
            ax5.set_ylabel('Temperature (°C)', color=self.colors['danger'])
            ax5_twin.set_ylabel('Battery (%)', color=self.colors['success'])
            ax5.set_title('Temperature and Battery Trends During Validation', fontweight='bold')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='upper right')
            
            ax5.grid(True, alpha=0.3)
            
        # 6. State space coverage visualization
        ax6 = fig.add_subplot(gs[3, :])
        
        # Create a simplified state space coverage visualization
        coverage_data = []
        for combo_id, result in self.state_combination_results.items():
            parts = combo_id.split('|')
            coverage_data.append({
                'battery': parts[0],
                'temperature': parts[1],
                'threat': parts[2],
                'tested': 1,
                'safe': 1 if result['is_safe'] else 0,
                'agreement': 1 if result['agreement'] else 0
            })
            
        coverage_df = pd.DataFrame(coverage_data)
        
        # Group by battery and temperature for visualization
        pivot_data = coverage_df.groupby(['battery', 'temperature']).agg({
            'tested': 'sum',
            'safe': 'mean',
            'agreement': 'mean'
        }).reset_index()
        
        # Create a stacked bar chart
        battery_order = ['0-20%', '21-40%', '41-60%', '61-80%', '81-100%']
        x_pos = np.arange(len(battery_order))
        
        safe_bars = []
        agreement_bars = []
        
        for temp in self.temp_levels:
            temp_data = pivot_data[pivot_data['temperature'] == temp]
            safe_values = []
            agreement_values = []
            
            for battery in battery_order:
                row = temp_data[temp_data['battery'] == battery]
                if len(row) > 0:
                    safe_values.append(row['safe'].iloc[0])
                    agreement_values.append(row['agreement'].iloc[0])
                else:
                    safe_values.append(0)
                    agreement_values.append(0)
                    
            safe_bars.append(safe_values)
            agreement_bars.append(agreement_values)
            
        # Plot safety rates
        width = 0.25
        for i, temp in enumerate(self.temp_levels):
            ax6.bar([x + width * i for x in x_pos], safe_bars[i], width, 
                   label=f'{temp} (Safety)', alpha=0.7,
                   color=plt.cm.Reds(0.3 + i * 0.3))
                   
        ax6.set_xlabel('Battery Level')
        ax6.set_ylabel('Safety Rate')
        ax6.set_title('Safety Rate by Battery Level and Temperature', fontweight='bold')
        ax6.set_xticks([x + width for x in x_pos])
        ax6.set_xticklabels(battery_order)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim(0, 1.1)
        
        plt.savefig(self.plots_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_state_combination_analysis(self):
        """Create detailed analysis of all 45 state combinations."""
        # Prepare data matrix for all combinations
        results_matrix = np.zeros((len(self.battery_levels), len(self.temp_levels) * len(self.threat_levels)))
        agreement_matrix = np.zeros((len(self.battery_levels), len(self.temp_levels) * len(self.threat_levels)))
        
        col_labels = []
        for temp in self.temp_levels:
            for threat in self.threat_levels:
                col_labels.append(f"{temp}\n{threat}")
                
        for combo_id, result in self.state_combination_results.items():
            parts = combo_id.split('|')
            battery_idx = self.battery_levels.index(parts[0])
            temp_idx = self.temp_levels.index(parts[1])
            threat_idx = self.threat_levels.index(parts[2])
            col_idx = temp_idx * len(self.threat_levels) + threat_idx
            
            results_matrix[battery_idx, col_idx] = 1 if result['is_safe'] else 0
            agreement_matrix[battery_idx, col_idx] = 1 if result['agreement'] else 0
            
        fig, axes = plt.subplots(2, 1, figsize=(20, 12))
        fig.suptitle('Complete State Space Analysis (45 Combinations)', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Safety results
        ax1 = axes[0]
        im1 = ax1.imshow(results_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax1.set_title('Safety Results Across All State Combinations', fontweight='bold')
        ax1.set_ylabel('Battery Level')
        ax1.set_yticks(range(len(self.battery_levels)))
        ax1.set_yticklabels(self.battery_levels)
        ax1.set_xticks(range(len(col_labels)))
        ax1.set_xticklabels(col_labels, rotation=45, ha='right')
        plt.colorbar(im1, ax=ax1, label='Safe (1) / Unsafe (0)')
        
        # Add grid lines to separate temperature groups
        for i in range(1, len(self.temp_levels)):
            ax1.axvline(x=i * len(self.threat_levels) - 0.5, color='black', linewidth=2)
            
        # Agreement results
        ax2 = axes[1]
        im2 = ax2.imshow(agreement_matrix, cmap='Blues', aspect='auto', interpolation='nearest')
        ax2.set_title('Expert Agreement Across All State Combinations', fontweight='bold')
        ax2.set_ylabel('Battery Level')
        ax2.set_xlabel('Temperature Level | Threat Level')
        ax2.set_yticks(range(len(self.battery_levels)))
        ax2.set_yticklabels(self.battery_levels)
        ax2.set_xticks(range(len(col_labels)))
        ax2.set_xticklabels(col_labels, rotation=45, ha='right')
        plt.colorbar(im2, ax=ax2, label='Agreement (1) / Disagreement (0)')
        
        # Add grid lines
        for i in range(1, len(self.temp_levels)):
            ax2.axvline(x=i * len(self.threat_levels) - 0.5, color='black', linewidth=2)
            
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'state_combination_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for the validation suite."""
    parser = argparse.ArgumentParser(description='UAV Cybersecurity RL Validation Suite')
    parser.add_argument('--model_path', required=True, help='Path to trained model JSON file')
    parser.add_argument('--outdir', default='validation_results', help='Output directory for results')
    parser.add_argument('--scenarios_only', action='store_true', help='Run only safety scenarios (skip state combinations)')
    parser.add_argument('--plot_only', action='store_true', help='Generate plots only (skip validation)')
    
    args = parser.parse_args()
    
    try:
        # Initialize validation suite
        suite = ValidationSuite(args.model_path, args.outdir)
        
        if not args.plot_only:
            # Run validation
            metrics = suite.run_validation()
            
            # Generate plots
            suite.plot_results(metrics)
            
            # Print summary
            print("\n" + "="*80)
            print("VALIDATION SUITE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Results saved to: {suite.output_dir}")
            print(f"Expert Agreement: {metrics['validation_summary']['overall_expert_agreement']:.1%}")
            print(f"Safety Violation Rate: {metrics['validation_summary']['overall_safety_violation_rate']:.1%}")
            print(f"Scenarios Tested: {metrics['validation_summary']['total_scenarios_tested']}")
            print(f"State Combinations: {metrics['validation_summary']['total_state_combinations_tested']}")
            print("="*80)
        else:
            print("Plot-only mode not implemented. Please run full validation first.")
            
    except KeyboardInterrupt:
        print("\nValidation interrupted by user. Partial results may be available.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during validation: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()