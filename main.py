"""
Main Entry Point for UAV Cybersecurity RL System.

This script provides the main interface for training, evaluation, and deployment
of the Q-learning agent for UAV cybersecurity algorithm selection.
"""

import logging
import argparse
import os
import json
import sys
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.q_learning_agent import WarmStartQLearningAgent
from agents.expert_policy import ExpertPolicy
from environment.thermal_simulator import TunableUAVSimulator
from validation.safety_validator import SafetyValidator
from utils.state_discretizer import StateDiscretizer
from training.trainer import RLTrainer


def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"uav_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce matplotlib logging
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='UAV Cybersecurity RL System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py --mode train --episodes 1000 --output_dir results/experiment1
  
  # Evaluate an existing model
  python main.py --mode evaluate --model_path results/experiment1/checkpoints/final_model.json
  
  # Test expert policy
  python main.py --mode test_expert
  
  # Quick training test
  python main.py --mode train --episodes 50 --max_steps 100 --output_dir test_run
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'test_expert', 'demo', 'validate_system'],
                        help='Operation mode')
    
    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=300,
                        help='Maximum steps per episode')
    parser.add_argument('--eval_interval', type=int, default=100,
                        help='Evaluation interval (episodes)')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='Model saving interval (episodes)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='Logging interval (episodes)')
    
    # Agent parameters
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate (alpha)')
    parser.add_argument('--discount', type=float, default=0.99,
                        help='Discount factor (gamma)')
    parser.add_argument('--epsilon_start', type=float, default=0.3,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05,
                        help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995,
                        help='Exploration rate decay factor')
    parser.add_argument('--expert_bonus', type=float, default=2.0,
                        help='Reward bonus for expert agreement')
    
    # Configuration files
    parser.add_argument('--sim_config', type=str, default='config/simulator_params.json',
                        help='Simulator configuration file')
    parser.add_argument('--expert_config', type=str, default='config/expert_policy.json',
                        help='Expert policy configuration file')
    
    # Input/Output parameters
    parser.add_argument('--output_dir', type=str, 
                        default=f'results/{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default='uav_ddos_rl',
                        help='Experiment name')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for evaluate/demo modes)')
    
    # System parameters
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    parser.add_argument('--no_validation', action='store_true',
                        help='Skip safety validation during training')
    
    return parser.parse_args()


def create_components(args: argparse.Namespace) -> Dict:
    """Create and initialize all system components."""
    logger = logging.getLogger(__name__)
    
    # Create simulator
    sim_params = None
    if os.path.exists(args.sim_config):
        logger.info(f"Loading simulator config from {args.sim_config}")
        with open(args.sim_config, 'r') as f:
            sim_params = json.load(f)
    else:
        logger.warning(f"Simulator config not found: {args.sim_config}, using defaults")
    
    simulator = TunableUAVSimulator(params=sim_params)
    
    # Create expert policy
    expert_policy = ExpertPolicy(args.expert_config)
    
    # Create state discretizer
    state_discretizer = StateDiscretizer()
    
    # Create agent
    agent = WarmStartQLearningAgent(
        expert_policy=expert_policy,
        state_discretizer=state_discretizer,
        learning_rate=args.learning_rate,
        discount_factor=args.discount,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        expert_bonus_weight=args.expert_bonus
    )
    
    # Create validator (if not disabled)
    validator = None
    if not args.no_validation:
        validator = SafetyValidator(simulator)
    
    logger.info("All components created successfully")
    
    return {
        'simulator': simulator,
        'expert_policy': expert_policy,
        'state_discretizer': state_discretizer,
        'agent': agent,
        'validator': validator
    }


def mode_train(args: argparse.Namespace, components: Dict) -> None:
    """Training mode implementation."""
    logger = logging.getLogger(__name__)
    
    # Create trainer configuration
    trainer_config = {
        'num_episodes': args.episodes,
        'max_steps_per_episode': args.max_steps,
        'eval_interval': args.eval_interval,
        'save_interval': args.save_interval,
        'log_interval': args.log_interval,
        'output_dir': args.output_dir,
        'experiment_name': args.experiment_name
    }
    
    # Create trainer
    trainer = RLTrainer(
        agent=components['agent'],
        simulator=components['simulator'],
        validator=components['validator'],
        config=trainer_config
    )
    
    # Save arguments for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, 'run_arguments.json')
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run training
    logger.info("Starting training...")
    results = trainer.train()
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Episodes: {results['training_summary']['total_episodes']}")
    print(f"Training time: {results['training_summary']['training_time_minutes']:.1f} minutes")
    print(f"Final reward: {results['performance']['final_avg_reward']:.2f}")
    print(f"Expert agreement: {results['performance']['final_expert_agreement']:.1%}")
    print(f"Final effectiveness: {results['performance']['final_effectiveness']:.1%}")
    print(f"Safety violations: {results['safety']['total_safety_violations']}")
    print(f"Output directory: {results['output_directory']}")
    print("="*60)


def mode_evaluate(args: argparse.Namespace, components: Dict) -> None:
    """Evaluation mode implementation."""
    logger = logging.getLogger(__name__)
    
    if not args.model_path:
        logger.error("Model path required for evaluation mode")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    components['agent'].load(args.model_path)
    
    # Run evaluation
    if components['validator']:
        logger.info("Starting comprehensive evaluation...")
        metrics = components['validator'].validate(components['agent'], num_episodes=10)
        
        # Generate and save report
        report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
        os.makedirs(args.output_dir, exist_ok=True)
        components['validator'].generate_report(metrics, report_path)
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"Expert agreement: {metrics['expert_agreement']:.1%}")
        print(f"Safety violation rate: {metrics['safety_violation_rate']:.1%}")
        print(f"Success rate: {metrics['success_rate']:.1%}")
        print(f"Safety score: {metrics['safety_score']:.3f}")
        print(f"Status: {'PASSED' if metrics['passed'] else 'FAILED'}")
        print(f"Report saved to: {report_path}")
        print("="*60)
    else:
        logger.error("Validation disabled, cannot run evaluation")


def mode_test_expert(args: argparse.Namespace, components: Dict) -> None:
    """Test expert policy mode."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("EXPERT POLICY TEST")
    print("="*60)
    
    expert_policy = components['expert_policy']
    
    # Validate lookup table
    is_valid = expert_policy.validate_lookup_table()
    print(f"Lookup table valid: {is_valid}")
    
    # Print statistics
    stats = expert_policy.get_lookup_table_stats()
    print(f"Total lookup entries: {stats['total_entries']}")
    print("\nAction distribution:")
    for action, count in stats['action_distribution'].items():
        percentage = stats['action_percentages'][action]
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    # Test scenarios
    test_scenarios = [
        {'name': 'High battery, safe temp, normal', 'battery': 90, 'temperature': 55, 'threat': 0},
        {'name': 'Medium battery, warm, confirming', 'battery': 60, 'temperature': 70, 'threat': 1},
        {'name': 'Low battery, hot, confirmed', 'battery': 25, 'temperature': 80, 'threat': 2},
        {'name': 'Critical battery, any temp', 'battery': 15, 'temperature': 60, 'threat': 1},
        {'name': 'Good battery, critical temp', 'battery': 80, 'temperature': 85, 'threat': 1},
    ]
    
    print("\nTest scenarios:")
    print("-" * 40)
    
    for scenario in test_scenarios:
        state = {
            'battery': scenario['battery'],
            'temperature': scenario['temperature'],
            'threat': scenario['threat'],
            'cpu_usage': 30,
            'time_since_tst': 300,
            'power': 5
        }
        
        action = expert_policy.get_action(state)
        action_name = expert_policy.get_action_name(action)
        state_repr = expert_policy.get_state_representation(state)
        safe, reason = expert_policy.is_safe_action(state, action)
        
        print(f"\n{scenario['name']}:")
        print(f"  State: {state_repr}")
        print(f"  Action: {action} ({action_name})")
        print(f"  Safe: {safe} - {reason}")
    
    print("="*60)


def mode_demo(args: argparse.Namespace, components: Dict) -> None:
    """Demo mode - interactive demonstration."""
    logger = logging.getLogger(__name__)
    
    if args.model_path:
        if os.path.exists(args.model_path):
            logger.info(f"Loading trained model from {args.model_path}")
            components['agent'].load(args.model_path)
        else:
            logger.warning(f"Model file not found: {args.model_path}, using warm-started agent")
    
    print("\n" + "="*60)
    print("UAV CYBERSECURITY RL DEMO")
    print("="*60)
    
    simulator = components['simulator']
    agent = components['agent']
    expert_policy = components['expert_policy']
    
    # Reset simulator
    state_dict = simulator.reset(initial_temp=50, initial_battery=80)
    
    print("Running 20-step demonstration...")
    print("Threat pattern: Normal -> Confirming -> Confirmed -> Normal")
    
    threat_pattern = [0, 0, 1, 1, 2, 2, 2, 1, 1, 0, 0, 1, 2, 1, 0, 0, 0, 1, 1, 0]
    
    print("\nStep | Threat | Agent Action | Expert Action | Temp | Battery | Power | Safe")
    print("-" * 80)
    
    for step in range(20):
        # Current state
        current_threat = threat_pattern[step]
        current_state = {
            'temperature': state_dict['temperature'],
            'battery': state_dict['battery'],
            'threat': current_threat,
            'cpu_usage': state_dict['cpu_usage'],
            'time_since_tst': state_dict['time_since_tst'],
            'power': state_dict['power_consumption']
        }
        
        # Get actions
        agent_action = agent.get_action(current_state, training=False)
        expert_action = expert_policy.get_action(current_state)
        
        # Check safety
        safe, _ = expert_policy.is_safe_action(current_state, agent_action)
        
        # Action names
        action_names = ['No_DDoS', 'XGBoost', 'TST']
        agent_name = action_names[agent_action]
        expert_name = action_names[expert_action]
        
        # Threat names
        threat_names = ['Normal', 'Confirming', 'Confirmed']
        threat_name = threat_names[current_threat]
        
        # Display step
        agreement = "✓" if agent_action == expert_action else "✗"
        safety = "✓" if safe else "✗"
        
        print(f"{step+1:4d} | {threat_name:9s} | {agent_name:8s} {agreement} | {expert_name:9s} | "
              f"{state_dict['temperature']:4.1f} | {state_dict['battery']:5.1f} | "
              f"{state_dict['power_consumption']:5.1f} | {safety}")
        
        # Simulate step
        state_dict = simulator.simulate_step(agent_action, dt=10.0)  # 10-second steps
    
    # Final summary
    final_state = {
        'temperature': state_dict['temperature'],
        'battery': state_dict['battery'],
        'threat': 0,
        'cpu_usage': state_dict['cpu_usage'],
        'time_since_tst': state_dict['time_since_tst'],
        'power': state_dict['power_consumption']
    }
    
    print(f"\nFinal state:")
    print(f"  Temperature: {final_state['temperature']:.1f}°C")
    print(f"  Battery: {final_state['battery']:.1f}%")
    print(f"  Power: {final_state['power']:.1f}W")
    
    thermal_zone = simulator.get_thermal_zone()
    battery_zone = simulator.get_battery_zone()
    print(f"  Thermal zone: {thermal_zone}")
    print(f"  Battery zone: {battery_zone}")
    
    print("="*60)


def mode_validate_system(args: argparse.Namespace, components: Dict) -> None:
    """System validation mode - comprehensive system checks."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*60)
    print("SYSTEM VALIDATION")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Component initialization
    print("1. Component Initialization...")
    try:
        # Test simulator
        simulator = components['simulator']
        state = simulator.reset()
        state = simulator.simulate_step(1, dt=1.0)
        print("   ✓ Simulator: Working")
        
        # Test expert policy
        expert = components['expert_policy']
        action = expert.get_action({'battery': 80, 'temperature': 50, 'threat': 1, 
                                   'cpu_usage': 30, 'time_since_tst': 300, 'power': 5})
        print("   ✓ Expert Policy: Working")
        
        # Test state discretizer
        discretizer = components['state_discretizer']
        discrete = discretizer.discretize({'battery': 80, 'temperature': 50, 'threat': 1, 
                                          'cpu_usage': 30, 'time_since_tst': 300, 'power': 5})
        print("   ✓ State Discretizer: Working")
        
        # Test agent
        agent = components['agent']
        action = agent.get_action({'battery': 80, 'temperature': 50, 'threat': 1, 
                                  'cpu_usage': 30, 'time_since_tst': 300, 'power': 5}, training=False)
        print("   ✓ RL Agent: Working")
        
    except Exception as e:
        print(f"   ✗ Component test failed: {e}")
        all_passed = False
    
    # Test 2: Safety constraints
    print("\n2. Safety Constraints...")
    try:
        expert = components['expert_policy']
        
        # Critical temperature test
        unsafe_state = {'battery': 80, 'temperature': 85, 'threat': 1, 
                       'cpu_usage': 30, 'time_since_tst': 300, 'power': 5}
        action = expert.get_action(unsafe_state)
        if action == 0:  # Should be No_DDoS
            print("   ✓ Critical temperature handling: Working")
        else:
            print("   ✗ Critical temperature handling: Failed")
            all_passed = False
        
        # Critical battery test
        unsafe_state = {'battery': 15, 'temperature': 50, 'threat': 2, 
                       'cpu_usage': 30, 'time_since_tst': 300, 'power': 5}
        action = expert.get_action(unsafe_state)
        if action == 0:  # Should be No_DDoS
            print("   ✓ Critical battery handling: Working")
        else:
            print("   ✗ Critical battery handling: Failed")
            all_passed = False
            
    except Exception as e:
        print(f"   ✗ Safety test failed: {e}")
        all_passed = False
    
    # Test 3: Configuration files
    print("\n3. Configuration Files...")
    
    config_files = [
        ('Simulator config', args.sim_config),
        ('Expert policy config', args.expert_config),
    ]
    
    for name, path in config_files:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    json.load(f)
                print(f"   ✓ {name}: Valid")
            except Exception as e:
                print(f"   ✗ {name}: Invalid JSON - {e}")
                all_passed = False
        else:
            print(f"   ⚠ {name}: Not found (using defaults)")
    
    # Test 4: Quick training run
    print("\n4. Quick Training Test...")
    try:
        trainer = RLTrainer(
            agent=components['agent'],
            simulator=components['simulator'],
            validator=None,  # Skip validation for speed
            config={
                'num_episodes': 5,
                'max_steps_per_episode': 20,
                'eval_interval': 10,
                'save_interval': 10,
                'log_interval': 2,
                'output_dir': os.path.join(args.output_dir, 'validation_test'),
                'experiment_name': 'validation_test'
            }
        )
        
        results = trainer.train()
        if results['training_summary']['total_episodes'] == 5:
            print("   ✓ Training pipeline: Working")
        else:
            print("   ✗ Training pipeline: Failed")
            all_passed = False
            
    except Exception as e:
        print(f"   ✗ Training test failed: {e}")
        all_passed = False
    
    # Final result
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED - System ready for operation")
    else:
        print("✗ SOME TESTS FAILED - Check errors above")
    print("="*60)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting UAV RL System in {args.mode} mode")
    logger.info(f"Arguments: {vars(args)}")
    
    try:
        # Create components
        components = create_components(args)
        
        # Execute based on mode
        if args.mode == 'train':
            mode_train(args, components)
        elif args.mode == 'evaluate':
            mode_evaluate(args, components)
        elif args.mode == 'test_expert':
            mode_test_expert(args, components)
        elif args.mode == 'demo':
            mode_demo(args, components)
        elif args.mode == 'validate_system':
            mode_validate_system(args, components)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        logger.info(f"{args.mode} mode completed successfully")
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        raise


if __name__ == "__main__":
    main()