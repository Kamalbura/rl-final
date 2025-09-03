"""
Expert Policy Implementation for UAV Cybersecurity RL System.

This module implements the expert policy based on the lookup table for UAV algorithm scheduling.
The policy provides safe actions based on battery level, temperature, and threat state.
"""

import json
import os
from typing import Dict, Tuple, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ExpertPolicy:
    """
    Expert policy implementation based on lookup table for UAV algorithm scheduling.
    
    This class implements the decision logic for selecting cybersecurity algorithms
    (No_DDoS, XGBoost, TST) based on the current system state including battery level,
    temperature zone, and threat detection status.
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize expert policy with lookup table.
        
        Args:
            config_path: Path to expert policy configuration file
        """
        self.config_path = config_path or "config/expert_policy.json"
        
        # State space definitions
        self.battery_levels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
        self.temperatures = ["Safe", "Warning", "Critical"]
        self.threat_states = ["Normal", "Confirming", "Confirmed"]
        self.action_labels = ["No_DDoS", "XGBoost", "TST"]
        
        # Load lookup table
        self.lookup_table = {}
        self._load_lookup_table()
        
        # Safety rules for constraint checking
        self.safety_rules = {
            'critical_temp_threshold': 75.0,     # Temperature threshold for critical zone
            'warning_temp_threshold': 65.0,      # Temperature threshold for warning zone
            'critical_battery': 20.0,            # Critical battery threshold
            'low_battery': 40.0,                 # Low battery threshold
            'tst_temp_limit': 70.0,              # Temperature limit for TST
            'tst_battery_limit': 30.0,           # Battery limit for TST
            'recovery_time': 240.0               # Recovery time after TST
        }
        
        logger.info(f"Expert policy initialized with {len(self.lookup_table)} lookup entries")
    
    def _load_lookup_table(self):
        """Load lookup table from configuration file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.lookup_table = config.get('lookup_table', {})
                    logger.info(f"Loaded lookup table from {self.config_path}")
            else:
                # Create default lookup table if file doesn't exist
                self._create_default_lookup_table()
                self._save_lookup_table()
                logger.info("Created default lookup table")
        except Exception as e:
            logger.error(f"Error loading lookup table: {e}")
            self._create_default_lookup_table()
    
    def _create_default_lookup_table(self):
        """Create the default lookup table based on specifications."""
        self.lookup_table = {
            # Battery 0-20%: Always No_DDoS (safety constraint)
            "0-20%|Safe|Normal": 0,
            "0-20%|Safe|Confirming": 0,
            "0-20%|Safe|Confirmed": 0,
            "0-20%|Warning|Normal": 0,
            "0-20%|Warning|Confirming": 0,
            "0-20%|Warning|Confirmed": 0,
            "0-20%|Critical|Normal": 0,
            "0-20%|Critical|Confirming": 0,
            "0-20%|Critical|Confirmed": 0,
            
            # Battery 21-40%: Conservative approach
            "21-40%|Safe|Normal": 1,
            "21-40%|Safe|Confirming": 1,
            "21-40%|Safe|Confirmed": 1,
            "21-40%|Warning|Normal": 1,
            "21-40%|Warning|Confirming": 2,  # TST only if warning + confirming
            "21-40%|Warning|Confirmed": 1,
            "21-40%|Critical|Normal": 0,
            "21-40%|Critical|Confirming": 0,
            "21-40%|Critical|Confirmed": 0,
            
            # Battery 41-60%: More aggressive when safe
            "41-60%|Safe|Normal": 1,
            "41-60%|Safe|Confirming": 2,
            "41-60%|Safe|Confirmed": 1,
            "41-60%|Warning|Normal": 1,
            "41-60%|Warning|Confirming": 2,
            "41-60%|Warning|Confirmed": 1,
            "41-60%|Critical|Normal": 0,
            "41-60%|Critical|Confirming": 0,
            "41-60%|Critical|Confirmed": 0,
            
            # Battery 61-80%: Optimal performance zone
            "61-80%|Safe|Normal": 1,
            "61-80%|Safe|Confirming": 2,
            "61-80%|Safe|Confirmed": 1,
            "61-80%|Warning|Normal": 1,
            "61-80%|Warning|Confirming": 2,
            "61-80%|Warning|Confirmed": 1,
            "61-80%|Critical|Normal": 0,
            "61-80%|Critical|Confirming": 0,
            "61-80%|Critical|Confirmed": 0,
            
            # Battery 81-100%: High performance zone
            "81-100%|Safe|Normal": 1,
            "81-100%|Safe|Confirming": 2,
            "81-100%|Safe|Confirmed": 1,
            "81-100%|Warning|Normal": 1,
            "81-100%|Warning|Confirming": 2,
            "81-100%|Warning|Confirmed": 1,
            "81-100%|Critical|Normal": 0,
            "81-100%|Critical|Confirming": 0,
            "81-100%|Critical|Confirmed": 0,
        }
    
    def _save_lookup_table(self):
        """Save lookup table to configuration file."""
        try:
            config = {
                "lookup_table": self.lookup_table,
                "state_space": {
                    "battery_levels": self.battery_levels,
                    "temperatures": self.temperatures,
                    "threat_states": self.threat_states
                },
                "action_labels": self.action_labels,
                "timestamp": datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Saved lookup table to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving lookup table: {e}")
    
    def _discretize_battery(self, battery_level: float) -> str:
        """Convert continuous battery level to discrete category."""
        if battery_level <= 20:
            return "0-20%"
        elif battery_level <= 40:
            return "21-40%"
        elif battery_level <= 60:
            return "41-60%"
        elif battery_level <= 80:
            return "61-80%"
        else:
            return "81-100%"
    
    def _discretize_temperature(self, temperature: float) -> str:
        """Convert continuous temperature to discrete category."""
        if temperature <= self.safety_rules['warning_temp_threshold']:
            return "Safe"
        elif temperature <= self.safety_rules['critical_temp_threshold']:
            return "Warning"
        else:
            return "Critical"
    
    def _discretize_threat(self, threat_level: int) -> str:
        """Convert threat level to discrete category."""
        threat_map = {0: "Normal", 1: "Confirming", 2: "Confirmed"}
        return threat_map.get(threat_level, "Normal")
    
    def _create_state_key(self, battery: str, temperature: str, threat: str) -> str:
        """Create lookup table key from discrete state components."""
        return f"{battery}|{temperature}|{threat}"
    
    def get_action(self, state: Dict[str, float]) -> int:
        """
        Get expert-recommended action based on state.
        
        Args:
            state: Dictionary with temperature, battery, threat, etc.
            
        Returns:
            Recommended action (0=No_DDoS, 1=XGBoost, 2=TST)
        """
        # Extract and discretize state components
        battery_level = state.get('battery', 80.0)
        temperature = state.get('temperature', 50.0)
        threat_level = int(state.get('threat', 0))
        
        # Discretize continuous values
        battery_discrete = self._discretize_battery(battery_level)
        temp_discrete = self._discretize_temperature(temperature)
        threat_discrete = self._discretize_threat(threat_level)
        
        # Create lookup key
        state_key = self._create_state_key(battery_discrete, temp_discrete, threat_discrete)
        
        # Get action from lookup table
        action = self.lookup_table.get(state_key, 0)  # Default to No_DDoS
        
        # Apply additional safety checks
        if not self.is_safe_action(state, action)[0]:
            return 0  # Force No_DDoS if unsafe
        
        return action
    
    def is_safe_action(self, state: Dict[str, float], action: int) -> Tuple[bool, str]:
        """
        Check if an action is safe given the current state.
        
        Args:
            state: Dictionary with temperature, battery, threat, etc.
            action: Action to check (0=No_DDoS, 1=XGBoost, 2=TST)
            
        Returns:
            (safe, reason): Tuple with bool indicating safety and reason string
        """
        # Extract state components
        temperature = state.get('temperature', 50.0)
        battery = state.get('battery', 80.0)
        time_since_tst = state.get('time_since_tst', 999.0)
        
        # Hard safety constraints - critical temperature
        if temperature >= self.safety_rules['critical_temp_threshold']:
            if action != 0:  # Only No_DDoS allowed
                return False, f"Temperature {temperature:.1f}°C in critical zone"
        
        # Hard safety constraints - critical battery
        if battery <= self.safety_rules['critical_battery']:
            if action != 0:  # Only No_DDoS allowed
                return False, f"Battery {battery:.1f}% below critical threshold"
        
        # TST-specific constraints
        if action == 2:  # TST
            if temperature >= self.safety_rules['tst_temp_limit']:
                return False, f"Temperature {temperature:.1f}°C too high for TST"
            
            if battery <= self.safety_rules['tst_battery_limit']:
                return False, f"Battery {battery:.1f}% too low for TST"
            
            if time_since_tst < self.safety_rules['recovery_time']:
                return False, f"Still in TST recovery period ({time_since_tst:.1f}s < {self.safety_rules['recovery_time']}s)"
        
        # All checks passed
        return True, "Action is safe"
    
    def get_state_representation(self, state: Dict[str, float]) -> str:
        """
        Get human-readable state representation.
        
        Args:
            state: Dictionary with state values
            
        Returns:
            String representation of discrete state
        """
        battery_level = state.get('battery', 80.0)
        temperature = state.get('temperature', 50.0)
        threat_level = int(state.get('threat', 0))
        
        battery_discrete = self._discretize_battery(battery_level)
        temp_discrete = self._discretize_temperature(temperature)
        threat_discrete = self._discretize_threat(threat_level)
        
        return self._create_state_key(battery_discrete, temp_discrete, threat_discrete)
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable action name."""
        if 0 <= action < len(self.action_labels):
            return self.action_labels[action]
        return "Unknown"
    
    def get_lookup_table_stats(self) -> Dict[str, Any]:
        """Get statistics about the lookup table."""
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in self.lookup_table.values():
            action_counts[action] += 1
        
        total_entries = len(self.lookup_table)
        
        return {
            'total_entries': total_entries,
            'action_distribution': {
                'No_DDoS': action_counts[0],
                'XGBoost': action_counts[1],
                'TST': action_counts[2]
            },
            'action_percentages': {
                'No_DDoS': action_counts[0] / total_entries * 100,
                'XGBoost': action_counts[1] / total_entries * 100,
                'TST': action_counts[2] / total_entries * 100
            }
        }
    
    def validate_lookup_table(self) -> bool:
        """Validate that lookup table covers all state combinations."""
        expected_entries = len(self.battery_levels) * len(self.temperatures) * len(self.threat_states)
        actual_entries = len(self.lookup_table)
        
        if actual_entries != expected_entries:
            logger.warning(f"Lookup table incomplete: {actual_entries}/{expected_entries} entries")
            return False
        
        # Check that all entries have valid actions
        for key, action in self.lookup_table.items():
            if action not in [0, 1, 2]:
                logger.warning(f"Invalid action {action} for state {key}")
                return False
        
        logger.info("Lookup table validation passed")
        return True


if __name__ == "__main__":
    # Test the expert policy
    logging.basicConfig(level=logging.INFO)
    
    # Create expert policy
    expert = ExpertPolicy()
    
    # Validate lookup table
    expert.validate_lookup_table()
    
    # Print statistics
    stats = expert.get_lookup_table_stats()
    print("Lookup Table Statistics:")
    print(f"Total entries: {stats['total_entries']}")
    print("Action distribution:")
    for action, count in stats['action_distribution'].items():
        percentage = stats['action_percentages'][action]
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    # Test some states
    test_states = [
        {'battery': 90, 'temperature': 55, 'threat': 0},
        {'battery': 30, 'temperature': 70, 'threat': 1},
        {'battery': 15, 'temperature': 80, 'threat': 2},
        {'battery': 60, 'temperature': 45, 'threat': 1},
    ]
    
    print("\nTest Cases:")
    for i, state in enumerate(test_states):
        action = expert.get_action(state)
        action_name = expert.get_action_name(action)
        state_repr = expert.get_state_representation(state)
        safe, reason = expert.is_safe_action(state, action)
        
        print(f"Test {i+1}: {state_repr}")
        print(f"  Action: {action} ({action_name})")
        print(f"  Safe: {safe} - {reason}")
        print()