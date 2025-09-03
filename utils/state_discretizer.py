"""
State Discretizer for UAV RL System.

This module provides tools for discretizing continuous state spaces into discrete bins
suitable for tabular Q-learning while maintaining meaningful state representations.
"""

from typing import Dict, Tuple, List
import numpy as np
import itertools
import logging

logger = logging.getLogger(__name__)


class StateDiscretizer:
    """
    Discretizes continuous state space for tabular RL.
    
    This class handles the conversion between continuous state values and discrete
    state representations suitable for lookup table-based reinforcement learning.
    """
    
    def __init__(self, bins=None):
        """
        Initialize discretizer with bin definitions.
        
        Args:
            bins: Dictionary defining discretization bins for each state variable
        """
        # Default bin definitions based on UAV cybersecurity domain
        self.bins = bins or {
            'temperature': [0, 55, 65, 70, 75, 85, 100],      # 6 thermal zones
            'battery': [0, 20, 40, 60, 80, 100],              # 5 battery levels
            'threat': [0, 1, 2, 3],                           # 3 threat levels (discrete)
            'cpu_usage': [0, 30, 60, 90, 100],                # 4 CPU zones
            'time_since_tst': [0, 60, 240, 999],              # 3 recovery zones
            'power': [0, 2, 4, 6, 8, 12]                      # 5 power zones
        }
        
        # Calculate state space size
        self.state_space_size = 1
        for var, bins_list in self.bins.items():
            if var == 'threat':  # Threat is already discrete
                num_bins = len(bins_list) - 1
            else:
                num_bins = len(bins_list) - 1
            self.state_space_size *= num_bins
        
        # Create mappings for expert policy compatibility
        self.battery_level_map = {
            0: "0-20%",
            1: "21-40%", 
            2: "41-60%",
            3: "61-80%",
            4: "81-100%"
        }
        
        self.temperature_map = {
            0: "Safe",      # < 55°C
            1: "Safe",      # 55-65°C
            2: "Safe",      # 65-70°C 
            3: "Warning",   # 70-75°C
            4: "Warning",   # 75-85°C
            5: "Critical"   # > 85°C
        }
        
        self.threat_map = {
            0: "Normal",
            1: "Confirming", 
            2: "Confirmed"
        }
        
        logger.info(f"State discretizer initialized with {self.state_space_size} discrete states")
    
    def discretize(self, continuous_state: Dict) -> Tuple:
        """
        Discretize continuous state into tuple for Q-table lookup.
        
        Args:
            continuous_state: Dictionary with continuous values
            
        Returns:
            Tuple representing discrete state
        """
        discrete = []
        
        # Temperature zone
        temp = continuous_state.get('temperature', 50.0)
        temp_bin = np.digitize(temp, self.bins['temperature']) - 1
        temp_bin = max(0, min(temp_bin, len(self.bins['temperature']) - 2))
        discrete.append(temp_bin)
        
        # Battery level
        battery = continuous_state.get('battery', 80.0)
        battery_bin = np.digitize(battery, self.bins['battery']) - 1
        battery_bin = max(0, min(battery_bin, len(self.bins['battery']) - 2))
        discrete.append(battery_bin)
        
        # Threat level (already discrete)
        threat = int(continuous_state.get('threat', 0))
        threat = max(0, min(threat, len(self.bins['threat']) - 2))
        discrete.append(threat)
        
        # CPU usage
        cpu = continuous_state.get('cpu_usage', 30.0)
        cpu_bin = np.digitize(cpu, self.bins['cpu_usage']) - 1
        cpu_bin = max(0, min(cpu_bin, len(self.bins['cpu_usage']) - 2))
        discrete.append(cpu_bin)
        
        # Time since TST
        time_since_tst = continuous_state.get('time_since_tst', 999.0)
        tst_bin = np.digitize(time_since_tst, self.bins['time_since_tst']) - 1
        tst_bin = max(0, min(tst_bin, len(self.bins['time_since_tst']) - 2))
        discrete.append(tst_bin)
        
        # Power consumption
        power = continuous_state.get('power', 5.0)
        power_bin = np.digitize(power, self.bins['power']) - 1
        power_bin = max(0, min(power_bin, len(self.bins['power']) - 2))
        discrete.append(power_bin)
        
        return tuple(discrete)
    
    def tuple_to_continuous(self, state_tuple: Tuple) -> Dict:
        """
        Convert discrete state tuple to approximate continuous values.
        
        Args:
            state_tuple: Discrete state tuple
            
        Returns:
            Dictionary with continuous state approximation
        """
        if len(state_tuple) != 6:
            raise ValueError(f"Expected 6-element tuple, got {len(state_tuple)}")
        
        temp_idx, battery_idx, threat_idx, cpu_idx, tst_idx, power_idx = state_tuple
        
        # Calculate continuous values (use midpoints of bins)
        temp_bins = self.bins['temperature']
        battery_bins = self.bins['battery']
        cpu_bins = self.bins['cpu_usage']
        tst_bins = self.bins['time_since_tst']
        power_bins = self.bins['power']
        
        return {
            'temperature': self._get_bin_midpoint(temp_bins, temp_idx),
            'battery': self._get_bin_midpoint(battery_bins, battery_idx),
            'threat': threat_idx,  # Already discrete
            'cpu_usage': self._get_bin_midpoint(cpu_bins, cpu_idx),
            'time_since_tst': self._get_bin_midpoint(tst_bins, tst_idx),
            'power': self._get_bin_midpoint(power_bins, power_idx)
        }
    
    def _get_bin_midpoint(self, bins, idx):
        """Get midpoint of bin at index."""
        if idx >= len(bins) - 1:
            return bins[-1]
        if idx < 0:
            return bins[0]
        return (bins[idx] + bins[idx + 1]) / 2
    
    def get_all_possible_states(self) -> List[Tuple]:
        """Get all possible discrete states."""
        ranges = []
        for var in ['temperature', 'battery', 'threat', 'cpu_usage', 'time_since_tst', 'power']:
            if var == 'threat':
                ranges.append(range(len(self.bins[var]) - 1))
            else:
                ranges.append(range(len(self.bins[var]) - 1))
        
        return list(itertools.product(*ranges))
    
    def state_to_string(self, state_tuple: Tuple) -> str:
        """Convert state tuple to human-readable string."""
        if len(state_tuple) != 6:
            return f"Invalid state tuple: {state_tuple}"
        
        temp_idx, battery_idx, threat_idx, cpu_idx, tst_idx, power_idx = state_tuple
        
        # Temperature zones
        temp_zones = ['Very Cold', 'Cold', 'Cool', 'Warm', 'Hot', 'Critical']
        temp_zone = temp_zones[min(temp_idx, len(temp_zones)-1)]
        
        # Battery zones
        battery_zones = ['Critical', 'Low', 'Medium', 'High', 'Full']
        battery_zone = battery_zones[min(battery_idx, len(battery_zones)-1)]
        
        # Threat zones
        threat_zones = ['Normal', 'Confirming', 'Confirmed']
        threat_zone = threat_zones[min(threat_idx, len(threat_zones)-1)]
        
        # CPU zones
        cpu_zones = ['Idle', 'Low', 'Medium', 'High']
        cpu_zone = cpu_zones[min(cpu_idx, len(cpu_zones)-1)]
        
        # TST recovery zones
        tst_zones = ['Recent', 'Cooling', 'Ready']
        tst_zone = tst_zones[min(tst_idx, len(tst_zones)-1)]
        
        # Power zones
        power_zones = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        power_zone = power_zones[min(power_idx, len(power_zones)-1)]
        
        return (f"T:{temp_zone} B:{battery_zone} Th:{threat_zone} "
                f"CPU:{cpu_zone} TST:{tst_zone} P:{power_zone}")
    
    def get_expert_policy_state(self, state_tuple: Tuple) -> str:
        """
        Convert discrete state tuple to expert policy format.
        
        Args:
            state_tuple: Discrete state tuple
            
        Returns:
            String in format "battery|temperature|threat" for expert policy lookup
        """
        if len(state_tuple) != 6:
            raise ValueError(f"Expected 6-element tuple, got {len(state_tuple)}")
        
        temp_idx, battery_idx, threat_idx, _, _, _ = state_tuple
        
        # Map indices to expert policy categories
        battery_level = self.battery_level_map.get(battery_idx, "41-60%")
        
        # Temperature mapping needs to consider the finer bins
        if temp_idx <= 2:  # 0, 1, 2 -> Safe
            temperature = "Safe"
        elif temp_idx <= 4:  # 3, 4 -> Warning  
            temperature = "Warning"
        else:  # 5+ -> Critical
            temperature = "Critical"
        
        threat_state = self.threat_map.get(threat_idx, "Normal")
        
        return f"{battery_level}|{temperature}|{threat_state}"
    
    def get_state_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get the bounds for each continuous state variable."""
        bounds = {}
        for var, bins_list in self.bins.items():
            if var != 'threat':  # Threat is discrete
                bounds[var] = (bins_list[0], bins_list[-1])
            else:
                bounds[var] = (0, len(bins_list) - 2)
        return bounds
    
    def validate_state(self, state: Dict) -> bool:
        """Validate that a state dictionary contains all required variables."""
        required_vars = ['temperature', 'battery', 'threat', 'cpu_usage', 'time_since_tst', 'power']
        
        for var in required_vars:
            if var not in state:
                logger.warning(f"Missing required state variable: {var}")
                return False
        
        # Check bounds
        bounds = self.get_state_bounds()
        for var, value in state.items():
            if var in bounds:
                min_val, max_val = bounds[var]
                if not (min_val <= value <= max_val):
                    logger.warning(f"State variable {var}={value} out of bounds [{min_val}, {max_val}]")
                    return False
        
        return True
    
    def get_discretization_info(self) -> Dict:
        """Get information about the discretization scheme."""
        info = {
            'total_states': self.state_space_size,
            'variables': {},
        }
        
        for var, bins_list in self.bins.items():
            if var == 'threat':
                num_bins = len(bins_list) - 1
                info['variables'][var] = {
                    'type': 'discrete',
                    'bins': bins_list[:-1],  # Remove the extra element
                    'num_bins': num_bins
                }
            else:
                num_bins = len(bins_list) - 1
                info['variables'][var] = {
                    'type': 'continuous',
                    'bins': bins_list,
                    'num_bins': num_bins,
                    'ranges': [(bins_list[i], bins_list[i+1]) for i in range(num_bins)]
                }
        
        return info


if __name__ == "__main__":
    # Test the state discretizer
    logging.basicConfig(level=logging.INFO)
    
    # Create discretizer
    discretizer = StateDiscretizer()
    
    # Print discretization info
    info = discretizer.get_discretization_info()
    print(f"Total discrete states: {info['total_states']}")
    print("\nVariable discretization:")
    for var, var_info in info['variables'].items():
        print(f"  {var}: {var_info['num_bins']} bins ({var_info['type']})")
        if var_info['type'] == 'continuous':
            print(f"    Ranges: {var_info['ranges']}")
        else:
            print(f"    Values: {var_info['bins']}")
    
    # Test some states
    test_states = [
        {'temperature': 70, 'battery': 85, 'threat': 1, 'cpu_usage': 45, 'time_since_tst': 300, 'power': 5},
        {'temperature': 50, 'battery': 25, 'threat': 0, 'cpu_usage': 20, 'time_since_tst': 100, 'power': 3},
        {'temperature': 80, 'battery': 15, 'threat': 2, 'cpu_usage': 80, 'time_since_tst': 30, 'power': 8},
    ]
    
    print("\nTest Cases:")
    for i, state in enumerate(test_states):
        # Validate state
        valid = discretizer.validate_state(state)
        print(f"\nTest {i+1}: Valid={valid}")
        print(f"  Continuous: {state}")
        
        if valid:
            # Discretize
            discrete = discretizer.discretize(state)
            print(f"  Discrete: {discrete}")
            
            # Convert back
            continuous = discretizer.tuple_to_continuous(discrete)
            print(f"  Reconstructed: {continuous}")
            
            # Human readable
            readable = discretizer.state_to_string(discrete)
            print(f"  Human readable: {readable}")
            
            # Expert policy format
            expert_format = discretizer.get_expert_policy_state(discrete)
            print(f"  Expert format: {expert_format}")
    
    # Test all possible states
    all_states = discretizer.get_all_possible_states()
    print(f"\nGenerated {len(all_states)} possible discrete states")
    print("First 5 states:")
    for i, state in enumerate(all_states[:5]):
        readable = discretizer.state_to_string(state)
        expert_format = discretizer.get_expert_policy_state(state)
        print(f"  {i+1}: {state} -> {readable} -> {expert_format}")