"""
Thermal Simulator for UAV Cybersecurity RL System.

This module simulates the thermal and power dynamics of a UAV running
cybersecurity algorithms with realistic heating, cooling, and power consumption.
"""

import json
import os
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SimulationState:
    """Represents the current state of the UAV simulation."""
    temperature: float
    battery: float
    cpu_usage: float
    power_consumption: float
    time_since_tst: float
    current_algorithm: int
    algorithm_start_time: float
    total_time: float


class TunableUAVSimulator:
    """
    Simulates UAV thermal and power dynamics for cybersecurity algorithm evaluation.
    
    This simulator models:
    - Thermal dynamics with heating and cooling
    - Power consumption for different algorithms
    - Battery discharge
    - CPU usage patterns
    - Safety constraints and failure modes
    """
    
    def __init__(self, config_path: str = None, params: Dict = None):
        """
        Initialize the UAV simulator.
        
        Args:
            config_path: Path to configuration file
            params: Direct parameter dictionary (overrides config_path)
        """
        self.config_path = config_path or "config/simulator_params.json"
        
        # Load configuration
        if params:
            self.params = params
        else:
            self._load_config()
        
        # Extract key parameters
        self.thermal_params = self.params['thermal_model']
        self.power_params = self.params['power_model']
        self.timing_params = self.params['timing']
        self.safety_params = self.params['safety_constraints']
        
        # Algorithm mappings
        self.algorithm_names = ["No_DDoS", "XGBoost", "TST"]
        self.algorithm_heat = self.thermal_params['algorithm_heat_generation']
        self.algorithm_power = self.power_params['algorithm_power']
        
        # Simulation state
        self.reset()
        
        logger.info("UAV simulator initialized")
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.params = json.load(f)
                logger.info(f"Loaded simulator config from {self.config_path}")
            else:
                self._create_default_config()
                self._save_config()
                logger.info("Created default simulator configuration")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration parameters."""
        self.params = {
            "thermal_model": {
                "ambient_temperature": 25.0,
                "thermal_mass": 50.0,
                "cooling_coefficient": 0.02,
                "algorithm_heat_generation": {
                    "No_DDoS": 0.5,
                    "XGBoost": 2.0,
                    "TST": 4.0
                },
                "temperature_thresholds": {
                    "safe_max": 65.0,
                    "warning_max": 75.0,
                    "critical_max": 85.0
                }
            },
            "power_model": {
                "base_power": 3.0,
                "algorithm_power": {
                    "No_DDoS": 1.0,
                    "XGBoost": 3.5,
                    "TST": 6.0
                },
                "battery_capacity": 100.0,
                "discharge_rate": 0.1
            },
            "timing": {
                "simulation_timestep": 1.0,
                "tst_duration": 30.0,
                "tst_cooldown": 240.0
            },
            "safety_constraints": {
                "max_temperature": 85.0,
                "min_battery": 5.0,
                "emergency_shutdown_temp": 90.0
            }
        }
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.params, f, indent=2)
            logger.info(f"Saved simulator config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def reset(self, 
             initial_temp: Optional[float] = None,
             initial_battery: Optional[float] = None) -> Dict:
        """
        Reset the simulator to initial conditions.
        
        Args:
            initial_temp: Starting temperature (random if None)
            initial_battery: Starting battery level (random if None)
            
        Returns:
            Initial state dictionary
        """
        # Set initial conditions
        self.current_temp = initial_temp if initial_temp is not None else np.random.uniform(40, 60)
        self.current_battery = initial_battery if initial_battery is not None else np.random.uniform(70, 95)
        self.current_cpu = 20.0
        self.current_power = self.power_params['base_power']
        self.time_since_tst = 999.0  # Large value indicating no recent TST
        self.current_algorithm = 0   # Start with No_DDoS
        self.algorithm_start_time = 0.0
        self.total_time = 0.0
        
        # Algorithm execution tracking
        self.tst_active = False
        self.tst_remaining_time = 0.0
        
        # History tracking
        self.temperature_history = [self.current_temp]
        self.battery_history = [self.current_battery]
        self.power_history = [self.current_power]
        self.algorithm_history = [self.current_algorithm]
        
        logger.debug(f"Simulator reset: T={self.current_temp:.1f}°C, B={self.current_battery:.1f}%")
        
        return self.get_current_state()
    
    def simulate_step(self, action: int, dt: float = None) -> Dict:
        """
        Simulate one time step with the given action.
        
        Args:
            action: Algorithm to run (0=No_DDoS, 1=XGBoost, 2=TST)
            dt: Time step size (uses default if None)
            
        Returns:
            State dictionary after simulation step
        """
        if dt is None:
            dt = self.timing_params['simulation_timestep']
        
        # Validate action
        if action not in [0, 1, 2]:
            logger.warning(f"Invalid action {action}, defaulting to No_DDoS")
            action = 0
        
        # Handle TST special case (TST runs for fixed duration)
        if action == 2 and not self.tst_active:
            # Starting TST
            self.tst_active = True
            self.tst_remaining_time = self.timing_params['tst_duration']
            self.time_since_tst = 0.0
        elif self.tst_active:
            # TST is running, continue regardless of action
            action = 2
            self.tst_remaining_time -= dt
            if self.tst_remaining_time <= 0:
                # TST finished
                self.tst_active = False
                action = 0  # Switch to No_DDoS
        
        # Update algorithm state
        if action != self.current_algorithm:
            self.algorithm_start_time = self.total_time
            self.current_algorithm = action
        
        # Update time tracking
        self.total_time += dt
        if not self.tst_active:
            self.time_since_tst += dt
        
        # Simulate thermal dynamics
        self._update_temperature(action, dt)
        
        # Simulate power consumption
        self._update_power_and_battery(action, dt)
        
        # Update CPU usage (simplified model)
        self._update_cpu_usage(action, dt)
        
        # Record history
        self.temperature_history.append(self.current_temp)
        self.battery_history.append(self.current_battery)
        self.power_history.append(self.current_power)
        self.algorithm_history.append(action)
        
        # Check for safety violations
        safety_violation = self._check_safety_constraints()
        
        return self.get_current_state(safety_violation)
    
    def _update_temperature(self, action: int, dt: float):
        """Update temperature based on thermal model."""
        algorithm_name = self.algorithm_names[action]
        heat_generation = self.algorithm_heat[algorithm_name]
        
        # Environmental cooling (Newton's law of cooling)
        ambient_temp = self.thermal_params['ambient_temperature']
        cooling_rate = self.thermal_params['cooling_coefficient']
        thermal_mass = self.thermal_params['thermal_mass']
        
        # Heat generation from algorithm
        heat_input = heat_generation
        
        # Temperature change equation: dT/dt = (heat_input - cooling_rate * (T - T_ambient)) / thermal_mass
        temp_diff = self.current_temp - ambient_temp
        dT_dt = (heat_input - cooling_rate * temp_diff) / thermal_mass
        
        # Update temperature
        self.current_temp += dT_dt * dt
        
        # Ensure temperature doesn't go below ambient
        self.current_temp = max(self.current_temp, ambient_temp)
    
    def _update_power_and_battery(self, action: int, dt: float):
        """Update power consumption and battery level."""
        algorithm_name = self.algorithm_names[action]
        
        # Calculate power consumption
        base_power = self.power_params['base_power']
        algorithm_power = self.algorithm_power[algorithm_name]
        
        # Total power = base + algorithm + temperature penalty
        temp_penalty = max(0, (self.current_temp - 60) * 0.1)  # Extra power for cooling
        self.current_power = base_power + algorithm_power + temp_penalty
        
        # Update battery (simplified linear discharge)
        discharge_rate = self.power_params['discharge_rate']
        battery_drain = (self.current_power * discharge_rate * dt) / 60  # Convert to per-minute
        self.current_battery = max(0, self.current_battery - battery_drain)
    
    def _update_cpu_usage(self, action: int, dt: float):
        """Update CPU usage based on algorithm."""
        # Target CPU usage for each algorithm
        cpu_targets = {0: 15, 1: 70, 2: 95}  # No_DDoS, XGBoost, TST
        target_cpu = cpu_targets[action]
        
        # Smooth transition to target
        cpu_change_rate = 10.0  # CPU% per second
        if self.current_cpu < target_cpu:
            self.current_cpu = min(target_cpu, self.current_cpu + cpu_change_rate * dt)
        elif self.current_cpu > target_cpu:
            self.current_cpu = max(target_cpu, self.current_cpu - cpu_change_rate * dt)
        
        # Add some noise
        self.current_cpu += np.random.normal(0, 2) * dt
        self.current_cpu = np.clip(self.current_cpu, 0, 100)
    
    def _check_safety_constraints(self) -> bool:
        """Check if any safety constraints are violated."""
        violations = []
        
        # Temperature check
        if self.current_temp >= self.safety_params['emergency_shutdown_temp']:
            violations.append(f"Emergency temperature: {self.current_temp:.1f}°C")
        elif self.current_temp >= self.safety_params['max_temperature']:
            violations.append(f"Critical temperature: {self.current_temp:.1f}°C")
        
        # Battery check
        if self.current_battery <= self.safety_params['min_battery']:
            violations.append(f"Critical battery: {self.current_battery:.1f}%")
        
        if violations:
            logger.warning(f"Safety violations: {', '.join(violations)}")
            return True
        
        return False
    
    def get_current_state(self, safety_violation: bool = False) -> Dict:
        """Get current simulation state as dictionary."""
        return {
            'temperature': self.current_temp,
            'battery': self.current_battery,
            'cpu_usage': self.current_cpu,
            'power_consumption': self.current_power,
            'time_since_tst': self.time_since_tst,
            'current_algorithm': self.current_algorithm,
            'total_time': self.total_time,
            'tst_active': self.tst_active,
            'tst_remaining': self.tst_remaining_time if self.tst_active else 0,
            'safety_violation': safety_violation
        }
    
    def get_thermal_zone(self, temperature: float = None) -> str:
        """Get thermal zone classification."""
        temp = temperature if temperature is not None else self.current_temp
        thresholds = self.thermal_params['temperature_thresholds']
        
        if temp <= thresholds['safe_max']:
            return "Safe"
        elif temp <= thresholds['warning_max']:
            return "Warning"
        else:
            return "Critical"
    
    def get_battery_zone(self, battery: float = None) -> str:
        """Get battery level classification."""
        batt = battery if battery is not None else self.current_battery
        
        if batt <= 20:
            return "0-20%"
        elif batt <= 40:
            return "21-40%"
        elif batt <= 60:
            return "41-60%"
        elif batt <= 80:
            return "61-80%"
        else:
            return "81-100%"
    
    def predict_temperature(self, action: int, time_horizon: float = 60.0) -> float:
        """
        Predict temperature after running an algorithm for given time.
        
        Args:
            action: Algorithm to simulate
            time_horizon: Time to simulate (seconds)
            
        Returns:
            Predicted temperature
        """
        # Simple prediction model
        algorithm_name = self.algorithm_names[action]
        heat_generation = self.algorithm_heat[algorithm_name]
        ambient_temp = self.thermal_params['ambient_temperature']
        cooling_rate = self.thermal_params['cooling_coefficient']
        thermal_mass = self.thermal_params['thermal_mass']
        
        # Steady-state temperature for continuous operation
        steady_state_temp = ambient_temp + (heat_generation / cooling_rate)
        
        # Exponential approach to steady state
        tau = thermal_mass / cooling_rate  # Time constant
        temp_change = (steady_state_temp - self.current_temp) * (1 - np.exp(-time_horizon / tau))
        
        return self.current_temp + temp_change
    
    def get_simulation_summary(self) -> Dict:
        """Get summary statistics of the simulation."""
        if not self.temperature_history:
            return {}
        
        return {
            'duration': self.total_time,
            'temperature': {
                'min': min(self.temperature_history),
                'max': max(self.temperature_history),
                'mean': np.mean(self.temperature_history),
                'final': self.temperature_history[-1]
            },
            'battery': {
                'initial': self.battery_history[0],
                'final': self.battery_history[-1],
                'consumed': self.battery_history[0] - self.battery_history[-1]
            },
            'power': {
                'min': min(self.power_history),
                'max': max(self.power_history),
                'mean': np.mean(self.power_history)
            },
            'algorithms': {
                'distribution': {
                    'No_DDoS': self.algorithm_history.count(0),
                    'XGBoost': self.algorithm_history.count(1),
                    'TST': self.algorithm_history.count(2)
                }
            }
        }
    
    def set_parameters(self, param_dict: Dict):
        """Update simulator parameters dynamically."""
        # Update nested dictionary
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        update_nested_dict(self.params, param_dict)
        
        # Reload derived parameters
        self.thermal_params = self.params['thermal_model']
        self.power_params = self.params['power_model']
        self.timing_params = self.params['timing']
        self.safety_params = self.params['safety_constraints']
        
        self.algorithm_heat = self.thermal_params['algorithm_heat_generation']
        self.algorithm_power = self.power_params['algorithm_power']
        
        logger.info("Simulator parameters updated")


if __name__ == "__main__":
    # Test the simulator
    logging.basicConfig(level=logging.INFO)
    
    # Create simulator
    simulator = TunableUAVSimulator()
    
    # Run a test simulation
    print("Running test simulation...")
    state = simulator.reset(initial_temp=45, initial_battery=80)
    print(f"Initial state: T={state['temperature']:.1f}°C, B={state['battery']:.1f}%")
    
    # Simulate different algorithms
    actions = [1, 1, 1, 2, 2, 2, 0, 0, 0, 1]  # XGBoost, TST, No_DDoS, XGBoost
    
    for i, action in enumerate(actions):
        state = simulator.simulate_step(action, dt=30.0)  # 30-second steps
        algorithm_name = simulator.algorithm_names[action]
        thermal_zone = simulator.get_thermal_zone()
        battery_zone = simulator.get_battery_zone()
        
        print(f"Step {i+1}: {algorithm_name} -> "
              f"T={state['temperature']:.1f}°C ({thermal_zone}), "
              f"B={state['battery']:.1f}% ({battery_zone}), "
              f"P={state['power_consumption']:.1f}W, "
              f"CPU={state['cpu_usage']:.1f}%")
        
        if state.get('safety_violation', False):
            print("  *** SAFETY VIOLATION ***")
    
    # Get simulation summary
    summary = simulator.get_simulation_summary()
    print(f"\nSimulation Summary:")
    print(f"Duration: {summary['duration']:.0f}s")
    print(f"Temperature: {summary['temperature']['min']:.1f} - {summary['temperature']['max']:.1f}°C")
    print(f"Battery consumed: {summary['battery']['consumed']:.1f}%")
    print(f"Average power: {summary['power']['mean']:.1f}W")
    print(f"Algorithm usage: {summary['algorithms']['distribution']}")
    
    # Test temperature prediction
    print(f"\nTemperature predictions from current state ({state['temperature']:.1f}°C):")
    for action in [0, 1, 2]:
        predicted_temp = simulator.predict_temperature(action, 120.0)  # 2 minutes
        algorithm_name = simulator.algorithm_names[action]
        print(f"  {algorithm_name}: {predicted_temp:.1f}°C after 2 minutes")