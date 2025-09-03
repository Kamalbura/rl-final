"""
Generate educational visualizations for teaching reinforcement learning concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import seaborn as sns

# Create output directory
os.makedirs("teaching/visualizations", exist_ok=True)

# Professional color palette
colors = {
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

# Set up plotting style
plt.style.use('default')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 12,
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

def generate_rl_framework_diagram():
    """Generate a diagram illustrating the RL framework."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw agent and environment boxes
    agent_rect = plt.Rectangle((1, 3), 3, 2, fc=colors['primary'], ec='black', alpha=0.8)
    env_rect = plt.Rectangle((6, 3), 3, 2, fc=colors['secondary'], ec='black', alpha=0.8)
    ax.add_patch(agent_rect)
    ax.add_patch(env_rect)
    
    # Add text
    ax.text(2.5, 4, "Agent", color='white', ha='center', va='center', fontsize=14, weight='bold')
    ax.text(7.5, 4, "Environment", color='white', ha='center', va='center', fontsize=14, weight='bold')
    
    # Add arrows
    ax.arrow(4.1, 4, 1.8, 0, head_width=0.3, head_length=0.3, fc=colors['accent'], ec='black', width=0.1)
    ax.arrow(6, 3.5, -1.8, 0, head_width=0.3, head_length=0.3, fc=colors['warning'], ec='black', width=0.1)
    ax.arrow(6, 4.5, -1.8, 0, head_width=0.3, head_length=0.3, fc=colors['success'], ec='black', width=0.1)
    
    # Add text for arrows
    ax.text(5, 3.2, "Reward (rt)", color=colors['warning'], ha='center', va='center', fontsize=12)
    ax.text(5, 4.2, "State (st)", color=colors['success'], ha='center', va='center', fontsize=12)
    ax.text(5, 4.9, "Action (at)", color=colors['accent'], ha='center', va='center', fontsize=12)
    
    # Add title
    fig.suptitle('Reinforcement Learning Framework', fontsize=16)
    
    # Save figure
    plt.savefig('teaching/visualizations/rl_framework.png', bbox_inches='tight')
    plt.close()

def generate_q_learning_update_diagram():
    """Generate a diagram explaining the Q-learning update rule."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Remove axes
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Draw Q-update equation with parts highlighted
    eq = r"$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$"
    ax.text(6, 4.5, eq, fontsize=20, ha='center', va='center')
    
    # Draw boxes around different parts
    # Current Q-value
    current_q = plt.Rectangle((1.8, 4.3), 1.5, 0.6, fill=False, ec=colors['primary'], lw=2)
    ax.add_patch(current_q)
    ax.text(2.5, 3.7, "Current\nQ-value", color=colors['primary'], ha='center', va='center', fontsize=10)
    
    # Learning rate
    learning_rate = plt.Rectangle((5.3, 4.3), 0.3, 0.6, fill=False, ec=colors['secondary'], lw=2)
    ax.add_patch(learning_rate)
    ax.text(5.45, 3.7, "Learning\nRate", color=colors['secondary'], ha='center', va='center', fontsize=10)
    
    # Reward
    reward = plt.Rectangle((5.8, 4.3), 0.3, 0.6, fill=False, ec=colors['success'], lw=2)
    ax.add_patch(reward)
    ax.text(5.95, 3.7, "Reward", color=colors['success'], ha='center', va='center', fontsize=10)
    
    # Discount factor
    discount = plt.Rectangle((6.3, 4.3), 0.3, 0.6, fill=False, ec=colors['warning'], lw=2)
    ax.add_patch(discount)
    ax.text(6.45, 3.7, "Discount\nFactor", color=colors['warning'], ha='center', va='center', fontsize=10)
    
    # Future Q-value
    future_q = plt.Rectangle((6.8, 4.3), 2.3, 0.6, fill=False, ec=colors['accent'], lw=2)
    ax.add_patch(future_q)
    ax.text(7.95, 3.7, "Maximum Future Q-value", color=colors['accent'], ha='center', va='center', fontsize=10)
    
    # TD Error
    td_error = plt.Rectangle((5.8, 4.3), 3.8, 0.6, fill=False, ec=colors['danger'], lw=2)
    ax.add_patch(td_error)
    ax.text(7.7, 5.2, "TD Error", color=colors['danger'], ha='center', va='center', fontsize=10)
    
    # Add title
    fig.suptitle('Q-Learning Update Rule', fontsize=16)
    
    # Add explanation at bottom
    explanation = (
        "The Q-learning update rule adjusts our estimate of the value (Q-value) for a state-action pair.\n"
        "It moves our current estimate toward the 'target' value: immediate reward plus discounted future value."
    )
    ax.text(6, 2, explanation, ha='center', va='center', fontsize=12)
    
    # Save figure
    plt.savefig('teaching/visualizations/q_learning_update.png', bbox_inches='tight')
    plt.close()

def generate_exploration_exploitation_diagram():
    """Generate a diagram showing the exploration-exploitation tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define data for balance curve
    x = np.linspace(0, 100, 1000)
    y_explore = 100 * np.exp(-0.05 * x)
    y_exploit = 100 - y_explore
    
    # Plot exploration-exploitation curves
    ax.plot(x, y_explore, label='Exploration (ε)', color=colors['accent'], linewidth=3)
    ax.plot(x, y_exploit, label='Exploitation (1-ε)', color=colors['success'], linewidth=3)
    
    # Add shaded regions
    ax.fill_between(x, 0, y_explore, color=colors['accent'], alpha=0.3)
    ax.fill_between(x, 0, y_exploit, color=colors['success'], alpha=0.3)
    
    # Add annotations
    ax.annotate('Initial learning\nfocuses on\nexploration', xy=(10, 70), xytext=(20, 80),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax.annotate('Later learning\nfocuses on\nexploitation', xy=(80, 80), xytext=(60, 60),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    # Set labels and title
    ax.set_xlabel('Training Progress (%)')
    ax.set_ylabel('Strategy Focus (%)')
    ax.set_title('Exploration vs. Exploitation Balance Over Time', fontsize=14)
    
    # Add epsilon formula
    ax.text(75, 30, r'$\varepsilon = \varepsilon_{start} \times \varepsilon_{decay}^{episode}$', 
            fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add legend
    ax.legend(loc='center right')
    
    # Set grid and limits
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Save figure
    plt.savefig('teaching/visualizations/exploration_exploitation.png', bbox_inches='tight')
    plt.close()

def generate_expert_warmstart_comparison():
    """Generate a comparison between random initialization and expert warm-start."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Generate mock Q-table data
    state_labels = ['S1', 'S2', 'S3', 'S4', 'S5']
    action_labels = ['No_DDoS', 'XGBoost', 'TST']
    
    # Random initialization
    np.random.seed(42)
    random_q = np.random.uniform(0, 0.1, (5, 3))
    
    # Expert warm-start
    expert_q = np.zeros((5, 3))
    # Safe states (prefer XGBoost)
    expert_q[0, 1] = 10.0  # S1: XGBoost
    # Critical states (prefer No_DDoS)
    expert_q[1, 0] = 10.0  # S2: No_DDoS
    # Good states with threat (prefer TST)
    expert_q[2, 2] = 10.0  # S3: TST
    # Mixed states
    expert_q[3, 1] = 10.0  # S4: XGBoost
    expert_q[4, 0] = 10.0  # S5: No_DDoS
    # Add lower values for non-expert actions
    for i in range(5):
        for j in range(3):
            if expert_q[i, j] == 0:
                expert_q[i, j] = 2.0
    
    # Plot random initialization
    im1 = ax1.imshow(random_q, cmap='Blues')
    ax1.set_title('Random Initialization')
    ax1.set_xlabel('Action')
    ax1.set_ylabel('State')
    ax1.set_xticks(np.arange(len(action_labels)))
    ax1.set_yticks(np.arange(len(state_labels)))
    ax1.set_xticklabels(action_labels)
    ax1.set_yticklabels(state_labels)
    
    # Plot expert warm-start
    im2 = ax2.imshow(expert_q, cmap='Blues')
    ax2.set_title('Expert Warm-Start Initialization')
    ax2.set_xlabel('Action')
    ax2.set_xticks(np.arange(len(action_labels)))
    ax2.set_yticks(np.arange(len(state_labels)))
    ax2.set_xticklabels(action_labels)
    ax2.set_yticklabels(state_labels)
    
    # Add color bars
    fig.colorbar(im1, ax=ax1, label='Q-value')
    fig.colorbar(im2, ax=ax2, label='Q-value')
    
    # Annotate each cell with values
    for i in range(len(state_labels)):
        for j in range(len(action_labels)):
            ax1.text(j, i, f"{random_q[i, j]:.2f}", ha="center", va="center", color="black" if random_q[i, j] < 0.05 else "white")
            ax2.text(j, i, f"{expert_q[i, j]:.1f}", ha="center", va="center", color="black" if expert_q[i, j] < 5.0 else "white")
    
    # Add advantages text
    fig.suptitle('Comparing Initialization Strategies', fontsize=16)
    
    # Add explanation at bottom
    ax1.text(1.5, 5.8, "Random initialization provides no guidance.\nAll actions are equally likely at start.", 
             ha='center', va='center', fontsize=10)
    
    ax2.text(1.5, 5.8, "Expert warm-start encodes domain knowledge.\nPreferred actions start with higher values.", 
             ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('teaching/visualizations/expert_warmstart.png', bbox_inches='tight')
    plt.close()

def generate_safety_constraint_diagram():
    """Generate a diagram showing how safety constraints are implemented."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define state space (temperature and battery)
    temps = np.linspace(25, 90, 100)
    battery = np.linspace(0, 100, 100)
    T, B = np.meshgrid(temps, battery)
    
    # Define safety regions
    safety_value = np.ones_like(T)
    
    # Critical temperature regions (>85°C)
    safety_value[T > 85] = 0
    
    # Critical battery regions (<5%)
    safety_value[B < 5] = 0
    
    # Warning regions
    safety_value[(T > 80) & (T <= 85)] = 0.5
    safety_value[(B < 20) & (B >= 5)] = 0.5
    
    # TST restrictions (battery < 30% OR temp > 80°C)
    tst_allowed = np.ones_like(T)
    tst_allowed[(B < 30) | (T > 80)] = 0
    
    # Create a custom colormap
    colors_list = [(colors['danger'], colors['warning'], colors['success'])]
    safety_cmap = LinearSegmentedColormap.from_list("SafetyCmap", [(1, 0, 0), (1, 0.7, 0), (0, 0.7, 0)], N=3)
    
    # Plot the safety regions
    c = ax.pcolormesh(T, B, safety_value, cmap=safety_cmap, alpha=0.7)
    
    # Add TST restriction boundary
    ax.contour(T, B, tst_allowed, levels=[0.5], colors=[colors['gray_dark']], linestyles='dashed', linewidths=2)
    
    # Add labels and title
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Battery Level (%)')
    ax.set_title('Safety Constraints in the UAV State Space', fontsize=14)
    
    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_ticks([0.17, 0.5, 0.83])
    cbar.set_ticklabels(['Unsafe', 'Warning', 'Safe'])
    
    # Add annotations
    ax.annotate('Critical\nTemperature\nRegion', xy=(87, 50), xytext=(91, 50),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax.annotate('Critical\nBattery\nRegion', xy=(50, 3), xytext=(50, 10),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax.annotate('TST Restricted\nRegion', xy=(70, 25), xytext=(60, 15),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    # Add legend for TST restriction
    ax.plot([], [], color=colors['gray_dark'], linestyle='dashed', linewidth=2, label='TST Restriction Boundary')
    ax.legend(loc='upper left')
    
    # Add safety rules text
    rules_text = (
        "Safety Rules:\n"
        "1. Critical Temperature (>85°C): Only No_DDoS allowed\n"
        "2. Critical Battery (<5%): Only No_DDoS allowed\n"
        "3. TST Restrictions: Requires >30% battery AND <80°C"
    )
    
    # Add text box for rules
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(26, 3, rules_text, fontsize=10, verticalalignment='bottom', 
            horizontalalignment='left', bbox=props)
    
    # Save figure
    plt.savefig('teaching/visualizations/safety_constraints.png', bbox_inches='tight')
    plt.close()

def generate_learning_convergence_diagram():
    """Generate a diagram showing learning convergence over episodes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define episode data
    episodes = np.arange(1, 340)
    
    # Simulate expert agreement data
    np.random.seed(42)
    expert_agreement = 0.6 + 0.38 * (1 - np.exp(-0.01 * episodes))
    # Add some noise
    expert_agreement += np.random.normal(0, 0.05, size=len(episodes))
    # Ensure bounds
    expert_agreement = np.clip(expert_agreement, 0, 1)
    
    # Simulate rewards data
    rewards = 50 + 150 * (1 - np.exp(-0.008 * episodes))
    # Add some noise
    rewards += np.random.normal(0, 10, size=len(episodes))
    
    # Plot expert agreement
    ax1.plot(episodes, expert_agreement, color=colors['success'], linewidth=2)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Expert Agreement')
    ax1.set_title('Expert Agreement Convergence', fontsize=14)
    ax1.set_ylim(0, 1.05)
    
    # Add horizontal lines for convergence thresholds
    ax1.axhline(y=0.85, color=colors['warning'], linestyle='--', label='Target (85%)')
    ax1.axhline(y=0.95, color=colors['accent'], linestyle='--', label='Convergence (95%)')
    
    # Add annotations
    ax1.annotate('Initial Learning', xy=(50, 0.7), xytext=(20, 0.5),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax1.annotate('Rapid Improvement', xy=(150, 0.85), xytext=(100, 0.75),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax1.annotate('Convergence', xy=(300, 0.95), xytext=(250, 0.9),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax1.legend(loc='lower right')
    
    # Plot rewards
    ax2.plot(episodes, rewards, color=colors['secondary'], linewidth=2)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Reward Convergence', fontsize=14)
    
    # Add annotations
    ax2.annotate('Exploration Phase', xy=(50, 100), xytext=(20, 80),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax2.annotate('Policy Improvement', xy=(150, 160), xytext=(100, 130),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    ax2.annotate('Fine-tuning', xy=(300, 195), xytext=(250, 180),
                arrowprops=dict(facecolor=colors['primary'], shrink=0.05), fontsize=10)
    
    # Add early stopping marker
    ax2.axvline(x=339, color=colors['danger'], linestyle='-', linewidth=2)
    ax2.text(339, 80, 'Early\nStopping', rotation=90, color=colors['danger'], ha='right')
    
    fig.suptitle('Learning Convergence Over Episodes', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('teaching/visualizations/learning_convergence.png', bbox_inches='tight')
    plt.close()

def generate_state_space_visualization():
    """Generate a visualization of the UAV state space."""
    # Create figure with 3D projection
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define the state dimensions to show (Temperature, Battery, Threat)
    temps = [30, 45, 60, 75, 85]
    batteries = [10, 30, 50, 70, 90]
    threats = [0, 1, 2]
    threat_labels = ['Normal', 'Confirming', 'Confirmed']
    
    # Create the meshgrid
    T, B = np.meshgrid(temps, batteries)
    
    # Set up the plot
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Battery Level (%)')
    ax.set_zlabel('Threat Level')
    ax.set_title('UAV State Space Visualization', fontsize=14)
    
    # Set limits
    ax.set_xlim(25, 90)
    ax.set_ylim(0, 100)
    ax.set_zlim(-0.5, 2.5)
    
    # Set threat level ticks
    ax.set_zticks([0, 1, 2])
    ax.set_zticklabels(threat_labels)
    
    # Plot points for each threat level with different colors
    colors_by_threat = [colors['success'], colors['warning'], colors['danger']]
    
    # Action by state (simulated optimal policy)
    def get_action(temp, battery, threat):
        if battery <= 20 or temp >= 80:
            return 0  # No_DDoS (safety)
        elif threat == 2:
            return 1  # XGBoost (confirmed threat)
        elif threat == 1 and battery >= 40 and temp < 65:
            return 2  # TST (confirming threat, good conditions)
        else:
            return 1  # XGBoost (default balanced)
    
    # Plot each state point
    for threat in threats:
        xs, ys, zs = [], [], []
        colors_by_action = []
        
        for temp in temps:
            for battery in batteries:
                xs.append(temp)
                ys.append(battery)
                zs.append(threat)
                
                # Determine action color
                action = get_action(temp, battery, threat)
                if action == 0:  # No_DDoS
                    colors_by_action.append('blue')
                elif action == 1:  # XGBoost
                    colors_by_action.append('green')
                else:  # TST
                    colors_by_action.append('red')
        
        # Plot points for this threat level
        ax.scatter(xs, ys, zs, c=colors_by_action, s=100, alpha=0.7)
    
    # Add a legend
    no_ddos = mpatches.Patch(color='blue', label='No_DDoS')
    xgboost = mpatches.Patch(color='green', label='XGBoost')
    tst = mpatches.Patch(color='red', label='TST')
    ax.legend(handles=[no_ddos, xgboost, tst], title='Selected Action')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add text explaining state space
    fig.text(0.02, 0.02, (
        "State Space: 3 key dimensions shown (Temperature, Battery, Threat Level)\n"
        "Full state space also includes: CPU Usage, Time Since TST, Power Consumption\n"
        "Total theoretical states: 6 × 10 × 3 × 5 × 6 × 5 = 27,000 states\n"
        "Points are colored by the optimal action for each state"
    ), fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Adjust view angle
    ax.view_init(elev=20, azim=30)
    
    # Save figure
    plt.savefig('teaching/visualizations/state_space.png', bbox_inches='tight')
    plt.close()

def generate_algorithm_comparison_diagram():
    """Generate a diagram comparing the three UAV algorithms."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Algorithm names
    algorithms = ['No_DDoS', 'XGBoost', 'TST']
    
    # Metrics to compare
    metrics = ['Power\nConsumption', 'Heat\nGeneration', 'Threat\nDetection', 'Resource\nEfficiency', 'Usage\nFrequency']
    
    # Values for each algorithm on each metric (0-10 scale)
    data = np.array([
        [2, 1, 3, 10, 34.4],   # No_DDoS
        [6, 5, 8, 7, 65.5],    # XGBoost
        [10, 9, 10, 3, 0.1]    # TST
    ])
    
    # Normalize usage frequency to 0-10 scale
    data[0, 4] = data[0, 4] / 10
    data[1, 4] = data[1, 4] / 10
    data[2, 4] = data[2, 4] / 10 * 3  # Scale up TST slightly for visibility
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdYlGn_r')  # Reversed RdYlGn (red is high values)
    
    # Add labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(algorithms)))
    ax.set_xticklabels(metrics)
    ax.set_yticklabels(algorithms)
    
    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Value (higher is more)', rotation=270, labelpad=15)
    
    # Loop over data and add text annotations
    for i in range(len(algorithms)):
        for j in range(len(metrics)):
            if j < 4:  # Regular metrics
                text = ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", 
                              color="black" if data[i, j] < 7 else "white")
            else:  # Usage frequency
                original_val = [34.4, 65.5, 0.1][i]
                text = ax.text(j, i, f"{original_val:.1f}%", ha="center", va="center", 
                              color="black" if data[i, j] < 7 else "white")
    
    # Add title
    ax.set_title("Comparing UAV Cybersecurity Algorithms", fontsize=16)
    
    # Add explanatory text
    explanation = (
        "No_DDoS: Low resource usage, basic protection. Used in ~34% of decisions.\n"
        "XGBoost: Balanced performance and resource usage. Primary algorithm (~66% usage).\n"
        "TST: Highest protection but very resource-intensive. Used selectively (<0.1% of decisions)."
    )
    
    fig.text(0.5, 0.05, explanation, ha="center", fontsize=11, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('teaching/visualizations/algorithm_comparison.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Generate all diagrams
    print("Generating RL educational visualizations...")
    
    generate_rl_framework_diagram()
    generate_q_learning_update_diagram()
    generate_exploration_exploitation_diagram()
    generate_expert_warmstart_comparison()
    generate_safety_constraint_diagram()
    generate_learning_convergence_diagram()
    generate_state_space_visualization()
    generate_algorithm_comparison_diagram()
    
    print("All visualizations generated in teaching/visualizations/")
