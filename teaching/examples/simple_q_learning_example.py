"""
Simple Q-learning example in a grid world environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("teaching/examples", exist_ok=True)

class GridWorldEnv:
    """
    Simple 2D grid world environment for Q-learning demonstration.
    
    The agent starts at the bottom-left and needs to reach the goal at the top-right,
    avoiding obstacles along the way.
    """
    
    def __init__(self, width=5, height=5):
        """Initialize the grid world."""
        self.width = width
        self.height = height
        self.start_pos = (0, 0)  # Make sure this is a tuple
        self.goal_pos = (width-1, height-1)  # Make sure this is a tuple
        
        # Create grid with obstacles
        self.grid = np.zeros((height, width))
        
        # Set obstacles (value = -1)
        self.obstacles = [
            (1, 1), (3, 1),
            (1, 3), (2, 2), (3, 3)
        ]
        
        for obs in self.obstacles:
            self.grid[obs[1], obs[0]] = -1
            
        # Set goal (value = 1)
        self.grid[self.goal_pos[1], self.goal_pos[0]] = 1
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Changed to y,x format for clarity
        
        # Current position
        self.agent_pos = self.start_pos
    
    def reset(self):
        """Reset agent to start position and return state."""
        self.agent_pos = self.start_pos
        return self.agent_pos  # Already a tuple
    
    def step(self, action):
        """
        Take action and return new state, reward, done.
        
        Args:
            action: Integer 0-3 (up, right, down, left)
            
        Returns:
            next_state: New (x, y) position as a tuple
            reward: Reward for this step
            done: Whether episode is done
        """
        # Get action movement
        move = self.actions[action]
        
        # Calculate new position - ensure it's a tuple
        new_x = self.agent_pos[0] + move[0]
        new_y = self.agent_pos[1] + move[1]
        new_pos = (new_x, new_y)
        
        # Check if new position is valid
        if (0 <= new_x < self.width and 
            0 <= new_y < self.height and 
            new_pos not in self.obstacles):
            self.agent_pos = new_pos
        
        # Default small negative reward (encourages shortest path)
        reward = -0.1
        
        # Check for goal or obstacle
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        elif self.agent_pos in self.obstacles:
            reward = -10.0
            done = True
        else:
            done = False
            
        return self.agent_pos, reward, done

    def render(self, q_table=None, policy=None):
        """
        Render the grid world with matplotlib.
        
        Args:
            q_table: Q-table for visualization (optional)
            policy: Learned policy to show (optional)
            
        Returns:
            fig, ax: Figure and axis objects
        """
        from matplotlib.colors import ListedColormap
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create grid for visualization
        grid_vis = self.grid.copy()
        
        # Mark agent position
        if self.agent_pos != self.goal_pos:
            grid_vis[self.agent_pos[1], self.agent_pos[0]] = 2
        
        # Custom colormap for grid
        colors = ['white', 'green', 'blue', 'red']
        cmap = ListedColormap(colors)
        
        # Plot grid
        ax.imshow(grid_vis, cmap=cmap, vmin=-1, vmax=2)
        
        # Add grid lines
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-0.5, self.width, 1))
        ax.set_yticks(np.arange(-0.5, self.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Add cell text (coordinates)
        for i in range(self.width):
            for j in range(self.height):
                text_color = 'black'
                if (i, j) in self.obstacles:
                    text_color = 'white'
                    ax.text(i, j, "X", ha="center", va="center", color=text_color, fontsize=14, fontweight='bold')
                elif (i, j) == self.goal_pos:
                    ax.text(i, j, "GOAL", ha="center", va="center", color=text_color, fontsize=11)
                elif (i, j) == self.agent_pos:
                    ax.text(i, j, "AGENT", ha="center", va="center", color="white", fontsize=11)
                else:
                    ax.text(i, j, f"({i},{j})", ha="center", va="center", color=text_color, fontsize=9)
        
        # If Q-table is provided, show best actions
        if q_table is not None and policy is None:
            for i in range(self.width):
                for j in range(self.height):
                    state = (i, j)
                    if state != self.goal_pos and state not in self.obstacles:
                        # Fixed: Handle the Q-table structure correctly
                        if isinstance(q_table, dict) and state in q_table:
                            if isinstance(q_table[state], dict):  # Nested dict structure
                                q_values = [q_table[state].get(a, 0) for a in range(4)]
                            else:  # List structure
                                q_values = q_table[state]
                        else:
                            q_values = [0, 0, 0, 0]
                        
                        if sum(q_values) == 0:
                            continue  # Skip cells with no Q-values
                            
                        best_action = np.argmax(q_values)
                        
                        # Draw arrow for best action
                        dx, dy = 0, 0
                        if best_action == 0:  # up
                            dx, dy = 0, -0.4
                        elif best_action == 1:  # right
                            dx, dy = 0.4, 0
                        elif best_action == 2:  # down
                            dx, dy = 0, 0.4
                        elif best_action == 3:  # left
                            dx, dy = -0.4, 0
                            
                        ax.arrow(i, j, dx, dy, head_width=0.2, head_length=0.2, fc='black', ec='black')
        
        # If policy is provided, show path
        if policy is not None:
            current = self.start_pos
            path = [current]
            
            while current != self.goal_pos and len(path) < 100:  # Prevent infinite loop
                if current not in policy:
                    break  # No policy for this state
                    
                action = policy.get(current, 0)
                move = self.actions[action]
                next_pos = (current[0] + move[0], current[1] + move[1])
                
                # Check if move is valid
                if (0 <= next_pos[0] < self.width and 
                    0 <= next_pos[1] < self.height and 
                    next_pos not in self.obstacles):
                    path.append(next_pos)
                    current = next_pos
                else:
                    break
            
            # Draw path
            if len(path) > 1:  # Only if we have a valid path
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7)
                
                # Add arrows along the path
                for i in range(len(path)-1):
                    mid_x = (path_x[i] + path_x[i+1]) / 2
                    mid_y = (path_y[i] + path_y[i+1]) / 2
                    dx = path_x[i+1] - path_x[i]
                    dy = path_y[i+1] - path_y[i]
                    
                    # Normalize to get direction
                    mag = np.sqrt(dx**2 + dy**2)
                    if mag > 0:
                        dx, dy = dx/mag * 0.3, dy/mag * 0.3
                        ax.arrow(mid_x - dx/2, mid_y - dy/2, dx, dy, 
                                head_width=0.2, head_length=0.2, 
                                fc='red', ec='red')
        
        # Set title
        if policy is not None:
            ax.set_title("Grid World with Policy Path", fontsize=14)
        elif q_table is not None:
            ax.set_title("Grid World with Q-Values", fontsize=14)
        else:
            ax.set_title("Grid World Environment", fontsize=14)
            
        plt.tight_layout()
        return fig, ax

class QLearningAgent:
    """
    Simple Q-learning agent for the grid world.
    """
    
    def __init__(self, 
                 n_actions=4, 
                 learning_rate=0.1, 
                 discount_factor=0.9, 
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.95):
        """Initialize the Q-learning agent."""
        self.n_actions = n_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize empty Q-table as dictionary
        self.q_table = {}
        
        # Learning statistics
        self.rewards_per_episode = []
        self.steps_per_episode = []
    
    def get_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state (x, y) tuple
            explore: Whether to use exploration (epsilon-greedy) or pure exploitation
            
        Returns:
            Selected action (0-3)
        """
        # Ensure state is hashable (convert to tuple if it's a list)
        if isinstance(state, list):
            state = tuple(state)
            
        # Create state key if not exists
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
            
        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-values using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Ensure states are hashable
        if isinstance(state, list):
            state = tuple(state)
        if isinstance(next_state, list):
            next_state = tuple(next_state)
            
        # Create state keys if not exist
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.n_actions
            
        if next_state not in self.q_table:
            self.q_table[next_state] = [0.0] * self.n_actions
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        max_next_q = 0 if done else max(self.q_table[next_state])
        
        # Q-learning update
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error
        
        # Update Q-table
        self.q_table[state][action] = new_q
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    def get_policy(self):
        """Extract policy from Q-table."""
        policy = {}
        for state, q_values in self.q_table.items():
            policy[state] = np.argmax(q_values)
        return policy
    
    def get_value_function(self):
        """Extract value function from Q-table."""
        values = {}
        for state, q_values in self.q_table.items():
            values[state] = max(q_values)
        return values

def train_agent(env, agent, episodes=100, max_steps=100, render_interval=None):
    """
    Train the agent on the environment.
    
    Args:
        env: Environment
        agent: Q-learning agent
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        render_interval: If set, render environment every N episodes
    
    Returns:
        Trained agent
    """
    # Training loop
    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        for step in range(max_steps):
            # Get action
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            # Update agent
            agent.update(state, action, reward, next_state, done)
            
            # Update state and tracking
            state = next_state
            total_reward += reward
            steps += 1
            
            # Check if done
            if done:
                break
        
        # Track statistics
        agent.rewards_per_episode.append(total_reward)
        agent.steps_per_episode.append(steps)
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Render progress occasionally
        if render_interval and episode % render_interval == 0:
            print(f"Episode {episode}/{episodes}, Reward: {total_reward:.1f}, Steps: {steps}, Epsilon: {agent.epsilon:.3f}")
            
            # Optional: render environment
            if hasattr(env, 'render'):
                fig, ax = env.render(agent.q_table)
                plt.savefig(f"teaching/examples/grid_world_episode_{episode}.png")
                plt.close()
    
    return agent

def plot_learning_curves(agent, save_path="teaching/examples/learning_curves.png"):
    """Plot rewards and steps per episode."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(agent.rewards_per_episode, color='blue')
    ax1.set_title('Reward per Episode')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot steps
    ax2.plot(agent.steps_per_episode, color='green')
    ax2.set_title('Steps per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_policy_and_values(env, agent, save_path="teaching/examples/policy_visualization.png"):
    """Visualize the learned policy and value function."""
    # Get policy and value function
    policy = agent.get_policy()
    values = agent.get_value_function()
    
    # Create a grid for visualization
    value_grid = np.zeros((env.height, env.width))
    policy_grid = np.zeros((env.height, env.width))
    
    # Fill in the grid with values and policy
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state in values:
                value_grid[y, x] = values[state]
            if state in policy and state != env.goal_pos and state not in env.obstacles:
                policy_grid[y, x] = policy[state]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot value function
    im1 = ax1.imshow(value_grid, cmap='viridis')
    ax1.set_title('Value Function')
    fig.colorbar(im1, ax=ax1, label='Value')
    
    # Add grid lines
    for ax in [ax1, ax2]:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
        ax.set_xticks(np.arange(-0.5, env.width, 1))
        ax.set_yticks(np.arange(-0.5, env.height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    # Plot policy (arrows)
    ax2.imshow(np.zeros_like(policy_grid), cmap='viridis', alpha=0.1)
    ax2.set_title('Learned Policy')
    
    # Add arrows for policy
    for x in range(env.width):
        for y in range(env.height):
            state = (x, y)
            if state in policy and state != env.goal_pos and state not in env.obstacles:
                action = policy[state]
                dx, dy = 0, 0
                if action == 0:  # up
                    dx, dy = 0, -0.4
                elif action == 1:  # right
                    dx, dy = 0.4, 0
                elif action == 2:  # down
                    dx, dy = 0, 0.4
                elif action == 3:  # left
                    dx, dy = -0.4, 0
                    
                ax2.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc='red', ec='red')
    
    # Mark special cells
    # Goal
    ax2.text(env.goal_pos[0], env.goal_pos[1], "GOAL", ha="center", va="center", 
             color="white", fontweight='bold', fontsize=12)
    
    # Start
    ax2.text(env.start_pos[0], env.start_pos[1], "START", ha="center", va="center", 
             color="white", fontweight='bold', fontsize=12)
    
    # Obstacles
    for obs in env.obstacles:
        ax2.text(obs[0], obs[1], "X", ha="center", va="center", 
                 color="white", fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """Main function to demonstrate Q-learning."""
    print("Starting Q-learning demonstration...")
    
    # Create environment and agent
    env = GridWorldEnv(width=5, height=5)
    agent = QLearningAgent(
        n_actions=4,
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.95
    )
    
    # Render initial state
    fig, ax = env.render()
    plt.savefig("teaching/examples/grid_world_initial.png")
    plt.close()
    
    # Train the agent
    print("Training agent...")
    agent = train_agent(env, agent, episodes=200, max_steps=100, render_interval=50)
    
    # Plot learning curves
    plot_learning_curves(agent)
    
    # Visualize final policy
    print("Visualizing final policy...")
    fig, ax = env.render(policy=agent.get_policy())
    plt.savefig("teaching/examples/grid_world_final.png")
    plt.close()
    
    # Visualize policy and values
    visualize_policy_and_values(env, agent)
    
    print("Demonstration completed. Check 'teaching/examples/' directory for outputs.")

if __name__ == "__main__":
    # Create output directory
    os.makedirs("teaching/examples", exist_ok=True)
    main()