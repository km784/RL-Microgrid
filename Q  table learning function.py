import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
# Define the state space and action space sizes
n_states = 3 * 3 * 3 * 4  # battery_state * pv_state * load_state * time_state
n_actions = 2  # number of possible actions

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, discount_factor, epsilon):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_state_index(self, state):
        return state[0] * (3*3*4) + state[1] * (3*4) + state[2] * 4 + state[3]
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])
        state_idx = self.get_state_index(state)
        return int(np.argmax(self.q_table[state_idx, :]))
    
    def learn(self, state, action, reward, next_state):
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        old_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx, :])
        new_q = (1 - self.learning_rate) * old_q + \
                self.learning_rate * (reward + self.discount_factor * next_max_q)
        
        self.q_table[state_idx, action] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_microgrid(env, agent, n_episodes, max_steps):
    
    #Train the microgrid agent
        #env: MicrogridState instance
       # agent: QLearningAgent instance
        #n_episodes: Number of training episodes
       # max_steps: Maximum steps per episode
    
    episode_rewards = []
    
    for episode in range(n_episodes):
        logger.info(f"Starting episode {episode}")
        
        # Reset environment
        env.current_episode = episode
        env.current_step = 0
        total_reward = 0
        state = env.update_state()
        
        for step in range(max_steps):
            # Choose and take action
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Learn from experience
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Post-episode updates
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards

def main():
    # Initialize environment
    microgrid_env = MicrogridState()
    
    # Initialize agent
    n_states = 3 * 3 * 3 * 4  # battery_state * pv_state * load_state * time_state
    n_actions = 2  # number of possible actions
    
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0
    )
    
    # Training parameters
    n_episodes = 10000
    max_steps_per_episode = len(load.time_series)
    
    # Train the agent
    episode_rewards = train_microgrid(
        env=microgrid_env,
        agent=agent,
        n_episodes=n_episodes,
        max_steps=max_steps_per_episode
    )
    
    # Save the Q-table
    np.save('q_table.npy', agent.q_table)
    
    # Optional: Plot training progress
    try:
        import matplotlib.pyplot as plt
        plt.plot(episode_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('training_progress.png')
        plt.close()
    except ImportError:
        logger.warning("Matplotlib not installed, skipping plot generation")

if __name__ == "__main__":
    main()