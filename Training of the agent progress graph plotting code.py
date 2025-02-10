def plot_training_progress(episode_rewards, save_path=None):
    
    #Plot and display the training progress with moving averages
    #Args:
        #episode_rewards: List of rewards from each episode
        #save_path: Optional path to save the plot

    plt.figure(figsize=(12, 6))
    
    # Plot raw rewards
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    # Calculate and plot moving averages
    window_sizes = [100, 500]
    for window in window_sizes:
        moving_avg = np.convolve(episode_rewards, 
                                np.ones(window)/window, 
                                mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), 
                moving_avg, 
                label=f'{window}-Episode Moving Average')
    
    plt.title('Microgrid Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics as text
    stats_text = f'Final {window_sizes[-1]} Episodes:\n'
    stats_text += f'Average Reward: {np.mean(episode_rewards[-window_sizes[-1]:]):.2f}\n'
    stats_text += f'Max Reward: {np.max(episode_rewards[-window_sizes[-1]:]):.2f}\n'
    stats_text += f'Min Reward: {np.min(episode_rewards[-window_sizes[-1]:]):.2f}'
    
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()  # This will display the plot
    plt.close()

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
    
    # Plot and save the training progress
    save_path = os.path.join(log_dir, 'training_progress.png')
    try:
        plot_training_progress(episode_rewards, save_path)
    except Exception as e:
        logger.error(f"Error plotting training progress: {str(e)}")

if __name__ == "__main__":
    main()