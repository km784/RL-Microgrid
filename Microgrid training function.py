def train_microgrid(env, agent, n_episodes, max_steps):
    episode_rewards = []
    episode_costs = []  # Added to track costs
    env.battery_data = []

    for episode in range(n_episodes):
        logger.info(f"Starting episode {episode}")
        
        env.current_episode = episode
        env.current_step = 0
        
        # Randomize initial SOC between SOC_MIN and SOC_MAX for each episode
        env.state_space['battery_soc'] = np.random.uniform(env.SOC_MIN, env.SOC_MAX)
                
        total_reward = 0
        total_cost = 0
        state = env.update_state()
        episode_data = []

        for step in range(max_steps):
            step_data = {
                'Episode': episode,
                'Time_Hour': env.state_space['time_hour'],
                'Battery_SOC': env.state_space['battery_soc'],
                'PV_Power': env.state_space['pv_power'],
                'Load_Demand': env.state_space['load_demand']
            }

            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            # Store the cost from the environment
            total_cost += env.last_cost
            total_reward += reward

            step_data['Action'] = action
            step_data['Reward'] = reward
            episode_data.append(step_data)

            agent.learn(state, action, reward, next_state)
            state = next_state

            env.log_state_to_csv()  # Log state after each step

            if done:
                break

        # Convert episode data to DataFrame and append to battery_data list
        episode_df = pd.DataFrame(episode_data)
        env.battery_data.append(episode_df)

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards)
            logger.info(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    return episode_rewards, episode_costs, env.battery_data