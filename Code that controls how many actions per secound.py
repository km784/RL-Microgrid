import time  # Add this at the top of your file

# In your main training loop or where the actions are executed:
actions_per_second = 1  # Adjust this value to set actions per second
delay = 1.0 / actions_per_second  # Calculate delay between actions

# Inside your action execution loop:
while running:  # or your existing loop structure
    # Your existing action selection and execution code
    action = agent.choose_action(state)
    next_state, reward = env.step(action)
    
    # Add delay to control actions per second
    time.sleep(delay)
