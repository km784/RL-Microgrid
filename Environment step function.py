import sys
import pymgrid
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt as plt
import numpy as np
import os
import time
from helper_functions import *


def step(self, action):
    """
    step function to track costs
    """
    logger.info(f"Taking step with action: {action}")
    logger.debug(f"Current State Before Action: {self.state_space}")
    
    self.last_action = action
    control_dict = self.actions_agent(action)
    next_state = self.update_state()
    
    # Get both reward and cost
    reward, cost = self.calculate_reward(control_dict)
    self.last_reward = float(reward)
    self.last_cost = float(cost)  # Store the cost
    
    # Convert to scalar if it's a NumPy array
    reward = float(reward) if isinstance(reward, np.ndarray) else reward
    
    self.current_step += 1
    
    logger.debug(f"Next State After Action: {next_state}")
    logger.info(f"Reward Obtained: {reward}, Cost: {cost}")
    
    done = self.state_space['time_hour'] >= 23
    return next_state, reward, done