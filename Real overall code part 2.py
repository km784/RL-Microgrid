import sys
import pymgrid
import csv
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt 
import numpy as np
import logging
import pandas as pd
import stat
import os
from datetime import datetime
import time
from helper_functions import *

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

battery = BatteryModule(
    min_capacity=0,
    max_capacity=120,  # Maximum battery capacity in kWh
    max_charge=50,     # Maximum charging rate in kW
    max_discharge=30,  # Maximum discharging rate in kW                                                                                                      
    efficiency=1.0,    # Battery efficiency
    init_soc=0.5       # Initial state of charge (50%)
)
# Create renewable (PV) module with specific generation profile
renewable = RenewableModule(
    time_series=50*np.random.rand(100)  # Replace with your PV generation data
)
# Create load module with specific consumption profile
load = LoadModule(
    time_series=[30]*24  # Creates an array of 24 elements (for each hour) with the value 30kW
)
# Create and initialize the microgrid
microgrid = Microgrid([
    battery,
    ("pv", renewable),
    load
])


class BatteryDegradation:
    def __init__(self):
        # Battery specifications
        self.nominal_capacity = 10  # kWh
        self.total_throughput_limit = 1500000  # kWh throughput lifetime
        self.cycle_life_reference = 3000  # number of cycles
        self.temperature_reference = 25  # °C
        self.current_rate_reference = 0.5  # C-rate
        
        # Aging factors
        self.calendar_aging_factor = 0.00164  # can check for batteries
        self.temperature_coefficient = 0.06  # Arhenius equation coefficie
        self.current_rate_coefficient = 0.2     # coefficiencts
        self.depth_of_discharge_coefficient = 1.4    #coefficients
        
    
        self.cumulative_throughput = 0
        self.cycle_count = 0                     # tracking 
        self.total_capacity_loss = 0
        
    def calculate_cycle_degradation(self, energy_throughput, depth_of_discharge, temperature=25, current_rate=0.5):
        # Caluclate battery degradation through different factors
        
        # throughput-based degradation
        throughput_degradation = (energy_throughput / self.total_throughput_limit) * 100
        
        # mperature factor (Arrhenius relationship)
        temperature_factor = np.exp(self.temperature_coefficient * 
                                  (temperature - self.temperature_reference) / 8.314)
        
        # Current rate factor
        current_stress = (current_rate / self.current_rate_reference) ** self.current_rate_coefficient
        
        # Depth of discharge  factor (nonlinear relationship)
        dod_stress = (depth_of_discharge ** self.depth_of_discharge_coefficient)
        
        # calendar aging
        time_elapsed = energy_throughput / (self.nominal_capacity * 365 * 2)  # Approximate time in years
        calendar_degradation = self.calendar_aging_factor * time_elapsed
        
        # Combined degradation
        cycle_degradation = throughput_degradation * temperature_factor * current_stress * dod_stress
        total_degradation = cycle_degradation + calendar_degradation
        
        # Update state
        self.cumulative_throughput += energy_throughput
        self.cycle_count += energy_throughput / (self.nominal_capacity * depth_of_discharge)
        self.total_capacity_loss += total_degradation
        
        return total_degradation
    
    def get_remaining_capacity(self):
       # returns the capacity of the battery
        return max(0, 100 - self.total_capacity_loss)
    
    def get_battery_health_metrics(self):
      # returns health metrics of battery
        return {
            'remaining_capacity_percent': self.get_remaining_capacity(),
            'cumulative_throughput': self.cumulative_throughput,
            'cycle_count': self.cycle_count,
            'total_capacity_loss': self.total_capacity_loss,
            'estimated_remaining_cycles': max(0, self.cycle_life_reference - self.cycle_count)
        }
    
    def get_current_tariff(self):
        hour = self.state_space['time_hour']
        if (7 <= hour < 11) or (17 <= hour < 21):
            return {
                'consumption': 0.17,  # 17 pence per kWh for on-peak consumption
                'injection': 0.10     # 10 pence per kWh for on-peak injection
            }
        else:
            return {
                'consumption': 0.13,  # 13 pence per kWh for off-peak consumption
                'injection': 0.07     # 7 pence per kWh for off-peak injection
            }
        
class MicrogridState:
    def __init__(self):
        self.csv_headers = ['Episode', 'Step', 'Battery_SOC', 'PV_Power', 'Load_Demand', 
                       'Time_Hour', 'Net_Load', 'Action', 'Reward','Battery_Degradation', 'Remaining_Capacity', 'Cycle_Count']
        
        self.MAX_PENALTY = 50 #max penalty set
        self.battery_data = []
        self.state_space = {
            'battery_soc': 0.0,
            'pv_power': 0.0,
            'load_demand': 0.0,
            'time_hour': 0,
            'net_load': 0.0,
            'battery_health': 100.0
        }
        self.battery_degradation = BatteryDegradation()
        self.SOC_MIN = 0.2
        self.SOC_OPTIMAL_MIN = 0.3
        self.SOC_OPTIMAL_MAX = 0.85
        self.SOC_MAX = 0.9
        self.renewable = renewable
        self.battery = battery
        self.current_episode = 0
        self.current_step = 0
        self.last_action = None
        self.last_reward = 0
        self.last_cost = 0
        self.last_degradation = 0
       
    def get_current_tariff(self):
        hour = self.state_space['time_hour']
        if (7 <= hour < 11) or (17 <= hour < 21):
            return {
                'consumption': 0.17,  # 17 pence per kWh for on-peak consumption
                'injection': 0.10     # 10 pence per kWh for on-peak injection
            }
        else:
            return {
                'consumption': 0.13,  # 13 pence per kWh for off-peak consumption
                'injection': 0.07     # 7 pence per kWh for off-peak injection
            }       
         
    def update_state(self):
        
        # Update PV generation
        self.state_space['pv_power'] = self.renewable.current_obs
        
        # Update load demand
        self.state_space['load_demand'] = 30 # constant load of 30kW
        
        # Update time
        self.state_space['time_hour'] = (self.state_space['time_hour'] + 1) % 24
        
        # Update net load
        self.state_space['net_load'] = self.state_space['load_demand'] - self.state_space['pv_power']
        
        health_metrics = self.battery_degradation.get_battery_health_metrics()
        self.state_space['battery_health'] = health_metrics['remaining_capacity_percent']
        
        return self.discretize_state()
    
    def discretize_state(self):
        if self.state_space['battery_soc'] < self.SOC_MIN:
            battery_state = 0    # When below minimum point, damaging battery
        elif self.state_space['battery_soc'] < self.SOC_OPTIMAL_MIN:
            battery_state = 1    # lower than optiamum but alright
        elif self.SOC_OPTIMAL_MIN <= self.state_space['battery_soc'] < self.SOC_OPTIMAL_MAX:
            battery_state = 2    # Optimal range
        elif self.state_space['battery_soc'] < self.SOC_MAX:
            battery_state = 3    # High - approaching maximum
        else:
            battery_state = 4    # Critical - above maximum
            
        if self.state_space['pv_power'] < 20:
            pv_state = 0        # Low generation
        elif self.state_space['pv_power'] < 35:
            pv_state = 1        # Medium generation
        else:
            pv_state = 2        # High generation
            
        if self.state_space['load_demand'] < 25:
            load_state = 0      # Low demand
        elif self.state_space['load_demand'] < 45:
            load_state = 1      # Medium demand
        else:
            load_state = 2      # High demand
            
        if self.state_space['time_hour'] < 6:
            time_state = 0      # Night
        elif self.state_space['time_hour'] < 12:
            time_state = 1      # Morning
        elif self.state_space['time_hour'] < 18:
            time_state = 2      # Afternoon
        else:
            time_state = 3      # Evening
            
        return (battery_state, pv_state, load_state, time_state)
    
    def step(self, action):
        """
        step function to track costs
        """
        
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
        
        done = self.state_space['time_hour'] >= 23
        
        return next_state, reward, done
    
    def actions_agent(self, action):
    # Log initial state
        net_load = self.state_space['load_demand'] - self.state_space['pv_power']
        current_soc = self.state_space['battery_soc']
        
        # Calculating available capacity considering SOC limits
        available_to_charge = min(
            self.battery.max_charge,
            (self.SOC_MAX - current_soc) * self.battery.max_capacity
        )
        
        available_to_discharge = min(
            self.battery.max_discharge,
            (current_soc - self.SOC_MIN) * self.battery.max_capacity
        )
        control_dict = {}
        action_type = ''
        
        if action == 0:  # Charge battery from the grid
            action_type = 'CHARGE_BATTERY'
            if current_soc < self.SOC_MAX:
                charge_amount = min(available_to_charge, self.battery.max_charge)
                # Update battery SOC
                new_soc = current_soc + (charge_amount / self.battery.max_capacity)
                self.state_space['battery_soc'] = min(new_soc, self.SOC_MAX)
                
                control_dict = {
                    'pv_consumed': 0,
                    'battery_charge': charge_amount,
                    'battery_discharge': 0,
                    'grid_import': charge_amount,
                    'grid_export': 0
                }
                self.last_degradation = self.battery_degradation.calculate_cycle_degradation(
                    energy_throughput=charge_amount,
                    depth_of_discharge=1 - current_soc
                )
            else:
                control_dict = {
                    'pv_consumed': 0,
                    'battery_charge': 0,
                    'battery_discharge': 0,
                    'grid_import': 0,
                    'grid_export': 0
                }
                
        elif action == 1:  # Discharge battery to the grid
            action_type = 'DISCHARGE_BATTERY'
            if current_soc > self.SOC_MIN:
                discharge_amount = min(available_to_discharge, self.battery.max_discharge)
                # Update battery SOC
                new_soc = current_soc - (discharge_amount / self.battery.max_capacity)
                self.state_space['battery_soc'] = max(new_soc, self.SOC_MIN)
                
                control_dict = {
                    'pv_consumed': 0,
                    'battery_charge': 0,
                    'battery_discharge': discharge_amount,
                    'grid_import': 0,
                    'grid_export': discharge_amount
                }
                self.last_degradation = self.battery_degradation.calculate_cycle_degradation(
                    energy_throughput=discharge_amount,
                    depth_of_discharge=current_soc
                )
            else:
                control_dict = {
                    'pv_consumed': 0,
                    'battery_charge': 0,
                    'battery_discharge': 0,
                    'grid_import': 0,
                    'grid_export': 0
                }
                
        return control_dict
        # Remove the underscore from the method definition
        
    def calculate_reward(self, control_dict):
        """
        Calculate reward based on proximity to theoretical minimum cost.
        """
        # Get current tariff rates
        tariff = self.get_current_tariff()

        # Calculate grid interactions
        grid_import = control_dict.get('grid_import', 0)
        grid_export = control_dict.get('grid_export', 0)

        # Calculate costs and revenues
        grid_cost = grid_import * tariff['consumption']
        grid_revenue = grid_export * tariff['injection']
        actual_cost = grid_cost - grid_revenue

        # Calculate theoretical minimum cost for current timestep
        current_pv = self.state_space['pv_power']
        current_load = self.state_space['load_demand']
        theoretical_min = self.min_cost([current_pv], [current_load])  # Call min_cost as a method

        # Calculate proximity to theoretical minimum
        cost_difference = actual_cost - theoretical_min

        # Reward increases as cost gets closer to theoretical minimum
        if actual_cost == theoretical_min:
            cost_reward = 100  # Maximum reward when cost is at or below theoretical minimum
        else:
            cost_reward = 100 * (1 - cost_difference / theoretical_min)  # Linearly decrease reward as cost increases

        # Strategic battery management rewards
        strategic_reward = 0
        current_soc = self.state_space['battery_soc']
        hour = self.state_space['time_hour']

        # Peak hours check (7-11 and 17-21)
        is_peak = (7 <= hour < 11) or (17 <= hour < 21)

        # Modified peak/off-peak rewards
        if is_peak:
            if control_dict['battery_discharge'] > 0:
                strategic_reward += 15  # Peak discharge reward
            elif control_dict['battery_charge'] > 0:
                strategic_reward -= 15  # Penalty for peak charging
        else:
            if control_dict['battery_charge'] > 0 and current_soc < self.SOC_OPTIMAL_MAX:
                strategic_reward += 15  # Off-peak charging reward
            elif control_dict['battery_discharge'] > 0:
                strategic_reward -= 10  # Penalty for off-peak discharge

        # Battery health management with smoother transitions
        soc_reward = 0
        if self.SOC_OPTIMAL_MIN <= current_soc <= self.SOC_OPTIMAL_MAX:
            soc_reward += 15
        else:
            # Gradual penalty based on distance from optimal range
            distance_from_optimal = min(
                abs(current_soc - self.SOC_OPTIMAL_MIN),
                abs(current_soc - self.SOC_OPTIMAL_MAX)
            )
            soc_reward -= 15 * (distance_from_optimal / 0.1)  # Gradual penalty

        # Combine rewards with adjusted weights
        total_reward = (
            cost_reward * 1.2 +      # Increased emphasis on cost
            strategic_reward +        # Keep strategic rewards as is
            soc_reward * 0.8         # Slightly reduced SOC influence
        )

        # Smooth clipping of rewards
        total_reward = np.clip(total_reward, -100, 100)

        return total_reward, actual_cost
                         
        
    def min_cost(self, pv_generation, load_demand):
        """
        Calculate the theoretical minimum cost for a 24-hour period given perfect foresight
        and optimal battery operation.
        
        Args:
            pv_generation (list/array): PV generation profile for 24 hours in kW
            load_demand (list/array): Load demand profile for 24 hours in kW
                
        Returns:
            float: Theoretical minimum cost in pounds
        """
        # Convert inputs to numpy arrays if they aren't already
        pv_generation = np.array(pv_generation)
        load_demand = np.array(load_demand)
        
        # Handle single value inputs by repeating them 24 times
        if np.isscalar(pv_generation) or len(pv_generation) == 1:
            pv_generation = np.full(24, pv_generation if np.isscalar(pv_generation) else pv_generation[0])
        if np.isscalar(load_demand) or len(load_demand) == 1:
            load_demand = np.full(24, load_demand if np.isscalar(load_demand) else load_demand[0])
        
        # Ensure inputs are lists or arrays of length 24
        if not isinstance(pv_generation, (list, np.ndarray)) or not isinstance(load_demand, (list, np.ndarray)):
            raise ValueError("pv_generation and load_demand must be lists or arrays")
        
        if len(pv_generation) != 24 or len(load_demand) != 24:
            raise ValueError("pv_generation and load_demand must contain 24 values (one for each hour)")
        
        total_cost = 0
        total_revenue = 0
        
        # Define battery parameters
        battery_capacity = 120  # kWh
        max_charge_rate = 50    # kW
        max_discharge_rate = 50 # kW
        initial_soc = 0.5 * battery_capacity  # Start at 50% SOC
        current_soc = initial_soc
        
        # Create arrays for peak and off-peak periods
        hours = range(24)
        peak_periods = [(7, 11), (17, 21)]
        
        # Create a list of hourly periods with their associated tariffs
        period_tariffs = []
        for hour in hours:
            is_peak = any(start <= hour < end for start, end in peak_periods)
            if is_peak:
                tariff = {
                    'hour': hour,
                    'consumption': 0.17,  # Peak consumption rate (£/kWh)
                    'injection': 0.10    # Peak injection rate (£/kWh)
                }
            else:
                tariff = {
                    'hour': hour,
                    'consumption': 0.13,  # Off-peak consumption rate (£/kWh)
                    'injection': 0.07    # Off-peak injection rate (£/kWh)
                }
            period_tariffs.append(tariff)
        
        # Sort periods by price differential
        period_tariffs.sort(key=lambda x: x['hour'])
        
        # Calculate optimal battery operation for each hour
        for period in period_tariffs:
            hour = period['hour']
            net_load = load_demand[hour] - pv_generation[hour]
            
            if any(start <= hour < end for start, end in peak_periods):
                # During high-price periods, discharge the battery
                max_discharge = min(max_discharge_rate, current_soc)  # Discharge up to max_discharge_rate or current SOC
                discharge = min(max_discharge, net_load)  # Discharge only what is needed
                current_soc -= discharge
                
                # Calculate revenue from discharging
                revenue = discharge * period['injection']
                total_revenue += revenue
                
            else:
                # During low-price periods, charge the battery
                max_charge = min(max_charge_rate, battery_capacity - current_soc)  # Charge up to max_charge_rate or remaining capacity
                charge = min(max_charge, -net_load)  # Charge only if there is excess generation
                current_soc += charge
                
                # Calculate cost of charging
                cost = charge * period['consumption']
                total_cost += cost
                
            absolute_min_cost = total_cost - total_revenue
                    
        return absolute_min_cost

class QLearningAgent:       # creates Q-table and sets learning and exploration parameters
    def __init__(self, n_states, n_actions, learning_rate, discount_factor, epsilon):
        self.q_table = np.zeros((n_states, n_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def get_state_index(self, state):
        battery, pv, load, time = state  # Components
        return battery * (3 * 3 * 4) + pv * (3 * 4) + load * 4 + time
    
    def choose_action(self, state):                          # Implements epsilon-greedy strategy 
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.q_table.shape[1])     
        state_idx = self.get_state_index(state)
        return int(np.argmax(self.q_table[state_idx, :]))
    
    def learn(self, state, action, reward, next_state):                # Updates Q-values using on Bellman equation
        state_idx = self.get_state_index(state)                        # future reward and immediate reward
        next_state_idx = self.get_state_index(next_state)
        
        old_q = self.q_table[state_idx, action]
        next_max_q = np.max(self.q_table[next_state_idx, :])
        new_q = (1 - self.learning_rate) * old_q + \
                self.learning_rate * (reward + self.discount_factor * next_max_q)      # Q-learning fomrmula
        
        self.q_table[state_idx, action] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        # Exploration vs exploitation, agent starts to exploit when gains more knowledge
def train_microgrid(env, agent, n_episodes, max_steps):
    episode_rewards = []
    episode_costs = []  # Added to track costs
    env.battery_data = []
    for episode in range(n_episodes):
        
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
   
    return episode_rewards, episode_costs, env.battery_data

def plot_training_progress(episode_rewards, save_path=None):
    """Plot training progress with improved visualization"""
    plt.figure(figsize=(12, 6))
    
    # Convert episode_rewards to numpy array if it isn't already
    episode_rewards = np.array(episode_rewards)
    
    # Only plot if we have data
    if len(episode_rewards) > 0:
        # Plot raw rewards with reduced alpha for better visibility
        plt.plot(range(len(episode_rewards)), episode_rewards, alpha=0.2, color='gray', label='Raw Rewards')
        
        # Calculate and plot moving averages with distinct colors
        window_sizes = [100, 500]
        colors = ['#2196F3', '#4CAF50']  # Blue, Green
        
        for window, color in zip(window_sizes, colors):
            if len(episode_rewards) >= window:
                moving_avg = np.convolve(episode_rewards, 
                                    np.ones(window)/window, 
                                    mode='valid')
                plt.plot(range(window-1, len(episode_rewards)), 
                        moving_avg, 
                        color=color,
                        linewidth=2,
                        label=f'{window}-Episode Moving Average')
        
        plt.title('Microgrid Training Progress', fontsize=14, pad=20)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add statistics box if we have enough episodes
        if len(episode_rewards) >= window_sizes[-1]:
            stats_text = f'Final {window_sizes[-1]} Episodes Statistics:\n'
            stats_text += f'Average Reward: {np.mean(episode_rewards[-window_sizes[-1]:]):.2f}\n'
            stats_text += f'Max Reward: {np.max(episode_rewards[-window_sizes[-1]:]):.2f}\n'
            stats_text += f'Min Reward: {np.min(episode_rewards[-window_sizes[-1]:]):.2f}'
            
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    fontsize=10,
                    bbox=dict(boxstyle='round',
                            facecolor='white',
                            edgecolor='#CCCCCC',
                            alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_soc_across_episodes(self, battery_data, save_path=None):
    """Plot battery SOC profiles for selected episodes"""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    try:
        # Convert list of DataFrames to a single DataFrame
        all_episodes_df = pd.concat(battery_data, ignore_index=True)
        # Select the starting and final episodes to display
        start_episode = 10
        final_episode = all_episodes_df['Episode'].max()
        # Plot the SOC profiles for the selected episodes
        for episode in [start_episode, final_episode]:
            episode_data = all_episodes_df[all_episodes_df['Episode'] == episode]
            plt.plot(episode_data['Time_Hour'], episode_data['Battery_SOC'] * 100,
                    linewidth=2, label=f'Episode {episode}')
        # Add limit lines
        plt.axhline(y=self.SOC_MAX * 100, color='red', linestyle='--', alpha=0.5, label='Maximum SOC')
        plt.axhline(y=self.SOC_MIN * 100, color='red', linestyle='--', alpha=0.5, label='Minimum SOC')
        plt.axhline(y=self.SOC_OPTIMAL_MAX * 100, color='green', linestyle=':', alpha=0.5, label='Optimal Max')
        plt.axhline(y=self.SOC_OPTIMAL_MIN * 100, color='green', linestyle=':', alpha=0.5, label='Optimal Min')
        # Add grid lines
        plt.grid(True, alpha=0.3)
        # Customize the plot
        plt.xlabel('Time of Day (Hour)', fontsize=12)
        plt.ylabel('Battery State of Charge (%)', fontsize=12)
        plt.title('Battery SOC Profiles for Selected Episodes', fontsize=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, fontsize=10)
        # Adjust the y-axis scaling
        plt.ylim(20, 100)
        # Add key metrics annotations for the final episode
        final_episode_data = all_episodes_df[all_episodes_df['Episode'] == final_episode]
        stats_text = (
            f"Final Episode Metrics:\n"
            f"Average SOC: {final_episode_data['Battery_SOC'].mean() * 100:.1f}%\n"
            f"SOC Variance: {final_episode_data['Battery_SOC'].var() * 100:.2f}%\n"
            f"Time in Optimal Range: {((final_episode_data['Battery_SOC'] >= self.SOC_OPTIMAL_MIN) & (final_episode_data['Battery_SOC'] <= self.SOC_OPTIMAL_MAX)).mean() * 100:.1f}%"
        )
        plt.text(0.02, 0.98, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'), verticalalignment='top', fontsize=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        return plt.gcf()
    except Exception as e:
        logger.error(f"Error in plot_soc_across_episodes: {str(e)}")
        return None
    
def plot_tariff_battery_relationship(env, battery_data, episodes_to_compare=[10, 99], save_path=None):
    """
    Plot injection tariff rates and battery SOC against time, comparing two episodes
    
    Args:
        env: MicrogridState environment instance
        battery_data: List of DataFrames containing episode data
        episodes_to_compare: List of episode numbers to compare
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each episode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1])
    
    # Get all episode data
    all_data = pd.concat(battery_data, ignore_index=True)
    
    # Generate injection tariff data
    hours = range(24)
    injection_tariffs = []
    
    for hour in hours:
        if (7 <= hour < 11) or (17 <= hour < 21):
            injection_tariffs.append(0.10)    # Peak injection rate
        else:
            injection_tariffs.append(0.07)    # Off-peak injection rate
    
    # Process each episode
    for idx, episode_num in enumerate(episodes_to_compare):
        ax = ax1 if idx == 0 else ax2
        twin_ax = ax.twinx()
        
        # Get episode data
        episode_data = all_data[all_data['Episode'] == episode_num]
        
        # Plot injection tariff on primary y-axis
        line1 = ax.plot(hours, injection_tariffs, 'g-', label='Injection Tariff', linewidth=2.5)
        
        # Plot battery SOC on secondary y-axis
        line2 = twin_ax.plot(episode_data['Time_Hour'], 
                           episode_data['Battery_SOC'] * 100, 
                           'b-', 
                           label='Battery SOC', 
                           linewidth=2.5)
        
        # Add shaded regions for peak hours
        peak_periods = [(7, 11), (17, 21)]
        for start, end in peak_periods:
            ax.axvspan(start, end, color='yellow', alpha=0.2, label='Peak Hours' if start == 7 else "")
        
        # Customize the plot
        ax.set_xlabel('Time of Day (Hour)', fontsize=12)
        ax.set_ylabel('Injection Tariff Rate (£/kWh)', fontsize=12, color='g')
        twin_ax.set_ylabel('Battery State of Charge (%)', fontsize=12, color='b')
        
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 0.12)
        twin_ax.set_ylim(0, 100)
        
        # Add minor gridlines
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)
        
        # Combine legends from both axes
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')
        
        # Add episode-specific title
        ax.set_title(f'Episode {episode_num}', fontsize=14)
        
        # Calculate and add statistics
        optimal_range_time = ((episode_data['Battery_SOC'] >= env.SOC_OPTIMAL_MIN) & 
                            (episode_data['Battery_SOC'] <= env.SOC_OPTIMAL_MAX)).mean() * 100
        peak_hours_mask = ((episode_data['Time_Hour'].between(7, 10)) | 
                          (episode_data['Time_Hour'].between(17, 20)))
        peak_soc = episode_data[peak_hours_mask]['Battery_SOC'].mean() * 100
        off_peak_soc = episode_data[~peak_hours_mask]['Battery_SOC'].mean() * 100
        
        stats_text = (
            f"Episode Statistics:\n"
            f"Time in Optimal Range: {optimal_range_time:.1f}%\n"
            f"Avg Peak Hours SOC: {peak_soc:.1f}%\n"
            f"Avg Off-Peak SOC: {off_peak_soc:.1f}%"
        )
        
        plt.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         edgecolor='#CCCCCC',
                         alpha=0.9))
    
    # Add overall title
    fig.suptitle('Injection Tariff Rate and Battery SOC vs Time of Day: Learning Comparison', 
                fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()

def plot_battery_actions_soc_comparison(env, battery_data, episodes_to_compare=[10, 99], save_path=None):
    """
    Create a visualization comparing agent behavior across different episodes
    to demonstrate learning progress
    
    Args:
        env: MicrogridState environment instance
        battery_data: List of DataFrames containing episode data
        episodes_to_compare: List of episode numbers to compare
        save_path: Optional path to save the plot
    """
    # Set up the figure with a grid of subplots
    fig = plt.figure(figsize=(20, 12))
    gs = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.1, wspace=0.2)
    
    # Get all episode data
    all_data = pd.concat(battery_data, ignore_index=True)
    
    # Calculate learning metrics for both episodes
    learning_metrics = {}
    
    for episode_num in episodes_to_compare:
        episode_data = all_data[all_data['Episode'] == episode_num]
        
        # Calculate various performance metrics
        optimal_range_time = ((episode_data['Battery_SOC'] >= env.SOC_OPTIMAL_MIN) & 
                            (episode_data['Battery_SOC'] <= env.SOC_OPTIMAL_MAX)).mean() * 100
        
        peak_hours_mask = ((episode_data['Time_Hour'].between(7, 10)) | 
                          (episode_data['Time_Hour'].between(17, 20)))
        
        # Calculate strategic behavior metrics
        peak_discharges = (episode_data['Action'][peak_hours_mask] == 1).sum()
        off_peak_charges = (episode_data['Action'][~peak_hours_mask] == 0).sum()
        
        # Store metrics
        learning_metrics[episode_num] = {
            'optimal_range_time': optimal_range_time,
            'peak_discharges': peak_discharges,
            'off_peak_charges': off_peak_charges,
            'total_charges': (episode_data['Action'] == 0).sum(),
            'total_discharges': (episode_data['Action'] == 1).sum(),
            'average_soc': episode_data['Battery_SOC'].mean() * 100
        }
    
    # Create subplots for each episode
    for idx, episode_num in enumerate(episodes_to_compare):
        episode_data = all_data[all_data['Episode'] == episode_num]
        metrics = learning_metrics[episode_num]
        
        # Create SOC subplot
        ax1 = plt.subplot(gs[0, idx])
        ax2 = plt.subplot(gs[1, idx], sharex=ax1)
        
        # Plot battery SOC
        ax1.plot(episode_data['Time_Hour'], episode_data['Battery_SOC'] * 100, 
                'b-', linewidth=2.5, label='Battery SOC')
        
        # Add threshold lines
        ax1.axhline(y=env.SOC_MAX * 100, color='red', linestyle='--', alpha=0.5, label='Maximum SOC')
        ax1.axhline(y=env.SOC_MIN * 100, color='red', linestyle='--', alpha=0.5, label='Minimum SOC')
        ax1.axhline(y=env.SOC_OPTIMAL_MAX * 100, color='green', linestyle=':', alpha=0.5, label='Optimal Max')
        ax1.axhline(y=env.SOC_OPTIMAL_MIN * 100, color='green', linestyle=':', alpha=0.5, label='Optimal Min')
        
        # Plot actions
        for _, row in episode_data.iterrows():
            if row['Action'] == 0:  # Charge
                ax2.bar(row['Time_Hour'], 1, color='green', alpha=0.6, width=0.8)
            else:  # Discharge
                ax2.bar(row['Time_Hour'], -1, color='red', alpha=0.6, width=0.8)
        
        # Add peak hour shading
        peak_periods = [(7, 11), (17, 21)]
        for start, end in peak_periods:
            ax1.axvspan(start, end, color='yellow', alpha=0.2, label='Peak Hours' if start == 7 else "")
            ax2.axvspan(start, end, color='yellow', alpha=0.2)
        
        # Customize plots
        ax1.set_ylabel('Battery State of Charge (%)', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        if idx == 0:
            ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        ax2.set_xlabel('Time of Day (Hour)', fontsize=12)
        ax2.set_ylabel('Agent Actions', fontsize=12)
        ax2.set_yticks([-1, 1])
        ax2.set_yticklabels(['Discharge', 'Charge'])
        ax2.grid(True, alpha=0.3)
        
        # Add episode-specific title
        ax1.set_title(f'Episode {episode_num}', fontsize=14)
        
        # Add statistics textbox
        stats_text = (
            f"Episode {episode_num} Statistics:\n"
            f"Charges: {metrics['total_charges']}\n"
            f"Discharges: {metrics['total_discharges']}\n"
            f"Peak Hour Discharges: {metrics['peak_discharges']}\n"
            f"Off-Peak Charges: {metrics['off_peak_charges']}\n"
            f"Average SOC: {metrics['average_soc']:.1f}%\n"
            f"Time in Optimal Range: {metrics['optimal_range_time']:.1f}%"
        )
        
        plt.text(0.02, 0.98, stats_text,
                transform=ax1.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         edgecolor='#CCCCCC',
                         alpha=0.9))
    
    # Add overall title
    plt.suptitle('Agent Learning Progress: Early vs Late Episode Comparison', fontsize=16, y=1.02)
    
    # Add learning progress summary
    summary_text = (
        f"Learning Progress:\n"
        f"Optimal Range Time: {learning_metrics[10]['optimal_range_time']:.1f}% → {learning_metrics[99]['optimal_range_time']:.1f}%\n"
        f"Strategic Peak Discharges: {learning_metrics[10]['peak_discharges']} → {learning_metrics[99]['peak_discharges']}\n"
        f"Strategic Off-Peak Charges: {learning_metrics[10]['off_peak_charges']} → {learning_metrics[99]['off_peak_charges']}"
    )
    
    plt.figtext(0.02, 0.02, summary_text,
                fontsize=12,
                bbox=dict(boxstyle='round',
                         facecolor='lightgray',
                         edgecolor='gray',
                         alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def plot_convergence_to_min_cost(episode_rewards, episode_costs, theoretical_min_cost, save_path=None):
    """
    Plot both rewards and costs compared to theoretical minimum.
    
    Args:
        episode_rewards (list): List of rewards for each episode
        episode_costs (list): List of costs for each episode
        theoretical_min_cost (float): The theoretical minimum cost calculated
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(range(len(episode_rewards)), episode_rewards, 
            alpha=0.2, color='gray', label='Raw Rewards')
    
    # Calculate and plot reward moving average
    window_size = 100
    if len(episode_rewards) >= window_size:
        reward_ma = np.convolve(episode_rewards, 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        ax1.plot(range(window_size-1, len(episode_rewards)), 
                reward_ma, 
                color='#2196F3', 
                linewidth=2, 
                label=f'{window_size}-Episode Moving Average')
    
    ax1.set_title('Agent Rewards Over Time', fontsize=14)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot costs
    ax2.plot(range(len(episode_costs)), episode_costs, 
            alpha=0.2, color='gray', label='Actual Costs')
    
    # Calculate and plot cost moving average
    if len(episode_costs) >= window_size:
        cost_ma = np.convolve(episode_costs, 
                             np.ones(window_size)/window_size, 
                             mode='valid')
        ax2.plot(range(window_size-1, len(episode_costs)), 
                cost_ma, 
                color='#2196F3', 
                linewidth=2, 
                label=f'{window_size}-Episode Moving Average')
    
    # Plot minimum cost line
    ax2.axhline(y=theoretical_min_cost, color='#4CAF50', linestyle='--', 
                label='Theoretical Minimum Cost')
    
    # Calculate convergence metrics
    if len(episode_costs) >= window_size:
        final_avg_cost = np.mean(episode_costs[-window_size:])
        cost_gap = ((final_avg_cost - absolute_min_cost) / abs(absolute_min_cost)) * 100
        
        stats_text = (
            f'Convergence Metrics:\n'
            f'Theoretical Minimum: £{theoretical_min_cost:.2f}\n'
            f'Final Avg Cost: £{final_avg_cost:.2f}\n'
            f'Gap to Minimum: {cost_gap:.1f}%'
        )
        
        ax2.text(0.02, 0.98, stats_text,
                transform=ax2.transAxes,
                verticalalignment='top',
                fontsize=10,
                bbox=dict(boxstyle='round',
                         facecolor='white',
                         edgecolor='#CCCCCC',
                         alpha=0.9))
    
    ax2.set_title('Agent Costs vs. Theoretical Minimum', fontsize=14)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Cost (£)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def main():
    # Initialize environment and agent
    microgrid_env = MicrogridState()
    agent = QLearningAgent(
        n_states=5*3*3*4,
        n_actions=2,
        learning_rate=0.2,
        discount_factor=0.99,
        epsilon=1.0
    )
    
    # Training parameters
    n_episodes = 100
    max_steps_per_episode = len(load.time_series)
    
    # Define load demand and PV generation
    load_demand = [30] * 24
    pv_generation = 50*np.random.rand(24)
    
    # Calculate theoretical minimum cost
    theoretical_min_cost = microgrid_env.min_cost(pv_generation, load_demand)
    
    # Train the agent
    try:
        episode_rewards, episode_costs, battery_data = train_microgrid(
            env=microgrid_env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps_per_episode
        )
        
        # Save the Q-table
        np.save('q_table.npy', agent.q_table)
        
        # Create plots
        training_fig = plot_training_progress(episode_rewards, 
                                           save_path=os.path.join(log_dir, 'training_progress.png'))
        
        soc_fig = plot_soc_across_episodes(
            microgrid_env,
            battery_data,
            save_path=os.path.join(log_dir, 'soc_across_episodes.png')
        )
        
        tariff_battery_fig = plot_tariff_battery_relationship(
            microgrid_env,
            battery_data,
            save_path=os.path.join(log_dir, 'tariff_battery_relationship.png')
        )
        
        actions_soc_fig = plot_battery_actions_soc_comparison(
            microgrid_env,
            battery_data,
            save_path=os.path.join(log_dir, 'battery_actions_soc.png')
        )
        
        # Pass the theoretical_min_cost to the convergence plot
        convergence_fig = plot_convergence_to_min_cost(
            episode_rewards,
            episode_costs,
            theoretical_min_cost,  # Now properly defined and passed
            save_path=os.path.join(log_dir, 'convergence.png')
        )
        
        plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
