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
    max_charge=60,     # Maximum charging rate in kW
    max_discharge=30,  # Maximum discharging rate in kW                                                                                                      
    efficiency=1.0,    # Battery efficiency
    init_soc=1      # Initial state of charge (50%)
)
# Create renewable (PV) module with specific generation profile
renewable = RenewableModule(
    time_series=[0]*24  # Replace with your PV generation data
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

       
class MicrogridState:
    def __init__(self):
        self.csv_headers = ['Episode', 'Step', 'Battery_SOC', 'PV_Power', 'Load_Demand', 
                       'Time_Hour', 'Net_Load', 'Action', 'Reward',]
        
        self.battery_data = []
        self.state_space = {
            'battery_soc': 0.0,
            'pv_power': 0.0,
            'load_demand': 0.0,
            'time_hour': 0,
            'net_load': 0.0,
            'battery_health': 100.0
        }
        self.SOC_MIN = 0
        self.SOC_OPTIMAL_MIN = 0.1
        self.SOC_OPTIMAL_MAX = 0.9
        self.SOC_MAX = 1
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
                'injection': 0.11     # 10 pence per kWh for on-peak injection
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
        
        return self.discretize_state()
    
    def discretize_state(self):
            # Battery state discretization
       if self.state_space['battery_soc'] < self.SOC_MIN:
            battery_state = 0
       elif self.state_space['battery_soc'] < 0.1:
            battery_state = 1
       elif self.state_space['battery_soc'] < 0.3:
            battery_state = 2
       elif self.state_space['battery_soc'] < 0.5:
            battery_state = 3
       elif self.state_space['battery_soc'] < 0.7:
            battery_state = 4
       elif self.state_space['battery_soc'] < 0.9:
            battery_state = 5
       else:
            battery_state = 6
        # PV state is always 0 since we only have one state
       pv_state = 0
        
        # Load state is always 0 since we only have one state
       load_state = 0
       
       hour = self.state_space['time_hour']
    
       if 0 <= hour < 7:  # Night/Early Morning Off-Peak
            time_state = 0
       elif 7 <= hour < 11:  # Morning Peak
            time_state = 1
       elif 11 <= hour < 17:  # Midday Off-Peak
            time_state = 2
       elif 17 <= hour < 21:  # Evening Peak
            time_state = 3
       else:  # Late Night Off-Peak (21-24)
            time_state = 4
    
            
       return (battery_state, pv_state, load_state, time_state)
    
    def setup_debug_logging():
        """Setup debug logging configuration"""
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create file handler
        fh = logging.FileHandler(f'logs/debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        # Setup logger
        logger = logging.getLogger('microgrid_debug')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def print_state_transition(logger, step, episode, state_space, action, reward, next_state, control_dict):
        """Print detailed state transition information"""
        logger.debug(f"\n{'='*80}")
        logger.debug(f"Episode: {episode} | Step: {step}")
        logger.debug(f"{'='*80}")
        
        # Current State Information
        logger.debug("\nCURRENT STATE:")
        logger.debug(f"Time: {state_space['time_hour']:02d}:00")
        logger.debug(f"Battery SOC: {state_space['battery_soc']*100:.1f}%")
        logger.debug(f"PV Power: {state_space['pv_power']:.2f} kW")
        logger.debug(f"Load Demand: {state_space['load_demand']:.2f} kW")
        logger.debug(f"Net Load: {state_space['net_load']:.2f} kW")
        
        # Action and Control Information
        logger.debug("\nACTION & CONTROL:")
        action_map = {0: "CHARGE", 1: "DISCHARGE", 2: "DO_NOTHING"}
        logger.debug(f"Selected Action: {action_map.get(action, 'UNKNOWN')}")
        logger.debug("\nControl Dictionary:")
        for key, value in control_dict.items():
            logger.debug(f"  {key}: {value:.2f} kW")
        
        # Reward Information
        logger.debug(f"\nREWARD: {reward:.2f}")
        
        # Next State Information
        logger.debug("\nNEXT STATE:")
        battery_state, pv_state, load_state, time_state = next_state
        logger.debug(f"Battery State: {battery_state}")
        logger.debug(f"PV State: {pv_state}")
        logger.debug(f"Load State: {load_state}")
        logger.debug(f"Time State: {time_state}")
        
        # Tariff Information
        is_peak = (7 <= state_space['time_hour'] < 11) or (17 <= state_space['time_hour'] < 21)
        logger.debug(f"\nTariff Period: {'PEAK' if is_peak else 'OFF-PEAK'}")
        
        logger.debug(f"\n{'='*80}\n")

    def print_episode_summary(logger, episode, total_reward, total_cost, epsilon):
        """Print episode summary information"""
        logger.info(f"\n{'#'*80}")
        logger.info(f"Episode {episode} Summary")
        logger.info(f"{'#'*80}")
        logger.info(f"Total Reward: {total_reward:.2f}")
        logger.info(f"Total Cost: £{total_cost:.2f}")
        logger.info(f"Current Epsilon: {epsilon:.4f}")
        logger.info(f"{'#'*80}\n")
           
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
        
        done = self.state_space['time_hour'] > 23
        
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
        
        control_dict = {
            'grid_import': 30,  # Base load always needs to be imported
            'grid_export': 0,
            'battery_charge': 0,
            'battery_discharge': 0,
            'pv_consumed': 0
        }
        
        if action == 0:  # Charge battery from the grid
            action_type = 'CHARGE_BATTERY'
            if current_soc < self.SOC_MAX:
                charge_amount = min(available_to_charge, self.battery.max_charge)
                # Update battery SOC
                new_soc = current_soc + (charge_amount / self.battery.max_capacity)
                self.state_space['battery_soc'] = min(new_soc, self.SOC_MAX)
                
                control_dict.update({
                    'battery_charge': charge_amount,
                    'grid_import': 30 + charge_amount,  # Base load plus charging
                    'battery_discharge': 0,
                    'grid_export': 0
                })
                
        elif action == 1:  # Discharge battery to the grid
            action_type = 'DISCHARGE_BATTERY'
            if current_soc > self.SOC_MIN:
                discharge_amount = min(available_to_discharge, self.battery.max_discharge)
                # Update battery SOC
                new_soc = current_soc - (discharge_amount / self.battery.max_capacity)
                self.state_space['battery_soc'] = max(new_soc, self.SOC_MIN)
                
                control_dict.update({
                    'battery_charge': 0,
                    'battery_discharge': discharge_amount,
                    'grid_import': 30,  # Only base load
                    'grid_export': discharge_amount
                })
                
        else:  # action == 2: Do nothing (new action)
            action_type = 'DO_NOTHING'
            control_dict.update({
                'battery_charge': 0,
                'battery_discharge': 0,
                'grid_import': 30,  # Only base load
                'grid_export': 0
            })
            
        return control_dict

    def calculate_reward(self, control_dict):
        tariff = self.get_current_tariff()
        current_soc = self.state_space['battery_soc']
        is_peak = (7 <= self.state_space['time_hour'] < 11) or (17 <= self.state_space['time_hour'] < 21)
        
        # Base load cost (always need to import 30 kWh)
        base_cost = 30 * tariff['consumption']
        
        # Extra costs from battery charging
        battery_charge_cost = control_dict['battery_charge'] * tariff['consumption']
        
        # Revenue from battery discharging
        battery_export_revenue = control_dict['battery_discharge'] * tariff['injection']
        
        # Calculate total cost (this will always be >= base_cost)
        total_cost = base_cost + battery_charge_cost
        
        # Calculate net cost after considering revenue
        net_cost = total_cost - battery_export_revenue
        
        # Reward is negative of net cost minus any penalties
        reward = -net_cost 
        
        if not is_peak and current_soc < 1 and control_dict['battery_charge'] > 0:
            reward += 20.0 
            
        if not is_peak and current_soc == self.SOC_MAX and control_dict['battery_charge'] == 0 and control_dict['battery_discharge'] == 0:
            reward += 30.0 
    
        # Penalty for discharging during off-peak when SOC below
        if not is_peak and current_soc < 1 and control_dict['battery_discharge'] > 0:
            reward -= 20.0 * control_dict['battery_discharge']
        
        if is_peak and current_soc > 0 and control_dict['battery_charge'] > 0:         #penalty for charging on peak
            reward -= 20.0 * control_dict['battery_charge']
            
        if is_peak and current_soc == 0 and control_dict['battery_charge'] > 0:         #penalty for charging on peak
            reward += 20.0 * control_dict['battery_charge']
    
        # Penalty for discharging during off-peak when SOC is low
        if is_peak and current_soc > 0 and control_dict['battery_discharge'] > 0:
            reward += 20.0 * control_dict['battery_discharge']
            
        return reward, net_cost
        
        
    def min_cost(self, pv_generation, load_demand):
        """
        Calculate the theoretical minimum cost for a full day with detailed debugging output
        """
        total_cost = 0
        
        # Battery parameters
        max_charge_per_hour = self.battery.max_charge      # 60 kW
        max_discharge_per_hour = self.battery.max_discharge # 30 kW
        battery_capacity = self.battery.max_capacity        # 120 kWh
        battery_energy = battery_capacity * 0.5             # Start at 50% SOC
        
        # For debugging
        peak_total = 0
        offpeak_total = 0
        battery_revenue = 0
        battery_cost = 0
        
        print("\n=== Detailed Cost Calculation ===")
        print("Hour | Peak? | Base Load | Battery Action | Battery Level | Hour Cost")
        print("-" * 70)
        
        for hour in range(24):
            # Update time in state space for correct tariff calculation
            self.state_space['time_hour'] = hour
            tariffs = self.get_current_tariff()
            
            is_peak = (7 <= hour < 11) or (17 <= hour < 21)
            
            # Base load cost (always 30kW)
            base_load_cost = 30 * tariffs['consumption']
            hour_cost = base_load_cost
            
            battery_action = "None"
            
            if is_peak:
                # During peak, try to discharge battery
                potential_discharge = min(30, battery_energy)
                if potential_discharge > 0:
                    discharge_revenue = potential_discharge * tariffs['injection']
                    hour_cost -= discharge_revenue
                    battery_energy -= potential_discharge
                    battery_revenue += discharge_revenue
                    battery_action = f"Discharge {potential_discharge:.1f}kW"
                peak_total += hour_cost
            else:
                # During off-peak, charge battery if needed
                space_available = battery_capacity - battery_energy
                potential_charge = min(60, space_available)
                
                if potential_charge > 0 and battery_energy < battery_capacity * 0.9:
                    charge_cost = potential_charge * tariffs['consumption']
                    hour_cost += charge_cost
                    battery_energy += potential_charge
                    battery_cost += charge_cost
                    battery_action = f"Charge {potential_charge:.1f}kW"
                offpeak_total += hour_cost
            
            total_cost += hour_cost
            
            print(f"{hour:2d}   | {is_peak!r:6} | £{base_load_cost:6.2f} | {battery_action:13} | {battery_energy:6.1f}kWh | £{hour_cost:6.2f}")

        print("\n=== Summary ===")
        print(f"Peak Hours Total: £{peak_total:.2f}")
        print(f"Off-Peak Hours Total: £{offpeak_total:.2f}")
        print(f"Battery Charging Cost: £{battery_cost:.2f}")
        print(f"Battery Discharge Revenue: £{battery_revenue:.2f}")
        print(f"Net Battery Impact: £{battery_revenue - battery_cost:.2f}")
        print(f"Final Total Cost: £{total_cost:.2f}")
        
        return total_cost

class QLearningAgent:
    def __init__(self, n_states, n_actions, learning_rate, discount_factor, epsilon):
        # Calculate the actual state space size based on the discrete state components
        self.battery_states = 7# 0 to 4
        self.pv_states = 1      # 0 
        self.load_states = 1   # 0 
        self.time_states = 5  # 0 to 4
        
        # Calculate total number of states
        self.n_states = self.battery_states * self.pv_states * self.load_states * self.time_states
        
        # Initialize Q-table with correct dimensions
        self.q_table = np.zeros((self.n_states, n_actions))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        
        self.exploration_rates = []
        self.episode_count = 0

    def get_state_index(self, state):
        """
        Convert state tuple to a single index for Q-table lookup
        state format: (battery_state, pv_state, load_state, time_state)
        """
        try:
            battery, pv, load, time = state    
            # Verify state components are within bounds
            if not (0 <= battery < self.battery_states):
                print(f"Battery state {battery} out of bounds [0, {self.battery_states})")
                return 0
            if not (0 <= pv < self.pv_states):
                print(f"PV state {pv} out of bounds [0, {self.pv_states})")
                return 0
            if not (0 <= load < self.load_states):
                print(f"Load state {load} out of bounds [0, {self.load_states})")
                return 0
            if not (0 <= time < self.time_states):
                print(f"Time state {time} out of bounds [0, {self.time_states})")
                return 0
                
            # Calculate unique index using positional value method
            index = (battery * (self.pv_states * self.load_states * self.time_states) + 
                    pv * (self.load_states * self.time_states) + 
                    load * self.time_states + 
                    time)
            
            return index
        
        except Exception as e:
            print(f"Error in get_state_index: {e}")
            print(f"State received: {state}")
            # Return a safe default state index
            return 0      

    def choose_action(self, state):
        try:
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.q_table.shape[1])
            
            state_idx = self.get_state_index(state)
            return int(np.argmax(self.q_table[state_idx, :]))
            
        except Exception as e:
            print(f"Error in choose_action: {e}")
            # Return a safe default action
            return 0

    def learn(self, state, action, reward, next_state):
        try:
            state_idx = self.get_state_index(state)
            next_state_idx = self.get_state_index(next_state)
            
            old_q = self.q_table[state_idx, action]
            next_max_q = np.max(self.q_table[next_state_idx, :])
            
            new_q = (1 - self.learning_rate) * old_q + \
                    self.learning_rate * (reward + self.discount_factor * next_max_q)
            
            self.q_table[state_idx, action] = new_q
            
        except Exception as e:
            print(f"Error in learn: {e}")
            
    def adaptive_epsilon_decay(self, current_cost, min_cost, tolerance=0.05):
        """
        Decay epsilon based on how close we are to the minimum cost
        Args:
            current_cost: The cost achieved in the current episode
            min_cost: The theoretical minimum cost
            tolerance: How close to min_cost we need to be (as a percentage)
        """
        # Calculate how far we are from minimum cost as a percentage
        cost_gap = (current_cost - min_cost) / abs(min_cost)
        
        if cost_gap <= tolerance:
            # If we're within tolerance of minimum cost, decay epsilon faster
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.95)
        else:
            # Otherwise, decay epsilon slowly or not at all
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.999)
        
        self.exploration_rates.append(self.epsilon)
        self.episode_count += 1
        
def train_microgrid(env, agent, n_episodes, max_steps, min_cost):
    episode_rewards = []
    episode_costs = []
    env.battery_data = []
    initial_soc = 1
    
    for episode in range(n_episodes):
        env.current_episode = episode
        env.current_step = 0
        
        # Reset environment state for new episode
        env.state_space = {
            'battery_soc': initial_soc,
            'pv_power': 0.0,
            'load_demand': 30.0,
            'time_hour': 0,
            'net_load': 30.0,
            'battery_health': 100.0
        }
        
        total_reward = 0
        total_cost = 0
        state = env.update_state()
        episode_data = []
        
        for step in range(max_steps):
            step_data = {
                'Episode': episode,
                'Step': step,
                'Time_Hour': env.state_space['time_hour'],
                'Battery_SOC': env.state_space['battery_soc'],
                'PV_Power': env.state_space['pv_power'],
                'Load_Demand': env.state_space['load_demand'],
                'Net_Load': env.state_space['net_load'],
            }
            
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            
            step_data.update({
                'Action': action,
                'Reward': reward,
            })
            episode_data.append(step_data)
            
            total_cost += env.last_cost
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Use adaptive epsilon decay based on performance
        agent.adaptive_epsilon_decay(total_cost, min_cost)
        
        episode_df = pd.DataFrame(episode_data)
        env.battery_data.append(episode_df)
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes} completed. "
                  f"Total reward: {total_reward:.2f}, "
                  f"Cost: {total_cost:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    return episode_rewards, episode_costs, env.battery_data

def debug_agent_decisions(env, agent, episode_data: pd.DataFrame, episode_num: int, output_dir: str = "debug_output"):
    """
    Comprehensive debugging tool for analyzing agent decisions
    
    Args:
        env: MicrogridState environment instance
        agent: QLearningAgent instance
        episode_data: DataFrame containing episode data
        episode_num: Episode number being analyzed
        output_dir: Directory to save debug outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analyze Q-table values
    def analyze_q_values():
        """Analyze the learned Q-values for different states"""
        print("\n=== Q-Value Analysis ===")
        for battery_state in range(3):  # Low, Medium, High SOC
            for time_state in range(2):  # Off-peak, Peak
                state = (battery_state, 0, 0, time_state)
                state_idx = agent.get_state_index(state)
                q_values = agent.q_table[state_idx]
                
                # Convert states to readable format
                battery_desc = {0: "Low SOC", 1: "Medium SOC", 2: "High SOC"}[battery_state]
                time_desc = {0: "Off-Peak", 1: "Peak"}[time_state]
                
                print(f"\nState: {battery_desc}, {time_desc}")
                print(f"Q-values for actions:")
                print(f"  Charge: {q_values[0]:.3f}")
                print(f"  Discharge: {q_values[1]:.3f}")
                print(f"  Do Nothing: {q_values[2]:.3f}")
                print(f"Preferred action: {['Charge', 'Discharge', 'Do Nothing'][np.argmax(q_values)]}")
    
    # 2. Analyze episode decisions
    def analyze_episode_decisions():
        """Analyze the decisions made during the episode"""
        print("\n=== Episode Decision Analysis ===")
        
        # Count actions during peak/off-peak
        peak_hours = episode_data['Time_Hour'].apply(
            lambda x: (7 <= x < 11) or (17 <= x < 21))
        
        peak_actions = episode_data[peak_hours]['Action'].value_counts()
        offpeak_actions = episode_data[~peak_hours]['Action'].value_counts()
        
        print("\nPeak Hours Actions:")
        print(f"Charges: {peak_actions.get(0, 0)}")
        print(f"Discharges: {peak_actions.get(1, 0)}")
        print(f"No Action: {peak_actions.get(2, 0)}")
        
        print("\nOff-Peak Hours Actions:")
        print(f"Charges: {offpeak_actions.get(0, 0)}")
        print(f"Discharges: {offpeak_actions.get(1, 0)}")
        print(f"No Action: {offpeak_actions.get(2, 0)}")
        
        # Analyze SOC management
        print("\nSOC Management:")
        print(f"Average SOC: {episode_data['Battery_SOC'].mean()*100:.1f}%")
        print(f"Min SOC: {episode_data['Battery_SOC'].min()*100:.1f}%")
        print(f"Max SOC: {episode_data['Battery_SOC'].max()*100:.1f}%")
        
        optimal_range = ((episode_data['Battery_SOC'] >= env.SOC_OPTIMAL_MIN) & 
                        (episode_data['Battery_SOC'] <= env.SOC_OPTIMAL_MAX))
        print(f"Time in optimal range: {optimal_range.mean()*100:.1f}%")
    
    # 3. Visualize episode behavior
    def create_episode_visualization():
        """Create detailed visualization of episode behavior"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Battery SOC
        ax1.plot(episode_data['Time_Hour'], episode_data['Battery_SOC']*100, 'b-', linewidth=2)
        ax1.axhline(y=env.SOC_OPTIMAL_MIN*100, color='g', linestyle=':', label='Optimal Min')
        ax1.axhline(y=env.SOC_OPTIMAL_MAX*100, color='g', linestyle=':', label='Optimal Max')
        
        # Shade peak periods
        for start, end in [(7,11), (17,21)]:
            ax1.axvspan(start, end, color='yellow', alpha=0.2)
        
        ax1.set_ylabel('Battery SOC (%)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Actions taken
        colors = {0: 'green', 1: 'red', 2: 'gray'}
        for idx, row in episode_data.iterrows():
            ax2.scatter(row['Time_Hour'], row['Action'], 
                       color=colors[row['Action']], s=100)
        
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Charge', 'Discharge', 'No Action'])
        ax2.set_ylabel('Actions')
        ax2.grid(True)
        
        # Plot 3: Q-values for chosen actions
        for action in range(3):
            q_values = []
            for idx, row in episode_data.iterrows():
                state = (1, 0, 0, 1 if (7 <= row['Time_Hour'] < 11) or 
                        (17 <= row['Time_Hour'] < 21) else 0)
                state_idx = agent.get_state_index(state)
                q_values.append(agent.q_table[state_idx, action])
            ax3.plot(episode_data['Time_Hour'], q_values, 
                    label=['Charge', 'Discharge', 'No Action'][action])
        
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Q-Values')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/episode_{episode_num}_analysis.png")
        plt.close()
    
    # Run all analyses
    print(f"\n{'='*50}")
    print(f"Debugging Episode {episode_num}")
    print(f"{'='*50}")
    
    analyze_q_values()
    analyze_episode_decisions()
    create_episode_visualization()
    
    print(f"\nDebug visualizations saved to {output_dir}/episode_{episode_num}_analysis.png")


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
    
def plot_tariff_battery_relationship(env, battery_data, episodes_to_compare=[1000, 5000], save_path=None):
    """
    Plot injection tariff rates and battery SOC against time, comparing two episodes.
    Fixed version that prevents duplicate plotting of battery SOC.
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
        twin_ax = ax.twinx()  # Create secondary y-axis
        
        # Get episode data
        episode_data = all_data[all_data['Episode'] == episode_num]
        
        # Plot injection tariff on left y-axis (green)
        tariff_line = ax.plot(hours, injection_tariffs, 
                            'g-', 
                            label='Injection Tariff', 
                            linewidth=2.5)
        ax.set_ylabel('Injection Tariff Rate (£/kWh)', color='g', fontsize=12)
        ax.tick_params(axis='y', labelcolor='g')
        
        # Plot battery SOC only on right y-axis (blue)
        soc_line = twin_ax.plot(episode_data['Time_Hour'], 
                              episode_data['Battery_SOC'] * 100, 
                              'b-', 
                              label='Battery SOC', 
                              linewidth=2.5)
        twin_ax.set_ylabel('Battery State of Charge (%)', color='b', fontsize=12)
        twin_ax.tick_params(axis='y', labelcolor='b')
        
        # Add shaded regions for peak hours
        peak_periods = [(7, 11), (17, 21)]
        for start, end in peak_periods:
            ax.axvspan(start, end, color='yellow', alpha=0.2, 
                      label='Peak Hours' if start == 7 else "")
        
        # Set axis limits
        ax.set_xlim(0, 23)
        ax.set_ylim(0, 0.12)
        twin_ax.set_ylim(0, 100)
        
        # Add minor gridlines
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)
        
        # Combine legends
        lines = tariff_line + soc_line
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
        
        if idx == 0:  # Only add xlabel to bottom subplot
            ax.set_xlabel('Time of Day (Hour)', fontsize=12)
    
    # Add overall title
    fig.suptitle('Injection Tariff Rate and Battery SOC vs Time of Day: Learning Comparison', 
                fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def plot_battery_actions_soc_comparison(env, battery_data, episodes_to_compare=[1000, 5000], save_path=None):
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
        
        peak_hours_mask = ((episode_data['Time_Hour'].between(7, 11)) | 
                          (episode_data['Time_Hour'].between(17, 21)))
        
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
            elif row['Action'] == 1: # Discharge
                ax2.bar(row['Time_Hour'], -1, color='red', alpha=0.6, width=0.8)
            else:  #do nothing
                ax2.bar(row['Time_Hour'], 0, color='gray', alpha=0.3, width=0.8)
                
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
        ax2.set_yticks([-1 ,0, 1])
        ax2.set_yticklabels(['Discharge', 'No Action', 'Charge'])
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
        f"Optimal Range Time: {learning_metrics[1000]['optimal_range_time']:.1f}% → {learning_metrics[5000]['optimal_range_time']:.1f}%\n"
        f"Strategic Peak Discharges: {learning_metrics[1000]['peak_discharges']} → {learning_metrics[5000]['peak_discharges']}\n"
        f"Strategic Off-Peak Charges: {learning_metrics[1000]['off_peak_charges']} → {learning_metrics[5000]['off_peak_charges']}"
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
        cost_gap = ((final_avg_cost - theoretical_min_cost) / abs(theoretical_min_cost)) * 100
        
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
        n_states=35,
        n_actions=3,
        learning_rate=0.13,
        discount_factor=0.99,
        epsilon=1.0
    )
    
    n_episodes = 7000
    max_steps_per_episode = 24
    initial_soc = 1
    
    # Calculate theoretical minimum cost
    load_demand = [30] * 24
    pv_generation = [0] * 24
    theoretical_min_cost = microgrid_env.min_cost(pv_generation, load_demand)
    
    try:
        print("Starting training...")
        episode_rewards, episode_costs, battery_data = train_microgrid(
            env=microgrid_env,
            agent=agent,
            n_episodes=n_episodes,
            max_steps=max_steps_per_episode,
            min_cost=theoretical_min_cost
        )
        
        print("Training completed. Creating plots...")
        
        # Debug specific episodes after training is complete
        episodes_to_debug = [0, 1000, 5000]  # Debug start, middle, and end episodes
        for episode_num in episodes_to_debug:
            # Make sure episode_num is within the range of available data
            if episode_num < len(battery_data):
                debug_agent_decisions(
                    env=microgrid_env,  # Pass microgrid_env instead of env
                    agent=agent,
                    episode_data=battery_data[episode_num],
                    episode_num=episode_num
                )
        
        # Create and save plots
        plot_training_progress(episode_rewards, 
                             save_path=os.path.join(log_dir, 'training_progress.png'))
        
        plot_soc_across_episodes(
            microgrid_env,  # Use microgrid_env consistently
            battery_data,
            save_path=os.path.join(log_dir, 'soc_across_episodes.png')
        )
        
        plot_tariff_battery_relationship(
            microgrid_env,  # Use microgrid_env consistently
            battery_data,
            save_path=os.path.join(log_dir, 'tariff_battery_relationship.png')
        )
        
        plot_battery_actions_soc_comparison(
            microgrid_env,  # Use microgrid_env consistently
            battery_data,
            save_path=os.path.join(log_dir, 'battery_actions_soc.png')
        )
        
        plot_convergence_to_min_cost(
            episode_rewards,
            episode_costs,
            theoretical_min_cost,
            save_path=os.path.join(log_dir, 'convergence.png')
        )
        
        print("All plots saved successfully.")
        plt.show()
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

