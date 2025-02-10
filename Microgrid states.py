import sys
import pymgrid
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from helper_functions import *

class MicrogridState:
    def __init__(self):
        self.state_space = {
            'battery_soc': None,      # Current battery charge level (0-1)
            'pv_power': None,         # Current PV generation
            'load_demand': None,      # Current load demand
            'time_hour': None,        # Hour of day (0-23)
            'net_load': None          # Load demand minus PV generation
        }

    def discretize_state(self):    # battery SOC dsicretization (0-1 ranges into 3 levels)
        if self.state_space['battery_soc'] < 0.3:
            battery_state = 0    # Low
        elif self.state_space['battery_soc'] < 0.7:
            battery_state = 1    # Medium
        else:
            battery_state = 2    # High
            
        # PV power discretization (based on 50kW max from your initialization)
        if self.state_space['pv_power'] < 20:
            pv_state = 0        # Low generation
        elif self.state_space['pv_power'] < 35:
            pv_state = 1        # Medium generation
        else:
            pv_state = 2        # High generation
            
        # Load demand discretization (based on 60kW max from your initialization)
        if self.state_space['load_demand'] < 25:
            load_state = 0      # Low demand
        elif self.state_space['load_demand'] < 45:
            load_state = 1      # Medium demand
        else:
            load_state = 2      # High demand
            
        # Time discretization (24 hours into 4 periods)
        if self.state_space['time_hour'] < 6:
            time_state = 0      # Night
        elif self.state_space['time_hour'] < 12:
            time_state = 1      # Morning
        elif self.state_space['time_hour'] < 18:
            time_state = 2      # Afternoon
        else:
            time_state = 3      # Evening
            
        # Return discretized state as a tuple
        return (battery_state, pv_state, load_state, time_state)