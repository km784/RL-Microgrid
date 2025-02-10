import sys
import pymgrid
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt 
import numpy as np
import os
import time
from helper_functions import *

def update_state(self):
        # Update battery state of charge
        self.state_space['battery_soc'] = self.battery.soc
        
        # Update PV generation
        self.state_space['pv_power'] = self.renewable.current_power
        
        # Update load demand
        self.state_space['load_demand'] = self.load.current_demand
        
        # Update time
        self.state_space['time_hour'] = (self.state_space['time_hour'] + 1) % 24
        
        # Update net load
        self.state_space['net_load'] = self.state_space['load_demand'] - self.state_space['pv_power']
        
        return self.discretize_state()