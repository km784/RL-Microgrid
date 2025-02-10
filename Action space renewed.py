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

    def actions_agent(self, action):
    # Log initial state
        net_load = self.state_space['load_demand'] - self.state_space['pv_power']
    
        capa_to_charge = self.battery.max_charge
        p_charge_max = self.battery.max_charge
        p_charge = max(0, min(-net_load, capa_to_charge, p_charge_max))
        
        capa_to_discharge = self.battery.max_discharge
        p_discharge_max = self.battery.max_discharge
        p_discharge = max(0, min(net_load, capa_to_discharge, p_discharge_max))
    
        control_dict = {}
        action_type = ''
        
          
        if action == 0:  # Charge battery from the grid
            action_type = 'CHARGE_BATTERY'
            charge_amount = min(capa_to_charge, p_charge_max)
            control_dict = {
                'pv_consumed': 0,
                'battery_charge': charge_amount,
                'battery_discharge': 0,
                'grid_import': charge_amount,
                'grid_export': 0
            }
            logger.info(f"Executing action: {action_type}")
            logger.debug(f"Control dictionary: {control_dict}")
                    
        elif action == 1:  # Discharge battery to the grid
            action_type = 'DISCHARGE_BATTERY'
            discharge_amount = min(capa_to_discharge, p_discharge_max)
            control_dict = {
                'pv_consumed': 0,
                'battery_charge': 0,
                'battery_discharge': discharge_amount,
                'grid_import': 0,
                'grid_export': discharge_amount
            }