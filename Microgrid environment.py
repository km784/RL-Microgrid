import sys
import pymgrid
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from helper_functions import *

env = Microgrid([]) #Microgrid environment

# Create battery module with specific parameters
battery = BatteryModule(
    min_capacity=0,
    max_capacity=100,  # Maximum battery capacity in kWh
    max_charge=50,     # Maximum charging rate in kW
    max_discharge=50,  # Maximum discharging rate in kW
    efficiency=1.0,    # Battery efficiency
    init_soc=0.5       # Initial state of charge (50%)
)

# Create renewable (PV) module with specific generation profile
renewable = RenewableModule(
    time_series=50*np.random.rand(100)  # Replace with your PV generation data
)

# Create load module with specific consumption profile
load = LoadModule(
    time_series=60*np.random.rand(100),  # Replace with your load demand data
)

# Create and initialize the microgrid
microgrid = Microgrid([
    battery,
    ("pv", renewable),
    load
])