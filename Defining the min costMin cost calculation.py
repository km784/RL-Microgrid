def min_cost(pv_generation, load_demand):
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
    period_tariffs.sort(key=lambda x: x['injection'] - x['consumption'], reverse=True)
    
    # Calculate optimal battery operation for each hour
    for period in period_tariffs:
        hour = period['hour']
        net_load = load_demand[hour] - pv_generation[hour]
        
        if period['injection'] > period['consumption']:
            # During high-price periods, try to discharge the battery
            max_discharge = min(
                max_discharge_rate,
                current_soc - (0.2 * battery_capacity)  # Maintain minimum 20% SOC
            )
            discharge = min(max_discharge, load_demand[hour])
            current_soc -= discharge
            
            # Calculate revenue from discharging
            revenue = discharge * period['injection']
            total_cost -= revenue
            
        else:
            # During low-price periods, try to charge the battery
            max_charge = min(
                max_charge_rate,
                (0.9 * battery_capacity) - current_soc  # Maintain maximum 90% SOC
            )
            charge = min(max_charge, pv_generation[hour])
            current_soc += charge
            
            # Calculate cost of charging
            cost = charge * period['consumption']
            total_cost += cost
                
    return total_cost