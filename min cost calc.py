def calculate_theoretical_min_cost(env, pv_generation, load_demand):
    """
    Calculate the theoretical minimum cost possible for a 24-hour period
    given perfect foresight and optimal battery operation.
    """
    min_cost = 0
    battery_capacity = 100  # kWh
    max_charge_rate = 50    # kW
    max_discharge_rate = 50 # kW
    battery_soc = 0.5 * battery_capacity  # Start at 50% SOC
    
    # Create arrays for peak and off-peak periods
    hours = range(24)
    peak_periods = [(7, 11), (17, 21)]
    
    # Sort periods by tariff rates for optimal charging/discharging
    period_tariffs = []
    for hour in hours:
        is_peak = any(start <= hour < end for start, end in peak_periods)
        if is_peak:
            tariff = {
                'hour': hour,
                'consumption': 0.17,
                'injection': 0.10
            }
        else:
            tariff = {
                'hour': hour,
                'consumption': 0.13,
                'injection': 0.07
            }
        period_tariffs.append(tariff)
    
    # Sort periods by price differential (best times to charge/discharge)
    period_tariffs.sort(key=lambda x: x['injection'] - x['consumption'], reverse=True)
    
    # Calculate optimal battery operation for each hour
    for period in period_tariffs:
        hour = period['hour']
        net_load = load_demand[hour] - pv_generation[hour]
        
        if period['injection'] > period['consumption']:
            # Discharge during high price periods
            max_discharge = min(
                max_discharge_rate,
                battery_soc - (env.SOC_MIN * battery_capacity)
            )
            discharge = min(max_discharge, load_demand[hour])
            battery_soc -= discharge
            revenue = discharge * period['injection']
            min_cost -= revenue
        else:
            # Charge during low price periods
            max_charge = min(
                max_charge_rate,
                (env.SOC_MAX * battery_capacity) - battery_soc
            )
            charge = min(max_charge, pv_generation[hour])
            battery_soc += charge
            cost = charge * period['consumption']
            min_cost += cost
            
    return min_cost