def calculate_theoretical_min_cost(self, pv_generation, load_demand):
            """
            Calculate theoretical minimum cost with more realistic constraints
            """
            total_cost = 0
            battery_soc = 0.5 * self.battery.max_capacity  # Initial SOC
            
            # Create hourly periods with associated tariffs
            periods = []
            for hour in range(24):
                is_peak = (7 <= hour < 11) or (17 <= hour < 21)
                tariff = {
                    'hour': hour,
                    'consumption': 0.17 if is_peak else 0.13,
                    'injection': 0.10 if is_peak else 0.07
                }
                # Add degradation cost to consumption price
                tariff['effective_consumption'] = tariff['consumption'] + 0.01  # Account for battery wear
                periods.append(tariff)
            
            # Sort periods by price differential (considering degradation)
            periods.sort(key=lambda x: x['injection'] - x['effective_consumption'], reverse=True)
            
            for period in periods:
                hour = period['hour']
                net_load = load_demand[hour] - pv_generation[hour]
                
                # Consider battery operation costs in decision making
                if period['injection'] > period['effective_consumption']:
                    # Calculate optimal discharge amount
                    max_discharge = min(
                        self.battery.max_discharge,
                        battery_soc - (0.2 * self.battery.max_capacity)
                    )
                    discharge = min(max_discharge, load_demand[hour])
                    battery_soc -= discharge
                    
                    # Include degradation cost in revenue calculation
                    revenue = discharge * (period['injection'] - 0.01)  # Subtract degradation cost
                    total_cost -= revenue
                else:
                    # Calculate optimal charge amount
                    max_charge = min(
                        self.battery.max_charge,
                        (0.9 * self.battery.max_capacity) - battery_soc
                    )
                    charge = min(max_charge, pv_generation[hour])
                    battery_soc += charge
                    
                    # Include degradation cost in charging cost
                    cost = charge * (period['consumption'] + 0.01)  # Add degradation cost
                    total_cost += cost
            
            return total_cost
