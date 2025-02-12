def min_cost(self, pv_generation, load_demand):
        """
        Calculate the theoretical minimum cost with no PV generation
        """
        pv_generation = np.array(pv_generation)
        load_demand = np.array(load_demand)

        absolute_min_cost = 0

        # Check if we're calculating for a single timestep or full day
        if len(load_demand) == 1:
            # Single timestep calculation
            hour = self.state_space['time_hour']
            is_peak = (7 <= hour < 11) or (17 <= hour < 21)
            rate = 0.17 if is_peak else 0.13
            injection_rate = 0.10 if is_peak else 0.07# Peak or off-peak rate
            
            net_load = load_demand[hour]
            
            if net_load > 0:  # Buy from grid
                absolute_min_cost += net_load * consumption_rate
            else:  # Sell excess to grid
                absolute_min_cost += net_load * injection_rate  # Negative cost = revenue
            
            # Full day calculation
            for hour in range(24):
                is_peak = (7 <= hour < 11) or (17 <= hour < 21)
                rate = 0.17 if is_peak else 0.13  # Peak or off-peak rate
                absolute_min_cost += load_demand[hour] * rate

        return absolute_min_cost
