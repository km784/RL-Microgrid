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
        theoretical_min = min_cost([current_pv], [current_load])

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
            if 'battery_discharge' in control_dict and control_dict['battery_discharge'] > 0:
                strategic_reward += 15  # Peak discharge reward
            elif 'battery_charge' in control_dict and control_dict['battery_charge'] > 0:
                strategic_reward -= 15  # Penalty for peak charging
        else:
            if 'battery_charge' in control_dict and control_dict['battery_charge'] > 0 and current_soc < self.SOC_OPTIMAL_MAX:
                strategic_reward += 15  # Off-peak charging reward
            elif 'battery_discharge' in control_dict and control_dict['battery_discharge'] > 0:
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
