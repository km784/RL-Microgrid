def calculate_reward(self, control_dict):
        """
        Modified reward calculation to better incentivize peak-time discharging
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
        
        # Calculate cost-based reward with increased weight for revenue
        cost_difference = actual_cost - theoretical_min
        if grid_revenue > 0:  # Incentivize grid export
            cost_reward = 150 * (grid_revenue / (tariff['injection'] * self.battery.max_discharge))
        else:
            cost_reward = 100 * (1 - cost_difference / theoretical_min)
        
        # Strategic battery management rewards with higher peak discharge incentive
        strategic_reward = 0
        current_soc = self.state_space['battery_soc']
        hour = self.state_space['time_hour']
        
        # Peak hours check (7-11 and 17-21)
        is_peak = (7 <= hour < 11) or (17 <= hour < 21)
        
        if is_peak:
            if 'battery_discharge' in control_dict and control_dict['battery_discharge'] > 0:
                # Increased reward for peak discharge
                strategic_reward += 40 * (control_dict['battery_discharge'] / self.battery.max_discharge)
            elif 'battery_charge' in control_dict and control_dict['battery_charge'] > 0:
                strategic_reward -= 25  # Increased penalty for peak charging
        else:
            if 'battery_charge' in control_dict and control_dict['battery_charge'] > 0 and current_soc < self.SOC_OPTIMAL_MAX:
                strategic_reward += 20  # Increased off-peak charging reward
            elif 'battery_discharge' in control_dict and control_dict['battery_discharge'] > 0:
                strategic_reward -= 15  # Increased penalty for off-peak discharge
        
        # Modified SOC reward with reduced penalties for peak-time discharge
        soc_reward = 0
        if is_peak:
            # Reduced SOC penalties during peak hours to encourage discharge
            if self.SOC_MIN <= current_soc <= self.SOC_MAX:
                soc_reward += 10
            else:
                soc_reward -= 20
        else:
            # Normal SOC management during off-peak hours
            if self.SOC_OPTIMAL_MIN <= current_soc <= self.SOC_OPTIMAL_MAX:
                soc_reward += 15
            else:
                distance_from_optimal = min(
                    abs(current_soc - self.SOC_OPTIMAL_MIN),
                    abs(current_soc - self.SOC_OPTIMAL_MAX)
                )
                soc_reward -= 15 * (distance_from_optimal / 0.1)
        
        # Combine rewards with adjusted weights
        total_reward = (
            cost_reward * 1.5 +      # Increased weight for cost/revenue
            strategic_reward * 1.2 +  # Increased weight for strategic actions
            soc_reward * 0.6         # Reduced weight for SOC management
        )
        
        # Smooth clipping of rewards
        total_reward = np.clip(total_reward, -100, 100)
        
        return total_reward, actual_cost
