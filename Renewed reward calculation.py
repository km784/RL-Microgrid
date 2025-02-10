def calculate_reward(self, control_dict):
      tariff = self.get_current_tariff()
        current_soc = self.state_space['battery_soc']
        
        # Calculate grid costs and revenue
        grid_cost = control_dict['grid_import'] * tariff['consumption']
        grid_revenue = control_dict['grid_export'] * tariff['injection']
        
        # Introduce a more aggressive cost-based penalty
        # Calculate the deviation from the theoretical minimum cost
        theoretical_min_cost = 10  # This would be dynamically calculated based on your min_cost function
        actual_cost = grid_cost - grid_revenue
        
        # Cost penalty factor - increases quadratically with cost deviation
        cost_penalty = max(0, (actual_cost - theoretical_min_cost) ** 2 * 0.1)
        
        # Strategic operation rewards
        strategic_reward = 0
        hour = self.state_space['time_hour']
        
        # Reward for charging during off-peak and discharging during peak hours
        if (7 <= hour < 11) or (17 <= hour < 21):  # Peak hours
            if control_dict['battery_discharge'] > 0:
                strategic_reward += 5.0
        else:  # Off-peak hours
            if control_dict['battery_charge'] > 0:
                strategic_reward += 3.0
        
        # SOC management rewards and penalties
        soc_reward = 0
        if self.SOC_OPTIMAL_MIN <= current_soc <= self.SOC_OPTIMAL_MAX:
            soc_reward += 2.0
        elif current_soc < self.SOC_MIN or current_soc > self.SOC_MAX:
            soc_reward -= 1.0 * abs(current_soc - 0.5)  # Penalty increases with distance from optimal SOC
        
        # Combine reward components
        total_reward = (
            10.0  # Base reward
            + strategic_reward  # Reward for smart charging/discharging
            + soc_reward  # SOC management reward
            - cost_penalty  # Aggressive cost penalty
        )
        
        return max(0, total_reward), actual_cost