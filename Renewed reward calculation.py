def calculate_reward(self, control_dict):
        tariff = self.get_current_tariff()
        
        # Mandatory base load - always need to import 30 kWh
        base_cost = 30 * tariff['consumption']
        
        # Battery operations
        battery_charge_cost = control_dict['battery_charge'] * tariff['consumption']
        battery_export_revenue = control_dict['battery_discharge'] * tariff['injection']
        
        # Total cost: base load + battery charging - export revenue
        total_cost = base_cost + battery_charge_cost - battery_export_revenue
        
        penalty = 0
        current_soc = self.state_space['battery_soc']

        if current_soc < self.SOC_OPTIMAL_MIN:
            penalty += (self.SOC_OPTIMAL_MIN - current_soc) * 10  # Example weight for penalty
        elif current_soc > self.SOC_OPTIMAL_MAX:
            penalty += (current_soc - self.SOC_OPTIMAL_MAX) * 10  
        
        reward = -total_cost - penalty
        
        return reward, total_cost
