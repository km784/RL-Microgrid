def calculate_reward(self, control_dict):
        tariff = self.get_current_tariff()
        
        # Mandatory base load - always need to import 30 kWh
        base_cost = 30 * tariff['consumption']
        
        # Battery operations
        battery_charge_cost = control_dict['battery_charge'] * tariff['consumption']
        battery_export_revenue = control_dict['battery_discharge'] * tariff['injection']
        
        # Total cost: base load + battery charging - export revenue
        total_cost = base_cost + battery_charge_cost - battery_export_revenue
        
        
        reward = -total cost + penalty
        
        return -total_cost, total_cost
        total_reward = np.clip(total_reward, -100, 100)

        return total_reward, actual_cost
