 def actions_agent(self, action):
        # Log initial state
        net_load = self.state_space['load_demand'] - self.state_space['pv_power']
        current_soc = self.state_space['battery_soc']
        new_soc = current_soc
        # Calculate actual available energy in kWh based on current SOC
        actual_energy_available = current_soc * self.battery.max_capacity
        
        # Calculate maximum possible discharge considering both power and energy limits
        max_discharge_power = min(
            self.battery.max_discharge,  # Power limit (kW)
            actual_energy_available,     # Energy limit (kWh) for one hour
            (current_soc - self.SOC_MIN) * self.battery.max_capacity  # Energy above minimum SOC
        )
        
        # Calculate available capacity for charging
        available_to_charge = min(
            self.battery.max_charge,
            (self.SOC_MAX - current_soc) * self.battery.max_capacity
        )
        
        control_dict = {
            'grid_import': 30,  # Base load always needs to be imported
            'grid_export': 0,
            'battery_charge': 0,
            'battery_discharge': 0,
            'pv_consumed': 0
        }
        
        discharge_amount = 0
        
        if self.current_episode == 14990:
            print(f"\n=== Episode 14990 Action Debug ===")
            print(f"Time: {self.state_space['time_hour']:02d}:00")
            print(f"Current SOC: {current_soc * 100:.2f}%")
            print(f"Available Energy: {actual_energy_available:.2f} kWh")
            print(f"Max Discharge Power: {max_discharge_power:.2f} kW")
            print(f"Selected Action: {action}")
            
        if action == 0:  # Charge battery from the grid
            action_type = 'CHARGE_BATTERY'
            if current_soc < self.SOC_MAX:
                charge_amount = min(available_to_charge, self.battery.max_charge)
                # Update battery SOC
                new_soc = current_soc + (charge_amount / self.battery.max_capacity)						#look at this action
                self.state_space['battery_soc'] = min(new_soc, self.SOC_MAX)                       #change this equation
                
                control_dict.update({
                    'battery_charge': charge_amount,
                    'grid_import': 30 + charge_amount,  # Base load plus charging
                    'battery_discharge': 0,
                    'grid_export': 0
                })
                
        elif action == 1:  # Discharge battery to the grid
            action_type = 'DISCHARGE_BATTERY'
            if current_soc > self.SOC_MIN and max_discharge_power > 0 and current_soc * self.battery.max_capacity > 0:
                # Calculate actual available energy in kWh
                available_energy = current_soc * self.battery.max_capacity
                # Use the minimum between max discharge power and available energy
                discharge_amount = min(max_discharge_power, available_energy)
                
                new_soc = current_soc - (discharge_amount / self.battery.max_capacity)
                self.state_space['battery_soc'] = max(new_soc, self.SOC_MIN)
                
                control_dict.update({
                    'battery_charge': 0,
                    'battery_discharge': discharge_amount,
                    'grid_import': 30,  # Only base load
                    'grid_export': discharge_amount  # Export matches actual discharge
                })
                
            if self.current_episode == 14990:
                print(f"Discharge Amount: {discharge_amount:.2f} kW")
                print(f"New SOC: {new_soc * 100:.2f}%")
                print(f"Grid Export: {discharge_amount:.2f} kW")
                
        else:  # action == 2: Do nothing
            action_type = 'DO_NOTHING'
            control_dict.update({
                'battery_charge': 0,
                'battery_discharge': 0,
                'grid_import': 30,  # Only base load
                'grid_export': 0
            })
               
        if self.current_episode == 14990:
            print("Final Control Values:")
            for key, value in control_dict.items():
                print(f"{key}: {value:.2f} kW")
            print("=" * 50)
                
        return control_dict
