def log_state_to_csv(self):
    try:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
        
        # Properly convert NumPy arrays to scalars
        battery_soc = float(self.state_space['battery_soc'].item()) if isinstance(self.state_space['battery_soc'], np.ndarray) else float(self.state_space['battery_soc'])
        pv_power = float(np.asarray(self.state_space['pv_power']).item()) if isinstance(self.state_space['pv_power'], np.ndarray) else float(self.state_space['pv_power'])
        load_demand = float(np.asarray(self.state_space['load_demand']).item()) if isinstance(self.state_space['load_demand'], np.ndarray) else float(self.state_space['load_demand'])
        net_load = float(np.asarray(self.state_space['net_load']).item()) if isinstance(self.state_space['net_load'], np.ndarray) else float(self.state_space['net_load'])
        last_reward = float(np.asarray(self.last_reward).item()) if isinstance(self.last_reward, np.ndarray) else float(self.last_reward)
        
        # Prepare data before writing
        temp_data = [
            self.current_episode,
            self.current_step,
            battery_soc,
            pv_power,
            load_demand,
            int(self.state_space['time_hour']),
            net_load,
            self.last_action,
            last_reward
        ]
        
        # Write with error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(csv_file_path, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if file.tell() == 0:
                        writer.writerow(['Episode', 'Step', 'Battery_SOC', 'PV_Power', 
                                      'Load_Demand', 'Time_Hour', 'Net_Load', 'Action', 'Reward'])
                    writer.writerow(temp_data)
                break
            except PermissionError:
                if attempt == max_retries - 1:
                    print(f"Permission denied. Cannot write to {csv_file_path}")
                    raise
                time.sleep(1)  # Wait before retry
                
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
