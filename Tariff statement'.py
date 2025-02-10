def get_current_tariff(self):
    hour = self.state_space['time_hour']
    if (7 <= hour < 11) or (17 <= hour < 21):
        return {
            'consumption': 0.17,  # 17 pence per kWh for on-peak consumption
            'injection': 0.10     # 10 pence per kWh for on-peak injection
        }
    else:
        return {
            'consumption': 0.13,  # 13 pence per kWh for off-peak consumption
            'injection': 0.07     # 7 pence per kWh for off-peak injection
        }
