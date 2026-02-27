def calculate_cost(energy_kwh, tariff):
    return energy_kwh * tariff

def calculate_carbon(energy_kwh, emission_factor):
    return energy_kwh * emission_factor