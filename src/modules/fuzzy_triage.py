import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math

def build_fuzzy_system():
    heart_rate = ctrl.Antecedent(np.arange(40, 181, 1), 'heart_rate')
    spo2 = ctrl.Antecedent(np.arange(70, 101, 1), 'spo2')
    temperature = ctrl.Antecedent(np.arange(95, 106, 0.1), 'temperature')
    resp_rate = ctrl.Antecedent(np.arange(10, 41, 1), 'respiratory_rate')
    severity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'severity')

    heart_rate['low'] = fuzz.trimf(heart_rate.universe, [40, 60, 80])
    heart_rate['normal'] = fuzz.trimf(heart_rate.universe, [70, 85, 100])
    heart_rate['high'] = fuzz.trimf(heart_rate.universe, [90, 130, 180])

    spo2['low'] = fuzz.trimf(spo2.universe, [70, 85, 92])
    spo2['normal'] = fuzz.trimf(spo2.universe, [90, 96, 100])

    temperature['low'] = fuzz.trimf(temperature.universe, [95, 96.5, 97.5])
    temperature['normal'] = fuzz.trimf(temperature.universe, [97, 98.6, 99.5])
    temperature['high'] = fuzz.trimf(temperature.universe, [99, 101, 106])

    resp_rate['low'] = fuzz.trimf(resp_rate.universe, [10, 12, 16])
    resp_rate['normal'] = fuzz.trimf(resp_rate.universe, [15, 20, 24])
    resp_rate['high'] = fuzz.trimf(resp_rate.universe, [22, 30, 40])

    severity['low'] = fuzz.trimf(severity.universe, [0, 0.2, 0.4])
    severity['medium'] = fuzz.trimf(severity.universe, [0.3, 0.5, 0.7])
    severity['high'] = fuzz.trimf(severity.universe, [0.6, 0.8, 1.0])

    rules = [
        ctrl.Rule(heart_rate['high'] & spo2['low'], severity['high']),
        ctrl.Rule(temperature['high'] & resp_rate['high'], severity['high']),
        ctrl.Rule(heart_rate['normal'] & spo2['normal'] & temperature['normal'] & resp_rate['normal'], severity['low']),
        ctrl.Rule(heart_rate['low'] | spo2['low'], severity['medium']),
        ctrl.Rule(temperature['high'] | resp_rate['high'], severity['medium'])
    ]

    triage_ctrl = ctrl.ControlSystem(rules)
    return triage_ctrl  # return the system, not the simulation yet


def compute_severity(row, system_ctrl):
    """Safely compute severity for one row, even if missing data."""
    try:
        # Create a fresh simulation for each patient
        sim = ctrl.ControlSystemSimulation(system_ctrl)

        # Default values if any vital is missing
        hr = row.get('heart_rate', 80)
        spo2 = row.get('spo2', 95)
        temp = row.get('temperature', 98.6)
        rr = row.get('respiratory_rate', 18)

        # Handle NaN safely
        for v in [hr, spo2, temp, rr]:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return 0.5  # neutral severity if missing

        sim.input['heart_rate'] = float(hr)
        sim.input['spo2'] = float(spo2)
        sim.input['temperature'] = float(temp)
        sim.input['respiratory_rate'] = float(rr)

        sim.compute()
        return float(sim.output.get('severity', 0.5))

    except Exception as e:
        print("Fuzzy error:", e)
        return 0.5  # safe fallback
