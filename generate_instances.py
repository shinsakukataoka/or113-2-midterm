import pandas as pd
import numpy as np
import os
import math

# BASE_DATA_FILE = "OR113-2_midtermProject_data.xlsx"
INSTANCE_FOLDER = "instances"
NUM_INSTANCES_PER_SCENARIO = 30
NUM_SCENARIOS = 7
N_PRODUCTS = 10
T_MONTHS = 6 # March to August

def load_base_data():
    data = {}
    # Couldnt' read from Excel, so manually defining the data
    # Product Initial inventory Monthly forecast demands (Mar Apr May Jun Jul Aug)
    # 1 800 138 55 172 194 94 185
    data['initial_inventory'] = np.array([
        800, 600, 425, 350, 400, 524, 453, 218, 673, 200
    ]).astype(float)
    data['demand'] = np.array([
        # Mar Apr May Jun Jul Aug
        [138, 55, 172, 194, 94, 185], # Prod 1
        [190, 101, 68, 185, 13, 136],  # Prod 2
        [79, 179, 21, 49, 199, 200],   # Prod 3
        [142, 103, 78, 131, 146, 155], # Prod 4
        [35, 62, 83, 90, 197, 49],    # Prod 5
        [91, 95, 107, 127, 116, 183],  # Prod 6
        [105, 164, 19, 116, 119, 175], # Prod 7
        [37, 155, 10, 77, 168, 32],    # Prod 8
        [108, 185, 188, 176, 81, 172], # Prod 9
        [46, 178, 162, 200, 154, 199]  # Prod 10
    ]).astype(float)
    # --- Data from 'Inventory cost' Sheet ---
    # Product Purchasing cost Holding cost
    # 1 5000 100
    # ... (10 products)
    # Holding cost = 2% of purchasing cost, as stated in text (verify table values)
    # Table values: 100, 40, 180, 180, 40, 180, 140, 100, 180, 140
    # Purchase costs: 5k, 2k, 9k, 9k, 2k, 9k, 7k, 5k, 9k, 7k
    # 2% rule check: 5k*0.02=100, 2k*0.02=40, 9k*0.02=180 - It matches!
    data['purchase_cost_per_unit'] = np.array([
        5000, 2000, 9000, 9000, 2000, 9000, 7000, 5000, 9000, 7000
    ]).astype(float)
    data['holding_cost_per_unit'] = np.array([
        100, 40, 180, 180, 40, 180, 140, 100, 180, 140
    ]).astype(float)
    # Ignoring Sales price, Backorder cost/percentage for now

    # --- Data from 'Shipping cost' Sheet ---
    # Product Express delivery Air freight Cubic meter
    # 1 44 18 0.073
    # ... (10 products)
    # Variable costs for Express (j=0) and Air (j=1)
    data['shipping_variable_cost'] = np.array([
        # Express, Air
        [44, 18],  # Prod 1
        [89, 45],  # Prod 2
        [86, 38],  # Prod 3
        [91, 46],  # Prod 4
        [50, 21],  # Prod 5
        [51, 25],  # Prod 6
        [83, 46],  # Prod 7
        [96, 49],  # Prod 8
        [80, 35],  # Prod 9
        [49, 20]   # Prod 10
    ]).astype(float)

    data['volume_cbm'] = np.array([
        0.073, 0.005, 0.043, 0.063, 0.045, 0.086, 0.079, 0.082, 0.068, 0.098
    ]).astype(float)

    # --- Data from 'In-transit' Sheet ---
    # Product End of March End of April
    # 1 0 0
    # ... (10 products)
    # Map to arrivals at START of month t. T_MONTHS = 6 (0..5)
    # End of March -> arrives start April (t=1)
    # End of April -> arrives start May (t=2)
    in_transit_raw = {
        1: [0, 0], 2: [48, 0], 3: [0, 20], 4: [153, 0], 5: [0, 0],
        6: [18, 23], 7: [28, 45], 8: [0, 0], 9: [109, 34], 10: [0, 0]
    }
    in_transit_array = np.zeros((N_PRODUCTS, T_MONTHS))
    for i in range(N_PRODUCTS):
        prod_id = i + 1
        if prod_id in in_transit_raw:
            # Assign to correct columns (t=1 for April, t=2 for May)
            in_transit_array[i, 1] = in_transit_raw[prod_id][0] # End of March -> arrives start t=1
            in_transit_array[i, 2] = in_transit_raw[prod_id][1] # End of April -> arrives start t=2
    data['in_transit'] = in_transit_array.astype(float)

    # --- Fixed Costs and Parameters (from problem description/previous code) ---
    # Fixed Costs F: Express, Air, Ocean
    data['shipping_fixed_cost'] = np.array([100.0, 80.0, 50.0])
    # Ocean container cost C
    data['ocean_container_cost'] = 2750.0
    # Ocean container capacity V_cont
    data['ocean_container_capacity_cbm'] = 30.0
    # Lead Times T_lead: Express, Air, Ocean (in months)
    data['lead_times'] = np.array([1, 2, 3]) # months

    # Add N_PRODUCTS and T_MONTHS for potential use in solvers
    data['N_PRODUCTS'] = N_PRODUCTS
    data['T_MONTHS'] = T_MONTHS

    print("Base data loaded manually.") # Confirmation message
    return data

# --- Instance Generation Logic (Remains the same) ---
def generate_instance_data(base_data, scenario_id, instance_id):
    """Generates data for a specific instance based on scenario."""
    data = {}
    # Deep copy necessary parts of base_data to avoid modifying it across instances
    for key, val in base_data.items():
        if isinstance(val, np.ndarray):
            data[key] = val.copy()
        else:
            # Copy other types like numbers, lists if necessary
            # For this structure, most are ndarrays or fixed numbers
             data[key] = val # Simple assignment for numbers/primitives

    rng = np.random.RandomState(seed=scenario_id * 100 + instance_id) # Seed for reproducibility

    # --- Apply Scenario Variations ---
    # Scenario 1: Base Case (add minor noise)
    if scenario_id == 1:
        data['demand'] = np.maximum(0, data['demand'] * rng.uniform(0.95, 1.05, size=data['demand'].shape)).astype(int)
        data['initial_inventory'] = np.maximum(0, data['initial_inventory'] * rng.uniform(0.95, 1.05)).astype(int)

    # Scenario 2: High Demand Variability
    elif scenario_id == 2:
        data['demand'] = np.maximum(0, data['demand'] * rng.uniform(0.8, 1.2, size=data['demand'].shape)).astype(int)

    # Scenario 3: Low Demand Variability
    elif scenario_id == 3:
        data['demand'] = np.maximum(0, data['demand'] * rng.uniform(0.98, 1.02, size=data['demand'].shape)).astype(int)

    # Scenario 4: High Holding Costs
    elif scenario_id == 4:
        # Ensure holding cost is copied before modification if it's an array
        data['holding_cost_per_unit'] = data['holding_cost_per_unit'] * rng.uniform(1.5, 2.0)

    # Scenario 5: High Shipping Costs (Variable + Fixed)
    elif scenario_id == 5:
        data['shipping_variable_cost'] = data['shipping_variable_cost'] * rng.uniform(1.2, 1.5)
        data['shipping_fixed_cost'] = data['shipping_fixed_cost'] * rng.uniform(1.2, 1.5)
        data['ocean_container_cost'] = data['ocean_container_cost'] * rng.uniform(1.2, 1.5)

    # Scenario 6: Low Initial Inventory
    elif scenario_id == 6:
        data['initial_inventory'] = np.maximum(0, data['initial_inventory'] * rng.uniform(0.3, 0.6)).astype(int)

    # Scenario 7: High Initial Inventory
    elif scenario_id == 7:
        data['initial_inventory'] = np.maximum(0, data['initial_inventory'] * rng.uniform(1.5, 2.0)).astype(int)

    # --- Ensure data consistency (optional checks) ---
    # e.g., ensure costs are non-negative
    data['holding_cost_per_unit'] = np.maximum(0, data['holding_cost_per_unit'])
    data['shipping_variable_cost'] = np.maximum(0, data['shipping_variable_cost'])
    data['shipping_fixed_cost'] = np.maximum(0, data['shipping_fixed_cost'])
    data['ocean_container_cost'] = np.maximum(0, data['ocean_container_cost'])


    return data

# --- Main Script (Remains the same) ---
if __name__ == "__main__":
    print("Loading base data...")
    base_data = load_base_data() # Now calls the function with manual data

    print(f"Generating instances in folder '{INSTANCE_FOLDER}'...")
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)

    for s_id in range(1, NUM_SCENARIOS + 1):
        for i_id in range(1, NUM_INSTANCES_PER_SCENARIO + 1):
            instance_data = generate_instance_data(base_data, s_id, i_id)
            file_name = f"scenario_{s_id}_instance_{i_id}.npz"
            file_path = os.path.join(INSTANCE_FOLDER, file_name)
            try:
                # Save all relevant numpy arrays and other parameters
                save_dict = {key: val for key, val in instance_data.items() if isinstance(val, np.ndarray)}
                # Add non-array values, converting them first if they weren't already numbers
                save_dict['ocean_container_cost'] = np.array(instance_data['ocean_container_cost'])
                save_dict['ocean_container_capacity_cbm'] = np.array(instance_data['ocean_container_capacity_cbm'])
                save_dict['N_PRODUCTS'] = np.array(instance_data['N_PRODUCTS'])
                save_dict['T_MONTHS'] = np.array(instance_data['T_MONTHS'])

                np.savez_compressed(file_path, **save_dict)
            except Exception as e:
                print(f"Error saving instance {file_name}: {e}")

    print(f"Generated {NUM_SCENARIOS * NUM_INSTANCES_PER_SCENARIO} instances.")

