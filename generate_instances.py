import numpy as np
import os

INSTANCE_FOLDER              = "instances"
NUM_SCENARIOS                = 7
NUM_INSTANCES_PER_SCENARIO   = 30
BASE_FIXED_SHIPPING_COST     = np.array([100.0, 80.0, 50.0])
BASE_LEAD_TIMES              = np.array([1, 2, 3])
BASE_CONTAINER_CAPACITY      = 30.0

SCALES = {
    "small"  : (10,   6),
    "medium" : (100, 20),
    "large"  : (500, 50),
}
CONTAINER_COSTS = {
    "low"    : 1375.0,
    "medium" : 2750.0,
    "high"   : 5500.0,
}
HOLDING_COST_PERCENTAGES = {
    "low"    : 0.01,
    "medium" : 0.02,
    "high"   : 0.04,
}

SCENARIO_CONFIG = {
    1: ("medium", "medium", "medium"),   # baseline
    2: ("small",  "medium", "medium"),   # vary scale ↓
    3: ("large",  "medium", "medium"),
    4: ("medium", "low",    "medium"),   # vary container cost ↓
    5: ("medium", "high",   "medium"),
    6: ("medium", "medium", "low"),      # vary holding cost ↓
    7: ("medium", "medium", "high"),
}

def generate_instance_data(scenario_id: int, instance_id: int) -> dict:
    scale_level, cc_level, hc_level = SCENARIO_CONFIG[scenario_id]
    n_products, t_months            = SCALES[scale_level]
    container_cost                  = CONTAINER_COSTS[cc_level]
    holding_cost_factor             = HOLDING_COST_PERCENTAGES[hc_level]
    rng = np.random.RandomState(seed=scenario_id * 10_000 + instance_id)

    demand = rng.uniform(0.0, 200.0, size=(n_products, t_months))
    purchase_cost_per_unit = rng.uniform(1000, 10000, size=n_products)
    holding_cost_per_unit  = purchase_cost_per_unit * holding_cost_factor
    cv_express = rng.uniform(40.0, 100.0, size=n_products)
    alpha      = rng.uniform(0.4, 0.6,  size=n_products)
    cv_air     = cv_express * alpha
    shipping_variable_cost = np.stack((cv_express, cv_air), axis=-1)
    volume_cbm = rng.uniform(0.02, 0.10, size=n_products)
    I0_lower = 0.5 * demand[:, 0].max()
    initial_inventory = rng.uniform(I0_lower, 400.0, size=n_products)
    in_transit = np.zeros((n_products, t_months))
    for i in range(n_products):
        for t in range(min(2, t_months)):
            in_transit[i, t] = rng.choice([0.0, rng.uniform(0.0, 50.0)])
    data = {
        "demand"                       : demand.astype(float),
        "purchase_cost_per_unit"       : purchase_cost_per_unit.astype(float),
        "holding_cost_per_unit"        : holding_cost_per_unit.astype(float),
        "shipping_variable_cost"       : shipping_variable_cost.astype(float),
        "volume_cbm"                   : volume_cbm.astype(float),
        "initial_inventory"            : initial_inventory.astype(float),
        "in_transit"                   : in_transit.astype(float),
        "shipping_fixed_cost"          : BASE_FIXED_SHIPPING_COST.astype(float),
        "ocean_container_cost"         : np.array(container_cost, dtype=float),
        "ocean_container_capacity_cbm" : np.array(BASE_CONTAINER_CAPACITY, dtype=float),
        "lead_times"                   : BASE_LEAD_TIMES.astype(int),
        "N_PRODUCTS"                   : np.array(n_products, dtype=int),
        "T_MONTHS"                     : np.array(t_months,  dtype=int),
    }
    return data

if __name__ == "__main__":
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    total   = NUM_SCENARIOS * NUM_INSTANCES_PER_SCENARIO
    print(f"Generating {total} instances in “{INSTANCE_FOLDER}/”…")
    for s_id in range(1, NUM_SCENARIOS + 1):
        scale, cc, hc = SCENARIO_CONFIG[s_id]
        print(f"  Scenario {s_id}: scale={scale}, CC={cc}, HC={hc}")
        for inst in range(1, NUM_INSTANCES_PER_SCENARIO + 1):
            data = generate_instance_data(s_id, inst)
            fname = f"scenario_{s_id}_instance_{inst}.npz"
            fpath = os.path.join(INSTANCE_FOLDER, fname)
            np.savez_compressed(fpath, **data)
    print("Done.")