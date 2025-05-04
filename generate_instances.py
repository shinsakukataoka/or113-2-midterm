#!/usr/bin/env python3
"""
Instance generator – “first-script” logic with the original output format.
"""

import numpy as np
import os

# ── Global settings ──────────────────────────────────────────────────────────
INSTANCE_FOLDER              = "instances"
NUM_SCENARIOS                = 7
NUM_INSTANCES_PER_SCENARIO   = 30
np.random.seed(42)                           # base seed (instance-level seeds vary)

# Problem-wide constants (same as first script)
VC_CONTAINER_CAPACITY_CBM    = 30.0
FIXED_COSTS_F                = np.array([100.0, 80.0, 50.0])       # Express, Air, Ocean
LEAD_TIMES                   = np.array([1, 2, 3])                 # months
CONTAINER_COST_OPTIONS       = [1375.0, 2750.0, 5500.0]            # low/med/high
HOLDING_COST_FACTORS         = [0.01, 0.02, 0.04]                  # low/med/high
SCALES                       = [(10, 6), (100, 20), (500, 50)]     # (N_products, T_months)

# Scenario list matching the first script: (scale_idx, container_cost_idx, holding_cost_idx)
SCENARIOS = [
    (1, 1, 1),  # 1  baseline (medium everything)
    (0, 1, 1),  # 2  small scale
    (2, 1, 1),  # 3  large scale
    (1, 0, 1),  # 4  low container cost
    (1, 2, 1),  # 5  high container cost
    (1, 1, 0),  # 6  low holding cost
    (1, 1, 2),  # 7  high holding cost
]

# ── Instance generation ──────────────────────────────────────────────────────
def generate_instance(scale_idx, cc_idx, hc_idx, rng):
    """Return a dictionary with all arrays needed by solvers."""
    # Size
    N_products, T_months = SCALES[scale_idx]

    # Purchase & holding costs
    CP  = rng.uniform(1_000, 10_000, N_products)                 # purchase cost per unit
    CH  = CP * HOLDING_COST_FACTORS[hc_idx]                      # holding cost per unit

    # Variable shipping costs (Express, Air). Ocean variable cost set to zero.
    CV1 = rng.uniform(40, 100, N_products)                       # Express
    alpha = rng.uniform(0.4, 0.6, N_products)
    CV2 = CV1 * alpha                                           # Air
    CV3 = np.zeros(N_products)                                   # Ocean
    shipping_var_cost = np.vstack([CV1, CV2]).T                 # shape (N,2)

    # Product volumes (m³)
    volume_cbm = rng.uniform(0.02, 0.10, N_products)

    # Demand matrix
    demand = rng.uniform(0, 200, (N_products, T_months))

    # Initial on-hand inventory
    I0 = rng.uniform(demand[:, 0].max() * 0.5, 400, N_products)

    # In-transit inventory: only first two periods may have something
    in_transit = np.zeros((N_products, T_months))
    for i in range(N_products):
        for t in range(min(2, T_months)):
            in_transit[i, t] = rng.choice([0.0, rng.uniform(0.0, 50.0)])

    # Container cost for Ocean
    ocean_container_cost = CONTAINER_COST_OPTIONS[cc_idx]

    return {
        # main arrays (names retained from original script)
        "initial_inventory"       : I0.astype(float),
        "demand"                  : demand.astype(float),
        "purchase_cost_per_unit"  : CP.astype(float),
        "holding_cost_per_unit"   : CH.astype(float),
        "shipping_variable_cost"  : shipping_var_cost.astype(float),
        "volume_cbm"              : volume_cbm.astype(float),
        "in_transit"              : in_transit.astype(float),
        # constants that vary by scenario
        "shipping_fixed_cost"     : FIXED_COSTS_F.astype(float),
        "ocean_container_cost"    : float(ocean_container_cost),
        "ocean_container_capacity_cbm": VC_CONTAINER_CAPACITY_CBM,
        "lead_times"              : LEAD_TIMES.astype(int),
        # meta
        "N_PRODUCTS"              : N_products,
        "T_MONTHS"                : T_months,
    }

# ── Main driver ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(INSTANCE_FOLDER, exist_ok=True)
    total = NUM_SCENARIOS * NUM_INSTANCES_PER_SCENARIO
    counter = 0

    for s_id, (scale_idx, cc_idx, hc_idx) in enumerate(SCENARIOS, start=1):
        for inst_id in range(1, NUM_INSTANCES_PER_SCENARIO + 1):
            rng = np.random.RandomState(seed=s_id * 10_000 + inst_id)  # reproducible
            data = generate_instance(scale_idx, cc_idx, hc_idx, rng)

            fname = f"scenario_{s_id}_instance_{inst_id}.npz"
            fpath = os.path.join(INSTANCE_FOLDER, fname)

            # Build save-dict: numpy arrays only
            save_dict = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
            # Scalars → 0-d arrays so that everything lives inside the .npz
            for k in ("ocean_container_cost",
                      "ocean_container_capacity_cbm",
                      "N_PRODUCTS",
                      "T_MONTHS"):
                save_dict[k] = np.array(data[k])

            np.savez_compressed(fpath, **save_dict)

            counter += 1
            if counter % 50 == 0 or counter == total:
                print(f"  [{counter}/{total}] generated")

    print(f"Done. {counter} instances written to '{INSTANCE_FOLDER}/'.")
