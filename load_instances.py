import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

INSTANCE_FOLDER              = "instances"
NUM_SCENARIOS                = 7
INSTANCES_PER_SCENARIO       = 30

VC_CONTAINER_CAPACITY_CBM    = 30.0          # matches generator
J                              = 3           # shipping methods (Exp, Air, Ocean)

def load_instance(s_id: int, i_id: int):
    """Load a single (scenario, instance) pair from NPZ."""
    path = os.path.join(
        INSTANCE_FOLDER, f"scenario_{s_id}_instance_{i_id}.npz"
    )
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with np.load(path) as data:
        N  = int(data["N_PRODUCTS"])
        T  = int(data["T_MONTHS"])
        D  = data["demand"]                       # (N, T)
        I0 = data["initial_inventory"]            # (N,)
        I  = data["in_transit"]                   # (N, T)

        CP = data["purchase_cost_per_unit"]       # (N,)
        CH = data["holding_cost_per_unit"]        # (N,)

        ship_var         = data["shipping_variable_cost"]  # (N, 2)
        CV1, CV2         = ship_var[:, 0], ship_var[:, 1]
        CV3              = np.zeros_like(CV1)              # ocean has no variable unit cost

        CF               = data["shipping_fixed_cost"]     # (3,)
        Vi               = data["volume_cbm"]              # (N,)
        container_cost   = float(data["ocean_container_cost"])
        lead_times       = data["lead_times"]              # (3,)

    return (
        N, T, D, I0, I, CH, CP,
        CV1, CV2, CV3, CF, Vi,
        container_cost, lead_times,
    )
