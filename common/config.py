import numpy as np

INSTANCE_FOLDER               = "instances"
NUM_SCENARIOS                 = 7
INSTANCES_PER_SCENARIO        = 30
BASE_FIXED_SHIPPING_COST      = np.array([100.0, 80.0, 50.0])   # Exp, Air, Ocean
BASE_LEAD_TIMES               = np.array([1, 2, 3])             # months
BASE_CONTAINER_CAPACITY       = 30.0                            # CBM
VC_CONTAINER_CAPACITY_CBM     = BASE_CONTAINER_CAPACITY
J                               = 3                             # # shipping methods

# Factor-level definitions for Problem 4
SCALES = {
    "small":  (10, 6),
    "medium": (100, 20),
    "large":  (500, 50),
}
CONTAINER_COSTS = {"low": 1375.0, "medium": 2750.0, "high": 5500.0}
HOLDING_COST_PERCENTAGES = {"low": 0.01, "medium": 0.02, "high": 0.04}

SCENARIO_CONFIG = {
    1: ("medium", "medium", "medium"),  # base
    2: ("small",  "medium", "medium"),
    3: ("large",  "medium", "medium"),
    4: ("medium", "low",    "medium"),
    5: ("medium", "high",   "medium"),
    6: ("medium", "medium", "low"),
    7: ("medium", "medium", "high"),
}
