import numpy as np
import pandas as pd
import time
import importlib
from load_instances import load_instance

try:
    p3 = importlib.import_module("solve_lp")
    solve_lp_relaxation = p3.solve_lp_relaxation
except (ModuleNotFoundError, AttributeError):
    raise ImportError(
        "Could not import solve_lp_relaxation from p3.py – "
        "make sure p3.py is in the same directory and defines it."
    )

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
NUM_SCENARIOS           = 7
INSTANCES_PER_SCENARIO  = 30
VC                      = 30.0                      # container capacity (CBM)
CF                      = [100, 80, 50]             # fixed shipping costs
T_LEAD                  = [1, 2, 3]                 # lead times (months)  ★ NEW: passed into LP
J                       = 3
VCOST_OPTIONS           = [1375, 2750, 5500]
HOLDING_FACTORS         = [0.01, 0.02, 0.04]
SCALES                  = [(10, 6), (100, 20), (500, 50)]

SCENARIOS = [
    (1, 1, 1),  # all medium
    (0, 1, 1),  # small scale
    (2, 1, 1),  # large scale
    (1, 0, 1),  # low container cost
    (1, 2, 1),  # high container cost
    (1, 1, 0),  # low holding cost
    (1, 1, 2),  # high holding cost
]

# ---------------------------------------------------------------------------
# instance generator
# ---------------------------------------------------------------------------
def generate_instance(scale_idx, cc_idx, hc_idx, rng):
    N, T = SCALES[scale_idx]

    CP  = rng.uniform(1000, 10000, N)
    CH  = CP * HOLDING_FACTORS[hc_idx]
    CV1 = rng.uniform(40, 100, N)
    alpha = rng.uniform(0.4, 0.6, N)
    CV2 = CV1 * alpha
    CV3 = np.zeros(N)

    D   = rng.uniform(0, 200, (N, T))
    I0  = rng.uniform(D[:, 0].max() * 0.5, 400, N)
    Vi  = rng.uniform(0, 1, N)

    I   = np.zeros((N, T))
    for i in range(N):
        for t in range(min(2, T)):
            I[i, t] = rng.choice([0, rng.uniform(0, 50)])

    cont_cost = VCOST_OPTIONS[cc_idx]
    return N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF, Vi, cont_cost

# ---------------------------------------------------------------------------
# heuristic
# ---------------------------------------------------------------------------
def heuristic_cost(N, T, D, I0, I, CP, CV1, CF):
    total = 0.0
    inv   = I0.copy()
    for t in range(T):
        need = np.maximum(D[:, t] - (inv + I[:, t]), 0)
        total += np.sum(need * (CP + CV1))
        if need.sum() > 0:
            total += CF[0]
        inv = np.zeros(N)
    return total

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
rows = []

for s_id, (sc_idx, cc_idx, hc_idx) in enumerate(SCENARIOS, start=1):
    for inst_id in range(1, INSTANCES_PER_SCENARIO + 1):
        rng = np.random.RandomState(seed=10_000 * s_id + inst_id)
        #(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF_vals, Vi, cont_cost) = load_instance(sc_idx, cc_idx, hc_idx, rng)
        #(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF_vals, Vi, cont_cost) = generate_instance(sc_idx, cc_idx, hc_idx, rng)
        (N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF_vals, Vi, cont_cost, lead_times) = load_instance(s_id, inst_id)

        print(f"Running Scenario {s_id}  Instance {inst_id}")

        t0 = time.time()
        cost_h   = heuristic_cost(N, T, D, I0, I, CP, CV1, CF_vals)
        t_h      = time.time() - t0

        t0 = time.time()
        cost_lp  = solve_lp_relaxation(
            N, T, D, I0, I, CH, CP, CV1, CV2, CV3,
            CF_vals, Vi, cont_cost, T_LEAD     # ★ PASS lead_times
        )
        t_lp     = time.time() - t0

        gap = ((cost_h - cost_lp) / cost_lp) * 100 if cost_lp > 0 else np.nan
        rows.append([s_id, inst_id, cost_lp, cost_h, t_lp, t_h, gap])

# ---------------------------------------------------------------------------
# print tables
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows, columns=[
    "Scenario", "Instance", "LP Bound",
    "Heuristic", "Time LP (s)", "Time Heur. (s)", "Gap Heur. (%)"
])
summary = (df.groupby("Scenario")["Gap Heur. (%)"]
             .agg(['mean', 'std'])
             .rename(columns={'mean': 'Mean Gap (%)', 'std': 'Std Dev (%)'})
             .reset_index())

print("\n================ Scenario-level Gap Summary ================\n")
print(summary.to_string(index=False, float_format="%.2f"))

print("\n================ First 10 Raw Result Lines ================\n")
print(df.head(10).to_string(index=False, float_format="%.2f"))
