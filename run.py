import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from load_instances import load_instance
from solve_naive import naive_heuristic
from solve_lp import solve_lp_relaxation

INSTANCE_FOLDER              = "instances"
NUM_SCENARIOS                = 7
INSTANCES_PER_SCENARIO       = 30

VC_CONTAINER_CAPACITY_CBM    = 30.0          # matches generator
J                              = 3           # shipping methods (Exp, Air, Ocean)

results = []

for s_id in range(1, NUM_SCENARIOS + 1):
    for i_id in range(1, INSTANCES_PER_SCENARIO + 1):
        try:
            (N, T, D, I0, I, CH, CP,
             CV1, CV2, CV3, CF, Vi,
             container_cost, lead_times) = load_instance(s_id, i_id)

            # Naïve
            t0 = time.time()
            cost_naive = naive_heuristic(N, T, D, I0, I, CP, CV1, CF)
            t_naive    = time.time() - t0

            # LP relaxation
            t0        = time.time()
            cost_lp   = solve_lp_relaxation(
                N, T, D, I0, I, CH, CP, CV1, CV2, CV3,
                CF, Vi, container_cost, lead_times
            )
            t_lp      = time.time() - t0

            gap = ((cost_naive - cost_lp) / cost_lp) * 100 if cost_lp > 0 else np.nan

            results.append([
                s_id, i_id, cost_lp, cost_naive, t_lp, t_naive, gap
            ])

        except Exception as exc:
            print(f"Error on Scenario {s_id} Instance {i_id}: {exc}")

# ── Reporting ────────────────────────────────────────────────────────────────
df = pd.DataFrame(
    results,
    columns=[
        "Scenario ID", "Instance ID", "LP Relaxation",
        "Naive Heuristic", "Time LP (s)", "Time Naive (s)",
        "Gap Naive (%)"
    ]
)

summary = (
    df.groupby("Scenario ID")["Gap Naive (%)"]
      .agg(["mean", "std"])
      .reset_index()
      .rename(columns={"mean": "Mean Gap (%)", "std": "Std Dev Gap (%)"})
)

print("\n==== Scenario-level Gap Summary ====")
print(summary.to_string(index=False, float_format="%.2f"))

print("\n==== First 10 Raw Results ====")
print(df.head(10).to_string(index=False, float_format="%.2f"))
