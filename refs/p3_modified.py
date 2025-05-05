# run_experiment_from_npz.py
import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ── Global settings ──────────────────────────────────────────────────────────
INSTANCE_FOLDER              = "instances"
NUM_SCENARIOS                = 7
INSTANCES_PER_SCENARIO       = 30

VC_CONTAINER_CAPACITY_CBM    = 30.0          # matches generator
J                              = 3           # shipping methods (Exp, Air, Ocean)

# ── Helpers ──────────────────────────────────────────────────────────────────
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

def naive_heuristic(N, T, D, I0, I, CP, CV1, CF):
    """Simple ‘order-just-enough’ express heuristic."""
    total_cost = 0.0
    inventory  = I0.copy()

    for t in range(T):
        demand     = D[:, t]
        in_transit = I[:, t]
        available  = inventory + in_transit

        order_qty  = np.maximum(demand - available, 0)

        total_cost += np.sum(order_qty * (CP + CV1))
        if order_qty.sum() > 0:
            total_cost += CF[0]                   # express fixed

        inventory = np.zeros(N)                  # discard leftovers

    return total_cost

def solve_lp_relaxation(
    N, T, D, I0, I, CH, CP, CV1, CV2, CV3,
    CF, Vi, container_cost, lead_times
):
    """Continuous relaxation of the original MIP."""
    try:
        m = gp.Model()
        m.setParam("OutputFlag", 0)

        x = m.addVars(N, J, T, name="x")             # qty shipped
        v = m.addVars(N, T, name="v")                # inventory
        y = m.addVars(J, T, name="y")                # fixed-cost indicator
        z = m.addVars(T,     name="z")               # ocean container count

        # Objective
        hold = gp.quicksum(CH[i] * v[i, t]
                           for i in range(N) for t in range(T))
        ship = gp.quicksum(
            (CP[i] + [CV1[i], CV2[i], CV3[i]][j]) * x[i, j, t]
            for i in range(N) for j in range(J) for t in range(T)
        )
        fixed      = gp.quicksum(CF[j] * y[j, t] for j in range(J) for t in range(T))
        container  = gp.quicksum(container_cost * z[t] for t in range(T))
        m.setObjective(hold + ship + fixed + container, GRB.MINIMIZE)

        # Inventory balance
        for i in range(N):
            for t in range(T):
                arrivals = gp.LinExpr()
                for j in range(J):
                    t_order = t - int(lead_times[j]) + 1
                    if 0 <= t_order < T:
                        arrivals += x[i, j, t_order]

                if t == 0:
                    m.addConstr(
                        v[i, t] == I0[i] + I[i, t] + arrivals - D[i, t]
                    )
                else:
                    m.addConstr(
                        v[i, t] == v[i, t - 1] + I[i, t] + arrivals - D[i, t]
                    )

        # Link x and y (big-M)
        M = D.sum() * 2
        for j in range(J):
            for t in range(T):
                m.addConstr(
                    gp.quicksum(x[i, j, t] for i in range(N)) <= M * y[j, t]
                )

        # Ocean capacity
        for t in range(T):
            m.addConstr(
                gp.quicksum(Vi[i] * x[i, 2, t] for i in range(N))
                <= VC_CONTAINER_CAPACITY_CBM * z[t]
            )

        m.optimize()
        return m.ObjVal

    except Exception as exc:
        print(f"LP solve failed: {exc}")
        return np.nan

# ── Main experiment loop ─────────────────────────────────────────────────────
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
