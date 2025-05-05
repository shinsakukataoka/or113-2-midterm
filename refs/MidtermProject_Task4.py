
import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

# Constants
NUM_SCENARIOS = 7
INSTANCES_PER_SCENARIO = 30
VC = 30  # Volume of a container
CF = [100, 80, 50]  # Fixed shipping costs
T_lead = [1, 2, 3]  # Lead times
J = 3  # Shipping methods
VCost_options = [1375, 2750, 5500]  # Container cost options
HoldingCostFactors = [0.01, 0.02, 0.04]
Scales = [(10, 6), (100, 20), (500, 50)]  # (products, periods)

results = []

# Scenario definitions (7 scenarios)
scenarios = [
    (1, 1, 1),  # all medium
    (0, 1, 1),  # small scale
    (2, 1, 1),  # large scale
    (1, 0, 1),  # low container cost
    (1, 2, 1),  # high container cost
    (1, 1, 0),  # low holding cost
    (1, 1, 2),  # high holding cost
]

def generate_instance(scale_idx, container_cost_idx, holding_cost_idx):
    N, T = Scales[scale_idx]
    CP = np.random.uniform(1000, 10000, N)
    CH = CP * HoldingCostFactors[holding_cost_idx]
    CV1 = np.random.uniform(40, 100, N)
    alpha = np.random.uniform(0.4, 0.6, N)
    CV2 = CV1 * alpha
    CV3 = np.zeros(N)
    D = np.random.uniform(0, 200, (N, T))
    I0 = np.random.uniform(np.max(D[:, 0]) * 0.5, 400, N)
    Vi = np.random.uniform(0, 1, N)
    I = np.zeros((N, T))
    for i in range(N):
        for t in range(min(2, T)):
            I[i, t] = np.random.choice([0, np.random.uniform(0, 50)])
    container_cost = VCost_options[container_cost_idx]
    return N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF, Vi, container_cost

def naive_heuristic(N, T, D, I0, I, CP, CV1, CF):
    total_cost = 0
    inventory = I0.copy()

    for t in range(T):
        demand = D[:, t]
        in_transit = I[:, t]
        available = inventory + in_transit

        # Order only enough to meet remaining demand
        order_qty = np.maximum(demand - available, 0)

        # Cost = purchase + variable shipping (express) + fixed if any order
        total_cost += np.sum(order_qty * (CP + CV1))
        if np.sum(order_qty) > 0:
            total_cost += CF[0]  # fixed cost for express delivery

        # No inventory is carried over (discard leftover inventory)
        inventory = np.zeros(N)

    return total_cost


def solve_lp_relaxation(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF, Vi, container_cost):
    try:
        model = gp.Model("LPRelaxation")
        model.setParam('OutputFlag', 0)

        x = model.addVars(N, J, T, vtype=GRB.CONTINUOUS, name="x")
        v = model.addVars(N, T, vtype=GRB.CONTINUOUS, name="v")
        y = model.addVars(J, T, vtype=GRB.CONTINUOUS, name="y")
        z = model.addVars(T, vtype=GRB.CONTINUOUS, name="z")

        hold_cost = gp.quicksum(CH[i] * v[i, t] for i in range(N) for t in range(T))
        ship_cost = gp.quicksum((CP[i] + [CV1[i], CV2[i], CV3[i]][j]) * x[i, j, t]
                                for i in range(N) for j in range(J) for t in range(T))
        fixed_cost = gp.quicksum(CF[j] * y[j, t] for j in range(J) for t in range(T))
        container_cost_expr = gp.quicksum(container_cost * z[t] for t in range(T))
        model.setObjective(hold_cost + ship_cost + fixed_cost + container_cost_expr, GRB.MINIMIZE)

        for i in range(N):
            for t in range(T):
                in_transit_qty = gp.LinExpr()
                for j in range(J):
                    t_order = t - T_lead[j] + 1
                    if 0 <= t_order < T:
                        in_transit_qty += x[i, j, t_order]
                if t == 0:
                    model.addConstr(v[i, t] == I0[i] + I[i, t] + in_transit_qty - D[i, t])
                else:
                    model.addConstr(v[i, t] == v[i, t - 1] + I[i, t] + in_transit_qty - D[i, t])
                    model.addConstr(v[i, t - 1] >= D[i, t])

        M = np.sum(D) * 2
        for j in range(J):
            for t in range(T):
                model.addConstr(gp.quicksum(x[i, j, t] for i in range(N)) <= M * y[j, t])

        for t in range(T):
            model.addConstr(gp.quicksum(Vi[i] * x[i, 2, t] for i in range(N)) <= VC * z[t])

        model.optimize()
        return model.objVal
    except Exception as e:
        print(f"LP failed: {e}")
        return np.nan

# Run experiment
for scenario_id, (scale_idx, cc_idx, hc_idx) in enumerate(scenarios, 1):
    for instance_id in range(1, INSTANCES_PER_SCENARIO + 1):
        print(scenario_id,instance_id)
        N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF, Vi, container_cost = generate_instance(scale_idx, cc_idx, hc_idx)

        start_naive = time.time()
        naive_cost = naive_heuristic(N, T, D, I0, I, CP, CV1, CF)
        time_naive = time.time() - start_naive

        start_lp = time.time()
        lp_cost = solve_lp_relaxation(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF, Vi, container_cost)
        time_lp = time.time() - start_lp

        gap_naive = ((naive_cost - lp_cost) / lp_cost) * 100 if lp_cost and lp_cost > 0 else np.nan

        results.append([
            scenario_id, instance_id,
            lp_cost, naive_cost,
            time_lp, time_naive,
            gap_naive
        ])

df_results = pd.DataFrame(results, columns=[
    "Scenario ID", "Instance ID",
    "LP Relaxation", "Naive Heuristic",
    "Time LP", "Time Naive",
    "Gap Naive (%)"
])


#Create excel files
summary = df_results.groupby("Scenario ID")["Gap Naive (%)"].agg(["mean", "std"]).reset_index()
summary.columns = ["Scenario ID", "Mean Gap (%)", "Std Dev Gap (%)"]
summary.to_excel("scenario_gap_summary.xlsx", index=False)
df_results.to_excel("results.xlsx", index=False)

