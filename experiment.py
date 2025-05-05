import time, pandas as pd
import config as C
import os
from instance_io     import load_instance
from algo_naive      import naive_heuristic
from algo_heuristic  import heuristic_cost
from algo_lp         import solve_lp_relaxation

rows = []

for s in range(1, C.NUM_SCENARIOS + 1):
    for i in range(1, C.INSTANCES_PER_SCENARIO + 1):
        (N, T, D, I0, I, CH, CP,
         CV1, CV2, CV3, CF, Vi,
         cont_cost, lead) = load_instance(s, i)

        # ---------- Naive heuristic ------------------------------------------
        t0 = time.time()
        cost_naive = naive_heuristic(N, T, D, I0, I, CP, CV1, CF)
        t_naive    = time.time() - t0

        # ---------- NEW heuristic ------------------------------------------
        t0 = time.time()
        cost_heur  = heuristic_cost(
            N, T, D, I0, I,
            CH, CP, CV1, CV2, CF,
            Vi, cont_cost, C.VC_CONTAINER_CAPACITY_CBM
        )
        t_heur     = time.time() - t0

        # ---------- LP relaxation ------------------------------------------
        t0 = time.time()
        cost_lp = solve_lp_relaxation(
            N, T, D, I0, I, CH, CP,
            CV1, CV2, CV3, CF, Vi, cont_cost, lead
        )
        t_lp = time.time() - t0

        rows.append([
            s, i, cost_lp, cost_naive, cost_heur,
            t_lp, t_naive, t_heur,
            ((cost_naive - cost_lp) / cost_lp) * 100 if cost_lp > 0 else float("nan"),
            ((cost_heur  - cost_lp) / cost_lp) * 100 if cost_lp > 0 else float("nan"),
        ])

# ---------------------------------------------------------------------------
df = pd.DataFrame(rows, columns=[
    "Scenario", "Instance", "LP Bound",
    "Naive", "Heuristic",
    "Time LP (s)", "Time Naive (s)", "Time Heur (s)",
    "Gap Naive (%)", "Gap Heur (%)"
])

summary_naive = (df.groupby("Scenario")["Gap Naive (%)"]
                   .agg(['mean', 'std'])
                   .rename(columns={'mean': 'Mean Gap (%)', 'std': 'Std Dev (%)'}))

summary_heur  = (df.groupby("Scenario")["Gap Heur (%)"]
                   .agg(['mean', 'std'])
                   .rename(columns={'mean': 'Mean Gap (%)', 'std': 'Std Dev (%)'}))

print("\n========== Naive vs LP – Scenario Summary ==========")
print(summary_naive.to_string(float_format="%.2f"))
print("\\n========== Heuristic vs LP – Scenario Summary ==========")
print(summary_heur.to_string(float_format="%.2f"))

print("========== First 10 Raw Lines ==========")
print(df.head(10).to_string(index=False, float_format="%.2f"))

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

csv_path = os.path.join(OUTPUT_DIR, "p5_results.csv")
df.to_csv(csv_path, index=False, float_format="%.4f")
print(f"\nResults written to {csv_path}")