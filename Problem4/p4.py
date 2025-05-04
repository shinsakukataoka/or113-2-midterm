# Problem4/run_p4_experiment_from_npz.py
# Runs the P4 experiment (LP vs Naive) using pre-generated .npz instance files.

import numpy as np
import pandas as pd
import time
import gurobipy as gp
from gurobipy import GRB
import os
import glob # To find instance files

# --- Constants (Copied from original P4 script for context, some may not be needed directly) ---
# These might be implicitly defined by the loaded instance data now
VC = 30  # Volume of a container
CF = [100, 80, 50]  # Fixed shipping costs [Express, Air, Ocean] - Load from instance if needed
T_lead = [1, 2, 3]  # Lead times [Express, Air, Ocean] - Load from instance if needed
J = 3  # Shipping methods

# --- Naive Heuristic Function (Copied from original P4 script) ---
def naive_heuristic(N, T, D, I0, I, CP, CV1, CF_const): # Use CF_const to avoid name clash
    """Multi-period naive heuristic: order express to meet demand, discard inventory."""
    total_cost = 0
    inventory = I0.copy()

    for t in range(T):
        demand = D[:, t]
        in_transit = I[:, t] # In-transit arriving start of period t
        available = inventory + in_transit

        # Order only enough to meet remaining demand via express
        order_qty = np.maximum(demand - available, 0)

        # Cost = purchase + variable express shipping + fixed express cost if any order
        period_cost = np.sum(order_qty * (CP + CV1))
        if np.sum(order_qty) > 1e-6: # Use tolerance
            period_cost += CF_const[0]  # fixed cost for express delivery (method 0)

        total_cost += period_cost

        # No inventory is carried over (discard leftover inventory)
        inventory = np.zeros(N)

    return total_cost

# --- LP Relaxation Solver Function (Copied from original P4 script) ---
def solve_lp_relaxation(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF_const, Vi, container_cost): # Use CF_const
    """Solves the multi-period LP relaxation using Gurobi."""
    try:
        model = gp.Model("LPRelaxation_MultiPeriod")
        model.setParam('OutputFlag', 0) # Suppress Gurobi output

        # Variables
        x = model.addVars(N, J, T, vtype=GRB.CONTINUOUS, name="x", lb=0.0) # Order qty
        v = model.addVars(N, T, vtype=GRB.CONTINUOUS, name="v", lb=0.0) # Inventory EOP
        y = model.addVars(J, T, vtype=GRB.CONTINUOUS, name="y", lb=0.0, ub=1.0) # Use method flag
        z = model.addVars(T, vtype=GRB.CONTINUOUS, name="z", lb=0.0) # Ocean containers

        # Objective
        hold_cost = gp.quicksum(CH[i] * v[i, t] for i in range(N) for t in range(T))
        # Combine purchase and variable shipping cost
        ship_var_cost = gp.quicksum((CP[i] + [CV1[i], CV2[i], CV3[i]][j]) * x[i, j, t]
                                for i in range(N) for j in range(J) for t in range(T))
        ship_fixed_cost = gp.quicksum(CF_const[j] * y[j, t] for j in range(J) for t in range(T))
        ocean_cont_cost = gp.quicksum(container_cost * z[t] for t in range(T))
        model.setObjective(hold_cost + ship_var_cost + ship_fixed_cost + ocean_cont_cost, GRB.MINIMIZE)

        # Constraints
        for i in range(N):
            for t in range(T):
                # Calculate arrivals at START of period t from orders placed previously
                arrivals = I[i, t] # Initial in-transit for period t
                for j in range(J):
                    t_order = t - T_lead[j] + 1
                    if 0 <= t_order < T:
                        arrivals += x[i, j, t_order]

                # Inventory balance
                if t == 0:
                    model.addConstr(v[i, t] == I0[i] + arrivals - D[i, t], name=f"InvBal_{i}_{t}")
                else:
                    model.addConstr(v[i, t] == v[i, t - 1] + arrivals - D[i, t], name=f"InvBal_{i}_{t}")

                # Service level constraint (v[i,t-1] >= D[i,t]) - PRESENT in this version
                if t > 0:
                     model.addConstr(v[i, t - 1] >= D[i, t], name=f"SvcLvl_{i}_{t}")

        # Big-M constraints for fixed costs and ocean capacity
        M = np.sum(D) * 1.1 + 1 # Use a reasonable Big-M based on total demand

        for j in range(J):
            for t in range(T):
                # Link order quantity to fixed cost flag y
                model.addConstr(gp.quicksum(x[i, j, t] for i in range(N)) <= M * y[j, t], name=f"LinkY_{j}_{t}")

        for t in range(T):
            # Link ocean volume to container variable z
            ocean_cap_local = VC # Use constant VC for capacity
            if ocean_cap_local > 0:
                 model.addConstr(gp.quicksum(Vi[i] * x[i, 2, t] for i in range(N)) <= ocean_cap_local * z[t], name=f"OceanCap_{t}")
            else:
                 model.addConstr(gp.quicksum(x[i, 2, t] for i in range(N)) == 0, name=f"NoOcean_{t}")
            # Link ocean orders to container variable z (redundant if cap > 0, needed if cap=0)
            model.addConstr(gp.quicksum(x[i, 2, t] for i in range(N)) <= M * z[t], name=f"LinkZ_{t}")

        # Optimize
        model.optimize()

        # Check status and return result
        status = model.Status
        if status == GRB.OPTIMAL:
            return model.objVal
        else:
            status_map = {GRB.INFEASIBLE: "Infeasible", GRB.UNBOUNDED: "Unbounded",
                          GRB.INF_OR_UNBD: "Infeasible or Unbounded", GRB.NUMERIC: "Numeric Issues"}
            print(f"Warning: LP finished with status: {status_map.get(status, status)} for instance.")
            return np.nan # Return NaN for non-optimal

    except gp.GurobiError as e:
        print(f"Gurobi error during LP solve: {e}")
        return np.nan
    except Exception as e:
        print(f"Error in solve_lp_relaxation: {e}")
        return np.nan

# --- Main Experiment Execution Block ---
if __name__ == "__main__":

    print("Running Problem 4 Experiment from NPZ files...")
    INSTANCE_FOLDER = "./instances" # Assume instances are in parent dir's instances folder
    results = [] # Initialize list to store results

    # Find all instance files
    instance_files = sorted(glob.glob(os.path.join(INSTANCE_FOLDER, "scenario_*_instance_*.npz")))

    if not instance_files:
        print(f"Error: No instance files found in {INSTANCE_FOLDER}. Run generate_instances.py first.")
        exit()

    print(f"Found {len(instance_files)} instance files.")

    # Loop through instance files
    for file_path in instance_files:
        # Extract scenario and instance ID from filename (optional, but good for tracking)
        filename = os.path.basename(file_path)
        try:
            parts = filename.replace(".npz","").split('_')
            scenario_id = int(parts[1])
            instance_id = int(parts[3])
            print(f"Processing Scenario {scenario_id}, Instance {instance_id}...")
        except Exception as e:
            print(f"Warning: Could not parse scenario/instance ID from {filename}: {e}")
            scenario_id = -1
            instance_id = -1

        # Load data from .npz file
        try:
            with np.load(file_path) as loaded_data:
                # Extract all necessary parameters for the solvers
                # Use .item() for 0-d arrays saved from scalars
                N = int(loaded_data['N_PRODUCTS'].item())
                T = int(loaded_data['T_MONTHS'].item())
                D = loaded_data['demand']
                I0 = loaded_data['initial_inventory']
                I = loaded_data['in_transit'] # Renamed from 'transit' in generator
                CH = loaded_data['holding_cost_per_unit']
                CP = loaded_data['purchase_cost_per_unit']
                # Reconstruct CV1, CV2, CV3
                svc = loaded_data['shipping_variable_cost'] # Shape (N, 2)
                CV1 = svc[:, 0]
                CV2 = svc[:, 1]
                CV3 = np.zeros(N)
                # Use the fixed CF constant defined at the top
                CF_const = np.array(CF) # Ensure it's a numpy array
                Vi = loaded_data['volume_cbm'] # Renamed from 'vol' in generator
                container_cost = float(loaded_data['ocean_container_cost'].item())
                # T_lead is constant, defined at top
                # Ocean capacity VC is constant, defined at top

                # --- Data Validation (Optional but Recommended) ---
                # Add checks for shapes, types, etc. if needed
                # Example:
                if D.shape != (N, T): raise ValueError("Demand shape mismatch")
                if I0.shape != (N,): raise ValueError("I0 shape mismatch")
                # ... more checks ...

        except FileNotFoundError:
            print(f"Error: Instance file not found: {file_path}")
            continue
        except KeyError as e:
            print(f"Error: Missing key {e} in instance file {filename}")
            continue
        except Exception as e:
            print(f"Error loading or processing data from {filename}: {e}")
            continue

        # Run Naive Heuristic
        start_naive = time.time()
        # Pass the extracted/reconstructed parameters
        naive_cost = naive_heuristic(N, T, D, I0, I, CP, CV1, CF_const)
        time_naive = time.time() - start_naive

        # Run LP Relaxation
        start_lp = time.time()
        # Pass the extracted/reconstructed parameters
        lp_cost = solve_lp_relaxation(N, T, D, I0, I, CH, CP, CV1, CV2, CV3, CF_const, Vi, container_cost)
        time_lp = time.time() - start_lp

        # Calculate Naive Gap
        gap_naive = np.nan # Default to NaN
        if pd.notna(lp_cost) and lp_cost > 1e-6: # Check if lp_cost is valid and positive
             if pd.notna(naive_cost):
                 gap_naive = ((naive_cost - lp_cost) / lp_cost) * 100

        # Append results
        results.append([
            scenario_id, instance_id,
            lp_cost, naive_cost,
            time_lp, time_naive,
            gap_naive
        ])

    # Create DataFrame
    df_results = pd.DataFrame(results, columns=[
        "Scenario ID", "Instance ID",
        "LP Relaxation", "Naive Heuristic",
        "Time LP", "Time Naive",
        "Gap Naive (%)"
    ])

    # Save results to CSV files
    output_dir = "output" # Define an output directory
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "p4_results_from_npz.csv") # New filename
    summary_path = os.path.join(output_dir, "p4_scenario_gap_summary_from_npz.csv") # New filename

    try:
        # Calculate summary statistics (handle potential NaNs in gap)
        summary = df_results.groupby("Scenario ID")["Gap Naive (%)"].agg(
            mean=lambda x: x.mean(skipna=True),
            std=lambda x: x.std(skipna=True)
        ).reset_index()
        summary.columns = ["Scenario ID", "Mean Gap (%)", "Std Dev Gap (%)"]

        # Save files
        df_results.to_csv(results_path, index=False, float_format="%.4f")
        summary.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\nExperiment complete. Results saved to '{results_path}' and '{summary_path}'.")

    except Exception as e:
        print(f"\nError saving results to CSV: {e}")

