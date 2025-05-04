import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_lp(data):
    try:
        N        = int(data['N_PRODUCTS'])
        T        = int(data['T_MONTHS'])
        J        = 3                                  # 0-Exp, 1-Air, 2-Ocean

        D        = data['demand']                     # (N, T)
        I0       = data['initial_inventory']          # (N,)
        transit  = data['in_transit']               # (N, T)

        CH       = data['holding_cost_per_unit']      # (N,)
        CP       = data['purchase_cost_per_unit']     # (N,)

        svc      = data['shipping_variable_cost']     # (N, 2) [Express, Air]
        # Ensure correct dimensions for variable costs including ocean (which is 0)
        CV = np.hstack((svc, np.zeros((N, 1))))       # (N, 3) [Exp, Air, Ocean]

        CF       = data['shipping_fixed_cost']        # (3,) [Exp, Air, Ocean]
        vol      = data['volume_cbm']                 # (N,)

        ocean_cc = float(data['ocean_container_cost'])
        ocean_cap= float(data['ocean_container_capacity_cbm'])
        lead     = data['lead_times']                 # (3,) e.g. [1, 2, 3]

        # Define index lists for clarity
        product_indices = list(range(N))
        method_indices = list(range(J))
        month_indices = list(range(T))

    except KeyError as e:
        print(f"Error: Missing key in instance_data for LP: {e}")
        return float('inf'), f"Data key missing: {e}"
    except Exception as e:
        print(f"Error unpacking data in solve_lp: {e}")
        return float('inf'), f"Data unpacking error: {e}"

    # ---------------- model -----------------------------------------------
    try:
        m = gp.Model("LP_SingleDecision")
        m.setParam("OutputFlag", 0) # Suppress Gurobi output

        # --- Variables ---
        # x[i,j]: Order quantity of product i via method j placed ONLY at t=0
        x = m.addVars(N, J, vtype=GRB.CONTINUOUS, name="x", lb=0.0)
        # v[i,t]: Inventory level of product i at the END of period t
        v = m.addVars(N, T, vtype=GRB.CONTINUOUS, name="v", lb=0.0)
        # y[j]: 1 if shipping method j is used at t=0, 0 otherwise (relaxed)
        y = m.addVars(J, vtype=GRB.CONTINUOUS, name="y", lb=0.0, ub=1.0)
        # z: Number of ocean containers used for orders placed at t=0 (relaxed)
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z", lb=0.0)

        # --- Objective Function ---
        # Sum of: Purchase Cost (t=0) + Variable Shipping (t=0) + Fixed Shipping (t=0)
        #         + Ocean Container Cost (t=0) + Total Holding Cost (t=0..T-1)

        purchase_cost = gp.quicksum(CP[i] * x[i,j] for i in product_indices for j in method_indices)
        ship_var_cost = gp.quicksum(CV[i,j] * x[i,j] for i in product_indices for j in method_indices)
        ship_fix_cost = gp.quicksum(CF[j] * y[j] for j in method_indices)
        cont_cost     = ocean_cc * z
        hold_cost     = gp.quicksum(CH[i] * v[i,t] for i in product_indices for t in month_indices)

        m.setObjective(purchase_cost + ship_var_cost + ship_fix_cost + cont_cost + hold_cost, GRB.MINIMIZE)

        # --- Constraints ---
        # Inventory Balance Constraints
        for i in product_indices:
            for t in month_indices:
                # Arrivals AT START of period t from orders placed at t=0
                arrivals_t = transit[i, t] # In-transit arriving start of t
                for j in method_indices:
                    # Order placed month 0, lead time L -> arrives start of month L (index L)
                    # Check if current month 't' is the arrival month for method j
                    if lead[j] == t:
                        arrivals_t += x[i,j]

                # Balance equation
                if t == 0:
                    m.addConstr(v[i,0] == I0[i] + arrivals_t - D[i,0], name=f"InvBal_{i}_0")
                else:
                    m.addConstr(v[i,t] == v[i,t-1] + arrivals_t - D[i,t], name=f"InvBal_{i}_{t}")
                # Note: No service level constraint v[i,t-1] >= D[i,t] included here
                # as it wasn't in the original PuLP formulation for P5 comparison.
                # Add it if required by problem interpretation.

        # Ocean Capacity Constraint (for orders placed at t=0)
        if ocean_cap > 0:
            m.addConstr(gp.quicksum(vol[i] * x[i,2] for i in product_indices) <= ocean_cap * z, name="OceanCap")
        else:
             # If capacity is zero, no ocean orders allowed
             m.addConstr(gp.quicksum(x[i,2] for i in product_indices) == 0, name="NoOcean")


        # Linking Constraints for Fixed Costs (Big-M)
        # Estimate M: Sum of all demand could be an upper bound on total ordered quantity
        M = D.sum() * 1.1 # A slightly larger value than total demand

        for j in method_indices:
            # If sum(x[i,j]) > 0, then y[j] must be > 0 (objective forces it towards 1 if CF[j]>0)
            m.addConstr(gp.quicksum(x[i,j] for i in product_indices) <= M * y[j], name=f"LinkFixedCost_{j}")

        # Link ocean orders to ocean container variable z
        # If sum(x[i,2]) > 0, then z must be > 0
        m.addConstr(gp.quicksum(x[i,2] for i in product_indices) <= M * z, name="LinkOceanCont")
        # Also ensure ocean fixed cost flag y[2] is linked
        # The LinkFixedCost_2 constraint above already handles this.

        # --- Solve ---
        m.optimize()

        # --- Results ---
        status = m.Status
        if status == GRB.OPTIMAL:
            return m.objVal, "Optimal"
        elif status == GRB.INFEASIBLE:
            print(f"Warning: LP (Gurobi SingleDecision) Infeasible for instance.")
            return float('inf'), "Infeasible"
        elif status == GRB.UNBOUNDED:
            print(f"Warning: LP (Gurobi SingleDecision) Unbounded for instance.")
            return float('inf'), "Unbounded"
        else:
            print(f"Warning: LP (Gurobi SingleDecision) finished with status {status}")
            return float('inf'), f"Gurobi status {status}"

    except gp.GurobiError as e:
        print(f"Gurobi error code {e.errno}: {e}")
        return float('inf'), f"Gurobi Error {e.errno}"
    except Exception as e:
        print(f"Error building/solving Gurobi LP model: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), f"Model Build/Solve Error: {e}"

