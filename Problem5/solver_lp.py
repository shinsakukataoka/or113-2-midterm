import pulp
import numpy as np

def solve_lp(data):
    try:
        N = data['N_PRODUCTS'] # Number of products
        T = data['T_MONTHS']   # Number of months (planning horizon)
        J = 3                  # Number of shipping methods (0: Exp, 1: Air, 2: Ocean)

        product_indices = list(range(N)) # [0, 1, ..., N-1]
        month_indices = list(range(T))   # [0, 1, ..., T-1]
        method_indices = list(range(J))  # [0, 1, 2]

        D = data['demand']                     # Demand (N x T)
        I0 = data['initial_inventory']         # Initial Inventory (N)
        transit = data['in_transit']           # In-transit inventory (N x T)
        h = data['holding_cost_per_unit']      # Holding cost (N)
        p = data['purchase_cost_per_unit']     # Purchase cost (N)
        svc = data['shipping_variable_cost']   # Shipping variable cost (N x 2) [Express, Air]
        sfc = data['shipping_fixed_cost']      # Shipping fixed cost (J)
        vol = data['volume_cbm']               # Volume per unit (N)
        ocean_cc = data['ocean_container_cost'] # Ocean container cost
        ocean_cap = data['ocean_container_capacity_cbm'] # Ocean container capacity
        lead_times = data['lead_times']        # Lead times (J) [Express, Air, Ocean]

    except KeyError as e:
        print(f"Error: Missing key in instance_data: {e}")
        return float('inf'), "Data Error"
    except Exception as e:
        print(f"Error unpacking data in solve_lp: {e}")
        return float('inf'), "Data Error"

    # Create the LP problem
    prob = pulp.LpProblem("Inventory_Planning_LP", pulp.LpMinimize)

    # --- Decision Variables ---
    # Order quantities: x[i][j] = quantity of product i ordered via method j at start of month m=0
    # Use the explicit index lists - PuLP creates nested dicts: dict[i][j]
    order_vars = pulp.LpVariable.dicts("Order", (product_indices, method_indices), lowBound=0, cat='Continuous')

    # Inventory level: I[i][t] = inventory of product i at the *end* of month t
    inv_vars = pulp.LpVariable.dicts("Inventory", (product_indices, month_indices), lowBound=0, cat='Continuous')

    # Ocean containers: z = number of ocean containers used (ordered month 0)
    ocean_containers = pulp.LpVariable("OceanContainers", lowBound=0, cat='Continuous')

    # Auxiliary for fixed costs (Relaxation: continuous [0,1])
    use_method = pulp.LpVariable.dicts("UseMethod", method_indices, lowBound=0, upBound=1, cat='Continuous')


    # --- Objective Function ---
    # Minimize: Purchase Cost + Shipping Cost (Var + Fixed) + Holding Cost
    try:
        # Access variables using nested dictionary style: [i][j]
        purchase_cost = pulp.lpSum(order_vars[i][j] * p[i] for i in product_indices for j in method_indices)

        shipping_var_cost_exp_air = pulp.lpSum(order_vars[i][j] * svc[i, j] for i in product_indices for j in method_indices if j < 2) # Express(0), Air(1)
        # Ocean variable cost is tied to container cost, handled below

        shipping_fixed_cost = pulp.lpSum(use_method[j] * sfc[j] for j in method_indices)
        ocean_shipping_cost = ocean_containers * ocean_cc

        # Access inventory variables using nested dictionary style: [i][t]
        holding_cost = pulp.lpSum(inv_vars[i][t] * h[i] for i in product_indices for t in month_indices)

        prob += purchase_cost + shipping_var_cost_exp_air + ocean_shipping_cost + shipping_fixed_cost + holding_cost, "Total_Cost"

    except KeyError as e:
         # Simplified error message as loop variables might not be set
         print(f"Error accessing variable or parameter in objective function. Potential key issue: {e}")
         return float('inf'), "Objective Error"
    except IndexError as e:
         # Determine which index might be out of bounds if possible (tricky here)
         print(f"Error: Index out of bounds in objective function: {e}")
         return float('inf'), "Objective Error"
    except Exception as e:
        print(f"Error building objective function: {e}")
        return float('inf'), "Objective Error"


    # --- Constraints ---
    try:
        for i in product_indices:
            for t in month_indices:
                # Inventory Balance: I[i][t] = I[i, t-1] + Arrivals[i,t] - Demand[i,t]
                if t == 0:
                    prev_inv = I0[i]
                else:
                    # Access inventory variables using nested dictionary style: [i][t-1]
                    prev_inv = inv_vars[i][t-1]

                # Calculate arrivals *during* month t (available at start of t)
                arrivals_in_month_t = transit[i, t] # In-transit scheduled to arrive start of month t
                # Add arrivals from orders placed at m=0
                for j in method_indices:
                     # Order placed month 0, lead time L -> arrives start of month L (index L)
                     arrival_month_index = lead_times[j]
                     if arrival_month_index == t: # Arrives start of current month t
                          # Access order variables using nested dictionary style: [i][j]
                          arrivals_in_month_t += order_vars[i][j]

                # Demand is met by inventory at start of month + arrivals during month
                # Inventory at end of month t is what's left over
                # Access inventory variables using nested dictionary style: [i][t]
                prob += inv_vars[i][t] == prev_inv + arrivals_in_month_t - D[i, t], f"InvBalance_{i}_{t}"

        # Ocean Capacity Constraint
        # Access order variables using nested dictionary style: [i][2]
        total_ocean_volume = pulp.lpSum(order_vars[i][2] * vol[i] for i in product_indices) # j=2 is Ocean
        prob += total_ocean_volume <= ocean_containers * ocean_cap, "OceanCapacity"

        # Linking Constraints for Fixed Costs (Relaxed)
        BIG_M = 1e7 # A reasonably large number
        for j in method_indices:
             # Access order variables using nested dictionary style: [i][j]
             prob += pulp.lpSum(order_vars[i][j] for i in product_indices) <= BIG_M * use_method[j], f"LinkUseMethod_{j}"

        # Special link for Ocean: If any ocean orders exist, ocean_containers must be > 0
        # Access order variables using nested dictionary style: [i][2]
        prob += pulp.lpSum(order_vars[i][2] for i in product_indices) <= BIG_M * ocean_containers, "LinkOceanContainers"
        # The LinkUseMethod_2 constraint above already links ocean orders to use_method[2]

    except KeyError as e:
         # Simplified error message as loop variables might not be set
         print(f"Error accessing variable or parameter in constraints. Potential key issue: {e}")
         return float('inf'), "Constraint Error"
    except IndexError as e:
         # Determine which index might be out of bounds if possible (tricky here)
         print(f"Error: Index out of bounds in constraints: {e}")
         return float('inf'), "Constraint Error"
    except Exception as e:
        print(f"Error building constraints: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), "Constraint Error"

    # --- Solve ---
    try:
        # Using the default CBC solver included with PuLP
        solver = pulp.PULP_CBC_CMD(msg=0) # msg=0 suppresses solver output
        prob.solve(solver)
    except Exception as e:
        print(f"Error during PuLP solve step: {e}")
        # Could indicate solver not found or other runtime issue
        return float('inf'), "Solver Error"

    # --- Extract Results ---
    status = pulp.LpStatus[prob.status]
    if prob.status == pulp.LpStatusOptimal:
        objective_value = pulp.value(prob.objective)
        return objective_value, "Optimal"
    elif prob.status == pulp.LpStatusNotSolved:
         print("Warning: LP solver did not solve the problem.")
         return float('inf'), "Not Solved"
    elif prob.status == pulp.LpStatusInfeasible:
         print("Warning: LP problem is infeasible.")
         return float('inf'), "Infeasible"
    elif prob.status == pulp.LpStatusUnbounded:
         print("Warning: LP problem is unbounded.")
         return float('inf'), "Unbounded"
    else: # Undefined, etc.
         print(f"Warning: LP solver finished with status: {status}")
         return float('inf'), status