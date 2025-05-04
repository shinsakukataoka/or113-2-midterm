import numpy as np
import math

def fractional_knapsack(capacity, values_per_unit, weights_per_unit, max_items):
    n_items = len(values_per_unit)
    # Calculate value-to-weight ratio, handle division by zero if weight is zero
    ratios = [values_per_unit[i] / weights_per_unit[i] if weights_per_unit[i] > 0 else float('inf')
              for i in range(n_items)]

    # Sort items by ratio in descending order
    indices = sorted(range(n_items), key=lambda k: ratios[k], reverse=True)

    selected_items = np.zeros(n_items)
    remaining_capacity = capacity

    for i in indices:
        if remaining_capacity <= 0:
            break
        if weights_per_unit[i] <= 0: # Cannot take items with zero or negative weight if capacity matters
             continue

        # Amount to take: minimum of remaining available, or what fits in capacity
        # Ensure max_items[i] is treated as a number
        current_max_item = max_items[i]
        # Ensure weights_per_unit[i] is treated as a number > 0
        current_weight = weights_per_unit[i]

        amount_to_take = min(current_max_item, remaining_capacity / current_weight)

        selected_items[i] += amount_to_take
        remaining_capacity -= amount_to_take * current_weight

    return selected_items

def calculate_total_cost(orders, data):

    N = data['N_PRODUCTS']
    T = data['T_MONTHS']
    D = data['demand']
    I0 = data['initial_inventory']
    transit = data['in_transit']
    h = data['holding_cost_per_unit']
    p = data['purchase_cost_per_unit']
    svc = data['shipping_variable_cost'] # Shape (N, 2) for Exp, Air
    sfc = data['shipping_fixed_cost'] # Shape (3,) for Exp, Air, Ocean
    vol = data['volume_cbm']
    ocean_cc = data['ocean_container_cost']
    ocean_cap = data['ocean_container_capacity_cbm']
    lead_times = data['lead_times']

    inventory = np.zeros((N, T))
    current_inv = I0.copy()
    total_holding_cost = 0

    for t in range(T):
        # Arrivals at start of month t
        arrivals_t = transit[:, t].copy()
        # Add orders placed at month 0 that arrive at start of month t
        for j in range(len(lead_times)): # 0:Exp, 1:Air, 2:Ocean
            # Order placed month 0, lead time L -> arrives start of month L
            arrival_month_index = lead_times[j] # lead_time 1 arrives month 1 (index 1), etc.
            if arrival_month_index == t: # Check if arrival month index matches current month index t
                 arrivals_t += orders[:, j]

        # Inventory available at start of month t to meet demand D[:, t]
        available_inv = current_inv + arrivals_t

        # Meet demand
        demand_met = np.minimum(available_inv, D[:, t])
        # NOTE: Assumes demand should be met. Shortfalls aren't costed separately here.

        # Inventory at end of month t
        end_inv = available_inv - D[:, t]
        inventory[:, t] = np.maximum(0, end_inv) # Cannot go below zero

        # Update inventory for next period start
        current_inv = inventory[:, t]

        # Accumulate holding cost for inventory held at end of month t
        total_holding_cost += np.sum(inventory[:, t] * h)


    # --- Calculate costs ---
    total_purchase_cost = np.sum(orders * p[:, np.newaxis]) # orders(N,3) * p(N,1) -> sum

    # Shipping Costs
    total_shipping_cost = 0
    # Express & Air
    for j in range(2): # Express (0), Air (1)
         if np.sum(orders[:, j]) > 1e-6: # If method j was used
              total_shipping_cost += sfc[j] # Fixed cost
              total_shipping_cost += np.sum(orders[:, j] * svc[:, j]) # Variable cost

    # Ocean
    ocean_orders = orders[:, 2]
    if np.sum(ocean_orders) > 1e-6:
        total_shipping_cost += sfc[2] # Fixed cost for using Ocean
        total_volume = np.sum(ocean_orders * vol)
        # Use math.ceil for number of containers
        num_containers = math.ceil(total_volume / ocean_cap) if total_volume > 1e-6 else 0
        total_shipping_cost += num_containers * ocean_cc # Container cost

    total_cost = total_purchase_cost + total_shipping_cost + total_holding_cost
    return total_cost


def solve_proposed(data):
    try:
        N = data['N_PRODUCTS']
        T = data['T_MONTHS']
        D = data['demand']
        I0 = data['initial_inventory']
        transit = data['in_transit']
        h = data['holding_cost_per_unit']
        p = data['purchase_cost_per_unit']
        svc = data['shipping_variable_cost']
        sfc = data['shipping_fixed_cost']
        vol = data['volume_cbm']
        ocean_cc = data['ocean_container_cost']
        ocean_cap = data['ocean_container_capacity_cbm']
        lead_times = data['lead_times'] # [1, 2, 3] for Exp, Air, Ocean

        proj_inv = np.zeros((N, T))
        shortages = np.zeros((N, T))
        current_inv = I0.copy()

        for t in range(T):
            # Arrivals at start of month t (only from initial in-transit)
            arrivals_t = transit[:, t].copy()
            available_inv = current_inv + arrivals_t
            shortages[:, t] = np.maximum(0, D[:, t] - available_inv)
            end_inv = available_inv - D[:, t]
            proj_inv[:, t] = np.maximum(0, end_inv)
            current_inv = proj_inv[:, t] # Update inv for next step projection

        # Heuristic decides shipments to cover S[i,t] using methods that *arrive* by month t
        # Order Mar 1 (month 0), Lead Time L -> Arrives Start of Month L (index L)

        month_0_orders = np.zeros((N, 3)) # Orders placed now: [Exp, Air, Ocean]

        # Cover April (t=1) shortage S[:,1]: Use Express (Lead Time 1 -> arrives start month 1)
        S_april = shortages[:, 1]
        month_0_orders[:, 0] += S_april # Assign April shortage to Express

        # Cover May (t=2) shortage S[:,2]: Use Air (Lead Time 2 -> arrives start month 2)
        S_may = shortages[:, 2]
        month_0_orders[:, 1] += S_may # Assign May shortage to Air

        # Cover June (t=3) shortage S[:,3]: Use Ocean (Lead Time 3 -> arrives start month 3)
        S_june = shortages[:, 3].copy() # Work with a copy
        if np.sum(S_june) > 1e-6:
            V_june = np.sum(S_june * vol)
            C_june_max = math.floor(V_june / ocean_cap) if ocean_cap > 0 else 0
            ocean_fill_june = np.zeros(N)
            if C_june_max > 0 and ocean_cap > 0:
                 ocean_fill_june = fractional_knapsack(
                     capacity=C_june_max * ocean_cap,
                     values_per_unit=svc[:, 1], # Air cost as value proxy
                     weights_per_unit=vol,
                     max_items=S_june
                 )
            month_0_orders[:, 2] += ocean_fill_june
            R_june = S_june - ocean_fill_june
            if np.sum(R_june) > 1e-6:
                cost_residuals_air = np.sum(R_june * svc[:, 1]) + (sfc[1] if np.sum(month_0_orders[:, 1]) < 1e-6 else 0)
                cost_residuals_ocean = ocean_cc + (sfc[2] if np.sum(month_0_orders[:, 2]) < 1e-6 else 0)
                volume_residuals = np.sum(R_june * vol)
                if volume_residuals > ocean_cap or cost_residuals_air < cost_residuals_ocean:
                    month_0_orders[:, 1] += R_june # Add residuals to Air
                else:
                    month_0_orders[:, 2] += R_june # Add residuals to Ocean

        # Cover July (t=4) shortage S[:,4]: Use Ocean (Lead Time 3 -> arrives start month 3 - TOO LATE!)
        # Heuristic Limitation: This heuristic cannot cover July/Aug demand with Ocean ordered in Month 0.
        # Need to decide how to handle this.
        # Assume the heuristic *must* cover the demand using the *available* methods,
        # even if suboptimal according to its own logic. The fastest available is Express.
        S_july = shortages[:, 4].copy()
        if np.sum(S_july) > 1e-6:
             month_0_orders[:, 0] += S_july # Use Express as it's the only one arriving before July (at start month 1)

        # Cover August (t=5) shortage S[:,5]: Use Ocean (Lead Time 3 -> TOO LATE!)
        S_august = shortages[:, 5].copy()
        if np.sum(S_august) > 1e-6:
             month_0_orders[:, 0] += S_august # Use Express

        final_orders = month_0_orders # Shape (N, 3)
        total_cost = calculate_total_cost(final_orders, data)

        return total_cost

    # --- Corresponding except block ---
    except Exception as e:
        print(f"Error in proposed solver: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')