import numpy as np
import math

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

    # --- Simulate inventory flow ---
    inventory = np.zeros((N, T))
    current_inv = I0.copy()
    total_holding_cost = 0

    for t in range(T):
        # Arrivals at start of month t
        arrivals_t = transit[:, t].copy().astype(float) # Ensure arrivals_t starts as float
        for j in range(len(lead_times)): # 0:Exp, 1:Air, 2:Ocean
            arrival_month = lead_times[j] - 1
            if arrival_month == t:
                 arrivals_t += orders[:, j]

        available_inv = current_inv + arrivals_t
        demand_met = np.minimum(available_inv, D[:, t])
        end_inv = available_inv - D[:, t]
        inventory[:, t] = np.maximum(0, end_inv)
        current_inv = inventory[:, t]
        total_holding_cost += np.sum(inventory[:, t] * h)

    # --- Calculate costs ---
    total_purchase_cost = np.sum(orders * p[:, np.newaxis])

    # Shipping Costs
    total_shipping_cost = 0
    # Express & Air
    for j in range(2):
         if np.sum(orders[:, j]) > 1e-6:
              total_shipping_cost += sfc[j]
              total_shipping_cost += np.sum(orders[:, j] * svc[:, j])

    # Ocean
    ocean_orders = orders[:, 2]
    if np.sum(ocean_orders) > 1e-6:
        total_shipping_cost += sfc[2]
        total_volume = np.sum(ocean_orders * vol)
        num_containers = math.ceil(total_volume / ocean_cap) if total_volume > 1e-6 else 0
        total_shipping_cost += num_containers * ocean_cc

    total_cost = total_purchase_cost + total_shipping_cost + total_holding_cost
    return total_cost
# End of copied calculate_total_cost
def solve_naive(data):
    try:
        N = data['N_PRODUCTS']
        T = data['T_MONTHS']
        D = data['demand']
        I0 = data['initial_inventory']
        transit = data['in_transit']
        lead_times = data['lead_times'] # [1, 2, 3]

        proj_inv = np.zeros((N, T))
        shortages = np.zeros((N, T))
        current_inv = I0.copy()

        for t in range(T):
            arrivals_t = transit[:, t].copy()
            available_inv = current_inv + arrivals_t
            shortages[:, t] = np.maximum(0, D[:, t] - available_inv)
            end_inv = available_inv - D[:, t]
            proj_inv[:, t] = np.maximum(0, end_inv)
            current_inv = proj_inv[:, t]

        # Order S[i, t] using the method with lead time <= t, choosing the fastest.
        month_0_orders = np.zeros((N, 3)) # Orders placed now: [Exp, Air, Ocean]

        # Check shortages for each future month t and order using fastest possible method
        for t in range(T): # t=0 (Mar) to t=5 (Aug)
            S_t = shortages[:, t]
            if np.sum(S_t) < 1e-6:
                continue # No shortage this month

            # Find fastest method that arrives by start of month t
            # Order month 0, arrives start month lead_time
            best_method_idx = -1
            min_lead_time_for_t = float('inf')

            # Can Express (lead=1) arrive by month t? Yes if t >= 1
            if t >= lead_times[0]: # lead_times[0]=1
                 if lead_times[0] < min_lead_time_for_t:
                     min_lead_time_for_t = lead_times[0]
                     best_method_idx = 0 # Express

            # Can Air (lead=2) arrive by month t? Yes if t >= 2
            if t >= lead_times[1]: # lead_times[1]=2
                 if lead_times[1] < min_lead_time_for_t:
                     min_lead_time_for_t = lead_times[1]
                     best_method_idx = 1 # Air

            # Can Ocean (lead=3) arrive by month t? Yes if t >= 3
            if t >= lead_times[2]: # lead_times[2]=3
                 if lead_times[2] < min_lead_time_for_t:
                     min_lead_time_for_t = lead_times[2]
                     best_method_idx = 2 # Ocean

            # Assign the shortage S_t to the chosen method's order
            if best_method_idx != -1:
                 month_0_orders[:, best_method_idx] += S_t
            # Else: Shortage cannot be met by orders placed now (e.g., shortage for t=0)
            # This heuristic ignores unmet demand cost.

        final_orders = month_0_orders
        total_cost = calculate_total_cost(final_orders, data)

        return total_cost

    except Exception as e:
        print(f"Error in naive solver: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')