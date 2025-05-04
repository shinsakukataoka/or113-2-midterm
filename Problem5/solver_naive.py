import numpy as np
import math
import sys
import os

def calculate_total_cost(orders, data):
    try:
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
        lead_times = data['lead_times'] # [1, 2, 3]
    except KeyError as e:
        print(f"Error: Missing key in instance_data during cost calculation: {e}")
        return float('inf')

    # --- Simulate inventory flow ---
    inventory = np.zeros((N, T))
    current_inv = I0.copy()
    total_holding_cost = 0

    for t in range(T): # Loop through months t=0 (Mar) to t=5 (Aug)
        arrivals_t = transit[:, t].copy()
        for j in range(len(lead_times)):
            # Order placed month 0, lead time L -> arrives start of month L (index L)
            if lead_times[j] == t : # If arrival month index matches current month index t
                 arrivals_t += orders[:, j]

        available_inv = current_inv + arrivals_t
        demand_met = np.minimum(available_inv, D[:, t])
        end_inv = available_inv - D[:, t]
        inventory[:, t] = np.maximum(0, end_inv)
        current_inv = inventory[:, t]
        total_holding_cost += np.sum(inventory[:, t] * h)

    # --- Calculate total costs based on the simulation and orders ---
    total_purchase_cost = np.sum(orders * p[:, np.newaxis])

    total_shipping_cost = 0
    # Express
    if np.sum(orders[:, 0]) > 1e-6:
        total_shipping_cost += sfc[0] + np.sum(orders[:, 0] * svc[:, 0])
    # Air
    if np.sum(orders[:, 1]) > 1e-6:
        total_shipping_cost += sfc[1] + np.sum(orders[:, 1] * svc[:, 1])
    # Ocean
    ocean_orders = orders[:, 2]
    if np.sum(ocean_orders) > 1e-6:
        total_shipping_cost += sfc[2]
        total_volume = np.sum(ocean_orders * vol)
        num_containers = 0
        if ocean_cap > 0 and total_volume > 1e-6:
             num_containers = math.ceil(total_volume / ocean_cap)
        total_shipping_cost += num_containers * ocean_cc

    total_cost = total_purchase_cost + total_shipping_cost + total_holding_cost
    return total_cost
# --- End of copied calculate_total_cost ---


def solve_naive(data):
    """
    Implements a simple naive heuristic for the single-decision point problem:
    1. Calculate total net requirement for each product over the horizon.
    2. Order this entire quantity at t=0 using only Express delivery.
    3. Calculate the resulting total cost accurately.
    """
    try:
        # ---------- unpack data -------------------------------------------
        N = data['N_PRODUCTS']
        T = data['T_MONTHS']
        D = data['demand']                     # shape (N, T)
        I0 = data['initial_inventory']         # shape (N,)
        transit = data['in_transit']           # shape (N, T)

        # ---------- Calculate Net Requirements -----------------------------
        # Total demand over horizon T
        total_demand = D.sum(axis=1) # Sum demand across all months for each product
        # Total initial supply (on-hand + all in-transit)
        total_initial_supply = I0 + transit.sum(axis=1)
        # Net requirement = max(0, total demand - total initial supply)
        net_requirement = np.maximum(0, total_demand - total_initial_supply)

        # ---------- Determine Orders (t=0 only) ----------------------------
        # Naive strategy: Order everything via Express (method 0)
        month_0_orders = np.zeros((N, 3)) # Orders: [Express, Air, Ocean]
        month_0_orders[:, 0] = net_requirement # Assign net requirement to Express

        # ---------- Calculate Total Cost -----------------------------------
        # Use the accurate cost calculation function
        total_cost = calculate_total_cost(month_0_orders, data)

        return total_cost

    except Exception as e:
        print(f"Error in naive solver: {e}")
        import traceback
        traceback.print_exc()
        return float('inf') # Return infinity on error

