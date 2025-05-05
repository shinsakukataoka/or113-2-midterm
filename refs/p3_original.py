import numpy as np
import pandas as pd
import os # Added for file system operations

# --- Constants ---
INSTANCE_FOLDER = "instances" # Folder where .npz files are stored

# --- Helper Functions (modified slightly for clarity/consistency) ---

def express_cost(products, C_dict):
    """Calculates cost for express shipping."""
    # products: array of quantities for each product
    # C_dict: dictionary containing cost parameters
    variable_cost = np.dot(products, C_dict["V"][:, 0]) # Use Express variable costs (col 0)
    fixed_cost = C_dict["F"][0] if np.sum(products) > 1e-6 else 0 # Add fixed cost only if something is shipped
    return variable_cost + fixed_cost

def air_cost(products, C_dict):
    """Calculates cost for air freight."""
    variable_cost = np.dot(products, C_dict["V"][:, 1]) # Use Air variable costs (col 1)
    fixed_cost = C_dict["F"][1] if np.sum(products) > 1e-6 else 0
    return variable_cost + fixed_cost

def ocean_cost(containers, C_dict):
    """Calculates cost for ocean freight."""
    # containers: number of containers used
    fixed_cost_per_shipment = C_dict["F"][2] # Ocean fixed cost per shipment (not per container)
    cost_per_container = C_dict["C"]
    # Assuming fixed cost applies if at least one container is used
    total_cost = containers * cost_per_container
    if containers > 0:
        total_cost += fixed_cost_per_shipment
    return total_cost


def fractional_knapsack(capacity, items_to_potentially_pack, item_volumes, item_costs_per_unit):
    """
    Solves fractional knapsack: maximize value (cost reduction) within capacity.
    Here, 'value' is represented by the air freight cost saved by sending via ocean.
    We prioritize items with the highest air_cost / volume ratio.

    Args:
        capacity (float): Total volume capacity of the ocean containers.
        items_to_potentially_pack (np.array): Max quantity of each item that *could* be packed (e.g., missing inventory).
        item_volumes (np.array): Volume per unit for each item.
        item_costs_per_unit (np.array): Cost per unit for the alternative (Air Freight).

    Returns:
        np.array: Quantity of each item selected for the knapsack (ocean shipment).
    """
    n_items = len(item_volumes)
    selected_items = np.zeros(n_items)
    remaining_capacity = capacity

    # Calculate value density (air cost saved per unit volume)
    # Avoid division by zero for items with zero volume
    value_density = np.zeros(n_items)
    non_zero_volume_mask = item_volumes > 1e-9
    value_density[non_zero_volume_mask] = item_costs_per_unit[non_zero_volume_mask] / item_volumes[non_zero_volume_mask]

    # Sort items by value density in descending order
    # Keep track of original indices
    indices = np.argsort(value_density)[::-1]

    for i in indices:
        if remaining_capacity <= 1e-9: # Use tolerance for float comparison
            break
        if item_volumes[i] <= 1e-9: # Skip items with negligible volume
            continue

        # How much of item i can we potentially take?
        max_can_take = items_to_potentially_pack[i]

        # How much of item i fits in the remaining capacity?
        fits_in_volume = remaining_capacity / item_volumes[i]

        # Take the minimum of what's needed and what fits
        take_amount = min(max_can_take, fits_in_volume)

        if take_amount > 1e-9: # Only add if taking a non-negligible amount
            selected_items[i] = take_amount
            remaining_capacity -= take_amount * item_volumes[i]

    # Due to potential floating point inaccuracies, ensure we don't exceed capacity significantly
    # This step might not be strictly necessary with the fractional approach but is a safeguard.
    # total_volume_packed = np.dot(selected_items, item_volumes)
    # if total_volume_packed > capacity + 1e-6:
    #    print(f"Warning: Knapsack slightly exceeded capacity ({total_volume_packed} > {capacity}). Adjusting.")
       # Basic adjustment: slightly reduce the last added item, though more robust methods exist
       # For fractional, this overshoot should be minimal or zero if implemented correctly.

    return selected_items

# --- Main Loop ---

# Get list of instance files
try:
    instance_files = [f for f in os.listdir(INSTANCE_FOLDER) if f.endswith('.npz')]
    instance_files.sort() # Sort for consistent processing order
    if not instance_files:
        print(f"Error: No '.npz' instance files found in folder '{INSTANCE_FOLDER}'.")
        exit()
except FileNotFoundError:
    print(f"Error: Instance folder '{INSTANCE_FOLDER}' not found.")
    exit()


print(f"Found {len(instance_files)} instances in '{INSTANCE_FOLDER}'. Processing...")

# Loop through each instance file
for instance_file in instance_files:
    instance_path = os.path.join(INSTANCE_FOLDER, instance_file)
    print(f"\n--- Processing Instance: {instance_file} ---")

    try:
        # Load data from the .npz file
        data = np.load(instance_path)

        # --- Extract Data and Map to Script Variables ---
        N = int(data['N_PRODUCTS'])        # Number of products
        T = int(data['T_MONTHS'])         # Number of months (planning horizon)

        D = data['demand']               # Demand (N x T)
        I_0 = data['initial_inventory']  # Initial Inventory (N,)
        I_transit = data['in_transit']   # In-transit Inventory (N x T) - I_transit[:,t] arrives end of t / start of t+1

        V = data['volume_cbm']           # Volume per product (N,)
        V_C = float(data['ocean_container_capacity_cbm']) # Ocean container capacity

        # Lead times [Express, Air, Ocean]
        T_lead = data['lead_times'].astype(int) # Should be [1, 2, 3] usually

        # Cost dictionary
        C = {
            "H": data['holding_cost_per_unit'], # Holding cost (N,)
            "P": data['purchase_cost_per_unit'],# Purchase cost (N,)
            "F": data['shipping_fixed_cost'],  # Fixed costs [Express, Air, Ocean] (3,)
            "C": float(data['ocean_container_cost']), # Cost per Ocean container
            "V": np.zeros((N, 3))               # Variable ship cost (N x 3) - Initialize
        }
        # Populate variable shipping costs: Express (col 0), Air (col 1)
        C["V"][:, 0:2] = data['shipping_variable_cost']
        # Ocean variable cost is assumed 0 per unit (col 2 remains 0)

        # --- Initialization for the current instance ---
        # On-hand inventory simulation array (N x T), tracks inventory *at the end* of month t
        inventory_end_month = np.zeros([N, T])
        # In-transit inventory arriving at the *start* of month t
        inventory_arriving = np.zeros([N, T])

        # Populate inventory arriving based on lead times and in-transit data
        # I_transit[:, t] arrives end of month t, available start of t+1
        for t_arrival_end in range(T):
            t_available_start = t_arrival_end + 1
            if t_available_start < T: # Check if arrival month is within horizon
                 inventory_arriving[:, t_available_start] += I_transit[:, t_arrival_end]

        # Initialize order DataFrames and total cost for this instance
        months = [f"Month {t+1}" for t in range(T)] # Adjust based on actual T
        # Stores orders *placed* at the beginning of Month t+1 (index t)
        order_dataframes = {
            month: pd.DataFrame(
                {
                    "Product": [i + 1 for i in range(N)],
                    "Express Delivery": np.zeros(N),
                    "Air Freight": np.zeros(N),
                    "Ocean Freight": np.zeros(N),
                    "Containers": 0.0 # Track containers used for ocean freight placed this month
                }
            )
            for month in months
        }
        total_cost = 0.0
        total_purchase_cost = 0.0
        total_holding_cost = 0.0
        total_shipping_cost = 0.0

        # --- Simulation and Ordering Logic ---

        # Initial inventory at the start of month 1 (index 0)
        current_inventory = I_0.copy()

        # Arrays to store order decisions (N x T x 3 for prod, month_placed, method)
        # orders_placed[i, t, j] = quantity of product i ordered via method j, placed start of month t+1
        orders_placed = np.zeros((N, T, 3)) # 0: Express, 1: Air, 2: Ocean

        for t in range(T): # Loop through months 0 to T-1 (representing Month 1 to Month T)
            month_name = months[t]

            # 1. Inventory available at the START of month t+1
            inventory_start = current_inventory + inventory_arriving[:, t]

            # 2. Demand for month t+1
            demand_t = D[:, t]

            # 3. Calculate unmet demand (potential shortage) for month t+1
            unmet_demand = np.maximum(0, demand_t - inventory_start)

            # 4. Determine Orders to Place at START of Month t+1 (Index t)
            #    These orders will arrive later based on lead times.
            #    Simple Heuristic: Cover future projected shortages.

            # Orders placed now (start of t+1) to cover future needs
            # Need to look ahead based on lead times.

            # Express order placed now (t) arrives start of t+T_lead[0] (t+1)
            if t + T_lead[0] < T:
                # Project inventory for the arrival month of express order
                # This is complex; simplified heuristic: order unmet demand for arrival month
                # A better heuristic would project inventory more accurately
                projected_shortage_express_arrival = np.maximum(0, D[:, t + T_lead[0]] - 0) # Simplified: assume 0 inventory then
                # For this simple heuristic, we order based on immediate need if lead time is 1
                if T_lead[0] == 1:
                   order_express = unmet_demand.copy() # Order immediate unmet demand via express
                   orders_placed[:, t, 0] = order_express
                   inventory_arriving[:, t + T_lead[0]] += order_express # Arrives next month
                   cost_express = express_cost(order_express, C)
                   total_shipping_cost += cost_express
                   order_dataframes[month_name]["Express Delivery"] += order_express

            # Air order placed now (t) arrives start of t+T_lead[1] (t+2)
            if t + T_lead[1] < T:
                # Simplified heuristic: Order projected shortage for arrival month via Air
                projected_shortage_air_arrival = np.maximum(0, D[:, t + T_lead[1]] - 0) # Simplified
                # Let's refine: Use Air for need in t+2 if not covered otherwise
                # This heuristic needs rework - original one was specific to T=6
                # Let's use the original heuristic structure adapted for general T
                # Order Air now (t) for needs in month t+T_lead[1]
                if t + T_lead[1] < T:
                   needs_in_air_arrival_month = D[:, t + T_lead[1]].copy()
                   # Very simple: order all needs for that future month via Air now
                   # (Ignores inventory projection - needs improvement for better heuristic)
                   order_air = needs_in_air_arrival_month
                   orders_placed[:, t, 1] = order_air
                   inventory_arriving[:, t + T_lead[1]] += order_air # Arrives in T_lead[1] months
                   cost_air = air_cost(order_air, C)
                   total_shipping_cost += cost_air
                   order_dataframes[month_name]["Air Freight"] += order_air

            # Ocean order placed now (t) arrives start of t+T_lead[2] (t+3)
            # Use Ocean for needs in month t + T_lead[2] and potentially beyond
            # Use Knapsack for Ocean freight.
            items_needing_shipping_future = np.zeros(N)
            start_ocean_lookahead = t + T_lead[2]
            if start_ocean_lookahead < T:
                # Aggregate demand from the ocean arrival month onwards
                items_needing_shipping_future = np.sum(D[:, start_ocean_lookahead:], axis=1)
                # (Again, this is simplified - ignores projected inventory)

                if np.sum(items_needing_shipping_future) > 1e-6:
                    # Calculate total volume needed for these items
                    total_volume_needed = np.dot(items_needing_shipping_future, V)
                    # Calculate number of containers (round up)
                    num_containers = int(np.ceil(total_volume_needed / V_C)) if total_volume_needed > 1e-6 else 0

                    if num_containers > 0:
                        container_capacity_total = num_containers * V_C
                        # Costs if we use Air instead (for knapsack value)
                        air_costs_for_these_items = C["V"][:, 1] * items_needing_shipping_future

                        # Use knapsack to fill containers efficiently
                        ocean_shipment = fractional_knapsack(
                            capacity=container_capacity_total,
                            items_to_potentially_pack=items_needing_shipping_future,
                            item_volumes=V,
                            item_costs_per_unit=C["V"][:, 1] # Use air cost as 'value density' proxy
                        )

                        # Record the ocean order placed in month t
                        orders_placed[:, t, 2] = ocean_shipment
                        # Update arriving inventory for the future
                        inventory_arriving[:, t + T_lead[2]] += ocean_shipment
                        # Calculate and add ocean cost
                        cost_ocean = ocean_cost(num_containers, C)
                        total_shipping_cost += cost_ocean
                        # Store order details
                        order_dataframes[month_name]["Ocean Freight"] += ocean_shipment
                        order_dataframes[month_name]["Containers"] = num_containers # Store container count

                        # TODO: Handle items NOT selected by knapsack (items_needing_shipping_future - ocean_shipment)
                        # A simple approach: ship remaining via Air, placed in month t+1?
                        # Or adjust the next month's orders? This heuristic gets complex fast.
                        # For now, this simple heuristic *might leave some future demand unmet*
                        # or rely on later express/air orders if the logic repeats.


            # --- Update Inventory for End of Month t ---
            inventory_after_demand = inventory_start - demand_t
            inventory_end_month[:, t] = np.maximum(0, inventory_after_demand) # Cannot be negative

            # Calculate holding cost for inventory held at the END of month t
            holding_cost_t = np.dot(inventory_end_month[:, t], C["H"])
            total_holding_cost += holding_cost_t

            # Carry over inventory to the start of the next month (t+1)
            if t + 1 < T:
                current_inventory = inventory_end_month[:, t].copy()
            # Else: loop finishes, final inventory state is inventory_end_month[:, T-1]


        # --- Calculate Total Purchase Cost ---
        # Sum all orders placed across all months and methods
        total_ordered_per_product = np.sum(orders_placed, axis=(1, 2))
        total_purchase_cost = np.dot(total_ordered_per_product, C["P"])

        # --- Calculate Final Total Cost ---
        total_cost = total_purchase_cost + total_holding_cost + total_shipping_cost

        # --- Display Results for the Instance ---
        print("\nOrder Schedule (Orders placed at start of Month):")
        # Need pandas again just for nice display formatting
        import pandas as pd
        pd.options.display.float_format = '{:,.2f}'.format
        for t in range(T):
            month = months[t]
            df = order_dataframes[month]
            # Only print if orders were placed or containers used
            if df["Express Delivery"].sum() > 1e-6 or \
               df["Air Freight"].sum() > 1e-6 or \
               df["Ocean Freight"].sum() > 1e-6:
                 print(f"\n{month}:")
                 # Display only relevant columns if containers is always 0 for this month
                 cols_to_show = ["Product", "Express Delivery", "Air Freight", "Ocean Freight"]
                 if df["Containers"][0] > 0: # Check first row's container value as proxy
                     cols_to_show.append("Containers")
                     # Show containers only once per month's table
                     print(f"  Ocean Containers Used: {df['Containers'][0]:.0f}")
                 print(df[cols_to_show].round(2).to_string(index=False))


        print("\n--- Cost Summary ---")
        print(f"Total Purchase Cost: ${total_purchase_cost:,.2f}")
        print(f"Total Holding Cost:  ${total_holding_cost:,.2f}")
        print(f"Total Shipping Cost: ${total_shipping_cost:,.2f}")
        print(f"Total Cost:          ${total_cost:,.2f}")


    except FileNotFoundError:
        print(f"Error: Could not find instance file {instance_path}")
    except KeyError as e:
        print(f"Error: Missing expected key in {instance_file}: {e}")
        # Optionally print available keys: print(f"Available keys: {list(data.keys())}")
    except Exception as e:
        print(f"An unexpected error occurred processing {instance_file}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

print("\n--- Instance processing finished. ---")