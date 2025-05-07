import numpy as np
import pandas as pd

# Define cost functions
def express_cost(products):
    return np.dot(products, C["V"][:, 0]) + C["F"][0]

def air_cost(products):
    return np.dot(products, C["V"][:, 1]) + C["F"][1]

def ocean_cost(containers):
    return containers * C["C"] + C["F"][2] if containers > 0 else 0

# Fractional knapsack function
def fractional_knapsack(capacity, max_items):
    volumes = V
    prices = C["V"][:, 1]
    items = [(i, prices[i], volumes[i], prices[i] / volumes[i]) for i in range(len(volumes))]
    items.sort(key=lambda x: x[3], reverse=True)

    remaining_capacity = capacity
    selected_items = np.zeros(N)

    for i, price, volume, ratio in items:
        if remaining_capacity <= 0:
            break
        can_take = min(max_items[i], remaining_capacity / volume)
        selected_items[i] += can_take
        remaining_capacity -= can_take * volume

    return selected_items


# Read the Excel file from the 'Demand' sheet
file_path = "data/OR113-2_midtermProject_data.xlsx"
df_demand = pd.read_excel(file_path, sheet_name="Demand")
N = df_demand.shape[0] - 1  # Number of products
T = df_demand.shape[1] - 2  # Number of months

# Initialize demand and inventory arrays
I = np.zeros([N, T])
D = np.zeros([N, T])
I_0 = np.zeros([N])

for i in range(N):
    I_0[i] = df_demand.iloc[i + 1, 1]
    for t in range(T):
        D[i, t] = df_demand.iloc[i + 1, t + 2]

# Read the Excel file from the 'In-transit' sheet
df_in_transit = pd.read_excel(file_path, sheet_name="In-transit")
for i in range(N):
    for t in range(df_in_transit.shape[1] - 1):
        I[i, t] = df_in_transit.iloc[i + 1, t + 1]

# Read the Excel file from the 'Shipping cost' and 'Inventory cost' sheets
df_shipping_cost = pd.read_excel(file_path, sheet_name="Shipping cost")
df_inventory_cost = pd.read_excel(file_path, sheet_name="Inventory cost")
J = df_shipping_cost.shape[1] - 1  # Number of shipping methods

# Initialize cost and volume dictionaries
C = {
    "H": np.zeros([N]),
    "P": np.zeros([N]),
    "V": np.zeros([N, J]),
    "F": np.array([100, 80, 50]),
    "C": 2750,
}
V = np.zeros([N])
V_C = 30

for i in range(N):
    C["H"][i] = df_inventory_cost.iloc[i, 3]
    C["P"][i] = df_inventory_cost.iloc[i, 2]
    V[i] = df_shipping_cost.iloc[i, 3]
    for j in range(J):
        C["V"][i, j] = df_shipping_cost.iloc[i, j + 1] if j < J - 1 else 0

T_lead = np.array([1, 2, 3])  # Lead times for each shipping method

# Initialize order DataFrames and total cost
months = [f"Month {t+1}" for t in range(6)]
order_dataframes = {
    month: pd.DataFrame(
        {
            "Product": [i + 1 for i in range(N)],
            "Express Delivery": 0,
            "Air Freight": 0,
            "Ocean Freight": 0,
        }
    )
    for month in months
}
total_cost = 0

# Calculate missing inventory
missing_inventory = np.zeros([N, T])
current_inventory = I_0.copy()

for t in range(T):
    current_inventory += I[:, t]
    remaining_inventory = current_inventory - D[:, t]
    missing_inventory[:, t] = np.maximum(-remaining_inventory, 0)
    current_inventory = np.maximum(remaining_inventory, 0)
    total_cost += np.dot(current_inventory, C["H"])

# Place express delivery order for Month 2 (April)
express_delivery_for_april = missing_inventory[:, 1]
order_dataframes["Month 1"]["Express Delivery"] += express_delivery_for_april
total_cost += express_cost(express_delivery_for_april)

# Place air freight order for Month 3 (May)
air_freight_for_may = missing_inventory[:, 2]
order_dataframes["Month 1"]["Air Freight"] += air_freight_for_may
total_cost += air_cost(air_freight_for_may)

# Calculate needed containers
weights_per_month = np.zeros(T)
containers_per_month = np.zeros(T)

for t in range(3, T):
    weights_per_month[t] = np.dot(missing_inventory[:, t], V)
    containers_per_month[t] = int(weights_per_month[t] / V_C)

filled_items = np.zeros([N, T])
differences = np.zeros([N, T])

# Fill containers using knapsack algorithm
for t in range(3, T):
    total_capacity = containers_per_month[t] * V_C
    missing_items = missing_inventory[:, t]
    selected_items = fractional_knapsack(total_capacity, missing_items)
    filled_items[:, t] += selected_items
    differences[:, t] = missing_items - selected_items
    order_dataframes[f"Month {t-2}"]["Ocean Freight"] += selected_items

# Handle remaining differences
for t in range(3, T):
    remaining_differences = differences[:, t]
    air_freight_cost = air_cost(remaining_differences)
    add_container_cost = ocean_cost(containers_per_month[t] + 1) - ocean_cost(containers_per_month[t])

    # choose which option is cheaper and persue that order
    if air_freight_cost < add_container_cost:
        total_cost += air_freight_cost + ocean_cost(containers_per_month[t])
        order_dataframes[f"Month {t-1}"]["Air Freight"] += remaining_differences
    else:
        total_cost += ocean_cost(containers_per_month[t] + 1)
        order_dataframes[f"Month {t-2}"]["Ocean Freight"] += remaining_differences

# add expenses of buying and storing items
total_missing_products = np.sum(missing_inventory, axis=1)
total_cost += np.dot(total_missing_products, C["P"]) # buying items
total_cost += np.dot(total_missing_products, C["H"]) # storing items for one month

# Display the updated order schedule
print("\nOrder Schedule:")
for month, df in order_dataframes.items():
    print(f"\n{month}:")
    print(df.round(2).to_string(index=False))

# Display the total cost
print(f"\nTotal Cost: ${total_cost:,.2f}")