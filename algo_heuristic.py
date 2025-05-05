import numpy as np

def express_cost(products, cv1, cf):
    return np.dot(products, cv1) + (cf[0] if products.sum() > 0 else 0.0)


def air_cost(products, cv2, cf):
    return np.dot(products, cv2) + (cf[1] if products.sum() > 0 else 0.0)


def ocean_cost(num_containers, cont_cost, cf):
    return num_containers * cont_cost + (cf[2] if num_containers > 0 else 0.0)

def fractional_knapsack(capacity, volumes, prices, max_items):
    #volumes = V
    #prices = C["V"][:, 1]
    items = [(i, prices[i], volumes[i], prices[i] / volumes[i] if volumes[i] > 0 else 0) for i in range(len(volumes))]
    items.sort(key=lambda tup: tup[3], reverse=True)

    remaining_capacity = capacity
    selected_items = np.zeros_like(max_items, dtype=float)

    for i, price, vol, ratio in items:
        if remaining_capacity <= 0:
            break
        if vol <= 0:
            continue
        can_take = min(max_items[i], remaining_capacity / vol)
        selected_items[i] += can_take
        remaining_capacity -= can_take * vol

    return selected_items

def heuristic_cost(
    N, T, D, I0, I,
    CH, CP, CV1, CV2, CF,
    Vi, cont_cost,
    container_capacity=30
):
    missing_inventory = np.zeros((N, T))
    current_inventory = I0.copy()
    holding_cost_acc  = 0.0

    for t in range(T):
        current_inventory += I[:, t]                       # arrivals at start
        remaining = current_inventory - D[:, t]            # after demand
        missing_inventory[:, t] = np.maximum(-remaining, 0)
        current_inventory = np.maximum(remaining, 0)       # carry-over
        holding_cost_acc += np.dot(current_inventory, CH)  # end-month charge

    total_cost = 0

    # — Express for April shortages (t = 1) —
    if T > 1:
        express_for_april = missing_inventory[:, 1]
        total_cost += express_cost(express_for_april, CV1, CF)

    # — Air for May shortages (t = 2) —
    if T > 2:
        air_for_may = missing_inventory[:, 2]
        total_cost += air_cost(air_for_may, CV2, CF)

    # — From June onward (t ≥ 3): ocean first, decide leftovers —
    for t in range(3, T):
        need_t = missing_inventory[:, t].copy()

        # How many full containers can we fill?
        total_volume = np.dot(need_t, Vi)
        full_containers = int(total_volume // container_capacity)

        # Use fractional knapsack to fill those containers optimally
        packed = fractional_knapsack(
            full_containers * container_capacity,
            volumes=Vi,
            prices=CV2,
            max_items=need_t
        )
        leftovers = need_t - packed

        # Cost of shipping the packed containers
        Cost_ocean_base = ocean_cost(full_containers, cont_cost, CF)
        total_cost += Cost_ocean_base

        # Decide how to move the leftovers
        if leftovers.sum() > 0:
            cost_by_air   = air_cost(leftovers, CV2, CF)
            cost_extra_ct = (
                ocean_cost(full_containers + 1, cont_cost, CF) - Cost_ocean_base
            )

            if cost_by_air < cost_extra_ct:
                total_cost += cost_by_air          # keep container count
            else:
                total_cost += cost_extra_ct        # open one more container

    total_ordered = missing_inventory.sum(axis=1)
    total_cost += np.dot(total_ordered, CP)   # purchase
    total_cost += np.dot(total_ordered, CH)   # holding for one extra month

    # Add rolling holding cost from step (1)
    total_cost += holding_cost_acc

    return total_cost