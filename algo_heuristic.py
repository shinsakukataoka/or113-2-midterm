import numpy as np

def express_cost(q, cv1, cf):
    """q : (N,) quantities shipped express in a month"""
    return (q @ cv1) + (cf[0] if q.sum() > 0 else 0.0)


def air_cost(q, cv2, cf):
    """q : (N,) quantities shipped air in a month"""
    return (q @ cv2) + (cf[1] if q.sum() > 0 else 0.0)

def ocean_cost(containers, cont_cost, cf):
    return containers * cont_cost + (cf[2] if containers > 0 else 0.0)

def fractional_knapsack(capacity, volumes, prices, max_items):
    """
    Greedy fill on price/vol ratio (exactly the logic in the Excel script).
    Returns an (N,) vector of selected quantities.
    """
    items = [
        (i, prices[i] / volumes[i] if volumes[i] > 0 else 0.0)
        for i in range(len(volumes))
    ]
    items.sort(key=lambda x: x[1], reverse=True)

    remaining = capacity
    picked    = np.zeros_like(max_items, dtype=float)

    for i, _ in items:
        if remaining <= 0:
            break
        take = min(max_items[i], remaining / volumes[i]) if volumes[i] > 0 else 0
        picked[i]  += take
        remaining  -= take * volumes[i]

    return picked


# ---------- main heuristic cost -------------------------------------------

def heuristic_cost(
    N, T, D, I0, I,
    CH, CP, CV1, CV2, CF,
    Vi, cont_cost,
    container_capacity=30
):
    """
    Parameters mirror what `main_experiment.py` passes.

    Returns
    -------
    total_cost : float
    """
    # ------------------ 1. pre-compute shortages & holding cost -------------
    missing = np.zeros((N, T))
    current_inv   = I0.copy()
    holding_cost  = 0.0

    for t in range(T):
        current_inv += I[:, t]                       # arrivals at start of month
        remaining    = current_inv - D[:, t]
        missing[:, t] = np.maximum(-remaining, 0)
        current_inv   = np.maximum(remaining, 0)     # carry-over
        holding_cost += current_inv @ CH             # charge end-of-month inv

    # ------------------ 2. month-level decisions ----------------------------
    total_cost = 0.0

    # month indices: t = 0 → March, 1 → April, 2 → May, …
    # Express for April (t = 1)
    if T > 1:
        q_exp = missing[:, 1]
        total_cost += express_cost(q_exp, CV1, CF)

    # Air for May (t = 2)
    if T > 2:
        q_air = missing[:, 2]
        total_cost += air_cost(q_air, CV2, CF)

    # Months ≥ 4  (t ≥ 3) – ocean first, then residual decision
    for t in range(3, T):
        need           = missing[:, t].copy()
        weight         = need @ Vi
        containers     = int(weight // container_capacity)

        picked = fractional_knapsack(
            containers * container_capacity,
            volumes=Vi,
            prices=CV2,        # same as Excel: sort by air var-cost/volume
            max_items=need
        )

        residual = need - picked

        # cost if we ship only the packed containers
        cost_ocean_base = ocean_cost(containers, cont_cost, CF)
        total_cost     += cost_ocean_base

        # decide residual path
        if residual.sum() > 0:
            cost_air_residual = air_cost(residual, CV2, CF)
            cost_next_container = (
                ocean_cost(containers + 1, cont_cost, CF) - cost_ocean_base
            )

            if cost_air_residual < cost_next_container:
                total_cost += cost_air_residual           # keep same container count
            else:
                total_cost += cost_next_container         # take one more container

    # ------------------ 3. purchase + extra holding -------------------------
    total_ordered = missing.sum(axis=1)
    total_cost   += total_ordered @ CP          # purchase
    total_cost   += total_ordered @ CH          # one extra month holding

    # add the rolling holding cost we tracked earlier
    total_cost += holding_cost
    return total_cost
