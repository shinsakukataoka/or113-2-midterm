from __future__ import annotations

import math
from typing import Dict, Tuple
import numpy as np

# --------------------------------------------------
# Utility helpers
# --------------------------------------------------

def _fractional_knapsack(capacity: float,
                         volumes: np.ndarray,
                         savings: np.ndarray,
                         max_take: np.ndarray) -> Tuple[np.ndarray, float]:
    """Greedy fractional knapsack that maximises *total saving*.

    Parameters
    ----------
    capacity : float
        Remaining container capacity in CBM.
    volumes : (N,) array
        Volume per unit for each product.
    savings : (N,) array
        Saving per unit (versus Air) if shipped by Ocean.
        Negative values are ignored.
    max_take : (N,) array
        Maximum number of units we *could* move via Ocean for this round.

    Returns
    -------
    selected : (N,) array
        Units allocated to the current Ocean shipment.
    utilised_volume : float
        Total volume actually used (≤ capacity).
    """
    n = len(volumes)
    selected = np.zeros(n)

    # value density = saving per CBM
    with np.errstate(divide="ignore", invalid="ignore"):
        density = savings / volumes
    density[np.isnan(density)] = 0.0
    density[density < 0] = 0.0  # drop negative‑saving candidates

    order = np.argsort(-density)  # descending by density
    remaining_cap = capacity

    for i in order:
        if density[i] <= 0 or remaining_cap <= 1e-9:
            break
        take = min(max_take[i], remaining_cap / volumes[i])
        if take <= 1e-9:
            continue
        selected[i] = take
        remaining_cap -= take * volumes[i]

    utilised = capacity - remaining_cap
    return selected, utilised


# --------------------------------------------------
# Cost accounting
# --------------------------------------------------

def _per_unit_ocean_cost(container_cost: float,
                         fixed_cost: float,
                         capacity_cbm: float,
                         utilisation: float) -> float:
    """Return *effective* per‑unit cost for Ocean under given utilisation rate."""
    if utilisation <= 0:
        return math.inf  # cannot divide by zero
    return (container_cost + fixed_cost) / (capacity_cbm * utilisation)


def _arrival_month(current_month: int, lead: int) -> int:
    return current_month + lead


# --------------------------------------------------
# Main solver
# --------------------------------------------------

def solve_proposed(instance: Dict) -> float:
    """Improved heuristic.

    Parameters
    ----------
    instance : dict
        Data dictionary exactly as produced by `generate_instance_data`.

    Returns
    -------
    float
        Total cost for the instance; `math.inf` on error.
    """
    try:
        # --- Unpack ---------------------------------------------------------
        N: int = instance["N_PRODUCTS"]
        T: int = instance["T_MONTHS"]

        D:  np.ndarray = instance["demand"].astype(float)          # (N,T)
        I0: np.ndarray = instance["initial_inventory"].astype(float)
        IT: np.ndarray = instance["in_transit"].astype(float)      # (N,T)

        h:  np.ndarray = instance["holding_cost_per_unit"].astype(float)  # (N,)
        p:  np.ndarray = instance["purchase_cost_per_unit"].astype(float)
        svc: np.ndarray = instance["shipping_variable_cost"].astype(float)  # (N,2)
        sfc: np.ndarray = instance["shipping_fixed_cost"].astype(float)     # (3,)

        vol: np.ndarray = instance["volume_cbm"].astype(float)      # (N,)
        CC: float = float(instance["ocean_container_cost"])
        cap_cbm: float = float(instance["ocean_container_capacity_cbm"])
        lead: np.ndarray = instance["lead_times"].astype(int)       # (3,)

        # mode indices for readability
        EXPRESS, AIR, OCEAN = 0, 1, 2

        # Storage for accounting
        pending_arrivals = np.zeros((N, T + max(lead)))
        orders_placed    = np.zeros((N, T, 3))

        # Add initial in‑transit inventory into pending_arrivals
        for m in range(T):
            pending_arrivals[:, m+1] += IT[:, m]

        inventory = I0.astype(float).copy()

        total_purchase = 0.0
        total_ship_var = 0.0
        total_ship_fix = 0.0
        total_hold     = 0.0

        # ---------------- rolling months -----------------------------------
        for t in range(T):
            # 1. Arrivals available at *start* of month t
            inventory += pending_arrivals[:, t]

            # 2. Satisfy demand in month t
            demand_t = D[:, t]
            satisfied = np.minimum(inventory, demand_t)
            inventory -= satisfied
            shortage   = demand_t - satisfied   # unmet so far but *must* be filled by orders arriving later

            # 3. Forecast SHORTAGES for every future month
            future_short = np.zeros((N, T))  # will fill rows for n, cols for m
            inv_copy   = inventory.copy()
            arr_copy   = pending_arrivals[:, (t+1):].copy()  # arrivals after current month
            for m in range(t+1, T):
                inv_copy += arr_copy[:, 0]  # arrivals for month m
                arr_copy   = arr_copy[:, 1:]
                dem_m      = D[:, m]
                sat_m      = np.minimum(inv_copy, dem_m)
                inv_copy  -= sat_m
                future_short[:, m] = dem_m - sat_m

            # 4. Determine order quantities in current month ----------------
            # We will fill shortages for months >= t+lead[k] with mode k.
            # Build per‑unit cost matrices.
            ocean_unit_cost_est = _per_unit_ocean_cost(CC, sfc[OCEAN], cap_cbm, utilisation=1.0)  # assume full utilisation

            # Track whether we already paid a mode's fixed cost this month
            fixed_paid = [False, False, False]

            # a) Allocate Ocean first (cheapest but slow, capacity‑limited)
            ocean_arrival_month = _arrival_month(t, lead[OCEAN])
            if ocean_arrival_month < T:
                # Candidate shortages are months >= arrival
                cand_mask = future_short[:, ocean_arrival_month:] > 0
                if np.any(cand_mask):
                    # Aggregate by product (sum over months) to keep container calcs simple
                    cand_needed = future_short[:, ocean_arrival_month:].sum(axis=1)

                    # Potential saving vs Air (ignoring extra holding for now)
                    # Effective Air cost per unit if ordered now:
                    months_air_hold = (ocean_arrival_month - _arrival_month(t, lead[AIR]))
                    air_unit_cost = svc[:, AIR] + months_air_hold * h
                    save_vs_air   = air_unit_cost - ocean_unit_cost_est

                    # If saving <= 0 -> never beneficial for that product
                    selected, util_vol = _fractional_knapsack(capacity=cap_cbm,
                                                               volumes=vol,
                                                               savings=save_vs_air,
                                                               max_take=cand_needed)
                    utilisation_rate = util_vol / cap_cbm
                    if utilisation_rate >= 0.70:  # only ship if ≥70 % full
                        # Re‑compute per‑unit ocean cost with actual utilisation
                        ocean_unit_true = _per_unit_ocean_cost(CC, sfc[OCEAN], cap_cbm, utilisation_rate)

                        # Ship selected units
                        if util_vol > 1e-9:
                            # Pay container & fixed cost
                            total_ship_var += CC  # container cost itself
                            if not fixed_paid[OCEAN]:
                                total_ship_fix += sfc[OCEAN]
                                fixed_paid[OCEAN] = True

                            # Record arrivals and purchase cost
                            pending_arrivals[:, ocean_arrival_month] += selected
                            orders_placed[:, t, OCEAN] += selected

                            total_purchase += np.dot(selected, p)

                            # Zero out shipped units within future_short
                            remaining = selected.copy()
                            for m in range(ocean_arrival_month, T):
                                take = np.minimum(remaining, future_short[:, m])
                                future_short[:, m] -= take
                                remaining -= take
                                if remaining.sum() <= 1e-9:
                                    break

            # b) Allocate Air for shortages that can still wait lead[AIR] months
            air_arrival_month = _arrival_month(t, lead[AIR])
            if air_arrival_month < T:
                air_short = future_short[:, air_arrival_month]
                if np.sum(air_short) > 1e-9:
                    if not fixed_paid[AIR]:
                        total_ship_fix += sfc[AIR]
                        fixed_paid[AIR] = True
                    total_ship_var += np.dot(air_short, svc[:, AIR])
                    pending_arrivals[:, air_arrival_month] += air_short
                    orders_placed[:, t, AIR] += air_short
                    total_purchase += np.dot(air_short, p)
                    future_short[:, air_arrival_month] = 0

            # c) Allocate Express for next‑month shortages (lead 1) & immediate unmet
            exp_arrival_month = _arrival_month(t, lead[EXPRESS])
            if exp_arrival_month < T:
                exp_short = future_short[:, exp_arrival_month]
                exp_short += shortage  # add unmet demand from current month
                if np.sum(exp_short) > 1e-9:
                    if not fixed_paid[EXPRESS]:
                        total_ship_fix += sfc[EXPRESS]
                        fixed_paid[EXPRESS] = True
                    total_ship_var += np.dot(exp_short, svc[:, EXPRESS])
                    pending_arrivals[:, exp_arrival_month] += exp_short
                    orders_placed[:, t, EXPRESS] += exp_short
                    total_purchase += np.dot(exp_short, p)

            # 5. Holding cost for end‑of‑month inventory
            total_hold += np.dot(inventory, h)

        # ---------------- final cost ---------------------------------------
        total_cost = total_purchase + total_ship_var + total_ship_fix + total_hold
        return float(total_cost)

    except Exception as e:
        print("Error in improved heuristic:", e)
        import traceback; traceback.print_exc()
        return math.inf
