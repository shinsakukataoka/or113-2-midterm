import os, numpy as np, config as C

def load_instance(s_id: int, i_id: int):
    path = os.path.join(
        C.INSTANCE_FOLDER, f"scenario_{s_id}_instance_{i_id}.npz"
    )
    with np.load(path) as d:
        return (
            int(d["N_PRODUCTS"]),            # N
            int(d["T_MONTHS"]),              # T
            d["demand"],
            d["initial_inventory"],
            d["in_transit"],
            d["holding_cost_per_unit"],
            d["purchase_cost_per_unit"],
            d["shipping_variable_cost"][:, 0],   # CV1
            d["shipping_variable_cost"][:, 1],   # CV2
            np.zeros_like(d["shipping_variable_cost"][:, 0]),   # CV3 (ocean)
            d["shipping_fixed_cost"],
            d["volume_cbm"],
            float(d["ocean_container_cost"]),
            d["lead_times"],
        )
