import gurobipy as gp
from gurobipy import GRB
import numpy as np

def solve_lp(data):
    try:
        # ---------------- unpack -------------------------------------------------
        N        = int(data['N_PRODUCTS'])
        T        = int(data['T_MONTHS'])
        J        = 3                                          # 0-Exp,1-Air,2-Ocean

        D        = data['demand']                   # (N,T)
        I0       = data['initial_inventory']        # (N,)
        transit  = data['in_transit']               # (N,T)

        CH       = data['holding_cost_per_unit']    # (N,)
        CP       = data['purchase_cost_per_unit']   # (N,)

        svc      = data['shipping_variable_cost']   # (N,2)
        CV1, CV2 = svc[:,0], svc[:,1]
        CV3      = np.zeros(N)                      # variable ocean cost = 0

        CF       = data['shipping_fixed_cost']      # (3,)
        vol      = data['volume_cbm']               # (N,)

        ocean_cc   = float(data['ocean_container_cost'])
        ocean_cap  = float(data['ocean_container_capacity_cbm'])
        lead       = data['lead_times']             # (3,)  e.g. [1,2,3]

    except KeyError as e:
        return float('inf'), f"Data key missing: {e}"

    # ---------------- model ------------------------------------------------------
    try:
        m = gp.Model("LPRelaxation")
        m.setParam("OutputFlag", 0)

        x = m.addVars(N, J, T, vtype=GRB.CONTINUOUS, name="x")   # order qty
        v = m.addVars(N, T,        vtype=GRB.CONTINUOUS, name="v")   # inventory
        y = m.addVars(J, T,        vtype=GRB.CONTINUOUS, name="y")   # fixed-cost flags
        z = m.addVars(T,           vtype=GRB.CONTINUOUS, name="z")   # ocean boxes

        # ---------- objective ----------------------------------------------------
        hold_cost = gp.quicksum(CH[i] * v[i,t]                for i in range(N) for t in range(T))
        ship_var  = gp.quicksum((CP[i] + [CV1[i], CV2[i], CV3[i]][j]) * x[i,j,t]
                                                            for i in range(N) for j in range(J) for t in range(T))
        ship_fix  = gp.quicksum(CF[j] * y[j,t]               for j in range(J) for t in range(T))
        cont_cost = gp.quicksum(ocean_cc * z[t]              for t in range(T))

        m.setObjective(hold_cost + ship_var + ship_fix + cont_cost, GRB.MINIMIZE)

        # ---------- constraints --------------------------------------------------
        for i in range(N):
            for t in range(T):
                # arrivals from in-transit + own orders that arrive this period
                arrivals = transit[i, t]
                for j in range(J):
                    src = t - int(lead[j]) + 1
                    if 0 <= src < T:
                        arrivals += x[i,j,src]

                if t == 0:
                    m.addConstr(v[i,0] == I0[i] + arrivals - D[i,0], name=f"bal_{i}_0")
                else:
                    m.addConstr(v[i,t] == v[i,t-1] + arrivals - D[i,t], name=f"bal_{i}_{t}")
                    # same “service-level” constraint as original code
                    m.addConstr(v[i,t-1] >= D[i,t], name=f"svc_{i}_{t}")

        # big-M value
        BIG_M = float(D.sum() * 2)

        # link fixed costs
        for j in range(J):
            for t in range(T):
                m.addConstr(gp.quicksum(x[i,j,t] for i in range(N)) <= BIG_M * y[j,t],
                            name=f"link_{j}_{t}")

        # ocean capacity each period
        for t in range(T):
            m.addConstr(gp.quicksum(vol[i] * x[i,2,t] for i in range(N)) <= ocean_cap * z[t],
                        name=f"ocean_cap_{t}")

        # ---------- solve --------------------------------------------------------
        m.optimize()
        status = m.Status
        if status == GRB.OPTIMAL:
            return m.objVal, "Optimal"
        else:
            return float('inf'), f"Gurobi status {status}"

    except Exception as e:
        return float('inf'), f"Gurobi build/solve error: {e}"
