import gurobipy as gp
from gurobipy import GRB
import numpy as np
import config as C

def solve_lp_relaxation(
    N, T, D, I0, I, CH, CP, CV1, CV2, CV3,
    CF, Vi, container_cost, lead_times
):
    try:
        m = gp.Model(); m.setParam("OutputFlag", 0)

        x = m.addVars(N, C.J, T, name="x")         # qty shipped
        v = m.addVars(N, T, name="v")              # inventory
        y = m.addVars(C.J, T, name="y")            # fixed-cost indicator
        z = m.addVars(T,     name="z")             # ocean containers

        # Objective ------------------------------------------------------------------
        hold = gp.quicksum(CH[i] * v[i, t]
                           for i in range(N) for t in range(T))
        ship = gp.quicksum(
            (CP[i] + [CV1[i], CV2[i], CV3[i]][j]) * x[i, j, t]
            for i in range(N) for j in range(C.J) for t in range(T)
        )
        fixed     = gp.quicksum(CF[j] * y[j, t] for j in range(C.J) for t in range(T))
        container = gp.quicksum(container_cost * z[t] for t in range(T))
        m.setObjective(hold + ship + fixed + container, GRB.MINIMIZE)

        # Inventory balance -----------------------------------------------------------
        for i in range(N):
            for t in range(T):
                arrivals = gp.LinExpr()
                for j in range(C.J):
                    t_order = t - int(lead_times[j]) + 1
                    if 0 <= t_order < T:
                        arrivals += x[i, j, t_order]

                if t == 0:
                    m.addConstr(v[i, t] == I0[i] + I[i, t] + arrivals - D[i, t])
                else:
                    m.addConstr(v[i, t] == v[i, t-1] + I[i, t] + arrivals - D[i, t])

        # Link x and y (big-M)
        M = D.sum() * 2
        for j in range(C.J):
            for t in range(T):
                m.addConstr(gp.quicksum(x[i, j, t] for i in range(N)) <= M * y[j, t])

        # Ocean capacity
        for t in range(T):
            m.addConstr(
                gp.quicksum(Vi[i] * x[i, 2, t] for i in range(N))
                <= C.VC_CONTAINER_CAPACITY_CBM * z[t]
            )

        m.optimize()
        return m.ObjVal
    except Exception as exc:
        print(f"LP solve failed: {exc}")
        return np.nan
