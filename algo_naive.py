"""‘Order-just-enough’ express-only heuristic (original naive)."""
import numpy as np

def naive_heuristic(N, T, D, I0, I, CP, CV1, CF):
    total = 0.0
    inv   = I0.copy()

    for t in range(T):
        avail   = inv + I[:, t]
        order   = np.maximum(D[:, t] - avail, 0)
        total  += np.sum(order * (CP + CV1))
        if order.sum() > 0:
            total += CF[0]          # express fixed cost
        inv = np.zeros(N)           # discard leftovers each month
    return total
