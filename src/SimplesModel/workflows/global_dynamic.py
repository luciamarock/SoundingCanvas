"""
E = Σ (shape_i − 1)² + λ Σ (shape_i − shape_j)²

"""
def global_update(simples):
    # compute all variations first
    deltas = compute_deltas(simples)

    # then apply simultaneously
    for s, delta in zip(simples, deltas):
        s.shape += delta

def compute_deltas(simples, alpha=0.01):
    deltas = []
    for s in simples:
        local_force = -(s.shape - 1.0)
        coupling = 0.0
        for j in s.neighbors:
            coupling += (simples[j].shape - s.shape)
        deltas.append(alpha * (local_force + coupling))
    return deltas


