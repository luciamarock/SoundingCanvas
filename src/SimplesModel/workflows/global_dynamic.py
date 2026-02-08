"""
Module `global_dynamic` implements variational geometric evolution
for a lattice of Simples with realistic surface forces.

Forces:
  - Surface tension: gamma * H
  - Bending rigidity: -2k [Delta(H-H0) + (H-H0)(H^2 - K)]
  - Neighbor coupling: enforces area conservation

Updates are applied simultaneously to all Simples.
"""

def global_update(simples, alpha=0.01):
    """
    Perform a single global update of all Simples in the lattice.

    Computes the geometric forces for each Simple and updates their state
    simultaneously, ensuring path-independent evolution.

    Parameters
    ----------
    simples : list
        List of Simple objects in the lattice.
    alpha : float, optional
        Step size scaling factor (default 0.01).
    """
    # compute all force-induced variations first
    deltas = compute_deltas(simples, alpha=alpha)

    # then apply simultaneously
    for s, delta in zip(simples, deltas):
        s.H += delta  # update curvature/shape proxy
        # TODO: update actual surface geometry if stored explicitly


def compute_deltas(simples, alpha=0.01, gamma=1.0, k=0.1):
    """
    Compute the change in curvature/shape for each Simple due to variational forces.

    Parameters
    ----------
    simples : list
        List of Simple objects in the lattice.
    alpha : float, optional
        Step size scaling factor (default 0.01).
    gamma : float, optional
        Surface-tension constant (default 1.0).
    k : float, optional
        Bending rigidity constant (default 0.1).

    Returns
    -------
    deltas : list of float
        List of increments in curvature/shape for each Simple.
    """
    deltas = []
    for s in simples:
        # Surface-tension contribution
        surface_tension = gamma * s.H
        
        # Bending contribution (discrete Laplacian approximation)
        laplacian = 0.0
        for neighbor in s.neighbors:
            for item in enumerate(neighbor.neighbors):
                j =item[0]
                neighbor = simples[j]
                laplacian += (neighbor.H - neighbor.H0) - (s.H - s.H0)
                if j > 3:
                    break
        bending = -2 * k * (laplacian + (s.H - s.H0) * (s.H**2 - s.K))

        # Neighbor coupling to redistribute area deviations
        coupling = 0.0
        for neighbor in s.neighbors:
            for item in enumerate(neighbor.neighbors):
                j =item[0]
                neighbor = simples[j]
                coupling += (neighbor.H - s.H)  # encourages smooth transitions
                if j > 3:
                    break

        # Total delta scaled by alpha
        deltas.append(alpha * (surface_tension + bending + coupling))

    return deltas
