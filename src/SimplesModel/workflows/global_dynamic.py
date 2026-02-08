"""
Module `global_dynamic` implements variational geometric evolution
for a lattice of Simples with realistic surface forces and neighbor interactions.

Forces:
  - Surface tension: gamma * H
  - Bending rigidity: -2k [Δ(H-H0) + (H-H0)(H^2 - K)]
  - Neighbor coupling: interaction energy between adjacent patches
  - Area redistribution: ensures local area conservation

Updates are applied simultaneously to all Simples.
"""

def global_update(simples, alpha=0.01):
    """
    Perform a single global update of all Simples in the lattice.

    Computes the geometric forces for each Simple, including interactions
    with neighbors, and updates their state simultaneously.

    Parameters
    ----------
    simples : list of Simple
        List of Simple objects in the lattice.
    alpha : float, optional
        Step size scaling factor (default 0.01).
    """
    # Compute all variations first
    deltas = compute_deltas(simples, alpha=alpha)

    # Apply updates simultaneously
    for s, delta_dict in zip(simples, deltas):
        s.apply_update(delta_dict)

def global_update_with_area(network, alpha=0.01, gamma=1.0, k=0.1):
    """
    Perform a single global update, including curvature->area propagation.
    """
    deltas = network.compute_deltas(alpha=alpha, gamma=gamma, k=k)
    for s, delta in zip(network.simples, deltas):
        s.H += delta  # update curvature
        propagate_area(s, delta, network.simples)


def compute_deltas(simples, alpha=0.01, gamma=1.0, k=0.1):
    """
    Compute the proposed area redistribution for each Simple based on
    local energy and interaction with neighbors.

    Parameters
    ----------
    simples : list of Simple
        List of Simple objects in the lattice.
    alpha : float, optional
        Step size scaling factor (default 0.01).
    gamma : float, optional
        Surface-tension constant (default 1.0).
    k : float, optional
        Bending rigidity constant (default 0.1).

    Returns
    -------
    deltas : list of dict
        Each dict maps patch names ('xp','xm','yp','ym','zp','zm','free') to
        proposed changes in area for that Simple.
    """
    deltas = []

    for s in simples:
        # Initialize delta dict
        delta = {patch: 0.0 for patch in s.area}

        # Surface tension: proportional to mean curvature H
        for patch in s.area:
            delta[patch] += gamma * s.curvature[patch]

        # Bending rigidity: discrete Laplacian of curvature
        for patch in s.area:
            laplacian = 0.0
            for j in s.neighbors:
                neighbor = simples[j]
                laplacian += neighbor.curvature.get(patch, 0.0) - s.curvature[patch]
            bending = -2 * k * (laplacian + (s.curvature[patch] - s.H0) *
                                (s.curvature[patch]**2 - s.K))
            delta[patch] += bending

        # Neighbor interaction: energy gradient contribution from shared patches
        for j in s.neighbors:
            neighbor = simples[j]
            for patch in s.area:
                # assume a simple quadratic interaction on shared patch
                shared_area = min(s.area[patch], neighbor.area.get(opposite_patch(patch), 0.0))
                diff_curvature = s.curvature[patch] - neighbor.curvature.get(opposite_patch(patch), 0.0)
                delta[patch] += shared_area * diff_curvature  # energy gradient

        # Scale total delta by alpha
        for patch in delta:
            delta[patch] *= alpha

        deltas.append(delta)

    return deltas


def opposite_patch(patch):
    """
    Return the opposite patch name for a given patch.

    Parameters
    ----------
    patch : str
        One of 'xp','xm','yp','ym','zp','zm','free'.

    Returns
    -------
    str
        Opposite patch name.
    """
    mapping = {
        'xp': 'xm', 'xm': 'xp',
        'yp': 'ym', 'ym': 'yp',
        'zp': 'zm', 'zm': 'zp',
        'free': 'free'
    }
    return mapping.get(patch, 'free')


def propagate_area(simple, delta, network):
    """
    Propagate curvature change into the Simple's area patches.

    Parameters
    ----------
    simple : Simple
        The Simple whose area will be updated.
    delta : float
        Change in curvature/shape (ΔH) from global_update.
    network : list of Simple
        Full lattice, used for neighbor interaction.
    """
    # Distribute delta across patches proportionally to curvature deviation
    total_curv_dev = sum(abs(simple.curvature[p]) for p in simple.area)
    if total_curv_dev == 0.0:
        # equal redistribution if no prior curvature deviation
        weights = {p: 1.0 for p in simple.area}
    else:
        weights = {p: abs(simple.curvature[p]) / total_curv_dev for p in simple.area}

    # Compute area change per patch
    area_changes = {p: delta * weights[p] for p in simple.area}

    # Apply area change
    for p in simple.area:
        simple.area[p] += area_changes[p]
        # clamp to non-negative
        if simple.area[p] < 0.0:
            simple.area[p] = 0.0

    # Renormalize total area to preserve A0
    total_area = sum(simple.area.values())
    correction = simple.A0 / total_area
    for p in simple.area:
        simple.area[p] *= correction

    # Update patch curvatures (simple model: curvature proportional to area change)
    for p in simple.area:
        simple.curvature[p] += area_changes[p]

    # Propagate to neighbors (local geometric interaction)
    directions = ['xp','xm','yp','ym','zp','zm']
    for idx, neighbor_idx in enumerate(simple.neighbors[:6]):  # 6-connectivity
        neighbor = network[neighbor_idx]
        dir_patch = directions[idx]
        # Apply small fraction to neighbor's opposite patch
        opposite = directions[idx ^ 1]  # 0^1=1,1^1=0,2^1=3,...
        neighbor.area[opposite] += 0.2 * area_changes[dir_patch]  # fraction of transfer
        # clamp
        if neighbor.area[opposite] < 0.0:
            neighbor.area[opposite] = 0.0
