#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 14:03:25 2026

@author: luciamarock
"""

import numpy as np

def compute_transition_probabilities(simple, network, beta=1.0):
    """
    Compute the transition probability vector for a Simple based on local
    and neighbor curvature (energy gradients).

    Parameters
    ----------
    simple : Simple
        The Simple whose next state probabilities are computed.
    network : list of Simple
        Full lattice to access neighbors.
    beta : float, optional
        Inverse temperature / stiffness parameter controlling stochasticity.

    Returns
    -------
    P : dict
        Dictionary mapping patches to transition probabilities.
        Keys: 'xp','xm','yp','ym','zp','zm','free'
    """
    # Compute energy contribution for each patch
    delta_energies = {}
    for patch in simple.area:
        # local curvature energy
        dE_local = simple.curvature[patch]

        # neighbor interaction energy
        directions = ['xp','xm','yp','ym','zp','zm']
        if patch in directions:
            idx = directions.index(patch)
            try:
                neighbor_idx = simple.neighbors[idx]
                neighbor = network[neighbor_idx]
                opposite_patch = directions[idx ^ 1]  # opposite patch
                dE_neighbors = neighbor.curvature[opposite_patch] - simple.curvature[patch]
            except:
                dE_neighbors = 0.0

        delta_energies[patch] = -(dE_local + dE_neighbors)  # energy-decreasing directions

    # Convert to probabilities via softmax (stochastic selection)
    values = np.array(list(delta_energies.values()))
    # exponentiate with beta controlling randomness
    exp_vals = np.exp(beta * values)
    probs = exp_vals / np.sum(exp_vals)

    return dict(zip(simple.area.keys(), probs))


def stochastic_update(simple, network, alpha=0.01, beta=1.0):
    """
    Perform a stochastic update of a Simple using transition probabilities.

    Parameters
    ----------
    simple : Simple
        Simple to update
    network : list of Simple
        Full lattice of Simples
    alpha : float
        Scaling factor for area/curvature change
    beta : float
        Inverse temperature for stochastic selection
    """
    P = compute_transition_probabilities(simple, network, beta=beta)
    
    # select patch to update stochastically
    patches = list(P.keys())
    probabilities = list(P.values())
    selected_patch = np.random.choice(patches, p=probabilities)

    # apply a small delta along that patch
    delta = alpha * (1.0 if selected_patch != 'free' else -0.5)  # example scaling
    simple.area[selected_patch] += delta
    # ensure area non-negativity
    if simple.area[selected_patch] < 0.0:
        simple.area[selected_patch] = 0.0

    # renormalize total area
    total_area = sum(simple.area.values())
    correction = simple.A0 / total_area
    for p in simple.area:
        simple.area[p] *= correction

    # update curvature proportionally
    simple.curvature[selected_patch] += delta
