#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 20:03:35 2026

@author: luciamarock

Main program for the Simples lattice model.
"""

import random
import numpy as np
from entities.simples_factory import Simple  # your Simple class
from workflows.global_dynamic import global_update

# -----------------------------
# Parameters
# -----------------------------
N = 10  # lattice size along one axis
NUM_STEPS = 100
ALPHA = 0.01  # global update step size

def get_neighbors(x, y, z, n_limit):
    """Helper to find valid 3D lattice neighbors."""
    neighbors = []
    for dx, dy, dz in [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0), (0,0,-1), (0,0,1)]:
        nx, ny, nz = x + dx, y + dy, z + dz
        if 0 <= nx < n_limit and 0 <= ny < n_limit and 0 <= nz < n_limit:
            neighbors.append((nx, ny, nz))
    return neighbors

def run_simulation():
    """
    Encapsulates the simulation logic so it doesn't run automatically 
    when the file is imported (e.g., by Sphinx).
    """
    # 1. Initialize lattice
    simples_network = np.empty((N, N, N), dtype=object)
    
    # Initial pass to create objects
    for x in range(N):
        for y in range(N):
            for z in range(N):
                # Placeholder neighbors until second pass
                simples_network[x, y, z] = Simple(z, [], 1)

    # 2. Assign neighbors
    for x in range(N):
        for y in range(N):
            for z in range(N):
                idx_neighbors = get_neighbors(x, y, z, N)
                simples_network[x, y, z].neighbors = [
                    simples_network[nx, ny, nz] for nx, ny, nz in idx_neighbors
                ]

    # 3. Main evolution loop
    print(f"Starting simulation for {NUM_STEPS} steps...")
    for step in range(NUM_STEPS):
        # Pick a random Simple
        # Note: i, j, k are integers here, which avoids the IndexError
        i, j, k = random.randint(0, N-1), random.randint(0, N-1), random.randint(0, N-1)
        
        # Call global update
        global_update(simples_network.flatten(), alpha=ALPHA)

    print("Simulation complete.")

# -----------------------------
# Execution Guard
# -----------------------------
if __name__ == "__main__":
    # This block only runs if you execute the script directly.
    # Sphinx will skip this when generating documentation.
    run_simulation()