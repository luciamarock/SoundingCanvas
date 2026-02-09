#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main program for the Simples lattice model.

This script initializes a 3D lattice of Simples and runs the simulation
using either deterministic, stochastic, or hybrid updates.
"""

from entities.lattice_factory import SimpleLattice

# -----------------------------
# Parameters
# -----------------------------
N = 10           # lattice size along each axis
NUM_STEPS = 100  # total simulation steps
ALPHA = 0.01     # step size for curvature/area updates
GAMMA = 1.0      # surface tension constant
K = 0.1          # bending rigidity
BETA = 1.0       # stochastic inverse temperature
MODE = "deterministic"  # "deterministic", "stochastic", or "hybrid"

# -----------------------------
# Simulation
# -----------------------------
def run_simulation():
    print(f"Initializing {N}x{N}x{N} lattice of Simples...")
    lattice = SimpleLattice(N=N, area=1.0)

    print(f"Running simulation for {NUM_STEPS} steps (mode={MODE})...")
    lattice.run(
        steps=NUM_STEPS,
        alpha=ALPHA,
        gamma=GAMMA,
        k=K,
        beta=BETA,
        mode=MODE
    )

    print("Simulation complete.")

# -----------------------------
# Execution Guard
# -----------------------------
if __name__ == "__main__":
    run_simulation()
