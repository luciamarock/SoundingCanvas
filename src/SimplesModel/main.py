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
NUM_STEPS = 100  # total simulation steps, this is simply chosen on the basis of how fast the model answers
ALPHA = 0.01     # step size for curvature/area updates
GAMMA = 1.0      # surface tension constant
K = 0.1          # bending rigidity
BETA = 1.0       # stochastic inverse temperature
MODE = "deterministic"  # "deterministic", "stochastic", or "hybrid"

# -----------------------------
# Sounding Canvas Event
# -----------------------------
"""
idx = calculated on the basis of local/remote event and sensor channel id 
change = calculated on the basis of touch speed 
SC_answer = will be populated with the simple's mean curvature after simulation,
            this will be used for selecting a sound
"""
touch_event = {"idx":700,"change":0.1}
SC_answer = None

# -----------------------------
# Simulation
# -----------------------------
def run_simulation():
    global SC_answer
    print(f"Initializing {N}x{N}x{N} lattice of Simples...")
    lattice = SimpleLattice(N=N, area=1.0)

    print(f"Running simulation for {NUM_STEPS} steps (mode={MODE})...")
    SC_answer = lattice.run(
        steps=NUM_STEPS,
        alpha=ALPHA,
        gamma=GAMMA,
        k=K,
        beta=BETA,
        mode=MODE,
        event=touch_event
    )

    print("Simulation complete.")

# -----------------------------
# Execution Guard
# -----------------------------
if __name__ == "__main__":
    run_simulation()
    print(SC_answer)
