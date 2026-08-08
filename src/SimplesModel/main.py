#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the Simples lattice model.

The same sequence of interaction events is executed twice on the
same evolving lattice. The random generator is re-seeded before
each sequence so that both passes use the same stochastic realization.
"""

import json
from pathlib import Path

import numpy as np

from entities.lattice_factory import SimpleLattice


# -----------------------------
# Parameters
# -----------------------------

N = 10
NUM_STEPS = 10

ALPHA = 0.01
GAMMA = 1.0
K = 0.1
BETA = 1.0

MODE = "deterministic"

SEED = 42

SIMULATION_DATA = Path("simulation_data.json")


# -----------------------------
# Load simulation data
# -----------------------------

def load_simulation_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["touches"]


# -----------------------------
# Run one sequence
# -----------------------------

def run_sequence(lattice, touch_events, sequence_number):

    print(f"\n--- Sequence {sequence_number} ---")

    results = []

    for event_number, touch_event in enumerate(touch_events, start=1):

        print(
            f"\nTouch {event_number}: "
            f"idx={touch_event['idx']}, "
            f"change={touch_event['change']}"
        )

        SC_answer = lattice.run(
            steps=NUM_STEPS,
            alpha=ALPHA,
            gamma=GAMMA,
            k=K,
            beta=BETA,
            mode=MODE,
            event=touch_event
        )

        results.append({
            "sequence": sequence_number,
            "touch": event_number,
            "idx": touch_event["idx"],
            "change": touch_event["change"],
            "H": SC_answer
        })

        print(f"  -> Simple H = {SC_answer}")

    return results


# -----------------------------
# Simulation
# -----------------------------

def run_simulation():

    touch_events = load_simulation_data(SIMULATION_DATA)

    print(f"Loaded {len(touch_events)} touch events.")

    print(f"Initializing {N}x{N}x{N} lattice of Simples...")
    lattice = SimpleLattice(N=N, area=1.0)

    # -------------------------
    # First sequence
    # -------------------------

    print(f"\nSetting random seed to {SEED} for sequence 1.")
    np.random.seed(SEED)

    results_1 = run_sequence(
        lattice,
        touch_events,
        sequence_number=1
    )

    # -------------------------
    # Second sequence
    # -------------------------

    print(f"\nSetting random seed to {SEED} for sequence 2.")
    np.random.seed(SEED)

    results_2 = run_sequence(
        lattice,
        touch_events,
        sequence_number=2
    )

    # -------------------------
    # Results
    # -------------------------

    print("\n==============================")
    print("Simulation complete.")
    print("==============================")

    print("\nSequence 1:")
    for result in results_1:
        print(
            f"Touch {result['touch']}: "
            f"idx={result['idx']}, "
            f"change={result['change']}, "
            f"H={result['H']}"
        )

    print("\nSequence 2:")
    for result in results_2:
        print(
            f"Touch {result['touch']}: "
            f"idx={result['idx']}, "
            f"change={result['change']}, "
            f"H={result['H']}"
        )

    return results_1, results_2


# -----------------------------
# Execution Guard
# -----------------------------

if __name__ == "__main__":
    run_simulation()
