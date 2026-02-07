#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 20:03:35 2026

@author: luciamarock
"""

# simples_network = np.empty((N, N, N), dtype=object)

"""
This is:
gradient flow without time
Markovian but path-dependent
naturally branching
"""
    
for step in range(num_steps):
    pick random simple i
    compute admissible redistributions
    choose one (deterministic or stochastic)
    apply update

