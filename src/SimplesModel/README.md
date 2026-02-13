# Sounding Canvas Event Manager v3 – SimplesModel

This repository contains the **third version of the Sounding Canvas Event Manager**, a framework for modeling the real-time response of the Sounding Canvas to touch and interaction.

The foundation of this version is based on an idea developed during my **Physics Bachelor's degree**, formalized as a **Variational Geometric Network Framework**. This framework is documented in detail in the [Compositional Framework PDF](https://github.com/luciamarock/SoundingCanvas/blob/development/documentation/document_version_history/compositional_framework_20251120.pdf).

---

## Conceptual Overview

The framework introduces a **foundational model for a discrete, energy-driven network of surfaces**. Its key principles are:

- The universe of the model consists of **fundamental, closed, two-dimensional surfaces called _Simples_**.
- Simples are **geometric agents** whose behavior is governed by **energy minimization** and a **Per-Simple Area Conservation constraint**.
- A network of Simples forms a **3D lattice**, representing a discretized spacetime, where local geometry and neighbor interactions determine emergent dynamics.
- The dynamics of Simples encode **vibrational modes, contact geometry, and energy redistribution**, bridging formal ideas from:
  - **Loop Quantum Gravity (LQG)** through discrete network structures.
  - **String Theory (D-branes)** via collective vibrational modes.
- Within this framework, **Cognitive Quantum (CQ)** and **Directive Action (DA)** are formally defined to describe how awareness and interaction perturb the underlying geometry, guiding the system along paths of maximal informational complexity.

This framework provides a **physics-driven, mathematical model** suitable for **compositional generation** in music and interactive artworks.

---

## Computational Workflow in Deterministic Mode

The current deterministic model follows a clear, stepwise procedure:

1. **Compute energy-driven deltas**:
   - For each Simple, calculate proposed area/curvature changes (`ΔH`) on all patches (`xp`, `xm`, `yp`, `ym`, `zp`, `zm`, `free`) based on:
     - Surface tension
     - Bending rigidity
     - Neighbor interactions
     - Per-Simple Area Conservation

2. **Propagate patch-level updates**:
   - Within each Simple, the total delta is distributed across patches proportionally to current curvature deviations.
   - This redistribution may trigger further interactions with neighboring Simples in the lattice.

3. **Apply updates to Simples**:
   - Each Simple receives its computed patch deltas, updating both **area** and **curvature**.
   - The **mean curvature** of the Simple is updated as the patch-average of curvature deviations.
   - Per-Simple Area Conservation is enforced automatically by scaling the "free" patch.

4. **Iterate globally**:
   - Steps 1–3 are applied sequentially to all Simples in the lattice.
   - The evolution is **path-dependent**, as the order of Simple updates affects the propagation of local deformations across the network.

This workflow ensures that **all patches are accounted for**, and energy redistribution can propagate through neighbors naturally.

> Note: The stochastic evolution module exists but is currently **not used**; the framework is deterministic for the Sounding Canvas application.

---

## Purpose of This Repository

`SimplesModel` is designed to **simulate a network of deformable Simples** as the foundation of the Sounding Canvas interaction engine:

- Each Simple is a **closed, deformable surface** with patch-level energy-driven dynamics.
- The **lattice structure** enables the propagation of geometric interactions between neighbors.
- Local deformations (touch, pressure, gestures) **propagate through the network**, generating emergent responses in the Sounding Canvas.

---

## Future Directions

- Extend the **lattice-based network** to larger N×N×N grids of Simples.
- Refine **energy redistribution and curvature propagation** algorithms.
- Connect lattice dynamics to **audio synthesis**, mapping touch interactions to **dynamic sound responses**.
- Explore **optional stochastic updates** for alternative, path-dependent behaviors in interactive installations.

---

## References

- The program documentation is [Available Here](https://luciamarock.github.io/Academics/VariationalFramework/)
- Luciano Ciamarone – Foundational conceptual notes and derivations for Simples and their interactions.

