# Sounding Canvas Event Manager v3 – SimplesModel

This repository contains the **third version of the Sounding Canvas Event Manager**, a framework for modeling the real-time response of the Sounding Canvas to touch and interaction.  

The foundation of this version is based on an idea I developed during my **Physics Bachelor's degree**, formalized as a **Variational Geometric Network Framework**. This framework is documented in detail in the [Compositional Framework PDF](https://github.com/luciamarock/SoundingCanvas/blob/development/documentation/conceptual_framework/compositional_framework_20251120.pdf).



## Conceptual Overview

The framework introduces a **foundational model for a probabilistic spacetime ontology**. Its key principles are:

- The universe of the model consists of **fundamental, closed, two-dimensional surfaces called _Simples_**.  
- Simples are **geometric agents** whose behavior is governed purely by **energetic minimization** and a **Per-Simple Area Conservation constraint**.  
- A network of Simples forms a **lattice**, representing a discrete spacetime, where the local geometry and interactions of Simples determine emergent behavior.  
- The dynamics of Simples encode **vibrational modes, contact geometry, and energy redistribution**, providing a formal bridge between:
  - **Loop Quantum Gravity (LQG)**, through discrete network structures.
  - **String Theory (D-branes)**, via collective vibrational modes.
- Within this framework, **Cognitive Quantum (CQ)** and **Directive Action (DA)** are formally defined, representing how awareness and interaction perturb the underlying geometry, guiding the system along paths of maximal informational complexity.

This framework provides a **physics-driven, mathematical model** for generating **compositional forms** for music and interactive artworks.


## Computational Considerations: Path-Dependence vs. Global Rearrangement

A fundamental feature of the SimplesModel is the **global nature of the Per-Simple Area Conservation constraint**:

- Any local deformation of a Simple requires compensating changes across the entire network of Simples.
- On a **classical computer**, updates must be performed sequentially. This means the evolution of the network is **path-dependent**: the order in which Simples are updated affects the outcome.  
- In principle, a **quantum computer** could compute the **instantaneous global rearrangement**, naturally respecting the nonlocal correlations imposed by area conservation. This would allow the network to evolve in a way that is truly independent of update order, capturing the idealized dynamics of the Simples framework.

Currently, the simulation uses classical computation, producing **approximations of the true global dynamics**, with emergent behavior dependent on the chosen update sequence. Exploring quantum algorithms for this framework is an exciting avenue for future development.

## Purpose of This Repository

The primary goal of `SimplesModel` is to **simulate a network of Simples** as a basis for the **Sounding Canvas** interaction engine:

- Each Simple is modeled as a **deformable closed surface** with energy-driven dynamics.
- The **network structure** allows the propagation of geometric interactions between neighbors.
- The model is designed to explore how local deformations (touch, pressure, or gestures) **propagate through the lattice**, enabling complex and emergent responses in the Sounding Canvas.




> Note: The lattice implementation is under development and will define how Simples are spatially organized and interact across the network.



## Future Directions

- Implement **lattice-based network interactions** for large-scale N×N×N Simples.
- Model **energy redistribution and curvature dynamics** for each Simple.
- Connect the simulation output to **sound synthesis**, so that touch interactions on the canvas translate into **dynamic audio responses**.
- Explore **stochastic evolution rules** to generate branching, path-dependent behaviors.



## References

- The program documentation is [Available Here](https://luciamarock.github.io/Academics/VariationalFramework/)
- Luciano Ciamarone – Foundational conceptual notes and derivations for Simples and their interactions.

