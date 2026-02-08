"""
Module `entities.simples` defines the `Simple` class,
representing a closed geometric surface in the lattice network.

Each Simple interacts with neighbors via energy-driven rules:
- surface tension
- bending rigidity
- neighbor coupling
- area redistribution (local conservation)
"""

class Simple:
    """
    Represents a closed geometric surface (Simple) in the SimplesModel network.

    Attributes
    ----------
    idx : int
        Unique identifier for the Simple.
    neighbors : list of int
        Indices of neighboring Simples in the lattice.
    A0 : float
        Total area of the Simple.
    H0 : float
        Reference mean curvature.
    H : float
        Current mean curvature (patch-averaged if needed).
    K : float
        Current Gaussian curvature.
    area : dict
        Dictionary of area per patch:
        'xp', 'xm', 'yp', 'ym', 'zp', 'zm', 'free'.
    curvature : dict
        Curvature deviations per patch (for local energy and interactions).
    """

    def __init__(self, idx, neighbors=None, A0=1.0):
        self.idx = idx
        self.neighbors = neighbors or []  # list of neighbor indices
        self.A0 = A0
        self.H0 = 0.0
        self.H = 0.0
        self.K = 0.0

        # Area per patch
        self.area = {
            'xp': 0.0, 'xm': 0.0,
            'yp': 0.0, 'ym': 0.0,
            'zp': 0.0, 'zm': 0.0,
            'free': A0
        }

        # Curvature per patch
        self.curvature = {key: 0.0 for key in self.area}

    def local_energy(self):
        """
        Compute the local energy of the Simple due to surface tension and bending.
        """
        surface_energy = sum(self.curvature[patch] * self.area[patch] for patch in self.area)
        bending_energy = 0.0
        # placeholder: could implement discrete Laplacian over neighbors
        return surface_energy + bending_energy

    def interaction_energy(self, neighbor, direction):
        """
        Compute the interaction energy contribution with a neighbor Simple.

        Parameters
        ----------
        neighbor : Simple
            Neighboring Simple object.
        direction : str
            Patch direction of interaction ('xp','xm', etc.)
        """
        opp = opposite_patch(direction)
        shared_area = min(self.area[direction], neighbor.area.get(opp, 0.0))
        diff_curvature = self.curvature[direction] - neighbor.curvature.get(opp, 0.0)
        return 0.5 * shared_area * diff_curvature**2  # quadratic penalty

    def energy_gradient(self, network):
        """
        Compute the gradient of the energy w.r.t area redistribution.

        Returns
        -------
        grad : dict
            Dictionary mapping patch names to Δarea values.
        """
        grad = {patch: 0.0 for patch in self.area}

        # Surface tension contribution
        for patch in self.area:
            grad[patch] += self.curvature[patch]

        # Neighbor interaction contribution
        for j in self.neighbors:
            neighbor = network[j]
            for patch in self.area:
                opp = opposite_patch(patch)
                grad[patch] += self.area[patch] * (self.curvature[patch] - neighbor.curvature.get(opp, 0.0))

        return grad

    def propose_update(self, network, alpha=0.01):
        """
        Propose an update of the area patches based on energy gradient.

        Parameters
        ----------
        network : list of Simple
            All Simples in the lattice.
        alpha : float
            Step scaling factor.

        Returns
        -------
        delta : dict
            Proposed change per patch.
        """
        grad = self.energy_gradient(network)
        delta = {patch: -alpha * grad[patch] for patch in grad}
        return delta

    def apply_update(self, delta):
        """
        Apply the proposed area redistribution while conserving total area.

        Parameters
        ----------
        delta : dict
            Proposed change per patch.
        """
        total_delta = sum(delta.values())
        # Scale free patch to conserve total area
        delta['free'] -= total_delta

        for patch in self.area:
            self.area[patch] += delta[patch]
            # update curvature proxy as area-weighted
            self.curvature[patch] += delta[patch]  # or some mapping to curvature

        # Update mean curvature H as patch average
        self.H = sum(self.curvature.values()) / len(self.curvature)
