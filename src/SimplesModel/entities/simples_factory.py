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

    PATCHES = ['xp', 'xm', 'yp', 'ym', 'zp', 'zm', 'free']

    def __init__(self, idx, neighbors=None, A0=1.0):
        self.idx = idx
        self.neighbors = neighbors or []
        self.A0 = A0

        self.H0 = 0.0
        self.H = 0.0
        self.K = 0.0

        # Area per patch
        self.area = {p: 0.0 for p in self.PATCHES}
        self.area['free'] = A0

        # Curvature per patch
        self.curvature = {p: 0.0 for p in self.PATCHES}

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------

    def _redistribution_weights(self):
        """
        Compute redistribution weights for internal equilibration.

        Weights are proportional to curvature deviation magnitude.
        """
        total = sum(abs(self.curvature[p]) for p in self.PATCHES)

        if total == 0.0:
            return {p: 1.0 / len(self.PATCHES) for p in self.PATCHES}

        return {p: abs(self.curvature[p]) / total for p in self.PATCHES}

    def _renormalize_area(self):
        """
        Enforce total area conservation.
        """
        total_area = sum(self.area.values())
        if total_area == 0.0:
            return

        scale = self.A0 / total_area
        for p in self.area:
            self.area[p] *= scale

    def _update_mean_curvature(self):
        """
        Update mean curvature as patch average.
        """
        self.H = sum(self.curvature.values()) / len(self.curvature)

    # ------------------------------------------------------------------
    # Public API --> this is only one possible way in which the surface 
    # gets redistributed, all the possible ways constitute the vector V
    # ------------------------------------------------------------------

    def apply_update(self, delta):
        """
        Apply an external delta stimulus and redistribute it internally.

        Parameters
        ----------
        delta : dict
            External energy / curvature stimulus per patch.
        """

        # Total stimulus magnitude
        total_delta = sum(delta.values())

        if total_delta == 0.0:
            return

        # Internal redistribution weights
        weights = self._redistribution_weights()

        # Convert stimulus into area changes
        area_changes = {
            p: weights[p] * total_delta
            for p in self.PATCHES
        }

        # Apply area changes
        for p in self.PATCHES:
            self.area[p] += area_changes[p]
            if self.area[p] < 0.0:
                self.area[p] = 0.0

            # Update curvature proxy
            self.curvature[p] += area_changes[p]

        # Enforce closure
        self._renormalize_area()
        self._update_mean_curvature()
