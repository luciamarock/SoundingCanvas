"""
Module `entities.simples_lattice` defines the `SimpleLattice` class,
managing a 3D lattice of Simples and performing global energy-driven updates.
"""

from entities.simples_factory import Simple
from workflows.stochastic_update import stochastic_update


class SimpleLattice:
    """
    3D lattice of Simples with 6-connectivity.

    Attributes
    ----------
    N : int
        Number of Simples along each lattice dimension.
    simples : list of Simple
        Flattened list of all Simples in the lattice.
    """

    def __init__(self, N, area=1.0):
        self.N = N
        self.simples = []
        self._build_lattice(area)

    def _idx(self, x, y, z):
        """Convert 3D coordinates to linear index."""
        return x * self.N**2 + y * self.N + z

    def _inside(self, x, y, z):
        """Check if coordinates are inside the lattice."""
        return 0 <= x < self.N and 0 <= y < self.N and 0 <= z < self.N

    def _build_lattice(self, area):
        """Instantiate Simples and wire neighbors with patch-specific keys."""
        # Create all Simples
        for i in range(self.N**3):
            self.simples.append(Simple(idx=i, A0=area))

        # Map directions to patch names
        dirs = {
            'xm': (-1, 0, 0), 'xp': (1, 0, 0),
            'ym': (0, -1, 0), 'yp': (0, 1, 0),
            'zm': (0, 0, -1), 'zp': (0, 0, 1)
        }

        for x in range(self.N):
            for y in range(self.N):
                for z in range(self.N):
                    i = self._idx(x, y, z)
                    s = self.simples[i]
                    
                    # Re-initialize neighbor_instances as a dict for patch-mapping
                    s.neighbor_instances = {} 
                    
                    for patch_name, (dx, dy, dz) in dirs.items():
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if self._inside(nx, ny, nz):
                            neighbor_idx = self._idx(nx, ny, nz)
                            s.neighbors.append(neighbor_idx)
                            # Link the specific patch to the specific neighbor object
                            s.neighbor_instances[patch_name] = self.simples[neighbor_idx]
    
    def stochastic_step(self, alpha=0.01, beta=1.0):
        """
        Apply a single stochastic update step to all Simples.
    
        Parameters
        ----------
        alpha : float
            Step size for deterministic contribution (optional if used inside stochastic_update).
        beta : float
            Inverse temperature controlling stochasticity.
        """
        for simple in self.simples:
            stochastic_update(simple, self.simples, alpha=alpha, beta=beta)
    
    
    def get_opposite_patch(self,patch):
        """
        Return the opposite patch name for a given patch.
    
        Parameters
        ----------
        patch : str
            One of 'xp','xm','yp','ym','zp','zm','free'.
    
        Returns
        -------
        str
            Opposite patch name.
        """
        mapping = {
            'xp': 'xm', 'xm': 'xp',
            'yp': 'ym', 'ym': 'yp',
            'zp': 'zm', 'zm': 'zp',
            'free': 'free'
        }
        return mapping.get(patch, 'free')



    def compute_deltas(self, alpha=0.01, gamma=1.0, k=0.1):
        """
        Compute the proposed area redistribution for each Simple based on
        local energy and interaction with neighbors.
    
        Parameters
        ----------
        simples : list of Simple
            List of Simple objects in the lattice.
        alpha : float, optional
            Step size scaling factor (default 0.01).
        gamma : float, optional
            Surface-tension constant (default 1.0).
        k : float, optional
            Bending rigidity constant (default 0.1).
    
        Returns
        -------
        deltas : list of dict
            Each dict maps patch names ('xp','xm','yp','ym','zp','zm','free') to
            proposed changes in area for that Simple.
        """
        deltas = []
        for s in self.simples:
            # Initialize delta dict
            delta = {patch: 0.0 for patch in s.area}
            
            # Surface tension: proportional to mean curvature H
            for patch in s.area:
                delta[patch] += gamma * s.curvature[patch]        

            # Bending rigidity: discrete Laplacian of curvature
            for patch in s.area:
                laplacian = 0.0
                for j in s.neighbors:
                    neighbor = self.simples[j]
                    laplacian += neighbor.curvature[patch] - s.curvature[patch]
                bending = -2 * k * (laplacian + (s.curvature[patch] - s.H0) *
                                    (s.curvature[patch]**2 - s.K))
                delta[patch] += bending

            # Neighbor interaction: energy gradient contribution from shared patches
            for j in s.neighbors:
                neighbor = self.simples[j]
                for patch in s.area:
                    opposite_patch = self.get_opposite_patch(patch)
                    # assume a simple quadratic interaction on shared patch
                    shared_area = min(s.area[patch], neighbor.area[opposite_patch])
                    diff_curvature = s.curvature[patch] - neighbor.curvature[opposite_patch]
                    delta[patch] += shared_area * diff_curvature  # energy gradient

            # Scale total delta by alpha
            for patch in delta:
                delta[patch] *= alpha

            deltas.append(delta)
        return deltas

    def global_update(self, alpha=0.01, beta=1.0, gamma=1.0, k=0.1):
        """
        Perform a single global update of all Simples in the lattice.
    
        Computes the geometric forces for each Simple, including interactions
        with neighbors, and updates their state simultaneously.
    
        Parameters
        ----------
        simples : list of Simple
            List of Simple objects in the lattice.
        alpha : float, optional
            Step size scaling factor (default 0.01).
        """
        deltas = self.compute_deltas(alpha=alpha, gamma=gamma, k=k)
        for s, delta in zip(self.simples, deltas):
            s.apply_update(delta,beta)

    def run(self, steps=10, alpha=0.01, gamma=1.0, k=0.1, beta=1.0, mode="hybrid"):
        """
        Run the simulation for multiple steps.
    
        Parameters
        ----------
        steps : int
            Number of steps to run.
        mode : str
            "deterministic", "stochastic", or "hybrid".
        """
        for step in range(steps):
            if mode in ("deterministic", "hybrid"):
                self.global_update(alpha=alpha, beta=beta, gamma=gamma, k=k)
            if mode in ("stochastic", "hybrid"):
                self.stochastic_step(alpha=alpha, beta=beta)

