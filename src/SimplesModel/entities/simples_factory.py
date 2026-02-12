"""
Module `entities.simples` defines the `Simple` class.
Handles local area updates and neighbor negotiation for shared walls.
"""

class Simple:
    PATCHES = ['xp', 'xm', 'yp', 'ym', 'zp', 'zm', 'free']
    
    # Mapping to find which patch on a neighbor corresponds to mine
    OPPOSITE = {
        'xp': 'xm', 'xm': 'xp',
        'yp': 'ym', 'ym': 'yp',
        'zp': 'zm', 'zm': 'zp',
        'free': 'free'
    }

    def __init__(self, idx, A0=1.0):
        self.idx = idx
        self.neighbors = []  # List of indices
        self.neighbor_instances = {}  # Map: direction_patch -> Simple instance
        self.A0 = A0

        self.H0 = 0.0 # Reference Curvature
        self.H = 0.0 # Mean Curvature
        self.K = 0.0 # Gaussian Curvature

        self.area = {p: 0.0 for p in self.PATCHES}
        self.area['free'] = A0
        self.curvature = {p: 0.0 for p in self.PATCHES}
        
        # Buffer to store "pushes" or "pulls" from neighbors
        self.proposal_buffer = {p: 0.0 for p in self.PATCHES}

    def _renormalize_area(self):
        """Enforce total area conservation A0."""
        total_area = sum(self.area.values())
        if total_area <= 0: return
        
        scale = self.A0 / total_area
        for p in self.area:
            self.area[p] *= scale

    def _update_mean_curvature(self):
        """Update global H based on patch states."""
        self.H = sum(self.curvature.values()) / len(self.curvature)

    def receive_proposal(self, patch, delta_area):
        """
        Called by a neighbor to inform this Simple that the shared 
        wall area is changing.
        """
        self.proposal_buffer[patch] += delta_area
    
    def _compute_curvature_change(self):
        #TODO use old self._redistribution_weights() and or total_delta = sum(delta.values()) and or softmax
        # update this once the area is recalculated and before _update_mean_curvature
        pass

    def apply_update(self, delta):
        """
        Apply lattice forces and negotiate with neighbors via neighbor_instances.
        
        Parameters
        ----------
        delta : dict
            The force vector from the lattice for each patch.
        """
        # 1. Incorporate proposals from neighbors who have already updated
        for p in self.PATCHES:
            # If a neighbor pushed a change to our shared wall, apply it first
            self.area[p] += self.proposal_buffer[p]
            # Clear buffer after consuming
            self.proposal_buffer[p] = 0.0

        # 2. Apply the current lattice forces (deltas)
        for p, change in delta.items():
            self.area[p] += change
            
            # 3. Negotiation: Inform the neighbor of this change
            # We use the neighbor_instances map (assumed to be populated by Lattice)
            if p in self.neighbor_instances and p != 'free':
                neighbor = self.neighbor_instances[p]
                opposite = self.OPPOSITE[p]
                # If I grow my 'xp', my neighbor's 'xm' must grow by the same amount
                # to maintain the shared interface area.
                neighbor.receive_proposal(opposite, change)

        # 4. Physical Constraints: Area cannot be negative
        for p in self.area:
            if self.area[p] < 0:
                self.area[p] = 0.0

        # 5. Closure and Internal Consistency
        self._renormalize_area()
        self._compute_curvature_change()
        self._update_mean_curvature()