"""
- the transition probability vector is computed from the state of the object and the states of its neighbors
- store: a_x+, a_x-, a_y+, a_y-, a_z+, a_z-, a_free
  with constraint --> sum(all a_*) = A0

"""
class Simple:
    def __init__(self, idx, neighbors, A0):
        self.idx = idx
        self.neighbors = neighbors  # 6 indices
        self.A0 = A0

        # area allocation
        self.area = {
            'xp': 0.0, 'xm': 0.0,
            'yp': 0.0, 'ym': 0.0,
            'zp': 0.0, 'zm': 0.0,
            'free': A0
        }

        # curvature deviations per patch
        self.curvature = {key: 0.0 for key in self.area}

    def local_energy(self):
        # surface + bending contributions
        ...

    def interaction_energy(self, neighbor, direction):
        # depends on shared patch geometry
        ...

    def energy_gradient(self, network):
        # compute ∂E/∂(area redistribution)
        ...

    def propose_update(self, network):
        # propose area transfer between patches
        ...

    def apply_update(self, delta):
        # enforce area conservation
        ...

