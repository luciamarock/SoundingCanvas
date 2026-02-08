class SimpleLattice:
    def __init__(self, N, area=1.0):
        self.N = N
        self.simples = []
        self._build_lattice(area)

    def _build_lattice(self, area):
        for i in range(self.N**3):
            self.simples.append(Simple(i, area))

        # neighbor wiring (6-connectivity)
        for x in range(self.N):
            for y in range(self.N):
                for z in range(self.N):
                    i = self._idx(x, y, z)
                    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                        nx, ny, nz = x+dx, y+dy, z+dz
                        if self._inside(nx, ny, nz):
                            j = self._idx(nx, ny, nz)
                            self.simples[i].neighbors.append(j)

