# open-poisson-solver

Single-process FFT-based open-boundary Poisson solver using the Hockney doubled-grid algorithm.

## Quickstart

```python
import numpy as np
from open_poisson_solver import solve_open_poisson_hockney

particles = np.random.normal(size=(10000, 3))
out = solve_open_poisson_hockney(
    particles,
    charge_per_particle=1.0,
    grid_shape=(64, 64, 64),
    padding=0.2,
)

phi = out["phi_grid"]          # (Nx,Ny,Nz)
Egrid = out["E_grid"]          # (Nx,Ny,Nz,3)
Epart = out["E_particles"]     # (N,3)
```

