[![Latest Release](https://img.shields.io/github/v/release/aliemen/py-open-fft-poisson-solver)](https://github.com/aliemen/py-open-fft-poisson-solver/releases/latest)

# pyHockneySolver

Single-process FFT-based open-boundary Poisson solver using the Hockney doubled-grid algorithm, following the `FFTOpenPoissonSolver` Hockney implementation in the [IPPL](https://github.com/IPPL-framework/ippl) C++ library. For more information on the implementation, scroll to the bottom of this README.

## Quickstart

```python
import numpy as np
from pyHockneySolver import solve_open_poisson_hockney

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

## How to run

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/minimal_electron_bunch_3d_plot.py
```

This runs a minimal $32^3$ open-boundary solve for a normally distributed electron bunch
and shows a 3D surface plot of the potential averaged over the z-axis.

## Using in your own code (clone + import)

If you already have NumPy installed, you can clone this repository and use the package directly. Go into your project, where you want to make the package available.

1. Clone and install the Python package (editable install):

```bash
git clone https://github.com/aliemen/py-open-fft-poisson-solver.git 
cd py-open-fft-poisson-solver
python3 -m pip install -e .
```

2. Import and run a minimal solve from your own Python code:

```python
import numpy as np
from pyHockneySolver import solve_open_poisson_hockney

particles = np.random.normal(size=(1000, 3))
out = solve_open_poisson_hockney(
    particles,
    charge_per_particle=1.0,
    grid_shape=(16, 16, 16),
)
phi = out["phi_grid"]
E_particles = out["E_particles"]
print(E_particles.shape)
```

## License

This Python implementation is provided under the same license as IPPL: the GNU General Public License, version 3 or (at your option) any later version. See `LICENSE` for details and the IPPL project at
[github.com/IPPL-framework/ippl](https://github.com/IPPL-framework/ippl).

## Mathematical formulation

The solver works on a uniform Cartesian grid and computes a scalar potential
$\phi(\mathbf{x})$ from a charge density $\rho(\mathbf{x})$ by solving
the Poisson equation with free-space (open) boundary conditions:

$$
\nabla^2 \phi(\mathbf{x}) = -\rho(\mathbf{x})
$$

Numerically, this is implemented via the Hockney doubled-grid convolution method:

1. **Scatter**: particles with charges $q_i$ at positions $\mathbf{x}_i$ are
   deposited onto the mesh with a cloud-in-cell (CIC) kernel to form $\rho$.
2. **Green's function**: build a discrete free-space Green's function
   $G(\mathbf{r}) \approx -1/(4\pi\|\mathbf{r}\|)$ on a grid of size
   $(2N_x,2N_y,2N_z)$ using folded distances.
3. **Convolution by FFT**: compute
   $\phi = - G * \rho$ via FFTs on the doubled grid and restrict back to the
   physical $N_x\times N_y\times N_z$ domain, with normalization matched to IPPL.
4. **Field**: the electric field on the grid is recovered as
   $\mathbf{E} = -\nabla \phi$ using centered finite differences, and then
   interpolated back to particles with CIC.

If you supply a physical permittivity $\varepsilon_0$ via the `eps0` argument,
the Green's function is scaled accordingly, so the effective equation becomes

$$
\nabla^2 \phi = -\frac{\rho}{\varepsilon_0},
$$

matching the SI convention. Otherwise, the solver works in code units with
$\nabla^2 \phi = -\rho$.

For more information, consider reading [arXiv:2405.02603](https://doi.org/10.48550/arXiv.2405.02603) by my colleagues Mayani et al. (2025). The paper outlines the implementation and specifics of this algorithm together as implemented in IPPL-framework/ippl. 