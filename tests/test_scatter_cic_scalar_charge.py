import numpy as np


def main() -> None:
    from pyHockneySolver import GridSpec, scatter_cic

    grid = GridSpec(
        origin=np.array([0.0, 0.0, 0.0]),
        spacing=np.array([1.0, 1.0, 1.0]),
        shape=(4, 4, 4),
    )

    # Put a particle exactly at the center of cell (1,1,1):
    # center = origin + (i+0.5)*h
    p = np.array([[1.5, 1.5, 1.5]])
    rho = scatter_cic(p, grid=grid, charge_per_particle=2.0)

    # Should deposit entirely into that one cell for CIC at a cell center.
    assert rho[1, 1, 1] == 2.0  # dV=1
    assert np.isclose(rho.sum(), 2.0)

    # Sum rule: sum(rho)*dV == total charge
    dV = float(np.prod(grid.spacing))
    assert np.isclose(rho.sum() * dV, 2.0)

    print("OK")


if __name__ == "__main__":
    main()

