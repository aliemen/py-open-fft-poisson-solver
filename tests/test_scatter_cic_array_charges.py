import numpy as np


def main() -> None:
    from pyHockneySolver import GridSpec, scatter_cic

    grid = GridSpec(
        origin=np.array([0.0, 0.0, 0.0]),
        spacing=np.array([1.0, 1.0, 1.0]),
        shape=(4, 4, 4),
    )

    particles = np.array(
        [
            [1.5, 1.5, 1.5],  # cell (1,1,1)
            [2.5, 1.5, 1.5],  # cell (2,1,1)
        ]
    )
    charges = np.array([2.0, -1.0])

    rho = scatter_cic(particles, grid=grid, particle_charges=charges)
    assert np.isclose(rho.sum(), charges.sum())
    assert rho[1, 1, 1] == 2.0
    assert rho[2, 1, 1] == -1.0

    # Consistency: all-equal array charges matches scalar API.
    rho_scalar = scatter_cic(particles, grid=grid, charge_per_particle=3.0)
    rho_array = scatter_cic(particles, grid=grid, particle_charges=np.array([3.0, 3.0]))
    assert np.allclose(rho_scalar, rho_array)

    print("OK")


if __name__ == "__main__":
    main()

