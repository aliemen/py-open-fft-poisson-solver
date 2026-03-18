import numpy as np


def main() -> None:
    from pyHockneySolver import convolve_open_poisson_hockney

    Nx = Ny = Nz = 4
    spacing = np.array([2.0, 3.0, 4.0], dtype=float)

    # Use a single-cell delta in rho with value 1.0 (charge density).
    # With the discrete free-space Green's function regularization used by the
    # Hockney algorithm, the potential at the same cell should match 1/(4*pi)
    # for any spacing.
    rho = np.zeros((Nx, Ny, Nz), dtype=float)
    rho[1, 1, 1] = 1.0

    phi = convolve_open_poisson_hockney(rho, spacing)

    expected = 1.0 / (4.0 * np.pi)
    assert np.isfinite(phi).all()
    assert np.isclose(phi[1, 1, 1], expected, rtol=0.0, atol=1e-12)

    # eps0 scaling: convolve uses G/eps0, so phi should scale by 1/eps0.
    eps0 = 2.5
    phi_eps = convolve_open_poisson_hockney(rho, spacing, eps0=eps0)
    assert np.isclose(phi_eps[1, 1, 1], expected / eps0, rtol=0.0, atol=1e-12)

    print("OK")


if __name__ == "__main__":
    main()

