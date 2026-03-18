import numpy as np


def main() -> None:
    from pyHockneySolver import convolve_open_poisson_hockney

    spacing = np.array([1.0, 1.0, 1.0])
    rho = np.zeros((16, 16, 16), dtype=float)

    # Put a single-cell delta at the center.
    rho[8, 8, 8] = 1.0

    phi = convolve_open_poisson_hockney(rho, spacing)
    assert phi.shape == rho.shape
    assert np.isfinite(phi).all()

    # Symmetry: along axes, values should be symmetric around center.
    cx, cy, cz = 8, 8, 8
    for d in range(1, 5):
        assert np.isclose(phi[cx + d, cy, cz], phi[cx - d, cy, cz])
        assert np.isclose(phi[cx, cy + d, cz], phi[cx, cy - d, cz])
        assert np.isclose(phi[cx, cy, cz + d], phi[cx, cy, cz - d])

    print("OK")


if __name__ == "__main__":
    main()

