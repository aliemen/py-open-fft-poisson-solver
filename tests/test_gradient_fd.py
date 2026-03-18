import numpy as np


def main() -> None:
    from open_poisson_solver import gradient_fd

    Nx, Ny, Nz = 6, 5, 4
    spacing = np.array([2.0, 3.0, 5.0])

    x = (np.arange(Nx) + 0.5) * spacing[0]
    y = (np.arange(Ny) + 0.5) * spacing[1]
    z = (np.arange(Nz) + 0.5) * spacing[2]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Linear potential: phi = ax + by + cz -> grad = [a,b,c] everywhere.
    a, b, c = 1.25, -0.5, 0.1
    phi = a * X + b * Y + c * Z

    g = gradient_fd(phi, spacing)
    assert g.shape == (Nx, Ny, Nz, 3)

    # Interior should be exact for linear functions with centered differences.
    gi = g[1:-1, 1:-1, 1:-1]
    assert np.allclose(gi[..., 0], a)
    assert np.allclose(gi[..., 1], b)
    assert np.allclose(gi[..., 2], c)

    print("OK")


if __name__ == "__main__":
    main()

