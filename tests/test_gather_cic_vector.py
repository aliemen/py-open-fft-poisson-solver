import numpy as np


def main() -> None:
    from pyHockneySolver import GridSpec, gather_cic_vector

    grid = GridSpec(
        origin=np.array([0.0, 0.0, 0.0]),
        spacing=np.array([1.0, 1.0, 1.0]),
        shape=(5, 5, 5),
    )

    Nx, Ny, Nz = grid.shape
    x = grid.origin[0] + (np.arange(Nx) + 0.5) * grid.spacing[0]
    y = grid.origin[1] + (np.arange(Ny) + 0.5) * grid.spacing[1]
    z = grid.origin[2] + (np.arange(Nz) + 0.5) * grid.spacing[2]

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Define a linear field E = [x, y, z] on cell centers.
    vec = np.stack([X, Y, Z], axis=3)

    # Gather at exact cell centers should reproduce exactly.
    p = np.array([[x[2], y[3], z[1]], [x[0], y[0], z[0]]], dtype=float)
    out = gather_cic_vector(p, vec, grid=grid)
    assert np.allclose(out[0], [x[2], y[3], z[1]])
    assert np.allclose(out[1], [x[0], y[0], z[0]])

    print("OK")


if __name__ == "__main__":
    main()

