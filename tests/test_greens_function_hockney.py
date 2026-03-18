import numpy as np


def main() -> None:
    from pyHockneySolver import greens_function_hockney

    spacing = np.array([1.0, 2.0, 3.0])
    physical_shape = (4, 3, 2)
    shape2 = (8, 6, 4)

    G = greens_function_hockney(shape2, physical_shape, spacing, regularize_origin=True)
    assert G.shape == shape2
    assert np.isfinite(G).all()
    assert np.isclose(G[0, 0, 0], -1.0 / (4.0 * np.pi))

    # Folding symmetry checks (sample a few points).
    Nx, Ny, Nz = physical_shape
    Nx2, Ny2, Nz2 = shape2
    for i, j, k in [(1, 0, 0), (0, 2, 1), (3, 1, 0)]:
        mi = (-i) % Nx2
        mj = (-j) % Ny2
        mk = (-k) % Nz2
        assert np.isclose(G[i, j, k], G[mi, j, k])
        assert np.isclose(G[i, j, k], G[i, mj, k])
        assert np.isclose(G[i, j, k], G[i, j, mk])

    print("OK")


if __name__ == "__main__":
    main()

