import numpy as np


def main() -> None:
    from open_poisson_solver import GridSpec, gather_cic_vector, scatter_cic

    grid = GridSpec(
        origin=np.array([0.0, 0.0, 0.0]),
        spacing=np.array([1.0, 1.0, 1.0]),
        shape=(4, 4, 4),
    )

    # This is outside the valid CIC neighborhood (needs i0 in [0..Nx-2]).
    p_outside = np.array([[-0.1, 0.5, 0.5]])
    try:
        _ = scatter_cic(p_outside, grid=grid, charge_per_particle=1.0)
        raise AssertionError("Expected scatter_cic to raise for outside particle")
    except ValueError:
        pass

    vec = np.zeros((4, 4, 4, 3), dtype=float)
    try:
        _ = gather_cic_vector(p_outside, vec, grid=grid)
        raise AssertionError("Expected gather_cic_vector to raise for outside particle")
    except ValueError:
        pass

    print("OK")


if __name__ == "__main__":
    main()

