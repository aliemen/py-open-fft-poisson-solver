import numpy as np


def main() -> None:
    from open_poisson_solver import solve_open_poisson_hockney

    rng = np.random.default_rng(0)

    # Symmetric bunch centered at origin.
    N = 50_000
    sigma = np.array([1e-3, 1e-3, 2e-3])
    particles = rng.normal(size=(N, 3)) * sigma[None, :]

    grid_shape = (48, 48, 64)

    out1 = solve_open_poisson_hockney(
        particles,
        charge_per_particle=1.0,
        grid_shape=grid_shape,
        padding=0.3,
    )
    out2 = solve_open_poisson_hockney(
        particles,
        charge_per_particle=2.0,
        grid_shape=grid_shape,
        padding=0.3,
    )

    # Shape checks
    assert out1["phi_grid"].shape == grid_shape
    assert out1["rho_grid"].shape == grid_shape
    assert out1["E_grid"].shape == (*grid_shape, 3)
    assert out1["E_particles"].shape == (N, 3)

    # Finiteness checks
    assert np.isfinite(out1["phi_grid"]).all()
    assert np.isfinite(out1["E_grid"]).all()
    assert np.isfinite(out1["E_particles"]).all()

    # Symmetry sanity: mean E should be close to ~0 for a centered symmetric bunch.
    meanE = out1["E_particles"].mean(axis=0)
    assert np.linalg.norm(meanE) < 1e-2 * np.linalg.norm(out1["E_particles"]).mean()

    # Scaling check: doubling charge doubles potential and E (approximately).
    phi_ratio = np.nanmean(out2["phi_grid"] / out1["phi_grid"])
    E_ratio = np.nanmean(out2["E_particles"] / out1["E_particles"])
    assert 1.8 < phi_ratio < 2.2
    assert 1.8 < E_ratio < 2.2

    print("OK")


if __name__ == "__main__":
    main()

