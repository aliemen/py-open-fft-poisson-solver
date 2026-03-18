import numpy as np


def generate_truncated_normal_bunch(
    rng: np.random.Generator,
    *,
    N: int,
    sigma: np.ndarray,
    nsigma: float = 3.0,
) -> np.ndarray:
    """
    Generate particles from a normal distribution centered at 0, truncated to
    |x_d| <= nsigma*sigma_d for each coordinate d using rejection sampling.
    """
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (3,):
        raise ValueError(f"sigma must have shape (3,), got {sigma.shape}")
    if N <= 0:
        raise ValueError("N must be positive")

    particles = np.empty((0, 3), dtype=float)
    while particles.shape[0] < N:
        # Oversample to reduce loop iterations.
        batch = int(np.ceil((N - particles.shape[0]) * 1.3))
        cand = rng.normal(size=(batch, 3)) * sigma[None, :]
        mask = np.all(np.abs(cand) <= nsigma * sigma[None, :], axis=1)
        keep = cand[mask]
        if keep.shape[0] > 0:
            particles = np.vstack([particles, keep])

    return particles[:N]


def plot_phi_xy_surface(phi: np.ndarray, origin: np.ndarray, spacing: np.ndarray) -> None:
    """Plot the z-averaged potential phi(x,y) as a 3D surface."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    phi_xy = phi.mean(axis=2)  # (Nx,Ny)

    Nx, Ny, _ = phi.shape
    x = origin[0] + (np.arange(Nx) + 0.5) * spacing[0]
    y = origin[1] + (np.arange(Ny) + 0.5) * spacing[1]

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    vmin = float(np.nanmin(phi_xy))
    vmax = float(np.nanmax(phi_xy))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    surf = ax.plot_surface(
        X,
        Y,
        phi_xy,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        cmap="viridis",
        norm=norm,
    )

    ax.set_title("Electron bunch potential (z-averaged) surface")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("phi [V] (up to normalization)")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="phi")

    plt.show()


def main() -> None:
    from open_poisson_solver import solve_open_poisson_hockney

    rng = np.random.default_rng(42)

    # Normally distributed bunch, centered at origin (truncated to +/-3 sigma) --> same as IPPL samplers.
    N = 200_000
    sigma = np.array([1e-3, 1e-3, 1e-3], dtype=float)
    nsigma = 3.0

    particles = generate_truncated_normal_bunch(
        rng,
        N=N,
        sigma=sigma,
        nsigma=nsigma,
    )

    # print particle min and max positions
    print(f"Particle min positions: {particles.min(axis=0)}")
    print(f"Particle max positions: {particles.max(axis=0)}")

    # Fix total bunch charge and derive per-particle charge.
    total_bunch_charge = -1e-9  # Coulomb (1 nC)
    charge_per_particle = total_bunch_charge / N

    out = solve_open_poisson_hockney(
        particles,
        charge_per_particle=charge_per_particle,
        grid_shape=(32, 32, 32),
        padding=0.3,
        eps0=8.8541878128e-12,
    )

    plot_phi_xy_surface(phi=out["phi_grid"], origin=out["origin"], spacing=out["spacing"])

    # Info: can access E field at particle positions with out["E_particles"]
    print(f"Shape of E field at particle positions: {out['E_particles'].shape}")


if __name__ == "__main__":
    main()

