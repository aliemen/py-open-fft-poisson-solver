import numpy as np


def main() -> None:
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    from open_poisson_solver import solve_open_poisson_hockney

    rng = np.random.default_rng(1)

    # Normally distributed bunch.
    N = 200_000
    sigma = np.array([1e-3, 1e-3, 1e-3])
    particles = rng.normal(size=(N, 3)) * sigma[None, :]

    out = solve_open_poisson_hockney(
        particles,
        charge_per_particle=-1.0,
        grid_shape=(64, 64, 96),
        padding=0.3,
    )

    phi = out["phi_grid"]
    origin = out["origin"]
    spacing = out["spacing"]

    # Average potential over z.
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
    ax.set_title("Open Poisson potential (z-averaged) surface")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("phi")
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="phi")

    outpath = os.path.join(os.path.dirname(__file__), "phi_zavg.png")
    fig.savefig(outpath, dpi=150)
    print(f"Wrote {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    main()

