import pandas as pd
from os.path import abspath, dirname, join
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import umap
import numpy as np

ROOT = dirname(dirname(abspath(__file__)))
REDUCE = False
VERBOSE_CHART = False
COLOR_PTS = True
ZOOM_LEVEL = 0.5
NUM_FRAMES = 300


def get_data():
    csv_path = join(ROOT, "output", "vertices.csv")
    df = pd.read_csv(csv_path)

    data_3d = df[["x", "y", "z"]].values

    if COLOR_PTS:
        color_rgb = df[["r", "g", "b"]].values
    else:
        color_rgb = None

    return data_3d, color_rgb


def plot_3d_data(data_3d, color_rgb=None):

    if hasattr(data_3d, "values"):
        pts = data_3d.values
    else:
        pts = np.asarray(data_3d)

    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    z = pts[:, 2].copy()

    activation_time = (1 - (z - z.min()) / (z.max() - z.min() + 1e-8)) * 100

    vel = np.zeros_like(z)
    acc = 0.001
    t_active = np.zeros_like(z)
    
    ghost_positions = []
    ghost_decay = 0.02
    ghost_initial_alpha = 0.3
    ghost_size = 20
    ghost_scatters = []

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if COLOR_PTS:
        scatter = ax.scatter(x, y, z, marker="o", s=1, c=np.asarray(color_rgb))
    else:
        scatter = ax.scatter(x, y, z, s=5)

    if VERBOSE_CHART:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D points")

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5

    x_tuple = (mid_x - max_range / ZOOM_LEVEL,
            mid_x + max_range / ZOOM_LEVEL)

    y_tuple = (mid_y - max_range / ZOOM_LEVEL,
            mid_y + max_range / ZOOM_LEVEL)

    z_tuple = (mid_z - max_range,
            mid_z + 2* max_range / ZOOM_LEVEL)

    ax.set_xlim(x_tuple)
    ax.set_ylim(y_tuple)
    ax.set_zlim(z_tuple)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()

    def update(frame):
        nonlocal z, vel, t_active, ghost_positions, ghost_scatters

        active = frame > activation_time
        t_active[active] += 1
        vel[active] += acc
        z[active] += vel[active]

        for i in range(len(x)):
            ghost_positions.append([x[i], y[i], z[i], ghost_initial_alpha])

        for gs in ghost_scatters:
            gs.remove()
        ghost_scatters = []

        new_ghost_list = []
        for gx, gy, gz, alpha in ghost_positions:
            new_alpha = alpha - ghost_decay
            if new_alpha > 0:
                gs = ax.scatter(gx, gy, gz, s=ghost_size, c="gray", alpha=new_alpha)
                ghost_scatters.append(gs)
                new_ghost_list.append([gx, gy, gz, new_alpha])

        ghost_positions = new_ghost_list

        scatter._offsets3d = (x, y, z)

        fig.savefig(join(ROOT, "output", "plt_render", f"frame_{frame:04d}.png"), dpi=150)

    for frame in range(NUM_FRAMES):
        update(frame)


def reduce_data(data_3d: pd.DataFrame):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )

    data_2d = reducer.fit_transform(data_3d)

    return data_2d


def plot_2d_data(data_2d, color_rgb=None):
    plt.figure(figsize=(6, 6))

    if COLOR_PTS:
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=5, c=np.asarray(color_rgb))
    else:
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=5)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP 2D Projection")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data_3d, color_rgb = get_data()
    plot_3d_data(data_3d, color_rgb=color_rgb)

    if REDUCE:
        data_2d = reduce_data(data_3d)
        plot_2d_data(data_2d, color_rgb=color_rgb)
