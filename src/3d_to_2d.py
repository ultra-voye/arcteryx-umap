import pandas as pd
from os.path import abspath, dirname, join
import matplotlib.pyplot as plt
import umap
import numpy as np

VERBOSE_CHART = False
COLOR_PTS = True

def get_data():
    csv_path = join(dirname(dirname(abspath(__file__))), "output", "vertices.csv")
    df = pd.read_csv(csv_path)

    data_3d = df[["x", "y", "z"]].values

    if COLOR_PTS:
        color_rgb = df[["r", "g", "b"]].values
    else:
        color_rgb = None

    return data_3d, color_rgb


def plot_3d_data(data_3d, color_rgb=None):

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if hasattr(data_3d, "values"):
        pts = data_3d.values
    else:
        pts = np.asarray(data_3d)

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    if COLOR_PTS:
        ax.scatter(x, y, z, s=5, c=np.asarray(color_rgb))
    else:
        ax.scatter(x, y, z, s=5)

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

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


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

    data_2d = reduce_data(data_3d)
    plot_2d_data(data_2d, color_rgb=color_rgb)
