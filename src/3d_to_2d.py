import pandas as pd
from os.path import abspath, dirname, join
import matplotlib.pyplot as plt
import umap


def get_data():
    csv_path = join(dirname(dirname(abspath(__file__))), "output", "vertices.csv")
    df = pd.read_csv(csv_path)

    data_3d = df[["x", "y", "z"]].values

    return data_3d


def plot_3d_data(data_3d: pd.DataFrame):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], s=5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D points")

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
    print(data_2d.shape)

    return data_2d


def plot_2d_data(data_2d):
    plt.figure(figsize=(6, 6))

    plt.scatter(data_2d[:, 0], data_2d[:, 1], s=5)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP 2D Projection")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    data_3d = get_data()
    plot_3d_data(data_3d)

    data_2d = reduce_data(data_3d)
    plot_2d_data(data_2d)
