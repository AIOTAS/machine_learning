from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def kmeans():
    x, y = make_blobs(
        n_samples=2000,
        n_features=2,
        centers=(
            (-1, -1),
            (
                0,
                0,
            ),
            (1, 1),
            (2, 2),
        ),
        cluster_std=(0.4, 0.2, 0.2, 0.2),
        random_state=22,
    )

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker="o")
    plt.show()

    model = KMeans(n_clusters=4, init="k-means++", random_state=22)
    y_pred = model.fit_predict(x)

    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()


if __name__ == "__main__":
    kmeans()
