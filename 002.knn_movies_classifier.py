from sklearn.neighbors import KNeighborsClassifier


def knn_movies_classifier():
    # 电影分类
    x = [
        [39, 0, 31],
        [3, 2, 65],
        [2, 3, 55],
        [9, 38, 2],
        [8, 34, 17],
        [5, 2, 57],
        [21, 17, 5],
        [45, 2, 9],
    ]
    y = [0, 1, 2, 2, 2, 1, 0, 0]  # 0: 喜剧片 1: 动作片 2: 爱情片

    model = KNeighborsClassifier()

    model.fit(x, y)

    y_pred = model.predict([[3, 3, 37]])

    print(y_pred)


if __name__ == "__main__":
    knn_movies_classifier()
