from sklearn.linear_model import LinearRegression
import joblib


def linear_regression_train_predict():
    x = [[80, 86], [82, 80], [85, 78], [90, 90], [86, 82], [82, 90], [78, 80], [92, 94]]
    y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

    model = LinearRegression()

    model.fit(x, y)

    print(f"{model.coef_=}")
    print(f"{model.intercept_=}")

    # y_pred = model.predict([[90, 80]])
    y_pred = model.predict([[90, 80], [70, 75], [81, 82]])
    print(y_pred)

    # 保存模型
    joblib.dump(model, "models/linear_regression_checkpoint.bin")

    model = joblib.load("models/linear_regression_checkpoint.bin")

    y_pred = model.predict([[90, 80]])
    print(y_pred)


if __name__ == "__main__":
    linear_regression_train_predict()
