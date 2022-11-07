import bentoml


if __name__ == "__main__":
    # Create a Runner instance:
    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    # Runner#init_local initializes the model in current process, this is meant for development and testing only:
    iris_clf_runner.init_local()

    # This should yield the same result as the loaded model:
    print(iris_clf_runner.predict.run([[5.9, 3.0, 5.1, 1.8]]))