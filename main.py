import logging

from sklearn import svm
from sklearn import datasets

import bentoml

logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":

    # Load training data
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Model Training
    clf = svm.SVC()
    clf.fit(X, y)

    # Save model to BentoML local model store
    saved_model = bentoml.sklearn.save_model(
        "iris_clf", clf, signatures={"predict": {"batchable": True, "batch_dim": 0}}
    )
    print(f"Model saved: {saved_model}")



"""
Model object can be loaded by

model = bentoml.sklearn.load_model("iris_clf:2uo5fkgxj27exuqj")

# Alternatively, use `latest` to find the newest version
model = bentoml.sklearn.load_model("iris_clf:latest")
"""

