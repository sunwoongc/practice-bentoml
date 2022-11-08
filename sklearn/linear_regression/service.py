import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()

svc = bentoml.Service("linear_regression", runners=[reg_runner])

input_spec = NumpyNdarray(dtype="int", shape=(-1, 2))

@svc.api(input=input_spec, output=NumpyNdarray())
async def classify(input_series: np.ndarray) -> np.ndarray:
    return await reg_runner.predict.run(input_series)