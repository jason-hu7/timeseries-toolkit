from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


# Fitting is simply not needed in this category of naive regressos
class NaiveRegressor(ABC):
    @staticmethod
    @abstractmethod
    def predict(X: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def score(y: np.array, y_predict: np.array) -> np.float64:
        """
        RSME score
        """
        rmse = np.sqrt(((y - y_predict) ** 2).mean())
        return rmse


class Persist(NaiveRegressor):
    """
    The prediction is the same as the last recorded number of the time window.
    """

    @staticmethod
    def predict(X: pd.DataFrame) -> pd.DataFrame:
        _data_validate(X)

        for feature in X.columns:
            if feature[-1] == "1":
                return X[feature]


class MovingAverage(NaiveRegressor):
    """
    The prediction is the same as the average of the moving time window.
    """

    @staticmethod
    def predict(X: pd.DataFrame) -> pd.DataFrame:
        _data_validate(X)
        return X.mean(axis=1)


class ExponentialMovingAverage(NaiveRegressor):
    """
    The prediction is the same as the exponential average of the moving time
    window.
    """

    @staticmethod
    def predict(X: pd.DataFrame, span: Optional[int] = None) -> pd.DataFrame:
        _data_validate(X)
        return 0.5 * X.ewm(span=span, axis=1).mean()


def _data_validate(X: pd.DataFrame) -> None:
    """
    Make sure the data is univariate.
    """
    seen = set()
    for col in X.columns:
        seen.add(col.split("_goback_")[0])
    if len(seen) != 1:
        raise ValueError(
            "The data you used is not univariate, "
            "NaiveRegressor only works with univariate "
            "problems."
        )
