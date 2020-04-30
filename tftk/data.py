import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


class Data(ABC):
    """
    Base class for time series dataset.
    """

    @classmethod
    @abstractmethod
    def make_ts_features(
        cls, ts_df: pd.DataFrame, predictor_label: List[str], target_label: List[str]
    ):
        """
        Format the time series data to forecast features and forecast target
        DataFrames.
        """
        pass

    @abstractmethod
    def normalize(self):
        """
        Normalize the time series data.
        """
        pass

    @abstractmethod
    def smooth(self):
        """
        Smooth the time series data.
        """
        pass

    @abstractmethod
    def extract_stat_features(self):
        """
        Extract statistical features from the given data.
        """
        pass


class TsData(Data):
    """
    Order based time-series feature extraction.
    """

    def __init__(
        self,
        ts_features: pd.DataFrame,
        ts_target: pd.DataFrame,
        predictor_labels: List[str],
    ) -> None:
        super().__init__()
        self.ts_features = ts_features
        self.ts_target = ts_target
        self.predictor_labels = predictor_labels
        self.normalization_coef: Dict[str, float] = {}

    # Construct time-series features for the given data
    @classmethod
    def make_ts_features(
        cls,
        ts_df: pd.DataFrame,
        predictor_labels: List[str],
        target_label: List[str],
        goback_steps: int = 25,
        period_per_step: int = 1,
        forecast_period: int = 12,
        auto_feature_select: bool = True,
    ) -> "TsData":
        """Take a time series DataFrame and extract the features. This function
        should only be used when the time interval of the data is consistent.

        Args:
            ts_df: Time series DataFrame used to prepare time series features.
            predictor_labels: A list of the headers of the columns to use as
                predictors.
            target_label: A list of the headers of the columns to use as
                targets.
            goback_steps: The number of steps to walk backwards in time.
            period_per_step: Number of periods per step when walk backwards in
                time to extract features.
            forecast_period: The number of periods ahead of the current time
                where the forecast is cast.
            auto_feature_select: Whether to use correlation matrix and feature
                importance to select features.

        Returns:
            TsData: An instance of the time series class TsData
        """
        # TODO: consider the case when ts_df is a pandas series

        # Validate the DataFrame to ensure feasibility for following operations
        time_series_validation(ts_df)

        # Automatically get rid of highly correlated features
        if auto_feature_select:
            predictor_labels = process_redundancy(ts_df[predictor_labels], target_label)

        X_df = ts_df[target_label].copy()
        y_df = ts_df[target_label].copy()

        for predictor in predictor_labels:
            for i in range(goback_steps):
                col_str = f"{predictor}_goback_{str(i+1)}"
                sensor_train = ts_df[predictor]
                X_df[col_str] = sensor_train.shift(
                    forecast_period + period_per_step * i
                )

        X_df = X_df.dropna()
        X_df = X_df.drop(target_label, axis=1).copy()

        return cls(X_df, y_df.loc[X_df.index], predictor_labels)

    # Normalize sensor data
    def normalize(self) -> "TsData":
        norm_mean = self.ts_features.mean()
        norm_std = self.ts_features.std() + 1e-32
        self.normalization_coef["mean"] = norm_mean
        self.normalization_coef["std"] = norm_std
        self.ts_features = (self.ts_features - norm_mean) / norm_std
        return self

    # Smooth sensor data by using moving average
    def smooth(
        self,
        ma_group: Optional[List[str]] = None,
        ma_window: Optional[List[int]] = None,
        exponential: bool = True,
    ) -> "TsData":
        # TODO implement moving average

        # Exponential moving average
        for mag, maw in zip(ma_group, ma_window):
            ma_features = [s for s in self.ts_features if s.find(mag) == 0]
            self.ts_features.loc[:, ma_features] = (
                self.ts_features.loc[:, ma_features].ewm(span=maw).mean()
            )
        return self

    # Calculate statistics of each time window and use as features
    def extract_stat_features(
        self,
        predictors: Optional[List[str]] = None,
        mean: bool = True,
        median: bool = True,
        volatility: bool = True,
    ) -> "TsData":
        df = self.ts_features.copy()
        cols: List[str] = []
        for predictor in predictors:
            cols = [column for column in df.columns if column.find(predictor) == 1]
            mean_features = df[cols].mean(axis=1).rename(f"mean_{predictor}")
            median_features = df[cols].median(axis=1).rename(f"median_{predictor}")
            std_features = df[cols].std(axis=1).rename(f"std_{predictor}")
            if mean:
                df = pd.concat((df, mean_features), axis=1)
            if median:
                df = pd.concat((df, median_features), axis=1)
            if volatility:
                df = pd.concat((df, std_features), axis=1)
            self.ts_features = df
        return self


class Error(Exception):
    """Base class for exceptions in this module."""

    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message: str) -> None:
        self.expression = "There are problems with your input."
        self.message = message


def time_series_validation(df: pd.DataFrame) -> None:
    """
    checking sampling rate, NaN, and data size.
    """

    # Check if there is any NaN value in the dataset
    if df.isnull().values.any():
        raise InputError("NaN exist in your input.")

    # Check if the time spacing between adjacent points are consistent
    if np.unique(np.diff(df.index) / np.timedelta64(1, "s")).size > 1:
        warnings.warn(
            "The time interval of your index is inconsistent ,"
            "you need to resample the data so that it satisfies "
            "the requirements"
        )

    # Check if the index of the DataFrame is DatetimeIndex
    if type(df.index) is not pd.DatetimeIndex:
        raise InputError(
            "The index of your input is not DatetimeIndex, please"
            " parse the time stamp and set it as the index."
        )

    # Check if the index of the DataFrame is sorted in the correct order
    if not df.index.is_monotonic_increasing:
        raise InputError(
            "The input of the data is not sorted in the correct " "time order"
        )


def process_redundancy(ts_df: pd.DataFrame, target_label: List[str]) -> List[str]:
    """
    Get rid of predictors with high correlation, based on correlation matrix.
    """
    correlated_pairs = _get_correlated_sets(ts_df)

    # Randomly drop all the correlated sets
    predictor_to_drop = set()
    for pair in correlated_pairs:
        if target_label[0] in pair:
            pair.remove(target_label[0])
            drop = list(pair)
        else:
            drop = random.sample(pair, k=len(pair) - 1)
        predictor_to_drop.update(drop)

    predictor_to_keep = [i for i in ts_df.columns if i not in predictor_to_drop]

    return predictor_to_keep


def _get_correlated_sets(ts_df: pd.DataFrame) -> List[Set[str]]:
    """
    Get a list of correlated predictors.
    """
    corr = ts_df.corr().abs().unstack()

    # Get diagonal and lower triganular pairs of correlation matrix.
    pairs_to_drop = set()
    for i in range(0, ts_df.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((ts_df.columns[i], ts_df.columns[j]))

    # Drop the lower triangular pairs from the correlation matrix
    corr2 = corr.drop(labels=pairs_to_drop).sort_values(ascending=False)

    # Get corrlelations higher than 0.9
    corr2 = corr2[corr2 > 0.9]
    num_pairs = len(corr2)

    # Get the correlated sets
    level1_labels = corr2.index.levels[0][corr2.index.labels[0]]
    level2_labels = corr2.index.levels[1][corr2.index.labels[1]]
    correlated_pairs: List[Set[str]] = []
    for i in range(num_pairs):
        current_pair = set((level1_labels[i], level2_labels[i]))
        for j in range(num_pairs):
            if level1_labels[j] in current_pair or level2_labels[j] in current_pair:
                second_connection = set((level1_labels[j], level2_labels[j]))
                current_pair = current_pair.union(second_connection)

        if current_pair not in correlated_pairs:
            correlated_pairs.append(current_pair)

    return correlated_pairs
