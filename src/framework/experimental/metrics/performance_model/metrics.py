# -----------------------------------------------------------------------------
#  Project: Spark Self-Tuning Framework (STL-PARN-ILS-TS-BO)
#  File: yoro_perf_model_runner.py
#  Copyright (c) 2025 Mariano Garralda Barrio
#  Affiliation: Universidade da Coruña
#  SPDX-License-Identifier: CC-BY-NC-4.0 OR LicenseRef-Commercial
#
#  Associated publication:
#    "A hybrid metaheuristics–Bayesian optimization framework with safe transfer learning for continuous Spark tuning"
#    Mariano Garralda Barrio, Verónica Bolón Canedo, Carlos Eiras Franco
#    Universidade da Coruña, 2025.
#
#  Academic & research use: CC BY-NC 4.0
#    https://creativecommons.org/licenses/by-nc/4.0/
#  Commercial use: requires prior written consent.
#    Contact: mariano.garralda@udc.es
#
#  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, max_error, median_absolute_error, \
    mean_squared_log_error, mean_absolute_percentage_error


class NearestNeighborsDistanceMetrics:
    def __init__(self, distances: List[float]):
        self.distances = distances

    def mean_distance(self) -> float:
        return np.mean(self.distances)

    def standard_deviation(self) -> float:
        return np.std(self.distances)

    def max_distance(self) -> float:
        return np.max(self.distances)

    def min_distance(self) -> float:
        return np.min(self.distances)

    def gradient(self) -> List[float]:
        return np.gradient(self.distances)

    @staticmethod
    def get_statistic(distances: 'NearestNeighborsDistanceMetrics' | List['NearestNeighborsDistanceMetrics']) -> str:
        """
        Analyzing the distance parser directly gives us information about the absolute positioning of the sample in the data space,
        :param distances:
        :return:
        """

        if isinstance(distances, list):
            return (
                f"\tMean Distance: {np.mean([d.mean_distance() for d in distances]):.3f}\n"
                f"\tStd Distance: {np.std([d.standard_deviation() for d in distances]):.3f}\n"
                f"\tMax Distance: {np.max([d.max_distance() for d in distances]):.3f}\n"
                f"\tMin Distance: {np.min([d.min_distance() for d in distances]):.3f}\n"
            )
        else:
            return (
                f"\tMean Distance: {distances.mean_distance():.3f}\n"
                f"\tStd Distance: {distances.standard_deviation():.3f}\n"
                f"\tMax Distance: {distances.max_distance():.3f}\n"
                f"\tMin Distance: {distances.min_distance():.3f}\n"
            )

    def get_gradient_statistic(self) -> List[float]:
        """
        Gradient function calculates the n-th discrete difference along the given axis.
        The first order difference is given by out[n] = (a[n+1] - a[n]).
        Analyzing the gradient gives us information about the relative positioning and variation of the distances.
        :return:
        """
        gradient = np.gradient(self.distances)

        # # Print some basic statistics
        # print(f"Mean of gradient: {np.mean(gradient)}")
        # print(f"Standard deviation of gradient: {np.std(gradient)}")
        # print(f"Max of gradient: {np.max(gradient)}")
        # print(f"Min of gradient: {np.min(gradient)}")
        return gradient

    def get_cumsum_distribution(self) -> List[float]:
        return list(np.cumsum(self.distances))

    @staticmethod
    def get_outliers(values: List[float], threshold: float = 3) -> List[int]:  # List[Tuple[int, float]]:
        """
        The interquartile range (IQR) is a measure of statistical dispersion, being equal to the difference between the third and first quartiles.
        IQR = Q3 − Q1.
        The IQR is used to identify outliers by defining limits on the sample values that are a factor k of the IQR below the first quartile or above the third quartile.
        The common value for the factor k is 1.5.
        :param values:
        :param threshold:
        :return:
        """
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)

        # Return the indexes and values of the outliers
        # return [(i, x) for i, x in enumerate(values) if x < lower_bound or x > upper_bound]

        # Return the indexes of the outliers
        return [i for i, x in enumerate(values) if x < lower_bound or x > upper_bound]

    def __str__(self) -> str:
        return (
            f"\tMean Distance: {self.mean_distance():.3f}\n"
            f"\tStandard Deviation: {self.standard_deviation():.3f}\n"
            f"\tMax Distance: {self.max_distance():.3f}\n"
            f"\tMin Distance: {self.min_distance():.3f}\n"
        )


class RegressionMetrics:
    def __init__(self, y_true: List[float], y_pred: List[float]):
        self.y_true = y_true
        self.y_pred = y_pred

    def mean_absolute_error(self) -> float:
        return mean_absolute_error(self.y_true, self.y_pred)

    def mean_squared_error(self) -> float:
        return mean_squared_error(self.y_true, self.y_pred)

    def root_mean_squared_error(self) -> float:
        return np.sqrt(self.mean_squared_error())

    def r2_score(self) -> float:
        return r2_score(self.y_true, self.y_pred)

    def explained_variance_score(self) -> float:
        return explained_variance_score(self.y_true, self.y_pred)

    def max_error(self) -> float:
        return max_error(self.y_true, self.y_pred)

    def mean_absolute_percentage_error(self):
        return mean_absolute_percentage_error(self.y_true, self.y_pred)

    def median_absolute_error(self) -> float:
        return median_absolute_error(self.y_true, self.y_pred)

    def mean_squared_log_error(self) -> Optional[float]:
        """ Mean Squared Logarithmic cannot be used when targets contain negative values."""
        try:
            return mean_squared_log_error(self.y_true, self.y_pred)
        except ValueError:
            return None

    def harmonic_mean_errors(self) -> float:
        return 2 * (self.mean_absolute_error() * self.root_mean_squared_error()) / (self.mean_absolute_error() + self.root_mean_squared_error())

    def __str__(self):
        msle = self.mean_squared_log_error()
        return (
            f"\tMAE: {self.mean_absolute_error():.2f}\n"
            f"\tMSE: {self.mean_squared_error():.2f}\n"
            f"\tRMSE: {self.root_mean_squared_error():.2f}\n"
            f"\tR2 Score: {self.r2_score():.2f}\n"
            f"\tExplained Variance Score: {self.explained_variance_score():.2f}\n"
            f"\tMax Error: {self.max_error():.2f}\n"
            f"\tMean Absolute Percentage Error: {self.mean_absolute_percentage_error():.2f}\n"
            f"\tMedian Absolute Error: {self.median_absolute_error():.2f}\n"
            f"\tMean Squared Log Error: {'None' if msle is None else f'{msle:.2f}'}\n"
            f"\tHarmonic Mean Errors: {self.harmonic_mean_errors():.2f}\n"
        )
