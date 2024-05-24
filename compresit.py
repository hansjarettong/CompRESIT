from lingam import RESIT
import numpy as np
import sys
from lingam.hsic import hsic_test_gamma
from numpy.core.multiarray import array as array
from scipy.stats import rankdata
import Levenshtein

from nonlinear_dgp import generate_gp_cam
from sklearn.ensemble import RandomForestRegressor

import zlib
import gzip


# Make the mutual information swappable
class _BaseRESIT(RESIT):
    def _estimate_order(self, X):
        """Determine topological order"""
        S = np.arange(X.shape[1])
        pa = {}
        pi = []
        for _ in range(X.shape[1]):

            if len(S) == 1:
                pa[S[0]] = []
                pi.insert(0, S[0])
                continue

            hsic_stats = []
            for k in S:
                # Regress Xk on {Xi}
                predictors = [i for i in S if i != k]
                self._reg.fit(X[:, predictors], X[:, k])
                residual = X[:, k] - self._reg.predict(X[:, predictors])
                # Measure dependence between residuals and {Xi}
                hsic_stat = self._get_mutual_info(residual, X[:, predictors])
                hsic_stats.append(hsic_stat)

            k = S[np.argmin(hsic_stats)]
            S = S[S != k]
            pa[k] = S.tolist()
            pi.insert(0, k)

        return pa, pi

    def _get_mutual_info(self, residual, predictors):
        raise NotImplementedError


class HSIC_RESIT(_BaseRESIT):
    def _get_mutual_info(self, residual, predictors):
        hsic_stat, hsic_p = hsic_test_gamma(residual, predictors)
        return hsic_stat


class ZLIB_RESIT(_BaseRESIT):
    def __init__(self, regressor, random_state=None, mi_agg=max, alpha=0.01):
        super().__init__(regressor, random_state, alpha)
        self.mi_agg = mi_agg

    def _pairwise_mi(self, a: np.array, b: np.array):
        combined = np.outer(a, b)
        compressed_combined = sys.getsizeof(zlib.compress(combined)) / sys.getsizeof(
            combined
        )
        compressed_a = sys.getsizeof(zlib.compress(a)) / sys.getsizeof(a)
        compressed_b = sys.getsizeof(zlib.compress(b)) / sys.getsizeof(b)

        return compressed_a + compressed_b - compressed_combined

    def _get_mutual_info(self, residual, predictors):
        residual = np.array(residual).flatten()
        predictors = np.array(predictors)

        M = []
        for i in range(predictors.shape[1]):
            M.append(self._pairwise_mi(residual, predictors[:, i].flatten()))

        return self.mi_agg(M)


class Simple_ZLIB_RESIT(ZLIB_RESIT):
    def _pairwise_mi(self, a: np.array, b: np.array):
        combined = np.outer(a, b)
        compressed = sys.getsizeof(zlib.compress(combined))

        return -compressed


class SuperSimple_ZLIB_RESIT(ZLIB_RESIT):
    def _pairwise_mi(self, a: np.array, b: np.array):
        combined = np.concatenate([a, b])
        compressed = sys.getsizeof(zlib.compress(combined))

        return -compressed


class NCD_ZLIB_RESIT(ZLIB_RESIT):
    def _pairwise_mi(self, a: np.array, b: np.array):
        compressed_a = sys.getsizeof(zlib.compress(np.outer(a, a)))
        compressed_b = sys.getsizeof(zlib.compress(np.outer(b, b)))
        compressed_ab = sys.getsizeof(zlib.compress(np.outer(a, b)))
        return -(compressed_ab - min(compressed_a, compressed_b)) / max(
            compressed_a, compressed_b
        )


def floatarr2bytes(arr):
    return ("|".join([str(num) for idx, num in enumerate(arr)])).encode("utf-8")


class MaxNormNCDRESIT(ZLIB_RESIT):
    def __init__(
        self, regressor, compressor=gzip, random_state=None, mi_agg=max, alpha=0.01
    ):
        self.compressor = compressor
        super().__init__(regressor, random_state, mi_agg, alpha)

    def _pairwise_mi(self, a, b):
        a = np.abs(a.reshape(-1, 1) - a.reshape(1, -1)).flatten()
        b = np.abs(b.reshape(-1, 1) - b.reshape(1, -1)).flatten()
        combined = floatarr2bytes(np.maximum(a, b).flatten())
        a = floatarr2bytes(a.flatten())
        b = floatarr2bytes(b.flatten())

        compressed_ab = sys.getsizeof(self.compressor.compress(combined))
        compressed_a = sys.getsizeof(self.compressor.compress(a))
        compressed_b = sys.getsizeof(self.compressor.compress(b))
        return (compressed_ab - min(compressed_a, compressed_b)) / max(
            compressed_a, compressed_b
        )


class LevenshteinRESIT(ZLIB_RESIT):
    def __init__(
        self, regressor, use_ratio=False, random_state=None, mi_agg=max, alpha=0.01
    ):
        # if use_ratio is False, use Levenshtein distance. else use ratio
        self.use_ratio = use_ratio
        super().__init__(regressor, random_state, mi_agg, alpha)

    def _pairwise_mi(self, a: np.array, b: np.array):
        a = rankdata(a).astype(int)
        b = rankdata(b).astype(int)
        a = np.abs(a.reshape(-1, 1) - a.reshape(1, -1)).flatten()
        b = np.abs(b.reshape(-1, 1) - b.reshape(1, -1)).flatten()
        a = floatarr2bytes(a)
        b = floatarr2bytes(b)
        return (
            Levenshtein.ratio(a, b) if self.use_ratio else -Levenshtein.distance(a, b)
        )
