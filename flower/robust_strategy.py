from typing import List, Tuple, Union, Optional, Dict
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class RobustFedAvg(FedAvg):
    """Byzantine-robust Federated Averaging Strategy for Flower.

    Supports:
      - Coordinate-wise Median
      - Coordinate-wise Trimmed Mean
      - Krum (Blanchard et al., NeurIPS 2017)
    """

    def __init__(
        self,
        *args,
        aggregation_rule: str = "median",
        beta: int = 1,
        krum_f: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.aggregation_rule = aggregation_rule
        self.beta = beta
        self.krum_f = krum_f

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        weights_list = [
            parameters_to_ndarrays(fit_res.parameters)
            for _, fit_res in results
        ]

        if self.aggregation_rule == "median":
            aggregated_ndarrays = self._aggregate_median(weights_list)
        elif self.aggregation_rule == "trimmed_mean":
            aggregated_ndarrays = self._aggregate_trimmed_mean(weights_list, self.beta)
        elif self.aggregation_rule == "krum":
            aggregated_ndarrays = self._aggregate_krum(weights_list, self.krum_f)
        else:
            # Fall back to standard FedAvg
            return super().aggregate_fit(server_round, results, failures)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return parameters_aggregated, metrics_aggregated

    @staticmethod
    def _aggregate_median(weights_list: List[List[np.ndarray]]) -> List[np.ndarray]:
        num_layers = len(weights_list[0])
        aggregated = []
        for layer_idx in range(num_layers):
            stacked = np.stack([w[layer_idx] for w in weights_list], axis=0)
            aggregated.append(np.median(stacked, axis=0))
        return aggregated

    @staticmethod
    def _aggregate_trimmed_mean(weights_list: List[List[np.ndarray]], beta: int) -> List[np.ndarray]:
        n = len(weights_list)
        if 2 * beta >= n:
            return RobustFedAvg._aggregate_median(weights_list)

        num_layers = len(weights_list[0])
        aggregated = []
        for layer_idx in range(num_layers):
            stacked = np.stack([w[layer_idx] for w in weights_list], axis=0)
            sorted_stacked = np.sort(stacked, axis=0)
            trimmed = sorted_stacked[beta:n - beta]
            aggregated.append(np.mean(trimmed, axis=0))
        return aggregated

    @staticmethod
    def _aggregate_krum(weights_list: List[List[np.ndarray]], f: int) -> List[np.ndarray]:
        n = len(weights_list)
        if n <= 2:
            return RobustFedAvg._aggregate_median(weights_list)

        flat_vectors = [
            np.concatenate([layer.ravel() for layer in w])
            for w in weights_list
        ]
        flat_mat = np.stack(flat_vectors, axis=0)  # (n, D)

        diffs = flat_mat[:, np.newaxis, :] - flat_mat[np.newaxis, :, :]  # (n, n, D)
        dists = np.sum(diffs ** 2, axis=-1)  # (n, n)

        num_neighbors = max(1, n - f - 2)
        scores = []
        for i in range(n):
            other_dists = np.concatenate([dists[i, :i], dists[i, i + 1:]])
            sorted_dists = np.sort(other_dists)
            scores.append(np.sum(sorted_dists[:num_neighbors]))

        best_idx = int(np.argmin(scores))
        return [np.copy(layer) for layer in weights_list[best_idx]]
