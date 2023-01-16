from statistics import mean
from typing import Optional

import numpy as np

from accuracy import normalize_accuracy


def get_difficulty_per_trial(
    accuracy_per_instance: dict,
    exclude_models: Optional[list[str]] = None,
    num_options: Optional[int] = None,
):
    if exclude_models is None:
        exclude_models = set()
    result = {}
    for instance_id, instance_results in accuracy_per_instance.items():
        filtered_results = [
            instance_result
            for instance_result in instance_results
            if instance_result["model"] not in exclude_models
        ]
        accuracy = mean(result["is_correct"] for result in filtered_results)
        if num_options is not None:
            accuracy = normalize_accuracy(accuracy, num_options)
        difficulty = 1 - accuracy
        result[instance_id] = difficulty
    return result


def quantize_difficulties(instance_difficulties: dict[str, float], num_bins: int = 6):
    bin_edges = np.linspace(0, 1, num_bins)
    bins = np.digitize(np.array(list(instance_difficulties.values())), bin_edges)
    bins = np.array([bin_edges[bin - 1] for bin in bins])
    return bins.tolist()


def convert_difficulties_to_quantiles(
    difficulties: dict[str, float], num_bins: int = 100
):
    difficulties = np.array(list(difficulties.values()))
    difficulties += +np.random.normal(scale=0.001, size=difficulties.shape)
    bin_edges = sorted(set(np.quantile(difficulties, np.linspace(0, 1, num_bins))))
    return (np.digitize(difficulties, bin_edges) * 100 / num_bins).tolist()
