from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Dict, Any


def generate_confusion_matrix_plot_from_classification_results(
    plot_axis: Axes,
    gt_classes: List[int],
    pred_classes: List[int],
    class_names: List[str],
) -> None:
    """Plots confusion matrix on given axis based on provided prediction results"""
    conf_mat: np.ndarray = confusion_matrix(
        y_true=gt_classes,
        y_pred=pred_classes,
        labels=class_names,
    )
    conf_mat_dataframe: pd.DataFrame = pd.DataFrame(
        data=conf_mat, index=class_names, columns=class_names
    )
    plot_axis = sns.heatmap(data=conf_mat_dataframe, ax=plot_axis, annot=True, fmt="g")


def generate_figure_with_conf_matrix(
    class_names: List[str], y_true: List[str], y_pred: List[str]
) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax = generate_confusion_matrix_plot_from_classification_results(
        plot_axis=ax, gt_classes=y_true, pred_classes=y_pred, class_names=class_names
    )
    return fig


class MMActionScalarsDictInteraction:
    @staticmethod
    def mark_compressed_representation_by_prefix(
        compressed_dict: Dict[str, Any], prefix: str
    ) -> Dict[str, float]:
        repr_marked_by_prefixes: Dict[str, Any] = {}
        for key, value in compressed_dict.items():
            prefixed_key = f"{prefix}{key}"
            repr_marked_by_prefixes[prefixed_key] = value
        return repr_marked_by_prefixes

    @staticmethod
    def extract_compressed_representation_by_prefix(
        scalars_dict: Dict[str, Any], prefix: str
    ) -> Dict[str, float]:
        compressed_dict: Dict[str, float] = {}
        for k, v in scalars_dict.items():
            if not k.startswith(prefix):
                continue
            k_without_prefix = k.removeprefix(prefix)
            compressed_dict[k_without_prefix] = v
        return compressed_dict

    @staticmethod
    def purge_metrics_from_prefixed_keys(
        scalar_dict: Dict[str, float], prefixes: List[str]
    ):
        keys = list(scalar_dict.keys())
        for key in keys:
            if not any([key.startswith(p) for p in prefixes]):
                continue
            scalar_dict.pop(key)
