from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


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
