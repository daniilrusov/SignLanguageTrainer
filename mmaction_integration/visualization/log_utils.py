import os
from typing import List, Dict
from clearml import Logger

SCKLEARN_CLASSIFICATION_REPORT_TYPE = Dict[str, Dict[str, float]]


def log_classififcation_report_to_clearml(
    clearml_logger: Logger,
    classification_report: SCKLEARN_CLASSIFICATION_REPORT_TYPE,
    class_names: List[str],
    iteration: int,
) -> None:
    report_metrics_names: List[str] = ["f1-score", "precision", "recall", "support"]
    for metric_name in report_metrics_names:
        title = "Per class " + metric_name
        for class_name in class_names:
            logged_value: float = classification_report[class_name][metric_name]
            clearml_logger.report_scalar(
                title=title,
                series=class_name,
                iteration=iteration,
                value=logged_value,
            )

    # log aggregated metrics
    aggregated_metrics_keys: List[str] = list(
        set(classification_report.keys()) - set(class_names) - set(["accuracy"])
    )
    for aggregated_metrics_key in aggregated_metrics_keys:
        aggregated_metrics = classification_report[aggregated_metrics_key]
        for series_name, series_value in aggregated_metrics.items():
            clearml_logger.report_scalar(
                title=aggregated_metrics_key,
                series=series_name,
                value=series_value,
                iteration=iteration,
            )

    # log accuracy
    clearml_logger.report_scalar(
        title="accuracy",
        series="accuracy",
        iteration=iteration,
        value=classification_report["accuracy"],
    )
