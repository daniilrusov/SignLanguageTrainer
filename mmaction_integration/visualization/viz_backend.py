from mmengine.visualization.vis_backend import ClearMLVisBackend
from typing import Optional
from mmengine.hooks.logger_hook import SUFFIX_TYPE
from SignLanguageTrainer.mmaction_integration.visualization.log_utils import (
    log_classififcation_report_to_clearml,
    SCKLEARN_CLASSIFICATION_REPORT_TYPE,
)
from mmengine.visualization.vis_backend import (
    VISBACKENDS,
    ClearMLVisBackend,
    force_init_env,
)
from typing import List, Dict


BASIC_CLASS_REPORT_KEYS = ["accuracy", "macro avg", "weighted avg"]


@VISBACKENDS.register_module()
class ExtendedClearMLVisBackend(ClearMLVisBackend):
    def __init__(
        self,
        save_dir: str | None = None,
        init_kwargs: dict | None = None,
        artifact_suffix: SUFFIX_TYPE = ...,
    ):
        super().__init__(save_dir, init_kwargs, artifact_suffix)

    @force_init_env
    def log_classification_report(
        self,
        classification_report: SCKLEARN_CLASSIFICATION_REPORT_TYPE,
        step: int = 0,
    ):
        class_names = set(classification_report.keys()) - set(BASIC_CLASS_REPORT_KEYS)
        class_names = sorted(list(class_names))
        log_classififcation_report_to_clearml(
            clearml_logger=self._logger,
            classification_report=classification_report,
            class_names=class_names,
            iteration=step,
        )
