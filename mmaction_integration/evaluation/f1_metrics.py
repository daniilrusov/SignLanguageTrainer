import json
from typing import Dict, List
from pprint import pprint
from mmaction.evaluation.metrics import AccMetric
from mmaction.registry import METRICS
from mmaction.evaluation import get_weighted_score
from sklearn.metrics import classification_report
import numpy as np


@METRICS.register_module()
class F1Metric(AccMetric):
    def __init__(
        self,
        label2id_mapping_path: str,
        collect_device: str = "cpu",
        prefix: str | None = None,
        collect_dir: str | None = None,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        with open(label2id_mapping_path, "r", encoding="utf-8") as f:
            self.label2id: Dict[str, int] = json.load(f)
        self.id2labels: Dict[int, str] = {
            label_id: label for label, label_id in self.label2id.items()
        }
        ids = sorted(list(self.id2labels.keys()))
        self.class_names: List[str] = [self.id2labels[id] for id in ids]

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x["label"] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]["pred"], dict):
            for item_name in results[0]["pred"].keys():
                predictions_scores_distribution = [
                    x["pred"][item_name] for x in results
                ]
                eval_result = self.calculate(predictions_scores_distribution, labels)
                eval_results.update(
                    {f"{item_name}_{k}": v for k, v in eval_result.items()}
                )

            if (
                len(results[0]["pred"]) == 2
                and "rgb" in results[0]["pred"]
                and "pose" in results[0]["pred"]
            ):
                rgb = [x["pred"]["rgb"] for x in results]
                pose = [x["pred"]["pose"] for x in results]

                predictions_scores_distribution = {
                    "1:1": get_weighted_score([rgb, pose], [1, 1]),
                    "2:1": get_weighted_score([rgb, pose], [2, 1]),
                    "1:2": get_weighted_score([rgb, pose], [1, 2]),
                }
                for k in predictions_scores_distribution:
                    eval_result = self.calculate(
                        predictions_scores_distribution[k], labels
                    )
                    eval_results.update(
                        {f"RGBPose_{k}_{key}": v for key, v in eval_result.items()}
                    )
            return eval_results

        # Simple Acc Calculation
        else:
            predictions_scores_distribution = np.array([x["pred"] for x in results])
            predcited_labels = np.argmax(predictions_scores_distribution, axis=1)
            report = classification_report(
                y_true=labels,
                y_pred=predcited_labels,
                target_names=self.class_names,
                output_dict=True,
            )
            output = {
                "classification_report_data": {
                    "classification_report": report,
                    "class_names": self.class_names,
                },
                "weighted_avg_f1_score": report["weighted avg"]["f1-score"],
            }
            pprint("\n\n\noutput")
            pprint(output)
            return output
