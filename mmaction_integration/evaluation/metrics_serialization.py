from abc import abstractmethod, ABC
from typing import Any, Optional, Sequence, Dict, List, Union, Mapping, Generic, TypeVar

T = TypeVar("T")
SCKLEARN_CLASSIFICATION_REPORT_TYPE = Dict[str, Dict[str, float]]
SERIALIZATION_SEPARATOR = "@"
SKLEARN_REPORT_PREFIX = "SKLEARN_REPORT_PREFIX"


def _flatten_dict(
    flattened_dict: Dict[str, Any],
    prefix: str,
    traversed_collection: Union[Dict, Any],
    separator: str = "@",
) -> None:
    # recursion base:
    if not isinstance(traversed_collection, Mapping):
        flattened_dict[prefix] = traversed_collection
        return

    extended_prefix: str
    for key, value in traversed_collection.items():
        if prefix:
            extended_prefix = f"{prefix}{separator}{key}"
        else:
            extended_prefix = key
        _flatten_dict(flattened_dict, extended_prefix, value, separator)


def compress_dictionary(
    nested_dict: Dict[str, Union[Dict[str, Any], Any]], separator: str = "@"
) -> Dict:
    for k in nested_dict.keys():
        if separator in k:
            raise ValueError(
                f"separator {separator} in {k}, will result in invalid compressed repr, choose different separator value"
            )

    output = {}
    _flatten_dict(
        flattened_dict=output,
        prefix="",
        traversed_collection=nested_dict,
        separator="@",
    )
    return output


def decompress_dictionary(
    compressed_dict: Dict[str, Any], separator: str = "@"
) -> Dict[str, Union[Dict[str, Any], Any]]:
    decompressed_dict: Dict = {}
    for key, value in compressed_dict.items():
        nested_keys_sequence = key.split(separator)
        # setup keys sequence
        dict_layer: Dict = decompressed_dict
        for nested_key in nested_keys_sequence[:-1]:
            if nested_key not in decompressed_dict:
                dict_layer[nested_key] = {}
            dict_layer = dict_layer[nested_key]
        dict_layer[nested_keys_sequence[-1]] = value
    return decompressed_dict


class MMActionFormatMetricsSerialization(ABC, Generic[T]):
    @abstractmethod
    def serialize(self, value: T) -> Dict[str, float]:
        pass

    @abstractmethod
    def deserialize(self, serialized_repr: Dict[str, float]) -> T:
        pass


class SklearnClassReportSerializer(MMActionFormatMetricsSerialization):
    def __init__(self, separator: str) -> None:
        super().__init__()
        self.separator = separator

    def serialize(self, value: SCKLEARN_CLASSIFICATION_REPORT_TYPE) -> Dict[str, float]:
        return compress_dictionary(value, separator=self.separator)

    def deserialize(
        self, serialized_repr: Dict[str, float]
    ) -> SCKLEARN_CLASSIFICATION_REPORT_TYPE:
        return decompress_dictionary(serialized_repr, separator=self.separator)


class ConfMatrixSerializer(MMActionFormatMetricsSerialization):
    def serialize(self, value: Any) -> Dict[str, float]:
        return super().serialize(value)

    def deserialize(self, serialized_repr: Dict[str, float]) -> Any:
        return super().deserialize(serialized_repr)
