from typing import Any, Callable

from molign.models.architecture import BinaryClassificationMLP
from molign.models.ml_utils import train
from molign.models.model_wrapper import BERTModel, ChemBERT, chemGPTs, gins
from molign.models.simple_dataset import SimpleDataset

models_config: list[tuple[Callable, str, int, dict[str, Any]]] = [
    (gins, "gin_supervised_contextpred", 5, {}),  # max 5
    # (gins, "gin_supervised_contextpred", 4, {}),
    # (gins, "gin_supervised_contextpred", 3, {}),
    # (gins, "gin_supervised_edgepred", 5, {}),  # max 5
    # (gins, "gin_supervised_edgepred", 4, {}),
    # (gins, "gin_supervised_edgepred", 3, {}),
    # (gins, "gin_supervised_infomax", 5, {}),  # max 5
    # (gins, "gin_supervised_infomax", 4, {}),
    # (gins, "gin_supervised_infomax", 3, {}),
    # (gins, "gin_supervised_masking", 5, {}),  # max 5
    # (gins, "gin_supervised_masking", 4, {}),
    (gins, "gin_supervised_masking", 3, {}),
    (chemGPTs, "ChemGPT-19M", 24, {}),  # max 24
    # (chemGPTs, "ChemGPT-19M", 23, {}),
    # (chemGPTs, "ChemGPT-19M", 22, {}),
    # (chemGPTs, "ChemGPT-4.7M", 24, {}),  # max 24
    # (chemGPTs, "ChemGPT-4.7M", 23, {}),
    (chemGPTs, "ChemGPT-4.7M", 22, {}),
    # (chemGPTs, "GPT2-Zinc480M-87M", 12, {}),  # max 12
    # (chemGPTs, "GPT2-Zinc480M-87M", 11, {}),
    # (chemGPTs, "GPT2-Zinc480M-87M", 10, {}),
]
