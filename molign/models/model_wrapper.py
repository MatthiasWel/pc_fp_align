from typing import List

import numpy as np
import torch
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from transformers import BertModel, BertTokenizer

def BERTModel(data: List[str], model_name: str, layer: int):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    encoded_smiles = tokenizer(data, return_tensors="pt", padding=True)
    output = model(**encoded_smiles, output_hidden_states=True)

    hidden_states = (
        output.hidden_states
    )  # Tuple with tensors (batch_size, seq_length, hidden_size)
    # The first token of each sequence is the [CLS] token, which is learned for the entire sequence and can
    # be used for classification tasks (see https://arxiv.org/pdf/1810.04805)
    cls_representation = hidden_states[layer][:, 0, :]
    return cls_representation


def chemGPTs(data: List[str], model_name: str, layer: int):
    transformer = PretrainedHFTransformer(
        kind=model_name, dtype=float, preload=True, concat_layers=layer
    )
    return torch.tensor(transformer(data))


def ChemBERT(data: List[str], model_name: str, layer: int):
    transformer = PretrainedHFTransformer(
        kind=model_name, dtype=float, preload=True, concat_layers=layer
    )
    return torch.tensor(transformer(data))


def gins(data: List[str], model_name: str, layer: int):
    transformer = PretrainedDGLTransformer(kind=model_name, dtype=float, preload=True)
    transformer.featurizer.num_layers = min(layer, transformer.featurizer.num_layers)
    transformer.featurizer.gnn_layers[layer - 1].activation = None
    return torch.tensor(transformer(data))