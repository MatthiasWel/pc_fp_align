from typing import List, Callable

import numpy as np
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from chembl_structure_pipeline import standardize_mol
from sklearn.preprocessing import StandardScaler

from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer
from transformers import BertModel, BertTokenizer

def neural_network(data: List[str], network: Callable):
    batch_size = 500
    res = []

    for batch_id in range(0, len(data), batch_size):
        batch = data[batch_id: batch_id + batch_size]
        res.append(
            torch.tensor(network(batch))
        )

    return torch.cat(res)


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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer = PretrainedHFTransformer(
        kind=model_name, dtype=float, preload=True, concat_layers=layer, device=device
    )
    return neural_network(data, transformer)


def ChemBERT(data: List[str], model_name: str, layer: int):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transformer = PretrainedHFTransformer(
        kind=model_name, dtype=float, preload=True, concat_layers=layer, device=device
    )
    return neural_network(data, transformer)


def gins(data: List[str], model_name: str, layer: int):
    transformer = PretrainedDGLTransformer(kind=model_name, dtype=float, preload=True)
    transformer.featurizer.num_layers = min(layer, transformer.featurizer.num_layers)
    transformer.featurizer.gnn_layers[layer - 1].activation = None
    return torch.tensor(transformer(data))

def check_pc_validity(t: torch.Tensor, max_val: float = 1e6):
    return bool(
        not torch.isnan(t).any() and
        torch.isfinite(t).all() and
        (torch.abs(t) < max_val).all()
    )

def physico_chemical(data: List[str], model_name: str, layer: int):
    features = _physico_chemical_raw(data)
    scaler = StandardScaler()
    return torch.tensor(scaler.fit_transform(features))

def _physico_chemical_raw(data):
    desc_list = [
        desc_name
        for desc_name, _ in Descriptors._descList
        if desc_name not in ("Ipc", "BertzCT")
    ]
    descgen = MolecularDescriptorCalculator(desc_list)
    features = np.stack(
        [descgen.CalcDescriptors(standardize_mol(Chem.MolFromSmiles(smi))) for smi in data]
    )
    
    return torch.tensor(features, dtype=torch.float64)

def fingerprints(data: List[str], model_name: str, layer: int):
    RDLogger.DisableLog("rdApp.*")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = np.stack(
        [np.array(mfpgen.GetFingerprint(standardize_mol(Chem.MolFromSmiles(smi)))) for smi in data]
    )
    return torch.tensor(features, dtype=torch.float64)
