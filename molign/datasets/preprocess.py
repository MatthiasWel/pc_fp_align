import pandas as pd
import torch
from chembl_structure_pipeline import standardize_mol
from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
from molign.models.model_wrapper import _physico_chemical_raw


def standardize_mol_safe(mol):
    try:
        mol = standardize_mol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        return None

def clean(data: pd.DataFrame):
    RDLogger.DisableLog("rdApp.*")
    data = data[~data.smiles.isna()]
    data = data[data.smiles.map(Chem.MolFromSmiles).map(bool)]
    data = data[~data.smiles.isna()]
    data.smiles = data.smiles.map(Chem.MolFromSmiles).map(standardize_mol_safe)
    data = data[~data.smiles.isna()]
    remover = SaltRemover()
    data.smiles = (
        data.smiles.map(Chem.MolFromSmiles).map(remover.StripMol).map(Chem.MolToSmiles)
    )
    data = data[~data.smiles.isna()]
    data = data[data.smiles != ""]
    data = data[data.smiles.map(Chem.MolFromSmiles).map(bool)]
    features = _physico_chemical_raw(data.smiles)  
    valid_mask = (
        ~torch.isnan(features).any(dim=1) &    # no NaNs
        torch.isfinite(features).all(dim=1) &  # all finite
        (features.abs() < 1e6).all(dim=1)      # within range
    )
    data = data[valid_mask.numpy()]
    # remove all tiny tasks
    for task_id in data.task_id.unique():
        task = data[data.task_id == task_id]
        if len(task) < 10:
            data = data[data.task_id != task_id]
    data = data[data.label.astype(str).str.replace(".", "").str.isnumeric()]
    data.label = data.label.astype(float)
    data['inchi'] = data.smiles.map(Chem.MolFromSmiles).map(Chem.MolToInchi)
    # remove all instances of mols, where the same mol appears multiple times in a dataset (task)
    duplicates = data.duplicated(subset=['task_id', 'inchi'], keep=False)
    data = data[~duplicates].reset_index(drop=True)
    return data
