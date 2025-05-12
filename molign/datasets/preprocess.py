import pandas as pd
from chembl_structure_pipeline import standardize_mol
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.SaltRemover import SaltRemover


def standardize_mol_safe(mol):
    try:
        mol = standardize_mol(mol)
        return Chem.MolToSmiles(mol)
    except Exception as e:
        print(e)
        return None


def clean(data: pd.DataFrame):
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
    data = data[data.smiles.map(Chem.MolFromSmiles).map(Descriptors.ExactMolWt) < 600]
    for task_id in data.task_id.unique():
        task = data[data.task_id == task_id]
        if len(task) < 10:
            data = data[data.task_id != task_id]
    data = data[data.label.astype(str).str.replace(".", "").str.isnumeric()]
    data.label = data.label.astype(float)
    data = data.drop_duplicates().reset_index(drop=True)
    return data
