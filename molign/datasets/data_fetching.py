import pandas as pd
import tdc.single_pred
from tdc.utils import retrieve_label_name_list

from molign.utils.utils import DATA_PATH


def tdc_adme(sample_size):
    adme = [
        "PAMPA_NCATS",
        "HIA_Hou",
        "Pgp_Broccatelli",
        "Bioavailability_Ma",
        "BBB_Martins",
        "CYP2C19_Veith",
        "CYP2D6_Veith",
        "CYP3A4_Veith",
        "CYP1A2_Veith",
        "CYP2C9_Veith",
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels",
    ]
    all_adme = pd.DataFrame()
    for dataset in adme:
        data = tdc.single_pred.ADME(name=dataset, path=DATA_PATH / "tdc_cyp").balanced()
        data["task_id"] = dataset + "_adme"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_adme = pd.concat([all_adme, data])
    all_adme = all_adme.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_adme


def tdc_hts(sample_size):
    all_hts = pd.DataFrame()
    for dataset in tdc.metadata.hts_dataset_names:
        data = tdc.single_pred.HTS(name=dataset, path=DATA_PATH / "tdc_cyp").balanced()
        data["task_id"] = dataset + "_hts"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_hts = pd.concat([all_hts, data])
    all_hts = all_hts.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_hts


def tdc_tox(sample_size):
    all_tox = pd.DataFrame()
    for dataset in [
        "DILI",
        "hERG",
        "ClinTox",
        "hERG_Karim",
        "Skin Reaction",
        "AMES",
        "Skin Reaction",
        "Carcinogens_Lagunin",
    ]:
        data = tdc.single_pred.Tox(name=dataset, path=DATA_PATH / "tdc_cyp").balanced()
        data["task_id"] = dataset + "_tox"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_tox = pd.concat([all_tox, data])
    all_tox = all_tox.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_tox


def tdc_tox21(sample_size):
    all_tox = pd.DataFrame()
    label_list = retrieve_label_name_list("Tox21")
    for dataset in label_list:
        data = tdc.single_pred.Tox(
            name="Tox21", path=DATA_PATH / "tdc_cyp", label_name=dataset
        ).balanced()
        data["task_id"] = dataset + "_Tox21"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_tox = pd.concat([all_tox, data])
    all_tox = all_tox.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_tox


def tdc_toxcast(sample_size):
    all_tox = pd.DataFrame()
    label_list = retrieve_label_name_list("ToxCast")
    for dataset in label_list:
        data = tdc.single_pred.Tox(
            name="ToxCast", path=DATA_PATH / "tdc_cyp", label_name=dataset
        ).balanced()
        data["task_id"] = dataset + "_ToxCast"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_tox = pd.concat([all_tox, data])
    all_tox = all_tox.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_tox


def tdc_herg_central(sample_size):
    all_tox = pd.DataFrame()
    for dataset in ["hERG_inhib"]:
        data = tdc.single_pred.Tox(
            name="herg_central", path=DATA_PATH / "tdc_cyp", label_name=dataset
        ).balanced()
        data["task_id"] = dataset + "_herg_central"
        data = data[["task_id", "Drug", "Y"]]
        n_samples_task = min(sample_size, len(data))
        data = data.sample(n_samples_task)
        all_tox = pd.concat([all_tox, data])
    all_tox = all_tox.rename({"Drug": "smiles", "Y": "label"}, axis=1).reset_index(
        drop=True
    )
    return all_tox


def tdc_tasks(
    sample_size, include=("adme", "hts", "tox", "tox21", "tox_cast", "herg_central")
):
    dataset_collections = []

    if "adme" in include:
        adme = tdc_adme(sample_size)
        dataset_collections.append(adme)
    if "hts" in include:
        hts = tdc_hts(sample_size)
        dataset_collections.append(hts)
    if "tox" in include:
        tox = tdc_tox(sample_size)
        dataset_collections.append(tox)
    if "tox21" in include:
        tox21 = tdc_tox21(sample_size)
        dataset_collections.append(tox21)
    if "tox_cast" in include:
        tox_cast = tdc_toxcast(sample_size)
        dataset_collections.append(tox_cast)
    if "herg_central" in include:
        herg_central = tdc_herg_central(sample_size)
        dataset_collections.append(herg_central)

    return pd.concat(dataset_collections).reset_index(drop=True)
