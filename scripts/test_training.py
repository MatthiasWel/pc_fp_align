import copy
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchmetrics
from chembl_structure_pipeline import standardize_mol
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from molign.align import linear_cka
from molign.datasets import clean, tdc_tasks
from molign.models import BinaryClassificationMLP, SimpleDataset, train

BASE_PATH = Path("/data/shared/exchange/mwelsch/fp_pc_align")
DATASET_PATH = BASE_PATH / "datasets"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
TENSORBOARD_PATH = BASE_PATH / "tensorboard"

for p in [BASE_PATH, DATASET_PATH, DATA_PATH, RESULTS_PATH, TENSORBOARD_PATH]:
    assert os.path.exists(p), f"{p} does not exist. Please create it."


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def keep_non_nan(arr):
    if any(np.isnan(arr)):
        return None
    return arr


def process_data(data):
    RDLogger.DisableLog("rdApp.*")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    desc_list = [
        desc_name
        for desc_name, _ in Descriptors._descList
        if desc_name not in ("Ipc", "BertzCT")
    ]
    descgen = MolecularDescriptorCalculator(desc_list)
    data["mol"] = data.smiles.apply(Chem.MolFromSmiles).apply(standardize_mol)
    data = data.assign(
        inchi=data.mol.apply(Chem.inchi.MolToInchi),
        FP=data.mol.apply(lambda mol: mfpgen.GetFingerprint(mol)).apply(np.array),
        PC=data.mol.apply(lambda mol: descgen.CalcDescriptors(mol))
        .apply(np.array)
        .apply(lambda x: keep_non_nan(x)),
    )
    data = data[~data.PC.isna()]
    scaler = StandardScaler()
    data["PC"] = list(scaler.fit_transform(np.stack(data["PC"])))
    data["PCFP"] = [np.concatenate([a, b]) for a, b in zip(data["PC"], data["FP"])]
    data = data.drop_duplicates(subset=["inchi"]).reset_index(drop=True)
    return data


def main():
    timestamp = get_timestamp()
    data = tdc_tasks(DATASET_PATH, 10)
    data = clean(data)
    data.to_csv(DATA_PATH / f"data_{timestamp}.csv", index=False)
    results = {}
    aligns = {}

    full_processed = process_data(data.drop_duplicates(subset="smiles"))[
        ["PC", "FP", "PCFP"]
    ]

    for task_id in data.task_id.unique():
        print("\n\n\n\n", task_id, "\n\n\n\n")
        current_data = data[data.task_id == task_id]
        processed_data = process_data(current_data)
        train_data, test_data = train_test_split(processed_data, random_state=42)
        for featurization_strategy, input_dim in [
            ("PC", 206),
            ("FP", 2048),
            ("PCFP", 2048 + 206),
        ]:
            model = BinaryClassificationMLP(
                dict(decoder_input=input_dim, batch_norm=True, hidden_dimension=128)
            )
            X_train = torch.tensor(
                np.stack(train_data[featurization_strategy].to_list()),
                dtype=torch.float,
            )
            X_test = torch.tensor(
                np.stack(test_data[featurization_strategy].to_list()), dtype=torch.float
            )
            assert (
                X_train.isnan().sum() == 0
            ), f"There are nans in {task_id} in the training set"
            assert (
                X_test.isnan().sum() == 0
            ), f"There are nans in {task_id} in the test set"

            pred, true, res = train(
                "test",
                timestamp,
                model,
                train=SimpleDataset(X_train, train_data.label.to_list()),
                val=SimpleDataset(X_test, test_data.label.to_list()),
                tensorboard_path=TENSORBOARD_PATH,
                epochs=50,
            )

            if featurization_strategy == "PC":
                emb_pc = model.embedding(X_test)
                emb_pc_full = model.embedding(
                    torch.tensor(
                        np.stack(full_processed["PC"].to_list()), dtype=torch.float
                    )
                )
                pred_pc = copy.deepcopy(pred)

            if featurization_strategy == "FP":
                emb_fp = model.embedding(X_test)
                emb_fp_full = model.embedding(
                    torch.tensor(
                        np.stack(full_processed["FP"].to_list()), dtype=torch.float
                    )
                )
                pred_fp = copy.deepcopy(pred)

            if featurization_strategy == "PCFP":
                emb_pcfp = model.embedding(X_test)
                emb_pcfp_full = model.embedding(
                    torch.tensor(
                        np.stack(full_processed["PCFP"].to_list()), dtype=torch.float
                    )
                )

                metrics = {
                    "accuracy": torchmetrics.Accuracy("binary").to("cpu"),
                    "mcc": torchmetrics.MatthewsCorrCoef("binary").to("cpu"),
                }

                pred_mean = torch.mean(torch.stack([pred_pc, pred_fp]), dim=0)
                pred_max = torch.max(torch.stack([pred_pc, pred_fp]), dim=0).values

                results[(task_id, "MEAN")] = {
                    metric_name: metric(pred_mean, true)
                    for metric_name, metric in metrics.items()
                }
                results[(task_id, "MAX")] = {
                    metric_name: metric(pred_max, true)
                    for metric_name, metric in metrics.items()
                }

                alignment_pc_fp = linear_cka(emb_pc, emb_fp)
                full_alignment_pc_fp = linear_cka(emb_pc_full, emb_fp_full)

                alignment_pc_pcfp = linear_cka(emb_pc, emb_pcfp)
                full_alignment_pc_pcfp = linear_cka(emb_pc_full, emb_pcfp_full)

                alignment_fp_pcfp = linear_cka(emb_fp, emb_pcfp)
                full_alignment_fp_pcfp = linear_cka(emb_fp_full, emb_pcfp_full)

                aligns[task_id] = {
                    "alignment_pc_fp": float(alignment_pc_fp),
                    "full_alignment_pc_fp": float(full_alignment_pc_fp),
                    "alignment_pc_pcfp": float(alignment_pc_pcfp),
                    "full_alignment_pc_pcfp": float(full_alignment_pc_pcfp),
                    "alignment_fp_pcfp": float(alignment_fp_pcfp),
                    "full_alignment_fp_pcfp": float(full_alignment_fp_pcfp),
                    "train_set_size": len(X_train),
                    "test_set_size": len(X_test),
                    "train_set_balance": np.mean(train_data.label.to_list()),
                    "test_set_balance": np.mean(test_data.label.to_list()),
                }

            results[(task_id, featurization_strategy)] = res

    performance = pd.DataFrame.from_dict(results, orient="index").reset_index()
    performance.columns = ["dataset", "feature_type", "accuracy", "mcc"]
    performance.accuracy = performance.accuracy.astype(float)
    performance.mcc = performance.mcc.astype(float)

    alignments = pd.DataFrame.from_dict(aligns, orient="index").reset_index()
    alignments.columns = [
        "dataset",
        "alignment_pc_fp",
        "full_alignment_pc_fp",
        "alignment_pc_pcfp",
        "full_alignment_pc_pcfp",
        "alignment_fp_pcfp",
        "full_alignment_fp_pcfp",
        "train_set_size",
        "test_set_size",
        "train_set_balance",
        "test_set_balance",
    ]

    df = performance.merge(alignments)
    print(df)
    df.to_csv(BASE_PATH / f"results/performance_{timestamp}.csv", index=False)


if __name__ == "__main__":
    main()
