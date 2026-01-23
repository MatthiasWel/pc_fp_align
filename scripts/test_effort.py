import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from molign.models.model_wrapper import BERTModel, ChemBERT, chemGPTs, gins, physico_chemical, fingerprints
from molign.models import BinaryClassificationMLP, SimpleDataset, train
from pathlib import Path
from datetime import datetime
import os
from tqdm.auto import tqdm
import pandas as pd
from tqdm.auto import tqdm
from FPSim2 import FPSim2Engine
from FPSim2.io import create_db_file

BASE_PATH = Path("/data/shared/exchange/mwelsch/fp_pc_align")
DATASET_PATH = BASE_PATH / "datasets"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
TENSORBOARD_PATH = BASE_PATH / "tensorboard"
LOGS_PATH = BASE_PATH / "logs"
INTERMEDIATE_RES_PATH = BASE_PATH / 'intermediate_results'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mol_iterator(smiles):
    for i, smi in enumerate(smiles):
        yield [smi, i]

def calc_separability(dataset):
    dataset = dataset.reset_index(drop=True)
    results = []
    for name, group in tqdm(dataset.groupby("task_id")):
        idx_act = group.index[group.label == 1].to_numpy()
        idx_inact = group.index[group.label == 0].to_numpy()
        if len(idx_act) == 0 or len(idx_inact) == 0:
            continue

        file = f"/tmp/{name}_fpsim2.h5"

        create_db_file(
            mols_source=mol_iterator(dataset.smiles.iloc[idx_inact]),
            filename=file,
            mol_format='smiles',
            fp_type='Morgan',
            fp_params={'radius': 2, 'fpSize': 2048}
        )
        fpe = FPSim2Engine(file)
        for active_id in idx_act:
            query = dataset.smiles.iloc[active_id]
            results_list = fpe.similarity(query, threshold=0.0)

            max_sim = max(r[1] for r in results_list)
            min_dist = 1 - max_sim

            results.append(
                (name, query, min_dist)
            )
        results = pd.DataFrame(results, columns=['name', 'query', 'min_dist'])
        return np.mean(results.min_dist)
    
best_model_per_task = {
    "CYP2D6_Veith_adme": (gins, "gin_supervised_infomax", 5, {}),
    "CYP3A4_Veith_adme": (gins, "gin_supervised_infomax", 5, {}),
    "CYP1A2_Veith_adme": (gins, "gin_supervised_masking", 5, {}),
    "serine_threonine_kinase_33_butkiewicz_hts": (gins, "gin_supervised_contextpred", 5, {}),
    "APR_HepG2_StressKinase_1h_up_ToxCast": (gins, "gin_supervised_infomax", 5, {}),
    "ATG_Oct_MLP_CIS_dn_ToxCast": (gins, "gin_supervised_infomax", 5, {}),
    "CLD_SULT2A_24hr_ToxCast": (chemGPTs, "GPT2-Zinc480M-87M", 10, {}),
    "NVS_ENZ_hPTPN11_ToxCast": (gins, "gin_supervised_edgepred", 5, {}),
    "NVS_GPCR_r5HT1_NonSelective_ToxCast": (gins, "gin_supervised_edgepred", 3, {}),
    "hERG_inhib_herg_central": (gins, "gin_supervised_edgepred", 3, {})
}

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    dataset = pd.read_csv('/data/shared/exchange/mwelsch/fp_pc_align/data/data_2025-11-17_11:39:53.csv')
    n_iters = 30
    dataset = dataset[dataset.task_id.isin(list(best_model_per_task.keys()))]
    results = {}
    for task in dataset.task_id.unique():
        current_data = dataset[dataset.task_id == task]

        train_data, test_data = train_test_split(current_data, random_state=42)
        
        emb_func, model_name, layer, kwargs = best_model_per_task[task]
        X_train = emb_func(train_data.smiles.to_list(), model_name, layer, **kwargs)

        input_dim = X_train.shape[1]

        X_test = emb_func(test_data.smiles.to_list(), model_name, layer, **kwargs)
        test_dataset = SimpleDataset(
            X_test,
            test_data.label.to_list()
        )

        
        for effort_level in tqdm(np.arange(0.75, 1.0, 0.05)):
            n_datapoints = int(train_data.shape[0] * effort_level)
            for iteration in range(n_iters):
                permutation = torch.randperm(X_train.size(0))[:n_datapoints]

                current_X_train = X_train[permutation]

                current_labels = train_data.label.iloc[permutation].to_list()
                full_data_iterate = pd.concat([train_data.iloc[permutation], test_data])
                separability = calc_separability(full_data_iterate)

                train_dataset = SimpleDataset(
                    current_X_train,
                    current_labels
                )
                model = BinaryClassificationMLP(
                    dict(
                        decoder_input=input_dim,
                        batch_norm=True,
                        hidden_dimensions=128,
                        num_linear_layers=3,
                    )
                )
                
                pred, true, res = train(
                    model_name + "_" + str(effort_level) + "_" + str(iteration),
                    timestamp,
                    model,
                    train=train_dataset,
                    val=test_dataset,
                    tensorboard_path=TENSORBOARD_PATH,
                    logs_path=LOGS_PATH,
                    epochs=30,
                )
                res['separability'] = separability
                print(task, effort_level, iteration, res)
                results[(task, effort_level, iteration)] = res
                if effort_level == 1:
                    break
    print(results)
    performance = pd.DataFrame.from_dict(results, orient="index").reset_index()
    performance.columns = ["task", "effort_level", "iteration", "accuracy", "mcc", "ece", "separability"]
    performance.accuracy = performance.accuracy.astype(float)
    performance.mcc = performance.mcc.astype(float)
    performance.ece = performance.ece.astype(float)
    performance.to_csv(f'effort_performance_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main()