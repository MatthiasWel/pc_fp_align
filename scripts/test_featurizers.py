import pandas as pd
import os
import torch
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split

from molign.align import linear_cka
from molign.datasets import clean, tdc_tasks
from molign.models import BinaryClassificationMLP, SimpleDataset, train
from molign.models.metrics import metrics_on_device
from molign.models import models_config

BASE_PATH = Path("/data/shared/exchange/mwelsch/fp_pc_align")
DATASET_PATH = BASE_PATH / "datasets"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
TENSORBOARD_PATH = BASE_PATH / "tensorboard"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def main():
    datasets_of_interest = [
        'CYP2C19_Veith_adme', 
        'CYP2D6_Veith_adme', 
        'CYP3A4_Veith_adme',
        'CYP1A2_Veith_adme', 
        'CYP2C9_Veith_adme',
        'TOX21_AhR_LUC_Agonist_ToxCast',
        'TOX21_TR_LUC_GH3_Antagonist_ToxCast', 
        'hERG_inhib_herg_central'
    ]
    timestamp = get_timestamp()
    data = tdc_tasks(DATASET_PATH, 60, include=("adme", "herg_central", 'tox_cast')) # 
    data = clean(data)
    data = data[data.task_id.isin(datasets_of_interest)]
    
    data.to_csv(DATA_PATH / f"data_{timestamp}.csv", index=False)

    results = {}
    alignments = {}
    for task_id in data.task_id.unique():
        print("\n\n\n\n", task_id, "\n\n\n\n")
        current_data = data[data.task_id == task_id]
        train_data, test_data = train_test_split(current_data, random_state=42)
        models = {}
        for emb_func, model_name, layer, kwargs in models_config:
            model_id = hash((model_name, layer))

            X_train = emb_func(train_data.smiles.to_list(), model_name, layer, **kwargs)
            train_dataset = SimpleDataset(
                X_train,
                train_data.label.to_list()
            )
            X_test = emb_func(test_data.smiles.to_list(), model_name, layer, **kwargs)
            test_dataset = SimpleDataset(
                X_test,
                test_data.label.to_list()
            )

            input_dim = X_train.shape[1]

            model = BinaryClassificationMLP(
                dict(
                    decoder_input=input_dim,
                    batch_norm=True,
                    hidden_dimensions=128,
                    num_linear_layers=3,
                )
            )

            pred, true, res = train(
                "test",
                timestamp,
                model,
                train=train_dataset,
                val=test_dataset,
                tensorboard_path=TENSORBOARD_PATH,
                epochs=2,
            )
            results[(task_id, model_id, model_name, layer)] = res
            models[model_id] = model
        print(f'\n\ntraining models done for {task_id}\n\n')
        for emb_func1, model_name1, layer1, kwargs1 in models_config:
            embs1 = emb_func1(data.smiles.to_list(), model_name1, layer1, **kwargs1)
            model_id1 = hash((model_name1, layer1))
            model1 = models[model_id1]
            embs1 = model1.embedding(embs1.float())
            for emb_func2, model_name2, layer2, kwargs2 in models_config:
                
                embs2 = emb_func2(data.smiles.to_list(), model_name2, layer2, **kwargs2)
                
                model_id2 = hash((model_name2, layer2))
                if model_id1 == model_id1:
                    continue
                model2 = models[model_id2]
                embs2 = model2.embedding(embs2.float())
                alignments[(task_id, model_id1, model_id2)] = {'full_alignment': linear_cka(embs1, embs2)}

    performance = pd.DataFrame.from_dict(results, orient="index").reset_index()
    performance.columns = ["dataset", "model_id", "model_name", "model_layer", "accuracy", "mcc", "ece"]
    performance.accuracy = performance.accuracy.astype(float)
    performance.mcc = performance.mcc.astype(float)
    performance.ece = performance.ece.astype(float)

    alignments = pd.DataFrame.from_dict(alignments, orient="index").reset_index()
    alignments.columns = ['dataset', 'model_id1', 'model_id2', 'alignment']
    df = performance.merge(alignments, left_on=['dataset', 'model_id'], right_on=['dataset', 'model_id1'])
    print(df)
    df.to_csv(BASE_PATH / f"results/performance_features_{timestamp}.csv", index=False)




    

if __name__ == "__main__":
    main()