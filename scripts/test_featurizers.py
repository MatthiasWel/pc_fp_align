import pandas as pd
import os
import pickle
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split

from molign.align import linear_cka
from molign.datasets import clean, tdc_tasks
from molign.models import BinaryClassificationMLP, SimpleDataset, train
from molign.models import physico_chemical, fingerprints, gins, chemGPTs

BASE_PATH = Path("/data/shared/exchange/mwelsch/fp_pc_align")
DATASET_PATH = BASE_PATH / "datasets"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
TENSORBOARD_PATH = BASE_PATH / "tensorboard"
LOGS_PATH = BASE_PATH / "logs"
INTERMEDIATE_RES_PATH = BASE_PATH / 'intermediate_results'

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def continue_run_from_timestamp(timestamp):
    data = pd.read_csv(DATA_PATH / f"data_{timestamp}.csv")
    unfiltered_data = data.copy()
    return timestamp, data, unfiltered_data

models_config = [
    (physico_chemical, "pc", -1, {}), 
    (fingerprints, "fp", -1, {}), 
    (gins, "gin_supervised_contextpred", 5, {}),  
    (chemGPTs, "ChemGPT-19M", 24, {})
]
def main():
    # timestamp, data, unfiltered_data = new_run()
    # '2025-09-29_14:20:52' is the short run with
    #   'pc', 'fp', 'gin_supervised_contextpred', 'ChemGPT-19M'

    # '2025-09-29_14:31:55' is the long run with 
    #   'pc', 'fp', 'gin_supervised_contextpred',
    #   'gin_supervised_edgepred', 'gin_supervised_infomax',
    #   'gin_supervised_masking', 'ChemGPT-19M', 'ChemGPT-4.7M',
    #   'GPT2-Zinc480M-87M'
    

    # 2025-11-06_12:58:39 is the replica
    # timestamp, data, unfiltered_data = continue_run_from_timestamp('2025-09-29_14:31:55') 
    # timestamp, data, unfiltered_data = new_run()
    timestamp, data, unfiltered_data = continue_run_from_timestamp('2025-11-06_12:58:39')
    results = {}
    alignments = {}
    for task_id in (pbar_task := tqdm(data.task_id.unique(), desc="Task_id")):
        pbar_task.set_postfix_str(task_id)
        results_int_filename = INTERMEDIATE_RES_PATH / f'results_{task_id}_{timestamp}.pkl'
        alignment_intermediate_filename = INTERMEDIATE_RES_PATH / f'alignment_{task_id}_{timestamp}.pkl'
        if os.path.exists(results_int_filename) and os.path.exists(alignment_intermediate_filename):
            with open(results_int_filename, mode='rb') as f:
                res = pickle.load(f)
            results.update(res)

            with open(alignment_intermediate_filename, mode='rb') as f:
                align = pickle.load(f)
            alignments.update(align)

            continue
        
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
                model_name,
                timestamp,
                model,
                train=train_dataset,
                val=test_dataset,
                tensorboard_path=TENSORBOARD_PATH,
                logs_path=LOGS_PATH,
                epochs=30,
            )
            results[(task_id, model_id, model_name, layer)] = res
            models[model_id] = model

        with open(results_int_filename, 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

        smiles = list(unfiltered_data.smiles.unique())
        for emb_func1, model_name1, layer1, kwargs1 in (pbar_align := tqdm(models_config, desc="Alignment", leave=False)):
            pbar_align.set_postfix_str(f"{model_name1}, layer={layer1}")
            embs1 = emb_func1(smiles, model_name1, layer1, **kwargs1)
            model_id1 = hash((model_name1, layer1))
            model1 = models[model_id1]
            embs1 = model1.embedding(embs1.float())
            for emb_func2, model_name2, layer2, kwargs2 in models_config:
                
                embs2 = emb_func2(smiles, model_name2, layer2, **kwargs2)
                
                model_id2 = hash((model_name2, layer2))
                if model_id1 == model_id2:
                    continue
                model2 = models[model_id2]
                embs2 = model2.embedding(embs2.float())
                alignments[(task_id, model_id1, model_id2)] = {'full_alignment': linear_cka(embs1, embs2)}
        with open(alignment_intermediate_filename, 'wb') as file:
            pickle.dump(alignments, file, protocol=pickle.HIGHEST_PROTOCOL)

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

def new_run():
    timestamp = get_timestamp()
    data = tdc_tasks(DATASET_PATH, 5000, log_path=LOGS_PATH, time=timestamp) # , include=("herg_central", "adme", 'tox_cast')) # , "adme", 'tox_cast')) 
    data = clean(data)
    unfiltered_data = data.copy()
    data.to_csv(DATA_PATH / f"data_{timestamp}.csv", index=False)
    return timestamp, data, unfiltered_data




    

if __name__ == "__main__":
    main()