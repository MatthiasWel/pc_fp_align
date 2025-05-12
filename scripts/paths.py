import os
from pathlib import Path

BASE_PATH = Path("/data/shared/exchange/mwelsch/fp_pc_align")
DATASET_PATH = BASE_PATH / "datasets"
DATA_PATH = BASE_PATH / "data"
RESULTS_PATH = BASE_PATH / "results"
TENSORBOARD_PATH = BASE_PATH / "tensorboard"

for p in [BASE_PATH, DATASET_PATH, DATA_PATH, RESULTS_PATH, TENSORBOARD_PATH]:
    assert os.path.exists(p), f"{p} does not exist. Please create it."
