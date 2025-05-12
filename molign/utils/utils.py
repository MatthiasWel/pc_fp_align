from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path("/data/shared/exchange/mwelsch/align_mols_reps/results")
DATA_PATH = Path("/data/shared/exchange/mwelsch/align_mols_reps/data/")
PROCESSED_DATA = Path("/data/shared/exchange/mwelsch/align_mols_reps/processed")
PLOT_PATH = Path("/data/shared/exchange/mwelsch/align_mols_reps/plots")


def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

