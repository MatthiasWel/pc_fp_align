import torchmetrics
import torchmetrics.classification

def metrics_on_device(device):
    return {
            "accuracy": torchmetrics.Accuracy("binary").to(device),
            "mcc": torchmetrics.MatthewsCorrCoef("binary").to(device),
            "ece": torchmetrics.classification.CalibrationError("binary", n_bins=15, norm='l1').to(device)
        }