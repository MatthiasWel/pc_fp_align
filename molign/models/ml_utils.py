import torch
import logging
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import BCELoss

from molign.models.lightning import LitDataModule, LitModel


def train(
    model_name: str,
    time: str,
    model,
    train,
    val,
    tensorboard_path,
    logs_path,
    unwrap_data=lambda data: (data[0], data[1]),
    loss_fn=BCELoss(),
    learning_rate=1e-4,
    batch_size=1000,
    epochs=100,
):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    
    log_file = f"lightning_{time}.log"

    logger = logging.getLogger("lightning.pytorch")
    logger.setLevel(logging.DEBUG)  
    logger.handlers = []
    logger.propagate = False
    file_handler = logging.FileHandler(logs_path / log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    lit_model = LitModel(
        model=model,
        loss_function=loss_fn,
        unwrap_data=unwrap_data,
        lr=learning_rate,
        batch_size=batch_size,
    )
    datamodule = LitDataModule(train=train, val=val, batch_size=batch_size)

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(
                save_top_k=1, monitor="val_loss", mode="max", save_last=True
            )
        ],
        log_every_n_steps=1,
        enable_progress_bar=False,
        logger=TensorBoardLogger(
            save_dir=tensorboard_path, name=time, version=model_name
        ),
    )
    trainer.fit(lit_model, datamodule)
    pred = torch.cat(trainer.predict(lit_model, datamodule.val_dataloader()))
    true = next(iter(datamodule.val_dataloader()))[1]
    metrics_d = {
        metric_name: metric(pred, true)
        for metric_name, metric in lit_model.metrics.items()
    }

    return pred, true, metrics_d
