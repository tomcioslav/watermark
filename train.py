import mlflow
import pytorch_lightning as pl
import watermark
import watermark.config as config
import watermark.examples as ex
from pytorch_lightning.loggers import MLFlowLogger

train_dataloader = ex.full_logo_test_dataloader
test_dataloader = ex.full_logo_train_dataloader

mlflow_logger = MLFlowLogger(
    experiment_name="lightning_logs", tracking_uri="file:./ml-runs"
)

watermark_net = watermark.WatermarkNet()

mlflow.pytorch.autolog(log_every_n_step=1)

trainer = pl.Trainer(
    default_root_dir=config.paths.MODELS,
    accelerator=config.training_settings.DEVICE,
    devices=1,
    max_epochs=config.training_settings.NUM_OF_EPOCHS,
    logger=mlflow_logger,
)
with mlflow.start_run() as run:
    trainer.fit(
        model=watermark_net,
        train_dataloaders=train_dataloader,
        val_dataloaders=test_dataloader,
    )
