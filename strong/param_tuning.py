from lib2to3.pytree import Base
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import hydra
from dataset import DataModule
import wandb

@hydra.main(config_path="configs",config_name='optuna-main')
def train(cfg):
    debug = cfg.debug

    if cfg.batch_size_and_model == "wav2vec2-base-4":
        cfg.model.feature_extractors[0]["cp_path"] = "fairseq/wav2vec_small.pt"
        cfg.train.train_batch_size = 4
    elif cfg.batch_size_and_model == "wav2vec2-base-8":
        cfg.model.feature_extractors[0]["cp_path"] = "fairseq/wav2vec_small.pt"
        cfg.train.train_batch_size = 8
    elif cfg.batch_size_and_model == "wav2vec2-base-16":
        cfg.model.feature_extractors[0]["cp_path"] = "fairseq/wav2vec_small.pt"
        cfg.train.train_batch_size = 16
    elif cfg.batch_size_and_model == "wav2vec2-base-32":
        cfg.model.feature_extractors[0]["cp_path"] = "fairseq/wav2vec_small.pt"
        cfg.train.train_batch_size = 32
    elif cfg.batch_size_and_model == "wavlm-large-4":
        cfg.model.feature_extractors[0]["cp_path"] = "fairseq/WavLM-Large.pt"
        cfg.train.train_batch_size = 4
    print(cfg.batch_size_and_model)
    print(cfg.model.feature_extractors[0]["cp_path"])
    print(cfg.train.train_batch_size)
    
    if cfg.dataset.use_data.main:
        cfg.dataset.data_sources.pop(0)
    if cfg.dataset.use_data.ood:
        cfg.dataset.data_sources.pop(1)
    if cfg.dataset.use_data.external:
        cfg.dataset.data_sources.pop(2)

    loggers = []
    loggers.append(CSVLogger(save_dir=cfg.train.out_dir, name="train_log"))
    loggers.append(TensorBoardLogger(save_dir=cfg.train.out_dir, name="tf_log"))
    if cfg.train.use_wandb:
        loggers.append(WandbLogger(project="voicemos",offline=debug))

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.out_dir,
        save_weights_only=True,
        save_top_k=1,
        save_last=False,
        every_n_epochs=1,
        monitor=cfg.train.model_selection_metric,
        mode='max'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback,lr_monitor]
    earlystop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.0, patience=cfg.train.early_stopping.patience, mode="min"
    )
    callbacks.append(earlystop_callback)

    trainer = Trainer(
        **cfg.train.trainer_args,
        default_root_dir=hydra.utils.get_original_cwd(),
        limit_train_batches=0.01 if debug else 1.0,
        limit_val_batches=0.5 if debug else 1.0,
        callbacks=callbacks,
        logger=loggers,
    )

    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module,cfg=cfg , _recursive_=False)    
    wandblogger.watch(lightning_module)
    datamodule = hydra.utils.instantiate(cfg.dataset.datamodule,cfg=cfg,_recursive_=False)
    trainer.fit(lightning_module,datamodule=datamodule)
    trainer.test(lightning_module, datamodule=datamodule,ckpt_path=checkpoint_callback.best_model_path)


    SRCC_system = trainer.logged_metrics[cfg.tuning_target]
    wandb.finish()

    return SRCC_system

if __name__ == "__main__":
    train()
