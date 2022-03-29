from collections import defaultdict
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from dataset import CVDataModule, TestDataModule
from lightning_module import BaselineLightningModule
import hydra
import wandb

@hydra.main(config_path="configs",config_name='cv')
def cross_validation(cfg):

    k_cv = cfg.dataset.k_cv
    i_cv = cfg.dataset.i_cv
    
    print("------- Using subset_{} out of {}-fold -------".format(i_cv, k_cv))
    d_metrics = fit_and_test(cfg, k_cv, i_cv)
    wandb.log(d_metrics)
    

def fit_and_test(cfg, k_cv, i_cv):
    debug = cfg.debug
    if debug:
        cfg.train.trainer_args.max_steps=10

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

    csvlogger = CSVLogger(save_dir=cfg.train.out_dir, name="log_subset{}".format(i_cv))
    tblogger = TensorBoardLogger(save_dir=cfg.train.out_dir, name="tf_log_subset{}".format(i_cv))
    wandblogger = WandbLogger(name="{}_{}_{}".format(cfg.testname,i_cv,k_cv), project="voiceMOS2022-cross-validation-main",entity='sarulab-voicemos',settings=wandb.Settings(start_method="fork"))

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.out_dir,
        save_weights_only=True,
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
        monitor="val_SRCC_system_main",
        mode='max'
    )
    callbacks = [checkpoint_callback]
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
        logger=[csvlogger, tblogger, wandblogger],
    )

    datamodule = CVDataModule(cfg=cfg, k_cv=k_cv, i_cv=i_cv)
    val_datamodule = TestDataModule(cfg=cfg, i_cv=i_cv, set_name='val')
    test_datamodule = TestDataModule(cfg=cfg, i_cv=i_cv, set_name='test')
    lightning_module = hydra.utils.instantiate(cfg.model.lightning_module,cfg=cfg, _recursive_=False)
        
    trainer.fit(lightning_module, datamodule=datamodule)

    if debug:
        trainer.test(lightning_module, verbose=True, datamodule=datamodule)
        print(trainer.logged_metrics.keys())
        result = trainer.logged_metrics["test_SRCC_SYS_main_i_cv_{}_set_name_{}".format(i_cv, "fold")]
        trainer.test(lightning_module, verbose=True, datamodule=val_datamodule)
        trainer.test(lightning_module, verbose=True, datamodule=test_datamodule)
    else:
        trainer.test(lightning_module, verbose=True, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)
        result = trainer.logged_metrics["test_SRCC_SYS_main_i_cv_{}_set_name_{}".format(i_cv, "fold")]
        trainer.test(lightning_module, verbose=True, datamodule=val_datamodule, ckpt_path=checkpoint_callback.best_model_path)
        trainer.test(lightning_module, verbose=True, datamodule=test_datamodule, ckpt_path=checkpoint_callback.best_model_path)

    return result

if __name__ == "__main__":
    cross_validation()
