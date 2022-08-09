from omegaconf import open_dict
from pytorch_lightning import Trainer
import hydra
import os
import pathlib
from lightning_module import UTMOSLightningModule
from dataset import TestDataModule, DataModule
import torch

@hydra.main(config_path="configs",config_name='default')
def predict(cfg):
    """
    Specify ckeckpoint path as follows:
    
    python predict.py +ckpt_path="outputs/${date}/${time}/train_outputs/hoge.ckpt"
    """

    trainer = Trainer(
        **cfg.train.trainer_args,
        default_root_dir=hydra.utils.get_original_cwd(),
    )

    ckpt_path = pathlib.Path(cfg.ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = (pathlib.Path(hydra.utils.get_original_cwd()) / ckpt_path)

    if 'paper_weights' in cfg.keys():
        with open_dict(cfg):
            ckpt = torch.load(ckpt_path)
            use_data = ckpt['hyper_parameters']['cfg']['dataset']['use_data']
            cfg.dataset.use_data['external'] = use_data['lancers']
            cfg.dataset.use_data['main'] = use_data['main']
            cfg.dataset.use_data['ood'] = use_data['ood']
            cfg.dataset.only_mean = ckpt['hyper_parameters']['cfg']['dataset']['only_mean']
        lightning_module = UTMOSLightningModule.load_from_checkpoint(ckpt_path,cfg=cfg,paper_weight=cfg.paper_weights)
        lightning_module.cfg
    else:
        lightning_module = UTMOSLightningModule.load_from_checkpoint(ckpt_path)
    print(lightning_module.cfg)
    datamodule = DataModule(lightning_module.cfg)
    test_datamodule = TestDataModule(cfg=lightning_module.cfg, i_cv=0, set_name='test')
    trainer.test(
        lightning_module,
        verbose=True,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )
    trainer.test(
        lightning_module,
        verbose=True,
        datamodule=test_datamodule,
        ckpt_path=ckpt_path
    )

if __name__ == "__main__":
    predict()
