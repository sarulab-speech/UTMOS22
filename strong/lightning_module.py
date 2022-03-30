import pytorch_lightning as pl
import torch
import torch.nn as nn
from WavLM import WavLM, WavLMConfig
import os
import fairseq
import numpy as np
import scipy.stats
import hydra
from transformers import AdamW, get_linear_schedule_with_warmup
from model import load_ssl_model, PhonemeEncoder, DomainEmbedding, LDConditioner, Projection
import wandb


class UTMOSLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.construct_model()
        self.prepare_domain_table()
        self.save_hyperparameters()
    
    def construct_model(self):
        self.feature_extractors = nn.ModuleList([
            hydra.utils.instantiate(feature_extractor) for feature_extractor in self.cfg.model.feature_extractors
        ])
        output_dim = sum([ feature_extractor.get_output_dim() for feature_extractor in self.feature_extractors])
        output_layers = []
        for output_layer in self.cfg.model.output_layers:
            output_layers.append(
                hydra.utils.instantiate(output_layer,input_dim=output_dim)
            )
            output_dim = output_layers[-1].get_output_dim()

        self.output_layers = nn.ModuleList(output_layers)

        self.criterion = self.configure_criterion()

    def prepare_domain_table(self):
        self.domain_table = {}
        data_sources = self.cfg.dataset.data_sources
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.external and datasource['name'] == 'external':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.main and datasource['name'] == 'main':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.ood and datasource['name'] == 'ood':
                data_sources.pop(idx)
        for i, datasource in enumerate(data_sources):
            if not hasattr(datasource,'val_mos_list_path'):
                continue
            self.domain_table[i] = datasource["name"]

    def forward(self, inputs):
        outputs = {}
        for feature_extractor in self.feature_extractors:
            outputs.update(feature_extractor(inputs))
        x = outputs
        for output_layer in self.output_layers:
            x = output_layer(x,inputs)
        return x

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'])
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.train.train_batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'])
        if outputs.dim() > 1:
            outputs = outputs.mean(dim=1).squeeze(-1)
        return {
            "loss": loss,
            "outputs": outputs.cpu().numpy()[0]*2 +3.0,
            "filename": batch["wavname"][0],
            "domain": batch["domain"][0],
            "utt_avg_score": batch["utt_avg_score"][0].item(),
            "sys_avg_score": batch["sys_avg_score"][0].item()
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out["loss"] for out in outputs]).mean().item()
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, logger=True)
        for domain_id in self.domain_table:
            outputs_domain = [out for out in outputs if out["domain"] == domain_id]
            if len(outputs_domain) == 0:
                continue
            _, SRCC, MSE = self.calc_score(outputs_domain)
            self.log(
                "val_SRCC_system_{}".format(self.domain_table[domain_id]),
                SRCC,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            self.log(
                "val_MSE_system_{}".format(self.domain_table[domain_id]),
                MSE,
                on_epoch=True,
                prog_bar=True,
                logger=True
            )
            if domain_id == 0:
                self.log(
                    "val_SRCC_system".format(self.domain_table[domain_id]),
                    SRCC,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True
                )

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch['score'])
        labels = batch['score']
        filenames = batch['wavname']
        loss = self.criterion(outputs, labels)
        if outputs.dim() > 1:
            outputs = outputs.mean(dim=1).squeeze(-1)
        return {
            "loss": loss,
            "outputs": outputs.cpu().detach().numpy()[0]*2 +3.0,
            "labels": labels.cpu().detach().numpy()[0] *2 +3.0,
            "filename": filenames[0],
            "domain": batch["domain"][0],
            "i_cv": batch["i_cv"][0],
            "set_name": batch["set_name"][0],
            "utt_avg_score": batch["utt_avg_score"][0].item(),
            "sys_avg_score": batch["sys_avg_score"][0].item()
        }

    def test_epoch_end(self, outputs):
        outfiles = [datasource["outfile"] + '{}_{}'.format(outputs[0]['set_name'],outputs[0]['i_cv']) for datasource in self.cfg.dataset.data_sources if hasattr(datasource,'outfile')]
        for domain_id in self.domain_table:
            outputs_domain = [out for out in outputs if out["domain"] == domain_id]
            predictions, SRCC, MSE = self.calc_score(outputs_domain)
            self.log(
                "test_SRCC_SYS_{}_i_cv_{}_set_name_{}".format(self.domain_table[domain_id], outputs[0]['i_cv'], outputs[0]['set_name']),
                SRCC,
            )
            if domain_id == 0:
                self.log(
                    "test_SRCC_SYS".format(self.domain_table[domain_id]),
                    SRCC
                )
            with open(outfiles[domain_id], "w") as fw:
                for k, v in predictions.items():
                    outl = k.split(".")[0] + "," + str(v) + "\n"
                    fw.write(outl)
            try:
                wandb.save(outfiles[domain_id])
            except:
                print('outfile {} saved'.format(outfiles[domain_id]))

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.cfg.train.optimizer,
            params=self.parameters()
        )
        scheduler = hydra.utils.instantiate(
            self.cfg.train.scheduler,
            optimizer=optimizer
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_criterion(self):
        return hydra.utils.instantiate(self.cfg.train.criterion,_recursive_=True)

    def calc_score(self, outputs, verbose=False):

        def systemID(uttID):
            return uttID.split("-")[0]

        predictions = {}
        true_MOS = {}
        true_sys_MOS_avg = {}
        for out in outputs:
            predictions[out["filename"]] = out["outputs"]
            true_MOS[out["filename"]] = out["utt_avg_score"]
            true_sys_MOS_avg[out["filename"].split("-")[0]] = out["sys_avg_score"]

        ## compute correls.
        sorted_uttIDs = sorted(predictions.keys())
        ts = []
        ps = []
        for uttID in sorted_uttIDs:
            t = true_MOS[uttID]
            p = predictions[uttID]
            ts.append(t)
            ps.append(p)

        truths = np.array(ts)
        print(ps)
        preds = np.array(ps)

        ### UTTERANCE
        MSE = np.mean((truths - preds) ** 2)
        LCC = np.corrcoef(truths, preds)
        SRCC = scipy.stats.spearmanr(truths.T, preds.T)
        KTAU = scipy.stats.kendalltau(truths, preds)
        if verbose:
            print("[UTTERANCE] Test error= %f" % MSE)
            print("[UTTERANCE] Linear correlation coefficient= %f" % LCC[0][1])
            print("[UTTERANCE] Spearman rank correlation coefficient= %f" % SRCC[0])
            print("[UTTERANCE] Kendall Tau rank correlation coefficient= %f" % KTAU[0])

        ### SYSTEM
        pred_sys_MOSes = {}
        for uttID in sorted_uttIDs:
            sysID = systemID(uttID)
            noop = pred_sys_MOSes.setdefault(sysID, [])
            pred_sys_MOSes[sysID].append(predictions[uttID])

        pred_sys_MOS_avg = {}
        for k, v in pred_sys_MOSes.items():
            avg_MOS = sum(v) / (len(v) * 1.0)
            pred_sys_MOS_avg[k] = avg_MOS

        ## make lists sorted by system
        pred_sysIDs = sorted(pred_sys_MOS_avg.keys())
        sys_p = []
        sys_t = []
        for sysID in pred_sysIDs:
            sys_p.append(pred_sys_MOS_avg[sysID])
            sys_t.append(true_sys_MOS_avg[sysID])

        sys_true = np.array(sys_t)
        sys_predicted = np.array(sys_p)

        MSE = np.mean((sys_true - sys_predicted) ** 2)
        LCC = np.corrcoef(sys_true, sys_predicted)
        SRCC = scipy.stats.spearmanr(sys_true.T, sys_predicted.T)
        KTAU = scipy.stats.kendalltau(sys_true, sys_predicted)
        if verbose:
            print("[SYSTEM] Test error= %f" % MSE)
            print("[SYSTEM] Linear correlation coefficient= %f" % LCC[0][1])
            print("[SYSTEM] Spearman rank correlation coefficient= %f" % SRCC[0])
            print("[SYSTEM] Kendall Tau rank correlation coefficient= %f" % KTAU[0])

        return predictions, SRCC[0], MSE



class DeepSpeedBaselineLightningModule(UTMOSLightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)

    def configure_optimizers(self):
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        return DeepSpeedCPUAdam(self.parameters())