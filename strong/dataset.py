from typing import Any, Dict, List
import augment
import torch
import torchaudio
import os
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import hydra
from phonemizer import phonemize
import pandas as pd
from data_augment import ChainRunner, random_pitch_shift, random_time_warp
import text.symbols as symbols
import numpy as np
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    def setup(self, stage):
        ocwd = hydra.utils.get_original_cwd()
        join = os.path.join
        data_sources = self.cfg.dataset.data_sources
        self.wavdir = {} 
        for datasource in data_sources:
            self.wavdir[datasource.name] = join(ocwd, datasource['wav_dir'])  
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.external and datasource['name'] == 'external':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.main and datasource['name'] == 'main':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.ood and datasource['name'] == 'ood':
                data_sources.pop(idx)
        self.datasources = data_sources

        train_paths = [join(ocwd,data_source.train_mos_list_path) for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        val_paths = [join(ocwd,data_source.val_mos_list_path) for data_source in data_sources if hasattr(data_source,'val_mos_list_path')]
        domains_train = [data_source.name for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        domains_val = [data_source.name for data_source in data_sources if hasattr(data_source,'val_mos_list_path')]

        self.mos_df = {}
        self.mos_df['train'] = self.get_mos_df(train_paths,domains_train,self.cfg.dataset.only_mean)
        self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])
        self.mos_df['train'].to_csv("listener_embedding_lookup.csv")
       

    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        assert len(domains) == len(paths)
        dfs = [] 
        for path, domain in zip(paths,domains):
            df = pd.read_csv(path,names=['sysID', 'filename','rating', 'ignore', 'listener_info'])
            listener_df = pd.DataFrame()
            listener_df['filename'] = df['filename']
            listener_df['rating'] = df['rating']
            listener_df['listener_name'] = df['listener_info'].str.split('_').str[2]
            listener_df['domain'] = domain
            mean_df = pd.DataFrame(listener_df.groupby('filename',as_index=False)['rating'].mean())
            mean_df['listener_name']= f"MEAN_LISTENER_{domain}"
            mean_df['domain'] = domain
            if only_mean:
                dfs.append(mean_df)
            else:
                dfs.extend([listener_df,mean_df])
            
        return_df = pd.concat(dfs,ignore_index=True)
        if id_reference is None:
            return_df['listener_id'] = return_df["listener_name"].factorize()[0]
            return_df['domain_id'] = return_df['domain'].factorize()[0]
        else:
            listener_id = []
            domain_id = []
            for idx, row in return_df.iterrows():
                listener_id.append( id_reference[id_reference['listener_name'] == row['listener_name']]['listener_id'].iloc[0])
                domain_id.append(id_reference[id_reference['domain'] == row['domain']]['domain_id'].iloc[0])
            return_df['listener_id'] = listener_id
            return_df['domain_id'] = domain_id
        return return_df

    def get_ds(self, phase):
        ds = MyDataset(
            self.wavdir,
            self.mos_df[phase],
            phase=phase,
            cfg=self.cfg,
        )
        return ds
    def get_loader(self, phase, batchsize):
        ds = self.get_ds(phase)
        dl = DataLoader(
            ds,
            batchsize,
            shuffle=True if phase == "train" else False,
            num_workers=8,
            collate_fn=ds.collate_fn,
        )
        return dl

    def train_dataloader(self):
        return self.get_loader(phase="train", batchsize=self.cfg.train.train_batch_size)

    def val_dataloader(self):
        return self.get_loader(phase="val", batchsize=self.cfg.train.val_batch_size)

    def test_dataloader(self):
        return self.get_loader(phase="val", batchsize=self.cfg.train.test_batch_size)

class TestDataModule(DataModule):
    '''
        DataModule used for CV and test
        This is only used for inference so it has no train data
        Args:
            i_cv: number of fold
            set_name: test, val
    '''
    def __init__(self, cfg, i_cv, set_name):
        super().__init__(cfg)

        self.i_cv = i_cv
        self.set_name = set_name

    def setup(self, stage):
        ocwd = hydra.utils.get_original_cwd()
        join = os.path.join
        data_sources = self.cfg.dataset.data_sources
        self.wavdir = {} 
        for datasource in data_sources:
            self.wavdir[datasource.name] = join(ocwd, datasource['wav_dir'])  
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.external and datasource['name'] == 'external':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.main and datasource['name'] == 'main':
                data_sources.pop(idx)
        for idx, datasource in enumerate(data_sources):
            if not self.cfg.dataset.use_data.ood and datasource['name'] == 'ood':
                data_sources.pop(idx)

        train_paths = [join(ocwd,data_source.train_mos_list_path) for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]
        domains_train = [data_source.name for data_source in data_sources if hasattr(data_source,'train_mos_list_path')]

        if self.set_name == 'val': 
            mos_list_path = 'val_mos_list_path'
        elif self.set_name == 'test':
            mos_list_path = 'test_mos_list_path'
        elif self.set_name == 'test_post':
            mos_list_path = 'test_post_mos_list_path'
        val_paths = [join(ocwd,getattr(data_source, mos_list_path)) for data_source in data_sources if hasattr(data_source, mos_list_path)]
        domains_val = [data_source.name for data_source in data_sources if hasattr(data_source, mos_list_path)]

        self.mos_df = {}
        self.mos_df['train'] = self.get_mos_df(train_paths,domains_train,self.cfg.dataset.only_mean)
        
        if self.set_name == 'val':
            self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])
        elif self.set_name == 'test':
            self.mos_df['val'] = self.get_test_df(val_paths,domains_val,id_reference=self.mos_df['train'])
        elif self.set_name == 'test_post':
            self.mos_df['val'] = self.get_mos_df(val_paths,domains_val,only_mean=True,id_reference=self.mos_df['train'])


    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        return_df = super().get_mos_df(paths, domains, only_mean, id_reference)
        return_df['i_cv'] = self.i_cv
        return_df['set_name'] = self.set_name
        return return_df
    
    def get_test_df(self, paths, domains, id_reference):
        assert len(domains) == len(paths)
        dfs = [] 
        for path, domain in zip(paths,domains):
            with open(path) as f:
                wavlist = [wavline.strip() for wavline in f]
            df = pd.DataFrame()
            df['filename'] = wavlist
            df['rating'] = -1
            df['listener_name'] = f'MEAN_LISTENER_{domain}'
            df['domain'] = domain
            dfs.append(df)
        return_df = pd.concat(dfs,ignore_index=True)
        listener_id = []
        domain_id = []
        for idx, row in return_df.iterrows():
            listener_id.append(id_reference[id_reference['listener_name'] == row['listener_name']]['listener_id'].iloc[0])
            domain_id.append(id_reference[id_reference['domain'] == row['domain']]['domain_id'].iloc[0])
        return_df['listener_id'] = listener_id
        return_df['domain_id'] = domain_id
        return_df['i_cv'] = self.i_cv
        return_df['set_name'] = self.set_name
        return return_df

class CVDataModule(DataModule):
    def __init__(self, cfg, k_cv, i_cv,fold_target='main'):
        super().__init__(cfg)

        self.k_cv = k_cv
        self.i_cv = i_cv
        self.seed_cv = 0
        self.fold_target_datset = fold_target


    def setup(self, stage):
        super().setup(stage)
        
        target_id = {}
        for idx, datasource in enumerate(self.datasources):
            target_id[datasource['name']] = idx
        target_df = self.mos_df["train"][self.mos_df["train"]['domain'] == self.fold_target_datset]
        not_target_df = self.mos_df["train"][self.mos_df["train"]['domain'] != self.fold_target_datset]
        shuffled_train_df = target_df.sample(frac=1,random_state=self.seed_cv)
        chuncked_train_df = np.array_split(shuffled_train_df,self.k_cv)
        self.mos_df['val'] = chuncked_train_df[self.i_cv][chuncked_train_df[self.i_cv]['listener_name'].str.contains("MEAN_LISTENER")]
        chuncked_train_df.pop(self.i_cv)
        self.mos_df['val'] = self.mos_df['val'].reset_index()
        chuncked_train_df.append(not_target_df)
        self.mos_df['train'] = pd.concat(chuncked_train_df,ignore_index=True).reset_index()
        print("-"*20)
        print(len(self.mos_df['train']))
        print("-"*20)
        self.mos_df["train"] = self.mos_df["train"][~self.mos_df["train"]["filename"].isin(self.mos_df['val']["filename"])]
        print(len(self.mos_df['train']))
        print("-"*20)
        
    
    def get_mos_df(self, paths:List[str], domains:List[str],only_mean=False,id_reference=None):
        return_df = super().get_mos_df(paths, domains, only_mean, id_reference)
        return_df['i_cv'] = self.i_cv
        return_df['set_name'] = 'fold'
        return return_df
        

class MyDataset(Dataset):
    def __init__(self, wavdir, mos_df,phase, cfg,padding_mode='repetitive'):

        self.wavdir = wavdir if type(wavdir) != str else list(wavdir)
        self.additional_datas = []
        self.cfg = cfg
        self.padding_mode = padding_mode
        self.mos_df = mos_df

        # calc mean score by utterance
        sys_ratings = defaultdict(list)
        utt_ratings = defaultdict(list)
        for _, row in mos_df.iterrows():
            wavname = row["filename"]
            utt_ratings[wavname.split("-")[1].split(".")[0]].append(row["rating"])
            sys_ratings[wavname.split("-")[0]].append(row["rating"])
        self.utt_avg_score_table = {}
        self.sys_avg_score_table = {}
        for key in utt_ratings:
            self.utt_avg_score_table[key] = sum(utt_ratings[key])/len(utt_ratings[key])
        for key in sys_ratings:
            self.sys_avg_score_table[key] = sum(sys_ratings[key])/len(sys_ratings[key])

        for i in range(len(self.cfg.dataset.additional_datas)):
            self.additional_datas.append(
                hydra.utils.instantiate(
                    self.cfg.dataset.additional_datas[i],
                    cfg=self.cfg,
                    phase=phase,
                    _recursive_=False,
                )
            )

    def __getitem__(self, idx):
        selected_row = self.mos_df.iloc[idx]
        wavname = selected_row['filename']
        domain = selected_row['domain']
        wavpath = os.path.join(self.wavdir[domain], wavname)
        wav = torchaudio.load(wavpath)[0]
        score = selected_row['rating']
        domain_id = selected_row['domain_id']
        listener_id = selected_row['listener_id']
        i_cv = selected_row['i_cv'] if 'i_cv' in selected_row else -1
        set_name = selected_row['set_name'] if 'set_name' in selected_row else ''
        utt_avg_score = self.utt_avg_score_table[wavname.split("-")[1].split(".")[0]]
        sys_avg_score = self.sys_avg_score_table[wavname.split("-")[0]]
        data = {
            'wav': wav,
            'score': score,
            'wavname': wavname,
            'domain': domain_id,
            'judge_id': listener_id,
            'i_cv': i_cv,
            'set_name': set_name,
            'utt_avg_score': utt_avg_score,
            'sys_avg_score': sys_avg_score
        }
        for additional_data_instances in self.additional_datas:
            data.update(additional_data_instances(data))
        return data

    def __len__(self):
        return len(self.mos_df)

    def collate_fn(self, batch):  # zero padding
        wavs = [b['wav'] for b in batch]
        scores = [b['score'] for b in batch]
        wavnames = [b['wavname'] for b in batch]
        domains = [b['domain'] for b in batch]
        judge_id = [b['judge_id'] for b in batch]
        i_cvs = [b['i_cv'] for b in batch]
        set_names = [b['set_name'] for b in batch]
        utt_avg_scores = [b['utt_avg_score'] for b in batch]
        sys_avg_scores = [b['sys_avg_score'] for b in batch]
        wavs = list(wavs)
        max_len = max(wavs, key=lambda x: x.shape[1]).shape[1]
        wavs_lengths = torch.from_numpy(np.array([wav.size(0) for wav in wavs]))
        output_wavs = []
        if self.padding_mode == 'zero-padding':
            for wav in wavs:
                amount_to_pad = max_len - wav.shape[1]
                padded_wav = torch.nn.functional.pad(
                    wav, (0, amount_to_pad), "constant", 0)
                output_wavs.append(padded_wav)
        else:
            for wav in wavs:
                amount_to_pad = max_len - wav.shape[1]
                padding_tensor = wav.repeat(1,1+amount_to_pad//wav.size(1))
                output_wavs.append(torch.cat((wav,padding_tensor[:,:amount_to_pad]),dim=1))
        output_wavs = torch.stack(output_wavs, dim=0)
        scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(scores)], dim=0)
        utt_avg_scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(utt_avg_scores)], dim=0)
        sys_avg_scores = torch.stack([torch.tensor(x,dtype=torch.float) for x in list(sys_avg_scores)], dim=0)
        domains = torch.stack([torch.tensor(x) for x in list(domains)], dim=0)
        judge_id = torch.stack([torch.tensor(x) for x in list(judge_id)], dim=0)
        collated_batch = {
            'wav': output_wavs,
            'score': scores,
            'utt_avg_score': utt_avg_scores,
            'sys_avg_score': sys_avg_scores,
            'wavname': wavnames,
            'domains': domains,
            'judge_id': judge_id,
            'wav_len': wavs_lengths,
            'domain': domains,
            'i_cv': i_cvs,
            'set_name': set_names
        } # judge id, domain, averaged score
        for additional_data_instance in self.additional_datas:
            additonal_collated_batch = additional_data_instance.collate_fn(batch)
            collated_batch.update(additonal_collated_batch)
        return collated_batch

class AdditionalDataBase():
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg

    def __call__(self, data: Dict[str, Any]):
        return self.process_data(data)

    def process_data(self, data: Dict[str, Any]):
        raise NotImplementedError

    def collate_fn(self, batch):
        return dict()


class PhonemeData(AdditionalDataBase):
    def __init__(self, transcription_file_path: str, with_reference=True, cfg=None,phase='train') -> None:
        super().__init__(cfg)
        self.text_df = pd.read_csv(
            os.path.join(
                hydra.utils.get_original_cwd(),
                transcription_file_path,
            )
        )
        self.with_reference = with_reference

    def process_data(self, data: Dict[str, Any]):
        wavname = data['wavname']
        phonemes = self.text_df[self.text_df['wav_name']
                                == wavname]['transcription']
        assert len(phonemes) == 1, 'wavname {} has more than one text'.format(
            wavname)
        try:
            phonemes = [symbols.symbols.index(p) for p in phonemes.iloc[0]]
        except:
            print(wavname, phonemes)
            raise ValueError
        if self.with_reference:
            reference = self.text_df[self.text_df['wav_name']
                                     == wavname]['reference']
            assert len(reference) == 1, 'wavname {} has more than one text'.format(
                wavname)
            try:
                reference = [symbols.symbols.index(
                    p) for p in reference.iloc[0]]
            except:
                print(wavname, reference)
                raise ValueError
        return {
            'phonemes': phonemes,
            'reference': reference
        }

    def collate_fn(self, batch):
        phonemes = [b['phonemes'] for b in batch]
        references = [b['reference'] for b in batch]

        lens = [len(p) for p in phonemes]
        phonemes = [torch.tensor(p) for p in phonemes]
        phoneme_batch = torch.nn.utils.rnn.pad_sequence(
            phonemes, batch_first=True)

        if self.with_reference:
            len_references = [len(p) for p in references]
            references = [torch.tensor(p) for p in references]
            reference_batch = torch.nn.utils.rnn.pad_sequence(
                references, batch_first=True)
            return {
                'phonemes': phoneme_batch,
                'phoneme_lens': lens,
                'reference': reference_batch,
                'reference_lens': len_references
            }
        else:
            return {
                'phonemes': phoneme_batch,
                'phoneme_lens': lens,
            }



class NormalizeScore(AdditionalDataBase):
    def __init__(self, org_max, org_min,normalize_to_max,normalize_to_min,phase,cfg=None) -> None:
        super().__init__()
        self.org_max = org_max
        self.org_min = org_min
        self.normalize_to_max = normalize_to_max
        self.normalize_to_min = normalize_to_min
    def process_data(self, data: Dict[str, Any]):
        score = data['score']
        score = (score - (self.org_max + self.org_min)/2.0) / (self.normalize_to_max - self.normalize_to_min)
        return {'score': score}

class AugmentWav(AdditionalDataBase):
    def __init__(self,pitch_shift_minmax:Dict[str, int],random_time_warp_f,phase='train', cfg=None) -> None:
        super().__init__(cfg)
        self.chain = augment.EffectChain()
        self.chain.pitch(random_pitch_shift(pitch_shift_minmax['min'], pitch_shift_minmax['max'])).rate(16000)
        self.chain.tempo(random_time_warp(random_time_warp_f))
        self.chain = ChainRunner(self.chain)
        self.phase = phase
    def process_data(self, data: Dict[str, Any]):
        if self.phase=='train':
            augmented_wav = self.chain(data['wav'])
        else:
            augmented_wav = data['wav']
        return {'wav': augmented_wav}

class SliceWav(AdditionalDataBase):
    def __init__(self, max_wav_seconds,cfg=None,phase=None) -> None:
        super().__init__()
        self.max_wav_len = int(max_wav_seconds*16000)
    def process_data(self, data: Dict[str, Any]):
        return {'wav': data['wav'][:, :self.max_wav_len]}
