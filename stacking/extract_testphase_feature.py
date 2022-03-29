
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import torchaudio
import fairseq
import torch
import os

import sys
sys.path.append('./external_libs/WavLM')
from WavLM import WavLM, WavLMConfig

def get_arg():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datatrack", type=str, help="phase1-main or phase1-ood",
        default='phase1-main')
    return parser.parse_args()


def get_mos_data(datatrack):
    mos_list_file = f'./data/{datatrack}/DATA/sets/test.scp'
    mos_data = {}
    for line in open(mos_list_file):
        file_id = line.rstrip()
        mos = -1.0 # dummy data
        mos_data[file_id] = mos
    return mos_data

def extract_mean(wavpath, ssl_model, device, use_wavlm):
    with torch.no_grad():
        if use_wavlm:
            wav = torchaudio.load(wavpath)[0]
            res = ssl_model.extract_features(wav.to(device))
            return res[0].squeeze(0).mean(dim=0)
        else:
            wav = torchaudio.load(wavpath)[0]
            res = ssl_model(wav.to(device), mask=False, features_only=True)
            return res['x'].squeeze(0).mean(dim=0)


def extract_feature(datatrack, ssl_type):

    device = torch.device('cuda')    

    wav_dir = Path(f'./data/{datatrack}/DATA/wav/')

    base_ckpt_file = {
        'w2v_small': './fairseq/wav2vec_small.pt',
        'w2v_xlsr': './fairseq/xlsr_53_56k.pt',
        'w2v_large': './fairseq/wav2vec_vox_new.pt',
        'w2v_large2': './fairseq/w2v_large_lv_fsh_swbd_cv.pt',
        'wavlm_base': './external_libs/WavLM/WavLM-Base.pt',
        'wavlm_large': './external_libs/WavLM/WavLM-Large.pt',
        'hubert_base': './fairseq/hubert_base_ls960.pt',
        'hubert_large': './fairseq/hubert_large_ll60k.pt',
        'w2v_small_dr': './fairseq/wav2vec_small.pt',
        'w2v_xlsr_dr': './fairseq/xlsr_53_56k.pt',
        'w2v_large_dr': './fairseq/wav2vec_vox_new.pt',
        'w2v_large2_dr': './fairseq/w2v_large_lv_fsh_swbd_cv.pt',
        'wavlm_base_dr': './external_libs/WavLM/WavLM-Base.pt',
        'wavlm_large_dr': './external_libs/WavLM/WavLM-Large.pt',
        'hubert_base_dr': './fairseq/hubert_base_ls960.pt',
        'hubert_large_dr': './fairseq/hubert_large_ll60k.pt',
    }[ssl_type]

    # param_file = f'./out/finetune/{datatrack}/{ssl_type}_ft/ssl_param.pt'

    print('base_ckpt_file: {}'.format(base_ckpt_file))
    # print('param_file: {}'.format(param_file))

    use_wavlm = ssl_type in ['wavlm_base', 'wavlm_large']

    if use_wavlm:
        checkpoint = torch.load(base_ckpt_file)
        cfg = WavLMConfig(checkpoint['cfg'])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint['model'])
    else:
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([base_ckpt_file])
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()

    ssl_model.to(device)
    ssl_model.eval()

    print(ssl_model)

    out_dir = Path(f'./out/utt_data/{ssl_type}')
    os.makedirs(out_dir, exist_ok=True)

    mos_data = get_mos_data(datatrack)

    for key, mos in tqdm(sorted(mos_data.items())):
        wavpath = wav_dir / key
        vec = extract_mean(wavpath, ssl_model, device, use_wavlm)
        outpath = out_dir / (wavpath.stem + '.npy')
        
        vec = vec.detach().cpu().numpy()
        np.save(outpath, vec)


def main():
    args = get_arg()

    # extract_feature(args.datatrack, 'w2v_small')
    # for ssl_type in ['wavlm_large', 'hubert_large', 'hubert_base', 'wavlm_base',
    #                     'w2v_small', 'w2v_xlsr', 'w2v_large', 'w2v_large2']:
    for ssl_type in ['wavlm_large', 'wavlm_base']:
        print('ssl_type: {}'.format(ssl_type))
        extract_feature(args.datatrack, ssl_type)



if __name__ == '__main__':
    main()