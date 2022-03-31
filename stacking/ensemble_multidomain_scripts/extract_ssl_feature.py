
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
    return parser.parse_args()


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

    wav_dir = Path(f'../data/{datatrack}/DATA/wav/')

    base_ckpt_file = {
        'w2v_small': '../pretrained_model_ckpt/wav2vec_small.pt',
        'w2v_xlsr': '../pretrained_model_ckpt/xlsr_53_56k.pt',
        'w2v_large': '../pretrained_model_ckpt/wav2vec_vox_new.pt',
        'w2v_large2': '../pretrained_model_ckpt/w2v_large_lv_fsh_swbd_cv.pt',
        'wavlm_base': '../pretrained_model_ckpt/WavLM-Base.pt',
        'wavlm_large': '../pretrained_model_ckpt/WavLM-Large.pt',
        'hubert_base': '../pretrained_model_ckpt/hubert_base_ls960.pt',
        'hubert_large': '../pretrained_model_ckpt/hubert_large_ll60k.pt',
    }[ssl_type]

    print('base_ckpt_file: {}'.format(base_ckpt_file))

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

    out_dir = Path(f'../out/utt_data/{ssl_type}')
    os.makedirs(out_dir, exist_ok=True)


    wavpath_list = list(wav_dir.glob('*.wav'))

    for wavpath in tqdm(wavpath_list):
        vec = extract_mean(wavpath, ssl_model, device, use_wavlm)
        outpath = out_dir / (wavpath.stem + '.npy')
        
        vec = vec.detach().cpu().numpy()
        np.save(outpath, vec)


def main():
    args = get_arg()

    ssl_types = ['wavlm_large', 'hubert_large', 'hubert_base', 'wavlm_base',
                        'w2v_small', 'w2v_xlsr', 'w2v_large', 'w2v_large2']
    datatracks = ['phase1-main', 'phase1-ood', 'tetsphase-main', 'testphase-ood']

    for datatrack in datatracks:
        for ssl_type in ssl_types:
            print('datatrack {}, ssl_type: {}'.format(
                datatrack, ssl_type))
            extract_feature(datatrack, ssl_type)



if __name__ == '__main__':
    main()
