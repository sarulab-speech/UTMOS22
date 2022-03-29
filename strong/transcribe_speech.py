from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
from datasets import set_caching_enabled


if __name__ == '__main__':
    set_caching_enabled(False)
    # load datasets

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    from tqdm import tqdm
    _ = model.to('cuda')
    batch_size = 1
    def collate_fn(batch):
        import numpy as np
        wav_name = batch[0]['audio']['path']
        wav_array = batch[0]['audio']['array']
        return wav_name, processor(wav_array,sampling_rate=16_000,return_tensors="pt").input_values
    for track in 'main', 'ood', 'unlabeled':
        wav_names = []
        transcriptions = []
        if track == 'main':
            dataset = load_dataset("sarulab-speech/bvcc-voicemos2022","main_track", data_dir="data/phase1-main/", use_auth_token=True,download_mode="force_redownload")
        elif track == 'ood':
            dataset = load_dataset("sarulab-speech/bvcc-voicemos2022","ood_track", data_dir="data/phase1-ood/", use_auth_token=True,download_mode="force_redownload")
        else:
            dataset = load_dataset("sarulab-speech/bvcc-voicemos2022","ood_track_unlabeled", data_dir="data/phase1-ood/", use_auth_token=True,download_mode='force_redownload')
        for stage in 'train', 'validation', 'test':
            if track == 'unlabeled' and stage != 'train':
                continue
            print('Processing {track} track {stage}'.format(track=track, stage=stage))
            dl = torch.utils.data.DataLoader(dataset[stage], batch_size=batch_size, num_workers=4,collate_fn=collate_fn)
            for wav_name, data in tqdm(dl):

                # retrieve logits
                with torch.no_grad():
                    logits = model(data.to('cuda')).logits

                # take argmax and decode
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)
                transcriptions.extend(transcription)
                wav_names.append(wav_name)
        del(dataset)
        import pandas as pd
        df = pd.DataFrame({"wav_name": wav_names, "transcription": transcriptions})
        df['wav_name'] = df['wav_name'].apply(lambda x: x.split("/")[-1])
        df.to_csv('transcriptions_{}.csv'.format(track), index=False)
