from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
from datasets import set_caching_enabled
import pandas as pd
import Levenshtein
import numpy as np
from sklearn.cluster import DBSCAN

def cluster_transcriptions(df:pd.DataFrame):
    data = df['transcription'].to_list()
    def lev_metric(x, y):
        i, j = int(x[0]), int(y[0])     # extract indices
        return Levenshtein.distance(data[i], data[j])/max(len(data[i]), len(data[j]))

    X = np.arange(len(data)).reshape(-1, 1)
    result = DBSCAN(eps=0.3, metric=lev_metric,n_jobs=20,min_samples=3).fit(X)
    df['cluster'] = result.labels_
    text_medians = df.groupby('cluster').apply(lambda x:Levenshtein.median(x['transcription'].to_list()))
    medians = []
    for idx, row in df.iterrows():
        if row['cluster'] == -1:
            medians.append(row['transcription'])
        else:
            medians.append(text_medians[row['cluster']])
    df['reference'] = medians
    return df



if __name__ == '__main__':
    set_caching_enabled(False)
    # load datasets

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
    from tqdm import tqdm
    _ = model.to('cuda')
    batch_size = 1
    all_df = pd.DataFrame()
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
        df['track'] = track if track != 'unlabeled' else 'ood'
        all_df = pd.concat([all_df, df],ignore_index=True)
    result = pd.concat(
        [
            cluster_transcriptions(all_df[all_df['track'] == 'main'].copy()),
            cluster_transcriptions(all_df[all_df['track'] == 'ood'].copy()),
        ],
        ignore_index=True
    )
    result.to_csv('transcriptions_clustered.csv'.format(track), index=False)
    