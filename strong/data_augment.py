import torch
import torchaudio
import augment
import numpy as np
import random


class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),   # length of the sequence
                    'precision': 32,       # precision (16, 32 bits)
                    'rate': 16000.0,       # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y = self.chain.apply(
            x, src_info=src_info, target_info=target_info)

        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y


def random_pitch_shift(a=-300, b=300):
    return random.randint(a, b)

def random_time_warp(f=1):
    # time warp range: [1-0.1*f, 1+0.1*f], default is [0.9, 1.1]
    return 1 + f * (random.random() - 0.5) / 5

if __name__ == '__main__':
    chain = augment.EffectChain()
    chain.pitch(random_pitch_shift).rate(16000)
    chain.tempo(random_time_warp)
    chain = ChainRunner(chain)
    wav = torch.randn((1, 16000))
    augmented = chain(wav)
    print(wav.shape, augmented.shape)
    print(wav[:, :-1000], augmented[:, :-1000])
