import os

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import meta_file, wave_folder


def TextAudioCollate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature
        padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (padded_input, padded_target, input_length)

    # sort it by input lengths (long to short)
    batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class LJSpeechDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        with open(meta_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        self.lines = lines
        print('loading {} {} samples...'.format(len(self.lines), split))

    def __getitem__(self, i):
        line = self.lines[i]
        tokens = line.split('|')
        wave = tokens[0]
        wave = os.path.join(wave_folder, wave + '.wav')
        trn = [text.strip() for text in tokens[1:]]

        return wave, trn

    def __len__(self):
        return len(self.lines)


if __name__ == "__main__":
    import torch
    from utils import parse_args

    args = parse_args()
    train_dataset = LJSpeechDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    print(train_dataset[0])
