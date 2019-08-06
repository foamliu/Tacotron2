import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

meta_file = 'data/LJSpeech-1.1/metadata.csv'
wave_folder = 'data/LJSpeech-1.1/wavs'
