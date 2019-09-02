# Tacotron 2

A PyTorch implementation of Tacotron2, described in [Natural TTS Synthesis By Conditioning Wavenet On Mel Spectrogram Predictions](https://arxiv.org/pdf/1712.05884.pdf), an end-to-end text-to-speech(TTS) neural network architecture, which directly converts character text sequence to speech.

## Dataset

[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

## Dependency

- Python 3.5.2
- PyTorch 1.0.0

## Usage
### Data Pre-processing
Extract data:
```bash
$ python extract.py
```

### Train
```bash
$ python train.py
```

If you want to visualize during training, run in your terminal:
```bash
$ tensorboard --logdir runs
```

### Demo
Generate mel-spectrogram for text "You, me, or nobody is gonna hit as hard as life. But it is not about how hard you hit. It’s about how hard you can get hit and keep moving forward."
```bash
$ python demo.py
```
![image](https://github.com/foamliu/Tacotron2/raw/master/images/mel_spec.jpg)
