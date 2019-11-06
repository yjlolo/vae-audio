# vae-audio
For variational auto-encoders (VAEs) and audio/music lovers, based on PyTorch.

## Overview
The repo is under construction.

The project is built to facillitate research on using VAEs to model audio. It provides 
 - [ ] code for the paper [Learning Disentangled Representations of Timbre and Pitch for Musical Instrument Sounds Using Gaussian Mixture Variational Autoencoders](https://arxiv.org/pdf/1906.08152.pdf)
 - [x] [vanilla VAE](https://arxiv.org/abs/1312.6114)
 - [x] [Gaussian mixture VAE](https://arxiv.org/abs/1611.05148)
 - [ ] [vector-quantized VAE](https://arxiv.org/abs/1711.00937)
 - [ ] customizable model options
 - [x] audio feature extracton
 - [ ] model testing and latent space visualization
 - [ ] end-to-end audio feature extraction and model training
 - [ ] higher-level wrappers for easier use
 - [ ] easier installation
 - [ ] documentation
 - [ ] ...

The project structure is based on [PyTorch Template](https://github.com/victoresque/pytorch-template).

## Requirements
* torch 1.1.0
* librosa 0.6.3

## Usage
### Audio Feature Extraction 
1. Define customized `Dataset` classes in `dataset/datasets.py`
2. Run `python dataset/audio_transform.py -c your_config_of_audio_transform.json` to compute audio features (e.g., spectrograms)
3. Define customized `DataLoader` classes in `data_loader/data_loaders.py`
### Model Training
Run `python train.py -c your_config_of_model_train.json`

## To Be Continued
