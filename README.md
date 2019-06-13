# vae-audio
For variational auto-encoders (VAEs) and audio/music lovers, based on PyTorch.

## Overview
The repo is just like myself, in a very early stage of development and is under construction. Suggestions are very welcome, as I am keen to learn and do it right.

The project is built to facillitate my own research on using VAEs to model music audio. It provides 
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

## Project Structure
The structure is based on [PyTorch Template](https://github.com/victoresque/pytorch-template); please take a deep read beforehand. It might change to allow for a easier usage and to be more structural for this specific library.

## Requirements
* torch 1.1.0
* librosa 0.6.3

## Usage
### Audio Feature Extraction 
1. Define customized `Dataset` classes in `dataset/datasets.py`
2. Run `python dataset/audio_transform.py -c config_of_audio_transform.json` to compute audio features (e.g., spectrograms) and store
3. Define customized `DataLoader` classes in `data_loader/data_loaders.py`
### Model Training
Run `python train.py -c config_of_model_train.json`

## To Be Continued
