# DinTaiFung Project

# Repository Overview

- `classes.py` and `constants.py` are for creating/rendering a `Microphone`, `SoundSource`, and `Scene` objects.
   These were pulled from `https://github.com/vivjay30/d3audiorecon/tree/8fd7181562af5f0e7eb84b38606c5209ef42bf2f`.
- `data.py` has a PyTorch's `Dataset` class that handles all required preprocessing steps.
- `demo.ipynb` is used for demo. The notebook has the code for creating output audio.
- `network.py` has the network architecture and the loss function.
- `train.py` is used for training the network.
- `sample_data/` is a sample dataset (for checking the input format)
- `dataset_generators/simple_1_voice_1_bg` has a dataset generator script of scenarios where there is only a single voice
  and a single background in the scene. The voice (same wav file) is the same across the entire dataset.
- `dataset_generators/1_voice_1_bg` has a dataset generator script of scenarios where there is only a single voice
  and a single background in the scene. This extends from `dataset_generators/simple_1_voice_1_bg` by using different voices
  from VCTK corpus.
- `dataset_generators/multiple_voices_multiple_backgrounds` has a dataset generator script of general use case scenarios where
  the scene has multiple voices and multiple backgrounds. __Note: this is outdated.__
  
# Result replication

__Dataset generation__

Here's the code for generating the dataset `1_voice_1_bg_hard`. Note that `--bg_factor` specifies the multiplicative factor
of the background signal.

```
cd dataset_generators/1_voice_1_bg
python generate.py \
    --input_voice_dir /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/train \
    --output_path /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/train \
    --bg_factor 10.0 \
    --n_outputs 10000
python generate.py \
    --input_voice_dir /projects/grail/audiovisual/datasets/VCTK-Corpus/wav48/test \
    --output_path /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/test \
    --bg_factor 10.0 \
    --n_outputs 2000
```


__Training__

Here's the sample code for training the network. This sample code uses 2 GPUs and trains the network with dataset
generated from `dataset_generators/1_voice_1_bg`. `--multiple_GPUs` and `--device_ids` are required for multiple GPUs.
Note that it is recommended that `batch_size` is approximately 20 per the number of GPUs used.

```
CUDA_VISIBLE_DEVICES=1,2 python train.py \
  /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/train \
  /projects/grail/audiovisual/datasets/DinTaiFung/1_voice_1_bg_hard/test \
  --name 1_voice_1_bg_hard \
  --n_sources 2 \
  --lr 1e-4 \
  --n_channels 6 \
  --batch_size 40 \
  --epochs 50 \
  --use_cuda \
  --shuffle \
  --device cuda:0 \
  --tensorboard \
  --multiple_GPUs --device_ids cuda:0 cuda:1
```


__Inference__

See `demo.ipynb`. This notebook will load a single checkpoint and run a forward-pass on a single example. 
The code is still under development and should be used with cautions.
