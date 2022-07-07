# physicsAwareTransformer
Official respository for [Array Camera Image Fusion using Physics-Aware Transformers](https://arxiv.org/abs/2207.02250), arXiv:2207.02250.

## TODOs
* document the repository in details and add comments

## Data synthesis
Method 1: Tested on Blender 2.92.0. Load *put-together.py*, change paths to the local machine. and run the script to generate the dataset.

Method 2: Download our dataset [here](https://doi.org/10.25422/azu.data.20217140)(powered by UA ReDATA)

Then run *trainingDataSynthesis/test/gen_train.py* to generate training patches.

## PAT pip requirements
under *PAT/requirements.txt*. The enviorment is exported from *pytorch/nvidia/20.01* docker on PUMA nodes of UA HPC.

## Train

```bash
pytorch train.py
```
OR
```bash
pytorch train_4inputs.py
```

## Test

```bash
pytorch demo_test.py --model_dir log_2inputs
```
OR
```bash
pytorch demo_test_4inputs.py --model_dir log_4inputs
```

## Inference

Use *Inference_as_a_whole_pittsburgh.ipynb* or *Inference_as_a_whole.ipynb*

## Courtesy
Some code is borrowed from https://github.com/The-Learning-And-Vision-Atelier-LAVA/PASSRnet.
