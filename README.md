# physicsAwareTransformer
Official respository for [Array Camera Image Fusion using Physics-Aware Transformers](https://arxiv.org/abs/2207.02250), arXiv:2207.02250.

## TODOs
* upload dataset -> waiting for UA reData approval
* document the repository and add comments

## Data synthesis
Tested on Blender 2.92.0. Load *put-together.py*, change paths to the local machine. and run the script to generate the dataset. Run *trainingDataSynthesis/test/gen_train.py* to generate training patches.

## PAT requirements
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
