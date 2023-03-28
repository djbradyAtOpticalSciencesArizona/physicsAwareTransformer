# Physics-Aware Transformers
Official respository for Qian Huang,  Minghao Hu,  David J. Brady,  [*Array Camera Image Fusion using Physics-Aware Transformers*](https://doi.org/10.2352/J.ImagingSci.Technol.2022.66.6.060401) in Journal of Imaging Science and Technology, 2022.

## Update -- Mar 27, 2023
The colab notebook of the experiment on wide-narrow field system has been uploaded to [PAT/Inference_as_a_whole_dual_vision.ipynb](PAT/Inference_as_a_whole_dual_vision.ipynb).

## Update -- Feb 6, 2023
A colab notebook to play the pretrained PAT with the local receptive field (similar to the receptive field of CNNs) has been uploaded to [PAT/Inference_pat_local_receptive_field.ipynb](PAT/Inference_pat_local_receptive_field.ipynb).

## Data synthesis
Method 1: Open *put-together.py* in Blender 2.92.0, change paths to the local machine, and run the script to generate the dataset.

Method 2: Download our dataset [here](https://doi.org/10.25422/azu.data.20217140) (powered by UA ReDATA)

Then run *trainingDataSynthesis/test/gen_patches.py* to generate patches.

## PAT pip requirements
under *PAT/requirements.txt*. The enviorment is exported from *pytorch/nvidia/20.01* docker on PUMA nodes of UA HPC.

## Train

```bash
pytorch train.py --trainset_dir [path to your training patches] --validset_dir [path to your validation patches]
```
OR
```bash
pytorch train_4inputs.py --trainset_dir [path to your training patches] --validset_dir [path to your validation patches]
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
