# physicsAwareTransformer
Official respository for *Array Camera Image Fusion using Physics-Aware Transformers*.

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

Use Inference_as_a_whole_pittsburgh.ipynb or Inference_as_a_whole.ipynb

## Courtesy
Some code is borrowed from https://github.com/The-Learning-And-Vision-Atelier-LAVA/PASSRnet.
