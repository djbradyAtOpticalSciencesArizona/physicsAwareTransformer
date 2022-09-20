# PAT Inference Workflow
## -- Using a multispectral array as an example
#### Qian Huang
#### 09/20/22

Cam4 was unfiltered and selected as the alpha view. Cam1, cam2, and cam3 were with 600nm, 550nm, 450nm filters, respectively.

## 1. Use MATLAB Stereo Vision Toolbox to calibrate the system

Collect fundamental matrixes of F1: 4->1, F2: 4->2, and F3: 4->3. In case images are of different spatial resolution, pad the right and bottom edges of the smaller image to be the same resolution. This operation does not affect the fundamental matrix (but may impact other parameters).

## 2. Inspect Calibrated Data

See CalibratedDataInspection.ipynb. The example shows the investigation of epipolar lines and truncated epipolar lines. Truncation is based on rescaling and translation.

## 3. Generate Physical Receptive Fields

See gen_dataset.py. Its parameters followed CalibratedDataInspection.ipynb.

## 4. Inference

See Inference_as_a_whole_multispectral.ipynb. This notebook relies on network scripts in the parent folder.