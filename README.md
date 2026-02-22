# EndoTrack

A multi-object tracker for colonoscopy video, built on multi-objective optimization with IoU, detection confidence, and appearance (ReID) cues.

## Requirements

```
Python >= 3.7
torch
onnxruntime
loguru
sympy
lap
inspyred
numpy
scipy
opencv-python
pycocotools
tqdm
tabulate
```

Install dependencies:

```bash
pip install torch onnxruntime loguru sympy lap inspyred numpy scipy opencv-python pycocotools tqdm tabulate
```

## Model Files

Two model files are required and are **not** included in this repository:

| Model | Config Key | Description |
|---|---|---|
| YOLO-OB detector | `weight path` | `yoloob/weights/yolov3_ckpt_15_ap_0.99570.pth` |
| ReID model (ONNX) | `onnx_model_path` | `reid-model/polyp-ce-adasp-db.onnx` |

To obtain the model files, contact the corresponding author: **seanyan622@outlook.com**

## Dataset

The original colonoscopy dataset is not publicly available. The dataset format follows **MOT20**. Organize your data as:

```
datasets/Union-tracking/
    <sequence_name>/
        img1/
            *.jpg
        det/
            det.txt       # public detections (MOT format)
        gt/
            gt.txt        # ground truth (optional)
annotations/
    val_half.json         # COCO-format annotation with video/frame metadata
    test.json
```

## Configuration

Edit `config/exp.ini` (INI format) or `colon.yaml` (YAML format) to set paths for your environment.

Key fields to update:

```ini
[detector]
public detector: false     # false = use YOLO-OB; true = use det.txt
oracle: true               # true = use ground-truth boxes
weight path: yoloob/weights/yolov3_ckpt_15_ap_0.99570.pth

[dataset]
dataset path: datasets/Union-tracking
ann path: annotations/val_half.json
dataset type: val

[features_args]
onnx_model_path: reid-model/polyp-ce-adasp-db.onnx

[output]
path: ./output/<exp_name>/
```

## Running

```bash
# Run tracking with INI config (recommended)
python tools/track.py --exp_cfg config/exp.ini

# Run tracking with YAML config (legacy entry point)
python nMO2Tracker.py --exp_cfg colon.yaml

# Optional flags
--debug      # visualize tracking results frame by frame
--log        # save logs to output directory
```

Results are written to the directory specified in `output.path` in MOT format.

## Evaluation

Evaluation is performed automatically after tracking. The tracker reports HOTA, MOTA, IDF1, and other standard MOT metrics via [TrackEval](https://github.com/JonathonLuiten/TrackEval).
