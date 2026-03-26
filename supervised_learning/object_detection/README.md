# Object Detection

This project implements object detection using the YOLO v3 (You Only Look Once) algorithm with a pre-trained Darknet Keras model.

## Requirements

- Python 3.9
- numpy 1.25.2
- tensorflow 2.15
- opencv-python 4.9.0.80

## Installation

```bash
pip install --user opencv-python==4.9.0.80
```

## Files

| File | Description |
|------|-------------|
| `0-yolo.py` | Initialize the Yolo class with model, class names, thresholds, and anchors |
| `1-yolo.py` | Add `process_outputs` to decode raw model predictions into bounding boxes |
| `2-yolo.py` | Add `filter_boxes` to threshold boxes by confidence score |
| `3-yolo.py` | Add `non_max_suppression` to remove overlapping duplicate detections |
| `4-yolo.py` | Add `load_images` static method to load all images from a folder |
| `5-yolo.py` | Add `preprocess_images` to resize and normalize images for the model |
| `6-yolo.py` | Add `show_boxes` to display detection results with labeled bounding boxes |
| `7-yolo.py` | Add `predict` to run full inference pipeline on a folder of images |

## Usage

```python
import numpy as np
Yolo = __import__('7-yolo').Yolo

anchors = np.array([[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]])

yolo = Yolo('yolo.h5', 'coco_classes.txt', 0.6, 0.5, anchors)
predictions, image_paths = yolo.predict('yolo_images/yolo/')
```

## Key Concepts

- **YOLO v3**: Single-shot detector that divides the image into a grid and predicts bounding boxes and class probabilities simultaneously
- **Anchor Boxes**: Pre-defined box shapes used as references for prediction
- **IOU (Intersection over Union)**: Metric used to measure overlap between bounding boxes
- **Non-Max Suppression (NMS)**: Algorithm that removes redundant overlapping boxes, keeping only the highest-scoring detection per object
- **mAP (mean Average Precision)**: Standard metric for evaluating object detection models
