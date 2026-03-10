# Transfer Learning

This project demonstrates how to apply **transfer learning** to classify the
CIFAR-10 dataset using a pretrained Keras application (EfficientNetB0).

## Objectives

- Understand what transfer learning is and when to use it
- Learn what fine-tuning means and how to freeze/unfreeze layers
- Apply a Keras application to a downstream classification task

## Project Structure

| File | Description |
|---|---|
| `0-transfer.py` | Training script + `preprocess_data` helper |

## Approach

1. **Base model** – EfficientNetB0 pretrained on ImageNet, `include_top=False`,
   `pooling='avg'`.  All base layers start frozen.
2. **Resize layer** – A `Lambda` layer resizes 32×32 CIFAR-10 images to
   224×224 (EfficientNetB0's native resolution) before the base model.
3. **Feature caching (Phase 1)** – The frozen base model's output is computed
   *once* for the entire dataset, then a small classification head is trained
   on these cached feature vectors.  This is extremely fast.
4. **Fine-tuning (Phase 2)** – The last 30 layers of EfficientNetB0 are
   unfrozen and the full model is fine-tuned end-to-end on the raw images
   with data augmentation (horizontal flip, rotation, zoom, translation).

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2

## Usage

```bash
# Train and save cifar10.h5
./0-transfer.py

# Evaluate the saved model
./0-main.py
```

## Results

Target validation accuracy: **≥ 87%**

Expected output (similar to):
```
10000/10000 [...] - loss: 0.3329 - acc: 0.8864
```
