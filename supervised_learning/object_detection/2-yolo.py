#!/usr/bin/env python3
"""Module for YOLO v3 object detection - Task 2: Filter Boxes."""
import numpy as np
import tensorflow as tf


class Yolo:
    """Class that uses the Yolo v3 algorithm to perform object detection."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize the Yolo object detection model.

        Args:
            model_path: path to where a Darknet Keras model is stored
            classes_path: path to where the list of class names is found
            class_t: float representing the box score threshold for filtering
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
                containing all of the anchor boxes
        """
        self.model = tf.keras.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Process the outputs from the Darknet model.

        Args:
            outputs: list of numpy.ndarrays containing predictions from
                Darknet model for a single image, each of shape
                (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing the image's original size
                [image_height, image_width]

        Returns:
            tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, n_anchors, _ = output.shape

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            col = np.arange(grid_w).reshape(1, grid_w, 1)
            row = np.arange(grid_h).reshape(grid_h, 1, 1)
            col = np.broadcast_to(col, (grid_h, grid_w, n_anchors))
            row = np.broadcast_to(row, (grid_h, grid_w, n_anchors))

            b_x = (1 / (1 + np.exp(-t_x)) + col) / grid_w
            b_y = (1 / (1 + np.exp(-t_y)) + row) / grid_h

            anchors_w = self.anchors[i, :, 0]
            anchors_h = self.anchors[i, :, 1]

            b_w = (np.exp(t_w) * anchors_w) / input_w
            b_h = (np.exp(t_h) * anchors_h) / input_h

            x1 = (b_x - b_w / 2) * image_width
            y1 = (b_y - b_h / 2) * image_height
            x2 = (b_x + b_w / 2) * image_width
            y2 = (b_y + b_h / 2) * image_height

            box = np.stack([x1, y1, x2, y2], axis=-1)
            boxes.append(box)

            confidence = 1 / (1 + np.exp(-output[..., 4:5]))
            box_confidences.append(confidence)

            class_probs = 1 / (1 + np.exp(-output[..., 5:]))
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter bounding boxes by score threshold.

        Args:
            boxes: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 4)
            box_confidences: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, 1)
            box_class_probs: list of numpy.ndarrays of shape
                (grid_height, grid_width, anchor_boxes, classes)

        Returns:
            tuple of (filtered_boxes, box_classes, box_scores):
                filtered_boxes: numpy.ndarray of shape (?, 4)
                box_classes: numpy.ndarray of shape (?,)
                box_scores: numpy.ndarray of shape (?)
        """
        all_boxes = []
        all_classes = []
        all_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_probs
            box_class = np.argmax(scores, axis=-1)
            box_score = np.max(scores, axis=-1)

            mask = box_score >= self.class_t

            all_boxes.append(box[mask])
            all_classes.append(box_class[mask])
            all_scores.append(box_score[mask])

        filtered_boxes = np.concatenate(all_boxes, axis=0)
        box_classes = np.concatenate(all_classes, axis=0)
        box_scores = np.concatenate(all_scores, axis=0)

        return filtered_boxes, box_classes, box_scores
