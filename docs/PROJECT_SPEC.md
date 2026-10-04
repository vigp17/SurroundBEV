# SurroundBEV

## Goal

Build a camera-only Bird's-Eye-View perception system using nuScenes.

## Completed Milestones

### Milestone 0: Six-Camera Sample Inspection

- Loaded synchronized images from all six nuScenes cameras.
- Created a labeled 2-by-3 surround-camera mosaic.
- Saved the output to `outputs/sample_mosaic.jpg`.

## Current Milestone

### Milestone 1: 3D Bounding-Box Projection

## Objective

Project nuScenes 3D ground-truth bounding boxes onto all six
camera images using calibrated sensor metadata.

## Deliverable

`outputs/sample_boxes_mosaic.jpg`

## Requirements

- Use the same six-camera ordering as Milestone 0.
- Select the sample through configuration.
- Use nuScenes calibration and ego-pose metadata.
- Draw visible ground-truth 3D boxes on each camera image.
- Exclude boxes that are fully behind the camera.
- Preserve image aspect ratio.
- Display the camera channel on every panel.
- Report the number of visible boxes per camera.
- Preserve the existing Milestone 0 functionality.

## Non-Goals

Do not implement:

- Object detection
- Model training
- BEV projection
- BEV model
- Semantic segmentation
- Transformers
- TensorRT
- Distributed training
