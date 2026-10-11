# SurroundBEV Project Specification

## Project Goal

Build a camera-only Bird's-Eye-View perception system using the
nuScenes autonomous-driving dataset.

The completed system will cover:

```text
Multi-camera data ingestion
            ↓
Calibration-aware geometry
            ↓
Ego-centric BEV target generation
            ↓
Camera-to-BEV model development
            ↓
Robustness and failure analysis
            ↓
Training and deployment optimization
```

---

## Completed Milestones

### Milestone 0: Six-Camera Sample Inspection

#### Objective

Load one synchronized nuScenes sample and create a labeled six-camera
surround-view mosaic.

#### Completed Capabilities

- Loaded samples through a configurable sample index
- Retrieved all six synchronized camera images
- Preserved an explicit camera ordering
- Reported scene, sample, timestamp, and image metadata
- Preserved camera-image aspect ratios
- Generated a labeled 2-by-3 surround-view mosaic
- Added automated configuration and visualization tests

#### Deliverable

```text
outputs/sample_mosaic.jpg
```

---

### Milestone 1: Calibration-Aware 3D Bounding-Box Projection

#### Objective

Project nuScenes ground-truth 3D bounding boxes onto all six camera
images using calibrated sensor and ego-pose metadata.

#### Completed Capabilities

- Retrieved ground-truth 3D annotations for each camera
- Used calibrated sensor and ego-pose metadata
- Represented annotations in the appropriate camera coordinate frame
- Projected 3D box corners using camera intrinsic parameters
- Rejected geometry behind the camera
- Drew the 12 edges of visible 3D bounding boxes
- Supported optional object-category labels
- Reported visible-box counts by camera
- Generated an annotated six-camera mosaic
- Added automated configuration, geometry, and visualization tests

#### Deliverable

```text
outputs/sample_boxes_mosaic.jpg
```

---

## Current Milestone

## Current Milestone

# Milestone 3: Geometry-Only Camera-to-BEV Projection

## Objective

Project camera imagery into an ego-centric Bird's-Eye-View space
using camera calibration and geometric transformations only.

No neural networks should be used.

## Deliverables

```text
outputs/ipm_front_camera.png
outputs/ipm_bev.png
```

## Concept

Transform:

```text
Image Pixels
      ↓
Camera Rays
      ↓
Ground Plane Intersection
      ↓
Ego Coordinates
      ↓
BEV Grid
```

using calibration information provided by nuScenes.

## Requirements

The implementation must:

- Reuse the existing configuration system.
- Reuse the existing nuScenes dataset loading.
- Reuse the existing ego-coordinate convention established in Milestone 2.
- Use camera calibration from nuScenes.
- Project image pixels onto the ground plane.
- Generate an inverse-perspective-mapping (IPM) visualization.
- Generate an ego-centric BEV representation.
- Support:
  - CAM_FRONT_LEFT
  - CAM_FRONT
  - CAM_FRONT_RIGHT
  - CAM_BACK_LEFT
  - CAM_BACK
  - CAM_BACK_RIGHT
- Create a stitched BEV visualization from all cameras.
- Preserve Milestone 0 functionality.
- Preserve Milestone 1 functionality.
- Preserve Milestone 2 functionality.

## Required Commands

```bash
python scripts/generate_ipm_bev.py --config configs/mini.yaml
```

## Required Outputs

### Single Camera Projection

```text
outputs/ipm_front_camera.png
```

### Multi-Camera Projection

```text
outputs/ipm_bev.png
```

## Coordinate Convention

Use the same convention as Milestone 2:

- Origin: ego vehicle
- Positive X: ego forward
- Positive Y: ego left
- Units: meters
- Image top: ego forward
- Image left: ego left

## Required Tests

Implement tests for:

- pixel-to-ground projection
- ego-to-grid transformation
- output dimensions
- invalid calibration
- invalid resolution
- unsupported camera channel
- camera ordering preservation

## Non-Goals

Do not implement:

- Neural networks
- PyTorch
- BEV segmentation
- Object detection
- Training
- TensorRT
- Temporal fusion

## Definition of Done

Milestone 3 is complete when:

1. A front-camera IPM image is generated.

```text
outputs/ipm_front_camera.png
```

2. A stitched multi-camera BEV projection is generated.

```text
outputs/ipm_bev.png
```

3. Existing Milestones 0–2 continue to work.

4. All tests pass.

5. Ruff passes.

6. Camera-to-ground geometry is documented and validated.