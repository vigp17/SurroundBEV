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

#### Command

```bash
python scripts/inspect_sample.py --config configs/mini.yaml
```

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

#### Command

```bash
python scripts/project_boxes.py --config configs/mini.yaml
```

#### Deliverable

```text
outputs/sample_boxes_mosaic.jpg
```

---

### Milestone 2: Ego-Centric BEV Target Generation

#### Objective

Rasterize the nuScenes drivable-area map layer into an ego-centric
Bird's-Eye-View training target for one sample.

#### Completed Capabilities

- Reused the existing configuration system through a `bev` section
- Discovered the semantic-map name through `sample → scene → log → location`,
  with no hardcoded map name
- Anchored the BEV window on the `LIDAR_TOP` keyframe ego pose
- Computed the global axis-aligned bounds covering the rotated ego window
- Loaded drivable-area polygons intersecting those bounds
- Transformed global polygons into the ego frame using the planar ego heading
- Rasterized a binary mask by testing each cell center
- Excluded polygon holes from the drivable class
- Kept the raw `uint8` training mask separate from its visualization
- Added automated grid, transform, and rasterization tests

#### Coordinate Convention

This milestone established the convention reused by every later milestone:

- Origin: ego vehicle
- Positive X: ego forward
- Positive Y: ego left
- Units: meters
- Image top: ego forward
- Image left: ego left

```text
row = (x_max − x) / resolution
col = (y_max − y) / resolution
```

#### Prerequisite

This milestone requires the nuScenes map expansion pack, which is a separate
download. The per-location JSON files must be present in
`<dataset_root>/maps/expansion/`.

#### Command

```bash
python scripts/generate_bev_target.py --config configs/mini.yaml
```

#### Deliverables

```text
outputs/sample_bev_target.npy
outputs/sample_bev_target.png
```

The `.npy` file is the raw `uint8` mask containing only `0` and `1`. The PNG is
a visualization copy carrying the ego marker and heading arrow, which are never
written into the training target.

---

### Milestone 3: Geometry-Only Camera-to-BEV Projection

#### Objective

Project camera imagery into an ego-centric Bird's-Eye-View space
using camera calibration and geometric transformations only.

No neural networks were used.

#### Concept

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

#### Completed Capabilities

- Reused the existing configuration system through an `ipm` section
- Reused the existing nuScenes dataset loading
- Reused the Milestone 2 ego-coordinate convention
- Built pinhole rays from the nuScenes camera intrinsic matrix
- Expressed each camera in the Milestone 2 ego frame, including the motion
  between that camera's ego pose and the `LIDAR_TOP` reference pose
- Intersected camera rays with the ego ground plane `z = 0`
- Rejected rays pointing upward and intersections behind the camera
- Generated an inverse-perspective-mapping visualization for a single camera
- Generated a stitched ego-centric BEV representation from all six cameras
- Resolved overlapping cameras by keeping the nearest one per cell
- Supported all six surround channels:
  - CAM_FRONT_LEFT
  - CAM_FRONT
  - CAM_FRONT_RIGHT
  - CAM_BACK_LEFT
  - CAM_BACK
  - CAM_BACK_RIGHT
- Preserved Milestone 0, 1, and 2 functionality

#### Rendering Direction

The forward chain above defines the geometry. The renderer evaluates its exact
inverse — each BEV cell center is projected into the image — because the two are
equivalent for a single plane, and the inverse fills the grid without leaving
gaps between splatted pixels.

#### Ground-Plane Basis

The nuScenes ego frame has its origin at road level under the rear axle, and
`ego_pose.translation` has `z = 0`. The six cameras sit roughly 1.48 m to 1.58 m
above that plane, so `z = 0` is the road surface rather than an arbitrary datum.

#### Command

```bash
python scripts/generate_ipm_bev.py --config configs/mini.yaml
```

#### Deliverables

```text
outputs/ipm_front_camera.png
outputs/ipm_bev.png
```

#### Tests Delivered

- pixel-to-ground projection
- pixel-to-ground and ground-to-pixel round trip
- ego-to-grid transformation
- output dimensions
- invalid calibration
- invalid resolution
- unsupported camera channel
- camera ordering preservation
- camera pose expressed in the yaw-only reference frame

#### Non-Goals Observed

The following were explicitly out of scope and were not implemented:

- Neural networks
- PyTorch
- BEV segmentation
- Object detection
- Training
- TensorRT
- Temporal fusion

#### Definition of Done

| # | Criterion | Status |
| --- | --- | --- |
| 1 | A front-camera IPM image is generated | Met — `outputs/ipm_front_camera.png` |
| 2 | A stitched multi-camera BEV projection is generated | Met — `outputs/ipm_bev.png` |
| 3 | Existing Milestones 0–2 continue to work | Met — all four scripts run and reproduce their outputs |
| 4 | All tests pass | Met |
| 5 | Ruff passes | Met |
| 6 | Camera-to-ground geometry is documented and validated | Met — see `src/surround_bev/ipm.py` module docstring and `tests/test_ipm.py` |

---

## Current Milestone

No milestone is currently in progress. Milestone 3 closed the geometry-only
portion of the project.

The next candidates, in the order listed under **Next Up** in the README, are:

1. Single-camera segmentation baseline
2. Lightweight image encoder
3. Learned multi-camera view transformation

A specification for the next milestone will be added to this document before
implementation begins.

---

## Invariants for Future Milestones

Every later milestone must preserve the following:

- Dataset paths and output paths come from the YAML config; nothing is hardcoded
  in the Python implementation.
- Map names are discovered from the dataset's own log records.
- The six camera channels keep their explicit order.
- The ego-centric convention from Milestone 2 is reused unchanged: origin at the
  ego vehicle, +x forward, +y left, meters, image top forward, image left left.
- Raw training targets stay separate from their visualizations.
- Coordinate frames and physical units stay documented at the module level.
- Milestones 0 through 3 continue to run and reproduce their outputs.
- `pytest` and `ruff check .` both pass.
