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

# Milestone 2: Ego-Centric BEV Target Generation

## Objective

Generate an ego-centric semantic Bird's-Eye-View target from nuScenes
map data for the configured sample.

The initial target will represent the local drivable area around the
ego vehicle.

## Deliverable

```text
outputs/sample_bev_target.png
```

## Coordinate Convention

The BEV representation must use the following convention:

- Origin: ego-vehicle position
- Positive X: ego forward
- Positive Y: ego left
- Distance unit: meters
- Image top: ego forward
- Image bottom: ego backward
- Image left: ego left
- Image right: ego right
- Array indexing: row first, column second

Conceptually:

```text
                    Ego forward
                         +X
                          ↑
                          |
             +Y left  ← Ego →  right -Y
                          |
                          ↓
                       backward
```

## Initial Semantic Target

The first target contains one semantic layer:

- `drivable_area`

The raw target must be a binary mask containing only:

```text
0 = not drivable
1 = drivable
```

## Configured BEV Extent

The initial BEV extent will be configured as:

```text
X minimum: -30 meters
X maximum:  50 meters
Y minimum: -30 meters
Y maximum:  30 meters
Resolution: 0.25 meters per pixel
```

This represents:

- 30 meters behind the ego vehicle
- 50 meters ahead of the ego vehicle
- 30 meters to the left
- 30 meters to the right

The expected dimensions are:

```text
Forward and backward dimension:
(50 - (-30)) / 0.25 = 320 pixels

Left and right dimension:
(30 - (-30)) / 0.25 = 240 pixels
```

Expected mask shape:

```text
320 rows × 240 columns
```

The ego vehicle will not be vertically centered because the configured
region contains more distance ahead of the vehicle than behind it.

## Requirements

The implementation must:

- Read the selected sample from configuration
- Determine the scene associated with the sample
- Determine the log associated with the scene
- Determine the map location from the log
- Avoid hardcoding a map location
- Retrieve an ego pose associated with the configured sample
- Document which sensor record provides the reference ego pose
- Initialize the nuScenes map API using the discovered location
- Extract the configured semantic map layers
- Generate a local map region around the ego vehicle
- Transform global map information into the ego coordinate frame
- Rotate the local map so ego forward points toward image top
- Map ego left toward image left
- Preserve the configured physical extent
- Preserve the configured metric resolution
- Generate a binary drivable-area target
- Keep the raw target separate from visualization overlays
- Mark the ego position only in a visualization copy
- Mark the ego-forward direction only in a visualization copy
- Report map and geometry metadata
- Preserve Milestone 0 behavior
- Preserve Milestone 1 behavior

## Required Console Output

The command should report:

- Sample index
- Sample token
- Scene name
- Map location
- Ego global X coordinate
- Ego global Y coordinate
- Ego yaw in degrees
- BEV metric extent
- BEV resolution
- Output mask shape
- Output path

## Required Command

```bash
python scripts/generate_bev_target.py --config configs/mini.yaml
```

## Raw Target and Visualization

The implementation should distinguish between:

### Raw Training Target

A binary array containing only `0` and `1`.

The raw target must not contain:

- Ego marker
- Heading arrow
- Text
- Grid lines
- Decorative colors

### Visualization Output

A human-readable PNG that may include:

- Drivable-area coloring
- Ego-vehicle origin marker
- Ego-forward direction marker
- Optional coordinate axes
- Optional metadata text

The visualization must be created from a copy of the raw target.

---

## Configuration Requirements

The `configs/mini.yaml` file must contain:

```yaml
bev:
  x_min: -30.0
  x_max: 50.0
  y_min: -30.0
  y_max: 30.0
  resolution: 0.25
  layers:
    - drivable_area
  output_file: outputs/sample_bev_target.png
```

The Python implementation must not 