# SurroundBEV

A camera-only Bird's-Eye-View perception system built using the nuScenes autonomous-driving dataset.

SurroundBEV is an end-to-end perception engineering project covering multi-camera data ingestion, calibration-aware geometry, Bird's-Eye-View target generation, perception model development, robustness evaluation, failure analysis, and deployment optimization.

> **Project status:** In active development  
> **Current milestone:** Ego-centric BEV target generation

---

## Project Objective

The objective of SurroundBEV is to develop a camera-only perception pipeline that transforms synchronized surround-view camera images into an ego-centric Bird's-Eye-View representation for autonomous-driving applications.

The project is being developed incrementally:

```text
nuScenes surround-camera data
            ↓
Calibration-aware preprocessing
            ↓
Camera-to-BEV transformation
            ↓
BEV segmentation and 3D detection
            ↓
Scenario-based robustness evaluation
            ↓
Failure mining and targeted retraining
            ↓
Training and deployment optimization
```

---

## Current Capabilities

SurroundBEV currently supports:

- Loading synchronized images from all six nuScenes cameras
- Selecting samples through a configurable sample index
- Reading scene, sample, sensor, and camera metadata
- Creating labeled six-camera surround-view mosaics
- Retrieving camera calibration and ego-pose information
- Loading 3D ground-truth annotations for each camera
- Projecting 3D bounding boxes into camera images
- Filtering projected geometry using camera-relative depth
- Drawing the 12 edges of visible 3D bounding boxes
- Generating multi-camera geometry-validation mosaics
- Validating configuration, projection, and visualization behavior through automated tests

---

## Dataset

The project currently uses the **nuScenes `v1.0-mini` dataset**.

The dataset is stored outside this Git repository and is not included with the source code.

Expected local dataset structure:

```text
datasets/
└── v1.0-mini/
    ├── LICENSE
    ├── maps/
    ├── samples/
    ├── sweeps/
    └── v1.0-mini/
```

The dataset path is provided through `configs/mini.yaml`. No local dataset path is hardcoded in the Python implementation.

---

## Camera Arrangement

SurroundBEV uses six synchronized surround-view cameras in the following layout:

```text
CAM_FRONT_LEFT  | CAM_FRONT | CAM_FRONT_RIGHT
CAM_BACK_LEFT   | CAM_BACK  | CAM_BACK_RIGHT
```

This explicit ordering is shared by the data loader, visualization pipeline, tests, and generated mosaics.

---

## Completed Milestones

### Milestone 0: Six-Camera Sample Inspection

Milestone 0 established the multi-camera data-ingestion and visualization foundation.

The implementation:

- Loads a nuScenes sample using a configurable sample index
- Retrieves all six synchronized camera images
- Preserves an explicit and validated camera ordering
- Reports the sample token, scene name, timestamp, and image dimensions
- Preserves camera-image aspect ratios
- Generates a labeled 2-by-3 surround-view mosaic

Run:

```bash
python scripts/inspect_sample.py
```

Expected output:

```text
outputs/sample_mosaic.jpg
```

### Six-Camera Mosaic

outputs/sample_mosaic.jpg

---

### Milestone 1: Calibration-Aware 3D Box Projection

Milestone 1 introduced camera geometry and 3D annotation projection.

The implementation:

- Retrieves nuScenes ground-truth 3D boxes for each camera
- Uses calibrated sensor and ego-pose metadata
- Represents boxes in the corresponding camera coordinate frame
- Filters geometry based on configurable minimum camera depth
- Projects 3D box corners through the camera intrinsic matrix
- Draws the 12 edges of each visible 3D bounding box
- Optionally displays object-category labels
- Reports the number of visible boxes for every camera
- Produces an annotated six-camera mosaic

Run:

```bash
python scripts/project_boxes.py --config configs/mini.yaml
```

Expected output:

```text
outputs/sample_boxes_mosaic.jpg
```

### Calibration-Aware 3D Projection

outputs/sample_boxes_mosaic.jpg

---

## Geometry Pipeline

The 3D annotation projection follows this conceptual transformation sequence:

```text
3D annotation in global coordinates
                ↓
Global-to-ego transformation
                ↓
Ego-to-camera transformation
                ↓
3D point in camera coordinates
                ↓
Camera intrinsic projection
                ↓
2D image pixel coordinates
```

For a point represented in the camera coordinate frame, pinhole-camera projection is conceptually defined as:

```text
u = fx * x / z + cx
v = fy * y / z + cy
```

Where:

- `x`, `y`, and `z` represent the point in camera coordinates
- `fx` and `fy` represent focal lengths
- `cx` and `cy` represent the camera principal point
- `u` and `v` represent the projected image coordinates
- `z` represents camera-relative depth

Geometry that does not satisfy the configured minimum positive depth is excluded from image projection.

---

## Repository Structure

```text
SurroundBEV/
├── configs/
│   └── mini.yaml
├── docs/
│   └── PROJECT_SPEC.md
├── outputs/
│   ├── sample_mosaic.jpg
│   └── sample_boxes_mosaic.jpg
├── scripts/
│   ├── inspect_sample.py
│   └── project_boxes.py
├── src/
│   └── surround_bev/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── projection.py
│       └── viz.py
├── tests/
├── AGENTS.md
├── CLAUDE.md
├── README.md
└── .gitignore
```

The exact source files may evolve as additional milestones introduce map processing, BEV target generation, model architectures, training, and deployment.

---

## Requirements

The project currently uses:

- Python 3.11
- nuScenes development kit
- NumPy
- Pillow
- PyYAML
- pytest
- Ruff

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vigp17/SurroundBEV.git
cd SurroundBEV
```

### 2. Create a Python virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install the current dependencies

```bash
pip install nuscenes-devkit pillow pyyaml pytest ruff
```

### 5. Verify the nuScenes development kit

```bash
python -c "from nuscenes.nuscenes import NuScenes; print('nuScenes devkit available')"
```

---

## Configuration

Dataset, sample, output, and projection settings are stored in:

```text
configs/mini.yaml
```

Example configuration:

```yaml
dataset:
  root: /absolute/path/to/v1.0-mini
  version: v1.0-mini

sample_index: 0

output:
  file: outputs/sample_mosaic.jpg
  boxes_mosaic: outputs/sample_boxes_mosaic.jpg

projection:
  minimum_depth: 1.0
  line_width: 3
  show_labels: false
```

Before running the project, update:

```yaml
dataset:
  root: /absolute/path/to/v1.0-mini
```

to point to the local nuScenes mini dataset.

The configuration contains:

- `dataset.root`: local dataset location
- `dataset.version`: nuScenes dataset version
- `sample_index`: sample selected for visualization
- `output.file`: Milestone 0 mosaic output
- `output.boxes_mosaic`: Milestone 1 mosaic output
- `projection.minimum_depth`: minimum camera-relative projection depth
- `projection.line_width`: projected box-edge width
- `projection.show_labels`: enables or disables object-category labels

---

## Usage

### Generate the original six-camera mosaic

```bash
python scripts/inspect_sample.py
```

The command prints:

- Sample index
- Sample token
- Scene name
- Sample timestamp
- Camera channels
- Image dimensions
- Final output location

The generated image is written to:

```text
outputs/sample_mosaic.jpg
```

### Generate the 3D box-projection mosaic

```bash
python scripts/project_boxes.py --config configs/mini.yaml
```

The command prints:

- Sample index
- Sample token
- Scene name
- Number of visible boxes per camera
- Final output location

The generated image is written to:

```text
outputs/sample_boxes_mosaic.jpg
```

### Change the selected sample

Update the following field in `configs/mini.yaml`:

```yaml
sample_index: 0
```

For example:

```yaml
sample_index: 25
```

Then rerun the desired script.

---

## Testing

Run the complete automated test suite:

```bash
python -m pytest -v
```

Run static analysis:

```bash
ruff check .
```

The automated tests cover:

- Configuration loading
- Projection-configuration validation
- Dataset-path resolution
- Output-path resolution
- Explicit camera ordering
- Invalid sample-index handling
- Missing camera-channel handling
- Synthetic pinhole projection
- Rejection of geometry behind the camera
- 3D bounding-box edge definitions
- Mosaic grid validation
- Camera-image labeling
- Output-directory creation
- Image-saving behavior

---

## Development Roadmap

### Completed

- [x] Python development environment
- [x] nuScenes mini integration
- [x] Configurable dataset loading
- [x] Six-camera sample inspection
- [x] Synchronized camera mosaic
- [x] Scene and camera metadata reporting
- [x] Calibration-aware 3D annotation projection
- [x] Depth-based projection filtering
- [x] Multi-camera 3D box mosaic
- [x] Automated configuration tests
- [x] Automated geometry tests
- [x] Automated visualization tests

### Current Development

- [ ] Ego-centric semantic-map extraction
- [ ] Drivable-area BEV target generation
- [ ] World-to-BEV grid transformations
- [ ] BEV target visualization and validation

### Planned Model Development

- [ ] Geometry-only camera-to-BEV baseline
- [ ] Single-camera segmentation baseline
- [ ] Lightweight image encoder
- [ ] Learned multi-camera view transformation
- [ ] BEV semantic segmentation
- [ ] Camera-only 3D object detection
- [ ] Multi-task detection and segmentation
- [ ] Multi-task loss-balancing experiments

### Planned Robustness Evaluation

- [ ] Detection and segmentation performance by distance
- [ ] Day and night performance analysis
- [ ] Object-scale and visibility analysis
- [ ] Missing-camera robustness evaluation
- [ ] Camera-calibration perturbation studies
- [ ] Automated failure collection
- [ ] Failure clustering
- [ ] Targeted hard-example selection
- [ ] Retraining and regression evaluation

### Planned Training and Deployment

- [ ] Mixed-precision training
- [ ] Experiment tracking
- [ ] Checkpoint and resume support
- [ ] PyTorch Distributed Data Parallel training
- [ ] Training-pipeline profiling
- [ ] ONNX export
- [ ] ONNX output-parity validation
- [ ] TensorRT optimization
- [ ] Latency, throughput, and memory benchmarking

---

## Current Limitations

- The project currently uses nuScenes `v1.0-mini`.
- Projected annotations are ground truth rather than model predictions.
- BEV target generation is not yet implemented.
- A learned camera-to-BEV model is not yet implemented.
- Temporal camera fusion is not yet implemented.
- Current outputs are intended for geometry validation and data inspection.
- Full-scale CUDA training and TensorRT benchmarking will require an NVIDIA GPU environment.
- Current results should not be interpreted as perception-model accuracy results.

---

## Design Principles

SurroundBEV follows the following engineering principles:

- Dataset paths are configuration-driven.
- Camera ordering is explicit and tested.
- Coordinate frames and physical units are documented.
- Dataset access, projection, visualization, and orchestration remain separate.
- Geometry functions are validated with synthetic tests.
- Visual outputs are manually inspected in addition to automated testing.
- New milestones must preserve existing functionality.
- Completed capabilities are clearly separated from planned capabilities.

---

## Intended Outcome

The final system is intended to demonstrate ownership of the complete camera-perception lifecycle:

```text
Data preparation
      ↓
Calibration-aware geometry
      ↓
BEV model development
      ↓
Scalable model training
      ↓
Scenario-based evaluation
      ↓
Failure-mode analysis
      ↓
Targeted data improvement
      ↓
Optimized deployment
```

---

## Data and Licensing

The nuScenes dataset is not distributed with this repository.

Users must download the dataset separately and comply with the applicable nuScenes terms and licensing requirements.

Third-party libraries and tools used by this project remain subject to their respective licenses.