# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Fire/Smoke False Positive Labeling System** - A web application for rapidly creating training datasets. Upload videos for automatic scanning with YOLO models, group similar events using ReID, and quickly label true fire/smoke vs false positives.

The project has been refactored from a monolithic 1200+ line application into a modular architecture with clear separation of concerns.

## Development Commands

### Environment Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Development mode installation (optional)
pip install -e .
```

### Code Quality
```bash
# Format code with black
black app.py core/ ui/ tools/

# Check code with flake8  
flake8 app.py core/ ui/ tools/

# Run tests (if available)
pytest tests/
```

### Running the Application
```bash
# Main web interface
python app.py

# Using startup script
./run.sh

# Alternative applications
python app_three_tabs.py    # Three-tab system with training/inference
python start_reid_labeling.py    # Alternative ReID labeling interface
```

### Version Switching
```bash
./switch-to-new.sh    # Switch to modular version
./switch-to-old.sh    # Switch to monolithic version
```

## Architecture Overview

### Core Components

The system follows a modular architecture with clear separation between core logic and UI:

```
app.py (63 lines) → Bootstraps the application
├── core/
│   ├── analyzer.py      → Video processing, YOLO detection, ReID grouping
│   ├── labeling.py      → Label management, progress tracking, data export
│   ├── inference.py     → Temporal classification inference
│   ├── training.py      → Model training pipeline
│   └── models/          → Deep learning models for temporal classification
│       ├── data_utils.py       → Dataset utilities and preprocessing
│       ├── temporal_classifier.py → Time-series classification models
│       └── temporal_trainer.py → Training pipeline for temporal models
└── ui/
    ├── builder.py              → Gradio interface construction
    ├── builder_new.py          → Three-tab interface with training/inference
    ├── interface.py            → User interaction logic
    ├── training_controller.py  → Training workflow management
    └── inference_controller.py → Inference workflow management
```

### Key Features
- **Detection**: Ultralytics YOLO (pretrained model at `best.pt`)
- **Tracking**: Supervision library for object tracking and ReID grouping
- **Interface**: Gradio web framework with automatic frame carousel
- **Export**: Structured dataset export as ZIP files
- **Temporal Classification**: Deep learning models for time-series fire/smoke analysis
- **Model Training**: Built-in training pipeline with multiple backbone architectures
- **Acceleration**: Supports Mac MPS, CUDA, and CPU inference

### Data Flow

#### Labeling Workflow
1. Upload video → Extract frames
2. Run YOLO detection on frames
3. Group detections via ReID similarity
4. Present event thumbnails for labeling
5. Export labeled dataset as ZIP

#### Training Workflow
1. Upload labeled datasets (ZIP files)
2. Configure training parameters and model architecture
3. Train temporal classification models
4. Monitor training progress and metrics
5. Save trained models for inference

#### Inference Workflow
1. Load trained temporal classification model
2. Upload video for temporal analysis
3. Generate classification results with confidence scores
4. Export inference results

### Output Structure

#### Labeling Dataset Output
```
dataset/
└── export_YYYYMMDD_HHMMSS.zip
    ├── true_positive/          # Real fire/smoke events
    │   ├── event_0/
    │   │   ├── crop_000_1.2s.jpg
    │   │   └── metadata.json
    └── false_positive/         # False positive events
```

#### Training Output
```
runs/
└── temporal_training/
    └── train_YYYYMMDD_HHMMSS/
        ├── best_model.pth      # Best trained model
        ├── config.yaml         # Training configuration
        ├── training_log.txt    # Training progress log
        └── metrics/            # Training metrics and plots
```

#### Inference Output
```
inference_workspace/
└── inference_YYYYMMDD_HHMMSS/
    ├── results.json           # Classification results
    ├── confidence_scores.csv  # Detailed confidence scores
    └── visualizations/        # Result visualizations
```

## Important Parameters

Adjustable parameters in the codebase:
- `conf=0.3`: Detection confidence threshold (analyzer.py:144)
- `sample_interval`: Video sampling interval (analyzer.py:80)
- ReID grouping threshold (analyzer.py:234)
- Min/max frames per event: Configurable in UI

## Multi-Application System

The project contains multiple entry points:
- `app.py`: Main simplified labeling interface (modular version)
- `app_old.py`: Original monolithic version (1200+ lines)  
- `app_new.py`: Intermediate refactored version
- `app_three_tabs.py`: **Comprehensive three-tab system with labeling/training/inference**
- `start_reid_labeling.py`: Alternative ReID labeling interface

### Training Progress Monitoring

When training is in progress, monitor status through:
- **Training logs**: Check `runs/temporal_training/train_*/training_log.txt`
- **Process monitoring**: Use `ps aux | grep python` to see active training processes
- **GPU usage**: Use `nvidia-smi` (if CUDA available) or Activity Monitor (if MPS)
- **Training workspace**: Check `training_workspace/` for current datasets and configs

## Environment Variables

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Allow MPS fallback to CPU
export OMP_NUM_THREADS=4              # Limit thread count
```

## Temporal Classification Models

### Supported Model Architectures

The system supports multiple backbone architectures optimized for different use cases:

#### Low Latency (Edge/Real-time)
- `temporal_mobilenetv3_large`: MobileNetV3-Large + Attention - Mobile optimized
- `temporal_ghostnet_100`: GhostNet-100 + Attention - Lightweight efficient
- `temporal_efficientnet_b0`: EfficientNet-B0 + Attention - Classic lightweight

#### Balanced (Recommended)
- `temporal_convnext_tiny`: ConvNeXt-Tiny + Attention - **Recommended starting point**
- `temporal_efficientnetv2_s`: EfficientNetV2-S + Attention - Speed/accuracy balance
- `temporal_resnet50`: ResNet50 + Attention - Proven backbone

#### High Accuracy
- `temporal_convnext_base`: ConvNeXt-Base + Attention - High accuracy
- `temporal_swin_base_patch4_window7_224`: Swin Transformer - SOTA performance

### Training Commands

```bash
# Monitor training progress
tail -f runs/temporal_training/train_*/training_log.txt

# Check active training processes
ps aux | grep python | grep training

# Monitor GPU usage (if available)
watch -n 1 nvidia-smi
```

## Current Status

- ✅ Modular web application architecture
- ✅ Video upload and concurrent analysis
- ✅ ReID event grouping implemented  
- ✅ Fast binary classification interface
- ✅ Dataset export functionality
- ✅ Three-tab system with training/inference capabilities
- ✅ Temporal classification model training pipeline
- ✅ Multiple backbone architecture support
- ⚠️ No formal test framework setup (pytest in requirements but no tests/ directory)
- 🔄 Active refactoring - multiple app versions coexist