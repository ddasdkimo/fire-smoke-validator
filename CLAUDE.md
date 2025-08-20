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

# Docker environment (alternative)
pip install -r requirements-docker.txt
```

**Note**: The project includes a `setup.py` for packaging as "temporal_classification_model" - a package for temporal fire/smoke classification models.

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

# Application launcher (used by Docker)
python start.py              # Interactive launcher
python start.py original     # Launch simple labeling interface
python start.py three-tabs   # Launch comprehensive system
```

### Version Switching
```bash
./switch-to-new.sh    # Switch to modular version
./switch-to-old.sh    # Switch to monolithic version (from archive/)
```

## Project Structure

```
fire-smoke-validator/
├── app.py                       # Main simplified labeling interface
├── app_three_tabs.py           # Comprehensive three-tab system
├── start.py                     # Application launcher (for Docker)
├── core/                       # Core business logic modules
│   ├── analyzer.py            # Video processing, YOLO detection, ReID grouping
│   ├── labeling.py            # Label management, progress tracking, data export
│   ├── inference.py           # Temporal classification inference
│   ├── training.py            # Model training pipeline
│   └── models/                # Deep learning models
│       ├── data_utils.py      # Dataset utilities and preprocessing
│       ├── temporal_classifier.py # Time-series classification models
│       └── temporal_trainer.py # Training pipeline for temporal models
├── ui/                        # User interface components
│   ├── builder.py             # Gradio interface construction
│   ├── builder_new.py         # Three-tab interface with training/inference
│   ├── interface.py           # User interaction logic
│   ├── training_controller.py # Training workflow management
│   └── inference_controller.py # Inference workflow management
├── tests/                     # Testing and debugging utilities
├── archive/                   # Archived legacy versions
└── docs/                     # Documentation files
```

## Architecture Overview

### Core Components

The system follows a modular architecture with clear separation between core logic and UI. The main applications are lightweight bootstrapping scripts that coordinate between the `core/` business logic modules and `ui/` interface components.

### Key Features
- **Detection**: Ultralytics YOLO (pretrained model at `best.pt`)
- **Tracking**: Supervision library for object tracking and ReID grouping
- **Interface**: Gradio web framework with automatic frame carousel
- **Export**: Structured dataset export as ZIP files
- **Temporal Classification**: Deep learning models for time-series fire/smoke analysis
- **Model Training**: Built-in training pipeline with multiple backbone architectures
- **TensorBoard Integration**: Comprehensive training metrics and visualization logging
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
    └── temporal_YYYYMMDD_HHMMSS/
        ├── best_model.pth      # Best trained model
        ├── final_model.pth     # Final trained model
        ├── training_history.json # Training metrics data
        ├── training_curves.png # Training/validation curves
        └── tensorboard/        # TensorBoard logs
            ├── events.out.tfevents.* # Training metrics
            └── hparams.yaml    # Hyperparameter logs
```

#### Inference Output
```
inference_workspace/
├── inference_YYYYMMDD_HHMMSS/
│   ├── results.json           # Classification results
│   ├── confidence_scores.csv  # Detailed confidence scores
│   └── visualizations/        # Result visualizations
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
- `app_three_tabs.py`: **Comprehensive three-tab system with labeling/training/inference**
- `start.py`: Application launcher with version selection (required for Docker)

**Archived versions** (moved to `archive/`):
- `archive/app_old.py`: Original monolithic version (1200+ lines)  
- `archive/app_new.py`: Intermediate refactored version
- `archive/start_reid_labeling.py`: Alternative ReID labeling interface
- `archive/tools/`: Legacy labeling tools

### Debugging and Testing Scripts

```bash
# Test inference functionality (moved to tests/)
python tests/test_inference_fix.py       # Test inference workflow fixes
python tests/test_actual_training.py     # Test complete training pipeline
```

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

## Docker Deployment

The project includes Docker support for containerized deployment:

```bash
# Build and run with Docker Compose
docker-compose up --build

# Manual Docker build
docker build -t fire-smoke-validator .
```

See `docker-deploy.md` and `Dockerfile` for detailed deployment instructions. The `requirements-docker.txt` contains Docker-specific dependencies.


## TensorBoard Integration

The system includes comprehensive TensorBoard integration for training visualization and monitoring:

### Training Metrics Logged
- **Loss Tracking**: Training and validation loss over epochs
- **Accuracy Monitoring**: Training, validation, and test accuracy
- **Learning Rate**: Learning rate scheduling visualization
- **Model Parameters**: Parameter and gradient histograms (every 10 epochs)
- **Hyperparameters**: Complete hyperparameter tracking with hparams plugin

### TensorBoard Features
- 📊 **Real-time Monitoring**: Live training progress visualization
- 📈 **Comprehensive Metrics**: Loss, accuracy, learning rate tracking
- 🎯 **Best Model Tracking**: Automatic marking of best validation accuracy
- 🧪 **Test Results**: Final test set evaluation metrics
- ⚙️ **Hyperparameter Logging**: Complete training configuration tracking
- 📷 **Training Curves**: Embedded training curve images

### Dataset Splitting
- **Training Set**: 70% - Used for model parameter updates
- **Validation Set**: 20% - Used for hyperparameter tuning and model selection
- **Test Set**: 10% - Used for final unbiased performance evaluation

The system automatically splits datasets and ensures no data leakage between sets.

### Accessing TensorBoard
```bash
# For a specific training run
tensorboard --logdir=runs/temporal_training/temporal_20250820_123456/tensorboard

# For all training runs
tensorboard --logdir=runs/temporal_training

# The system will show the exact command after training completion
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

# Launch TensorBoard for training visualization
tensorboard --logdir=runs/temporal_training/temporal_*/tensorboard

# Debugging and state management utilities (moved to tests/)
python tests/debug_training.py           # Debug training functionality and models
python tests/check_training_state.py     # Check current training status
python tests/create_training_state.py    # Create mock training completion state
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
- ✅ TensorBoard integration with comprehensive training metrics
- ✅ Automatic train/validation/test dataset splitting
- ⚠️ No formal test framework setup (pytest in requirements but no tests/ directory)
- 🔄 Active refactoring - multiple app versions coexist