# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

火煙誤判標註系統 (Fire/Smoke False Positive Labeling System) - A comprehensive deep learning system for fire/smoke detection, labeling, training, and inference. The system combines YOLO detection, ReID tracking, temporal classification models, and TensorBoard visualization to create a complete ML pipeline for fire/smoke analysis.

## Development Commands

### Running the Application
```bash
# Primary applications
python app.py               # Simplified labeling interface (recommended for pure labeling)
python app_three_tabs.py    # Full system with labeling/training/inference tabs

# Docker deployment
docker-compose up --build    # Standard deployment on port 7861
docker-compose -f docker-compose.triton.yml up  # With TensorRT/Triton support

# Alternative launchers
python start.py              # Interactive launcher with app selection
./run.sh                     # Shell script launcher
```

### Code Quality & Testing
```bash
# Format code
black app.py core/ ui/ tools/

# Lint check  
flake8 app.py core/ ui/ tools/

# Run tests (Note: No formal test suite, tests/ contains debugging utilities)
python tests/test_actual_training.py     # Test training pipeline
python tests/test_inference_fix.py       # Test inference workflow
```

### Training & Monitoring
```bash
# Monitor active training
tail -f runs/temporal_training/temporal_*/training_log.txt
ps aux | grep python | grep training

# Launch TensorBoard
tensorboard --logdir=runs/temporal_training

# GPU monitoring
nvidia-smi  # CUDA GPUs
# For Mac MPS: Check Activity Monitor
```

## High-Level Architecture

The system uses a **modular MVC architecture** with clear separation of concerns:

### Core Modules (`core/`)
- **analyzer.py**: Video frame extraction, YOLO detection (`best.pt`), and ReID-based event grouping using Supervision library
- **labeling.py**: Event management, labeling state tracking, and ZIP dataset export
- **training.py**: Orchestrates temporal model training with dataset loading and model initialization
- **inference.py**: Temporal classification inference pipeline for video analysis
- **models/temporal_classifier.py**: Multi-backbone temporal models (ResNet, ConvNeXt, EfficientNet, Swin, etc.) with attention-based temporal fusion
- **models/temporal_trainer.py**: Training loop with TensorBoard logging, early stopping, and learning rate scheduling
- **models/data_utils.py**: Dataset preprocessing, augmentation, and train/val/test splitting (70/20/10)

### UI Components (`ui/`)
- **builder.py**: Single-tab Gradio interface for labeling workflow
- **builder_new.py**: Three-tab interface integrating labeling, training, and inference
- **interface.py**: Event interaction handlers and state management
- **training_controller.py**: Training workflow coordination with background process management
- **inference_controller.py**: Inference execution and result visualization

### Key Workflows

**Labeling Pipeline**:
```
Video Upload → Frame Extraction (1 FPS) → YOLO Detection (conf=0.3) → 
ReID Grouping → Event Carousel UI → Binary Classification → ZIP Export
```

**Training Pipeline**:
```
ZIP Dataset Upload → Data Splitting (70/20/10) → Model Selection → 
Background Training → TensorBoard Logging → Model Checkpointing
```

**Inference Pipeline**:
```
Model Loading → Video Upload → Temporal Analysis (5-frame windows) → 
Confidence Scoring → Result Export
```

## Critical Configuration

### Detection Parameters
- **YOLO confidence**: `conf=0.3` in analyzer.py:144 - Lower values detect more events but increase false positives
- **Frame extraction**: 1 FPS sampling (analyzer.py:80) - Balances processing speed vs temporal resolution
- **ReID threshold**: analyzer.py:234 - Controls event grouping sensitivity
- **Event frame limits**: Min 2, Max 5 frames (configurable in UI) - Prevents single-frame noise and overly long events

### Model Files
- **best.pt**: YOLO fire/smoke detection model (required in project root)
- **Temporal models**: Saved to `runs/temporal_training/temporal_*/best_model.pth`

### Docker Configuration
- **Port mapping**: 7861:7860 (host:container)
- **GPU support**: NVIDIA CUDA with compute capability 8.0-9.0
- **Memory limits**: 8GB RAM, 2GB shared memory
- **Volumes**: Models, datasets, and uploads are persisted

## Temporal Model Selection Guide

### Model Architecture Decision Tree
```
Need real-time inference on edge device?
├─ Yes → temporal_mobilenetv3_large or temporal_ghostnet_100
└─ No → Need highest accuracy?
    ├─ Yes → temporal_swin_base_patch4_window7_224 or temporal_convnext_base
    └─ No → temporal_convnext_tiny (recommended default)
```

### Training Hyperparameters
- **Epochs**: 50-100 (early stopping at patience=10)
- **Batch size**: 32 (adjust based on GPU memory)
- **Learning rate**: 1e-4 with cosine annealing
- **Data augmentation**: Random crop, flip, color jitter
- **Temporal frames**: Fixed at 5 frames per sample

## TensorRT/Triton Support

For production deployment with optimized inference:
```bash
# Start Triton converter
./start-triton.sh

# Or with docker-compose
docker-compose -f docker-compose.triton.yml up
```

Converts TIMM models to TensorRT for 2-10x inference speedup. See `TRITON_USAGE.md` for details.