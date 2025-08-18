# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Fire/Smoke False Positive Labeling System** - a simplified web application for quickly building training datasets. Upload videos to automatically scan with the best.pt model, group similar events using ReID, and rapidly label true fire/smoke vs false positives.

## Development Commands

### Environment Setup
```bash
# Install all dependencies
pip install -r requirements.txt

# Install the package in development mode (optional)
pip install -e .
```

### Code Quality
```bash
# Format code with black
black app.py tools/

# Lint with flake8  
flake8 app.py tools/

# Run tests (if any)
pytest tests/
```

### Main Application
```bash
# Start the web interface
python app.py
```

## Architecture Overview

### Core Application
- **Main App**: `app.py` - Gradio-based web interface for video analysis and labeling
- **Video Analysis**: Automatically scans uploaded videos using best.pt YOLO model
- **ReID Grouping**: Groups similar detections across frames into events
- **Quick Labeling**: Binary classification interface (True Fire/Smoke vs False Positive)
- **Dataset Export**: Exports labeled data as structured training dataset

### Key Technologies
- **Detection**: Ultralytics YOLO (pre-trained model at `best.pt`)
- **Tracking**: Supervision library for object tracking and ReID grouping
- **UI**: Gradio for web interface
- **Vision**: OpenCV for video processing

### Data Flow
1. Upload video → Extract frames
2. Run YOLO detection on frames 
3. Group detections by ReID similarity
4. Present event thumbnails for labeling
5. Export labeled dataset as ZIP

### Output Structure
```
dataset/
└── export_YYYYMMDD_HHMMSS.zip
    ├── true_positive/          # Real fire/smoke events
    │   ├── event_0/
    │   │   ├── frame_000_1.2s.jpg
    │   │   ├── frame_001_1.7s.jpg  
    │   │   └── metadata.json
    │   └── event_1/
    └── false_positive/         # False positive events
        ├── event_2/
        └── event_3/
```

## Key Considerations

- Simplified from original complex temporal classification project
- Focuses on rapid false positive dataset creation
- Uses existing best.pt model for detection
- Binary labeling optimized for speed
- Lightweight architecture, removed unnecessary components
- Supports Mac MPS acceleration for model inference
- This is NOT a git repository

## Current Status

- ✅ Simplified web application complete
- ✅ Video upload and analysis functional
- ✅ ReID event grouping implemented  
- ✅ Quick binary labeling interface ready
- ✅ Dataset export functionality complete