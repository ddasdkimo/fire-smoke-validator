#!/usr/bin/env python3
"""
創建訓練完成狀態文件
"""

import json
from pathlib import Path

# 創建訓練完成的狀態
state = {
    'is_training': False,
    'training_progress': '✅ 時序模型訓練完成！最佳準確率: 0.875',
    'training_results': {
        'best_val_accuracy': 0.875,
        'final_train_accuracy': 0.9117647058823529,
        'save_path': 'runs/temporal_training/temporal_20250820_135523',
        'model_info': {
            'backbone': 'convnext_tiny',
            'temporal_frames': 5,
            'temporal_fusion': 'attention',
            'num_classes': 2,
            'feature_dim': 768,
            'total_parameters': 28953315,
            'trainable_parameters': 28953315,
            'frozen_backbone': False
        }
    }
}

# 保存狀態文件
state_file = Path("training_workspace/training_state.json")
state_file.parent.mkdir(exist_ok=True)

with open(state_file, 'w', encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False, indent=2)

print(f"✅ 已創建訓練狀態文件: {state_file}")
print(f"📊 訓練進度: {state['training_progress']}")
print(f"🎯 最佳準確率: {state['training_results']['best_val_accuracy']}")