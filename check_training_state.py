#!/usr/bin/env python3
"""
檢查當前訓練狀態
"""

import sys
sys.path.append('/home/ubuntu/fire-smoke-validator')

from core.training import ModelTrainer

trainer = ModelTrainer()
print(f"is_training: {trainer.is_training}")
print(f"training_progress: {repr(trainer.training_progress)}")
print(f"training_results: {trainer.training_results}")