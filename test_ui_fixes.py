#!/usr/bin/env python3
"""
測試UI修復是否有效
"""

import sys
import os
from pathlib import Path

# 添加項目路徑
sys.path.append('/home/ubuntu/fire-smoke-validator')

def test_ui_fixes():
    print("🔍 測試UI修復...")
    
    try:
        from core.training import ModelTrainer
        from ui.training_controller import TrainingController
        
        # 創建訓練器和控制器
        trainer = ModelTrainer()
        controller = TrainingController(trainer)
        
        # 測試模型列表
        print("\n📋 測試模型列表:")
        models_display = controller.refresh_models_list()
        print(models_display)
        
        # 測試訓練進度
        print("\n📊 測試訓練進度:")
        progress = controller.get_training_progress()
        print(f"進度顯示: {progress}")
        
        # 檢查模型列表數量
        models = trainer.list_trained_models()
        print(f"\n✅ 找到 {len(models)} 個已訓練模型")
        for model in models:
            print(f"   - {model['name']} ({model.get('type', 'unknown')})")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ui_fixes()