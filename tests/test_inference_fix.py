#!/usr/bin/env python3
"""
測試推論模組修復
"""

import sys
import os
from pathlib import Path

sys.path.append('/home/ubuntu/fire-smoke-validator')

def test_inference_fix():
    print("🔍 測試推論模組修復...")
    
    try:
        from core.inference import ModelInference
        
        inferencer = ModelInference()
        
        # 測試載入時序模型
        model_path = "runs/temporal_training/temporal_20250820_135523/best_model.pth"
        print(f"\n📦 嘗試載入模型: {model_path}")
        
        result = inferencer.load_model(model_path, device='cpu')
        print(result)
        
        if inferencer.current_model:
            print("\n✅ 模型載入成功！")
            model_info = inferencer.current_model.get_model_info()
            print(f"模型資訊: {model_info}")
        else:
            print("\n❌ 模型載入失敗")
            
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

def test_delete_function():
    print("\n🗑️ 測試刪除功能...")
    
    try:
        from core.training import ModelTrainer
        from ui.training_controller import TrainingController
        
        trainer = ModelTrainer()
        controller = TrainingController(trainer)
        
        # 列出模型
        models = trainer.list_trained_models()
        print(f"當前有 {len(models)} 個模型")
        for i, model in enumerate(models):
            print(f"  {i+1}. {model['name']}")
        
        # 測試刪除功能（不實際刪除）
        print("\n刪除功能準備就緒")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_fix()
    test_delete_function()