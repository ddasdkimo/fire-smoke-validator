#!/usr/bin/env python3
"""
調試訓練功能的腳本
"""

import sys
import os
from pathlib import Path

# 添加項目路徑
sys.path.append('/home/ubuntu/fire-smoke-validator')

def test_training():
    print("🔍 開始調試訓練功能...")
    
    try:
        # 導入必要模組
        from core.training import ModelTrainer
        print("✅ 成功導入 ModelTrainer")
        
        # 創建訓練器實例
        trainer = ModelTrainer()
        print("✅ 成功創建 ModelTrainer 實例")
        print(f"   is_training: {trainer.is_training}")
        print(f"   training_progress: {trainer.training_progress}")
        
        # 檢查可用模型
        models = trainer.get_available_models()
        print(f"✅ 可用模型數量: {len(models)}")
        
        # 檢查資料集
        dataset_path = "training_workspace/temporal_merged_dataset_20250820_134309"
        if Path(dataset_path).exists():
            print(f"✅ 資料集存在: {dataset_path}")
            
            # 嘗試導入 TemporalTrainer
            try:
                from core.models.temporal_trainer import TemporalTrainer
                print("✅ 成功導入 TemporalTrainer")
                
                # 測試創建 TemporalTrainer 實例
                config = trainer._get_temporal_model_config("temporal_convnext_tiny", 224)
                print(f"✅ 獲得模型配置: {config}")
                
                temp_trainer = TemporalTrainer(config)
                print("✅ 成功創建 TemporalTrainer 實例")
                
            except ImportError as e:
                print(f"❌ 導入 TemporalTrainer 失敗: {e}")
            except Exception as e:
                print(f"❌ 創建 TemporalTrainer 失敗: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"❌ 資料集不存在: {dataset_path}")
            
    except Exception as e:
        print(f"❌ 調試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training()