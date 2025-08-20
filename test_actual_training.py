#!/usr/bin/env python3
"""
測試實際訓練功能
"""

import sys
import os
from pathlib import Path

# 添加項目路徑
sys.path.append('/home/ubuntu/fire-smoke-validator')

def test_actual_training():
    print("🔥 開始測試實際訓練功能...")
    
    try:
        from core.training import ModelTrainer
        
        trainer = ModelTrainer()
        print(f"初始訓練狀態: is_training={trainer.is_training}")
        
        # 測試訓練路徑字符串
        dataset_path_str = """✅ 多資料集合併成功！
            
📊 處理結果:
- 成功處理: 3 個ZIP檔案
- 失敗檔案: 0 個

📈 合併後統計:
- 真實火煙事件: 8 個
- 誤判事件: 76 個  
- 總影像數: 514 張

📁 時序分類資料集路徑: training_workspace/temporal_merged_dataset_20250820_134309
            """
        
        print("🚀 嘗試啟動訓練...")
        result = trainer.start_training(
            dataset_path=dataset_path_str,
            model_name="temporal_convnext_tiny",
            epochs=2,  # 使用很少的 epoch 進行測試
            batch_size=4,  # 小的 batch size
            image_size=224
        )
        
        print(f"訓練啟動結果: {result}")
        print(f"訓練狀態: is_training={trainer.is_training}")
        
        # 等待一會兒看看進度
        import time
        print("等待 5 秒查看訓練進度...")
        time.sleep(5)
        
        progress = trainer.get_training_progress()
        print(f"訓練進度: {progress}")
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_training()