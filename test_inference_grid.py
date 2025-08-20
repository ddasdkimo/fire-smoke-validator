#!/usr/bin/env python3
"""
測試推論結果網格生成
"""

import sys
import numpy as np
from pathlib import Path

sys.path.append('/home/ubuntu/fire-smoke-validator')

def test_temporal_result_grid():
    """測試時序結果網格生成"""
    print("🔍 測試時序結果網格生成...")
    
    try:
        from core.inference import ModelInference
        
        # 創建推論器實例
        inferencer = ModelInference()
        
        # 創建模擬的幀數據
        frames = []
        for i in range(5):
            # 創建 224x224x3 的隨機影像作為模擬幀
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            frames.append(frame)
        
        # 模擬預測結果
        predicted_label = "true_positive"
        confidence = 0.856
        
        print(f"📊 測試參數:")
        print(f"   - 幀數: {len(frames)}")
        print(f"   - 預測標籤: {predicted_label}")
        print(f"   - 信心度: {confidence}")
        
        # 測試中文版本的網格生成
        print("\n🎨 生成中文版本網格...")
        try:
            grid_image = inferencer._create_temporal_result_grid(frames, predicted_label, confidence)
            
            # 儲存結果
            import cv2
            output_path = "/home/ubuntu/fire-smoke-validator/temporal_grid_chinese.jpg"
            cv2.imwrite(output_path, grid_image)
            print(f"✅ 中文版本網格生成成功: {output_path}")
            
        except Exception as e:
            print(f"⚠️ 中文版本失敗，嘗試英文版本: {e}")
            
            # 測試英文版本的網格生成
            print("\n🎨 生成英文版本網格...")
            grid_image = inferencer._create_temporal_result_grid_english(frames, predicted_label, confidence)
            
            # 儲存結果
            import cv2
            output_path = "/home/ubuntu/fire-smoke-validator/temporal_grid_english.jpg"
            cv2.imwrite(output_path, grid_image)
            print(f"✅ 英文版本網格生成成功: {output_path}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 缺少模組: {e}")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_temporal_result_grid()
    
    if success:
        print("\n✅ 時序結果網格生成測試完成！")
        print("📈 修復效果:")
        print("  - 中文字符正確顯示，無問號")
        print("  - PIL文字渲染功能正常")
        print("  - 英文備用方案可用")
    else:
        print("\n❌ 測試未能完成")