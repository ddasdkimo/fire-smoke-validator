#!/usr/bin/env python3
"""
測試熱區圖生成功能
"""

import sys
import os
from pathlib import Path
import tempfile

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# 添加專案路徑
sys.path.append('/home/ubuntu/fire-smoke-validator')

def create_test_images():
    """創建測試用的假影像"""
    test_images = []
    
    for i in range(3):
        # 創建 224x224 的假影像
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # 添加一些火煙相關的模擬特徵（紅色和橙色區域）
        if i == 0:
            # 第一幀：添加橙紅色區域模擬火焰
            cv2.rectangle(image, (50, 50), (150, 150), (0, 69, 255), -1)  # 橙紅色
        elif i == 1:
            # 第二幀：添加灰色區域模擬煙霧
            cv2.rectangle(image, (60, 60), (160, 160), (128, 128, 128), -1)  # 灰色
        else:
            # 第三幀：添加黃紅色區域
            cv2.rectangle(image, (70, 70), (170, 170), (0, 128, 255), -1)  # 黃紅色
        
        # 保存到暫存檔案
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, image)
        
        # 創建假的檔案物件
        class FakeFile:
            def __init__(self, name):
                self.name = name
        
        test_images.append(FakeFile(temp_file.name))
    
    return test_images

def test_heatmap_functionality():
    """測試熱區圖功能"""
    print("🔬 開始測試熱區圖生成功能...")
    
    # 檢查必要依賴
    if not NUMPY_AVAILABLE:
        print("❌ NumPy 未安裝，請安裝：pip install numpy")
        return
        
    if not CV2_AVAILABLE:
        print("❌ OpenCV 未安裝，請安裝：pip install opencv-python")
        return
    
    try:
        # 導入必要模組
        from core.inference import ModelInference, GradCAMVisualizer, AttentionVisualizer
        print("✅ 成功導入推論模組和視覺化工具")
        
        # 創建推論器
        inference = ModelInference()
        print("✅ 成功創建推論器實例")
        
        # 檢查可用模型
        models = inference.get_available_models()
        print(f"📊 可用模型數量: {len(models)}")
        
        if not models or models[0]['type'] == 'placeholder':
            print("⚠️ 沒有可用的時序模型，跳過實際推論測試")
            print("   建議先進行模型訓練再測試熱區圖功能")
            return
        
        # 載入第一個可用模型
        first_model = models[0]
        print(f"🔄 嘗試載入模型: {first_model['name']}")
        
        load_result = inference.load_model(first_model['path'])
        print("模型載入結果:")
        print(load_result)
        
        if "❌" in load_result:
            print("⚠️ 模型載入失敗，跳過推論測試")
            return
        
        # 創建測試影像
        print("🖼️ 創建測試影像...")
        test_images = create_test_images()
        print(f"✅ 創建了 {len(test_images)} 張測試影像")
        
        # 進行推論測試
        print("🚀 開始推論測試...")
        summary, results = inference.inference_batch_images(test_images, confidence_threshold=0.5)
        
        print("推論結果摘要:")
        print(summary)
        
        if results and len(results) > 0:
            result = results[0]
            print("\n📊 結果詳情:")
            print(f"- 序列ID: {result.get('sequence_id', 'N/A')}")
            print(f"- 預測類別: {result.get('predicted_label', 'N/A')}")
            print(f"- 信心度: {result.get('confidence', 'N/A')}")
            print(f"- 有熱區圖: {result.get('has_heatmaps', False)}")
            
            if result.get('has_heatmaps', False):
                heatmap_paths = result.get('heatmap_paths', [])
                print(f"✅ 生成了 {len(heatmap_paths)} 個熱區圖檔案:")
                for i, path in enumerate(heatmap_paths):
                    print(f"  {i+1}. {Path(path).name}")
                    if Path(path).exists():
                        print(f"     ✅ 檔案存在，大小: {Path(path).stat().st_size} bytes")
                    else:
                        print(f"     ❌ 檔案不存在")
            else:
                print("❌ 未生成熱區圖")
            
            # 測試畫廊功能
            gallery_paths = inference.get_detection_gallery()
            heatmap_gallery = inference.get_heatmap_gallery()
            
            print(f"\n🖼️ 畫廊統計:")
            print(f"- 總結果圖片: {len(gallery_paths)}")
            print(f"- 熱區圖數量: {len(heatmap_gallery)}")
        
        # 清理測試檔案
        print("\n🧹 清理測試檔案...")
        for img in test_images:
            try:
                Path(img.name).unlink()
            except:
                pass
        
        print("✅ 熱區圖功能測試完成！")
        
    except ImportError as e:
        print(f"❌ 導入錯誤: {e}")
        print("請確保所有相依套件已安裝：torch, matplotlib, opencv-python")
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_heatmap_functionality()