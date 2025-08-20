#!/usr/bin/env python3
"""
檢查熱區圖實作是否完成
"""

import sys
import inspect
from pathlib import Path

sys.path.append('/home/ubuntu/fire-smoke-validator')

def check_implementation():
    """檢查熱區圖實作完整性"""
    print("🔬 檢查熱區圖實作完整性...")
    
    # 檢查檔案是否存在且包含預期的類別和方法
    inference_file = Path('/home/ubuntu/fire-smoke-validator/core/inference.py')
    
    if not inference_file.exists():
        print("❌ inference.py 檔案不存在")
        return False
    
    # 讀取檔案內容
    content = inference_file.read_text(encoding='utf-8')
    
    # 檢查必要的類別
    required_classes = [
        'GradCAMVisualizer',
        'AttentionVisualizer',
        'ModelInference'
    ]
    
    for cls_name in required_classes:
        if f'class {cls_name}' in content:
            print(f"✅ 找到類別: {cls_name}")
        else:
            print(f"❌ 缺少類別: {cls_name}")
    
    # 檢查 GradCAMVisualizer 的必要方法
    gradcam_methods = [
        '_register_hooks',
        '_forward_hook', 
        '_backward_hook',
        'generate_cam'
    ]
    
    print("\n🔍 檢查 GradCAMVisualizer 方法:")
    for method in gradcam_methods:
        if f'def {method}' in content:
            print(f"✅ {method}")
        else:
            print(f"❌ {method}")
    
    # 檢查 AttentionVisualizer 的必要方法
    attention_methods = [
        '_register_attention_hooks',
        '_attention_hook',
        'get_attention_maps'
    ]
    
    print("\n🔍 檢查 AttentionVisualizer 方法:")
    for method in attention_methods:
        if f'def {method}' in content:
            print(f"✅ {method}")
        else:
            print(f"❌ {method}")
    
    # 檢查 ModelInference 的新增方法
    inference_methods = [
        '_generate_heatmaps',
        '_create_heatmap_overlay',
        '_save_heatmap_visualizations',
        '_create_combined_heatmap_visualization',
        'get_heatmap_gallery'
    ]
    
    print("\n🔍 檢查 ModelInference 新增方法:")
    for method in inference_methods:
        if f'def {method}' in content:
            print(f"✅ {method}")
        else:
            print(f"❌ {method}")
    
    # 檢查初始化相關代碼
    initialization_checks = [
        'self.gradcam_visualizer = None',
        'self.attention_visualizer = None',
        'self.gradcam_visualizer = GradCAMVisualizer(self.current_model)',
        'self.attention_visualizer = AttentionVisualizer(self.current_model)'
    ]
    
    print("\n🔍 檢查初始化代碼:")
    for check in initialization_checks:
        if check in content:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
    
    # 檢查推論結果是否包含熱區圖資訊
    result_checks = [
        '"heatmap_paths": heatmap_paths',
        '"has_heatmaps": len(heatmap_paths) > 0',
        '生成熱區圖視覺化',
        'Grad-CAM 熱區圖顯示模型關注的影像區域'
    ]
    
    print("\n🔍 檢查推論結果整合:")
    for check in result_checks:
        if check in content:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
    
    # 統計實作完成度
    all_checks = required_classes + gradcam_methods + attention_methods + inference_methods + initialization_checks + result_checks
    passed_checks = sum(1 for check in all_checks if check in content or f'class {check}' in content or f'def {check}' in content)
    
    completion_rate = passed_checks / len(all_checks) * 100
    
    print(f"\n📊 實作完成度: {passed_checks}/{len(all_checks)} ({completion_rate:.1f}%)")
    
    if completion_rate >= 80:
        print("🎉 熱區圖功能實作基本完成！")
        return True
    else:
        print("⚠️ 熱區圖功能實作尚未完整")
        return False

def check_dependencies():
    """檢查依賴套件"""
    print("\n🔍 檢查依賴套件:")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy')
    ]
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} 已安裝")
        except ImportError:
            print(f"⚠️ {name} 未安裝")

if __name__ == "__main__":
    success = check_implementation()
    check_dependencies()
    
    print("\n" + "="*50)
    if success:
        print("✅ 熱區圖功能實作檢查通過！")
        print("\n🔥 新功能說明：")
        print("1. 🎯 Grad-CAM 熱區圖：顯示模型關注的影像區域")
        print("2. 🧠 注意力視覺化：展示注意力機制的權重分佈")
        print("3. 📊 組合視覺化：原始圖片與熱區圖的對比顯示")
        print("4. 🖼️ 專用畫廊：熱區圖結果的獨立畫廊功能")
        print("5. 📁 自動儲存：熱區圖自動儲存到專用目錄")
        
        print("\n⚡ 使用方式：")
        print("- 載入時序模型後進行推論時會自動生成熱區圖")
        print("- 熱區圖檔案會儲存在 inference_workspace/heatmaps_* 目錄中")
        print("- 可通過 get_heatmap_gallery() 方法獲取熱區圖路徑")
    else:
        print("❌ 熱區圖功能實作需要進一步完善")