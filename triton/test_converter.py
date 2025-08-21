#!/usr/bin/env python3
"""
TensorRT 轉換器測試腳本
用於測試模型轉換功能
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# 添加 utils 路徑
sys.path.append(str(Path(__file__).parent))

from utils.model_converter import TimmToTensorRTConverter, get_supported_timm_models
import torch
import timm

def create_test_model(model_name: str = "efficientnet_b0", num_classes: int = 2) -> str:
    """創建測試模型"""
    print(f"📦 創建測試模型: {model_name}")
    
    try:
        # 創建 TIMM 模型
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # 儲存模型
        temp_dir = Path("/tmp/test_models")
        temp_dir.mkdir(exist_ok=True)
        
        model_path = temp_dir / f"test_{model_name}.pth"
        torch.save(model.state_dict(), model_path)
        
        print(f"✅ 測試模型已儲存: {model_path}")
        return str(model_path)
        
    except Exception as e:
        print(f"❌ 創建測試模型失敗: {e}")
        return None

def test_conversion(model_path: str, model_name: str):
    """測試模型轉換"""
    print(f"🔄 測試轉換: {model_name}")
    
    try:
        # 創建轉換器
        converter = TimmToTensorRTConverter(verbose=True)
        
        # 設定輸出路徑
        output_dir = Path("/tmp/test_converted")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"test_{model_name}.engine"
        
        # 執行轉換
        results = converter.convert_pth_to_tensorrt(
            pth_path=model_path,
            output_path=str(output_path),
            model_name=model_name,
            num_classes=2,
            input_shape=(1, 3, 224, 224),
            max_batch_size=1,
            fp16_mode=True
        )
        
        if results['success']:
            print("🎉 轉換成功！")
            
            # 顯示結果資訊
            if results['engine_path']:
                engine_size = os.path.getsize(results['engine_path'])
                print(f"📁 引擎檔案: {Path(results['engine_path']).name}")
                print(f"💾 檔案大小: {engine_size / (1024**2):.2f} MB")
            
            # 顯示模型資訊
            if 'model_info' in results:
                info = results['model_info']
                print(f"📊 總參數: {info.get('total_parameters', 0):,}")
                print(f"🎯 類別數: {info.get('num_classes', 0)}")
            
            # 測試引擎資訊
            engine_info = converter.get_model_info(results['engine_path'])
            if 'error' not in engine_info:
                print(f"⚡ 最大批次: {engine_info.get('max_batch_size', 'N/A')}")
                print(f"💾 設備記憶體: {engine_info.get('device_memory_size', 0) / (1024**2):.1f} MB")
        else:
            print(f"❌ 轉換失敗: {results.get('error', '未知錯誤')}")
        
        return results['success']
        
    except Exception as e:
        print(f"❌ 測試轉換失敗: {e}")
        return False

def main():
    """主測試函數"""
    print("🧪 TensorRT 轉換器測試")
    print("=" * 40)
    
    # 顯示支援的模型
    supported_models = get_supported_timm_models()
    print(f"📋 支援 {len(supported_models)} 個 TIMM 模型")
    
    # 測試幾個常用模型
    test_models = [
        "efficientnet_b0",
        "mobilenetv3_small_100",
        "resnet18"
    ]
    
    success_count = 0
    total_count = len(test_models)
    
    for model_name in test_models:
        print(f"\n🔍 測試模型: {model_name}")
        print("-" * 30)
        
        # 創建測試模型
        model_path = create_test_model(model_name)
        if not model_path:
            continue
        
        # 測試轉換
        success = test_conversion(model_path, model_name)
        if success:
            success_count += 1
        
        # 清理測試檔案
        try:
            os.unlink(model_path)
        except:
            pass
    
    print(f"\n📊 測試結果: {success_count}/{total_count} 成功")
    
    if success_count == total_count:
        print("🎉 所有測試通過！")
    else:
        print("⚠️ 部分測試失敗")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)