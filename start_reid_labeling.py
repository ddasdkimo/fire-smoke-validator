#!/usr/bin/env python3
"""
啟動 ReID 時序標記介面
"""

import sys
import os
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🔥 ReID 時序火災偵測標記工具")
print("=" * 60)
print("功能特色：")
print("✅ 支援上傳多種影片格式 (MP4, MOV, AVI, MKV, WebM)")
print("✅ 使用 ReID 技術自動分組偵測結果") 
print("✅ 生成時序連續的標記資料")
print("✅ 支援移除誤判幀和範圍選擇")
print("✅ 自動裁切物件區域並整理序列")
print("✅ 完成標記後可下載標記資料包")
print("✅ 一鍵清除會話資料重新開始")
print("✅ 適合訓練 timm 時序分類模型")
print()

# 檢查必要套件
missing_packages = []

try:
    import gradio
    print("✅ Gradio 已安裝")
except ImportError:
    missing_packages.append("gradio")

try:
    from ultralytics import YOLO
    print("✅ Ultralytics 已安裝")
except ImportError:
    missing_packages.append("ultralytics")

try:
    import supervision
    print("✅ Supervision 已安裝")
except ImportError:
    missing_packages.append("supervision")

if missing_packages:
    print(f"\n❌ 缺少必要套件: {', '.join(missing_packages)}")
    print("請執行: pip install " + " ".join(missing_packages))
    sys.exit(1)

# 檢查模型檔案
model_path = project_root / "best.pt"
if model_path.exists():
    print("✅ 找到模型檔案 best.pt")
else:
    print("⚠️  模型檔案 best.pt 不存在，將使用模擬偵測")

print("\n標記類別說明：")
print("🔥 動態火災: 真實的動態火災序列（時序變化明顯）")
print("💨 動態煙霧: 真實的動態煙霧序列（時序變化明顯）")
print("☁️  靜態雲朵: 被誤判為火災/煙霧的雲朵（靜態或緩慢變化）")
print("💡 靜態燈光: 被誤判為火災的燈光（靜態）")
print("📦 靜態物體: 其他被誤判的靜態物體")
print("❓ 不確定: 無法確定的情況")

print("\n輸出結構：")
print("data/labeled_sequences/")
print("├── true_dynamic_fire/     # 真實動態火災序列")
print("├── true_dynamic_smoke/    # 真實動態煙霧序列")
print("├── false_static_cloud/    # 靜態雲朵誤判序列")
print("├── false_static_light/    # 靜態燈光誤判序列")
print("├── false_static_object/   # 其他靜態物體誤判序列")
print("├── uncertain/             # 不確定序列")
print("└── labeling_history.json  # 完整標記記錄")

print("\n每個序列資料夾包含：")
print("- 000_1.2s.jpg, 001_1.7s.jpg, ... # 裁切的物件區域影像")
print("- metadata.json                    # 序列元資料")
print("\n新功能：")
print("✅ 可移除序列中的誤判幀 (支援範圍選擇：0,2,5 或 0-3)")
print("✅ 自動裁切並保存邊界框區域 (用於訓練)")
print("✅ 支援 Mac MPS 加速")

input("\n按 Enter 鍵啟動 ReID 標記介面...")

# 啟動介面
try:
    from tools.reid_labeling_interface import main
    main()
except Exception as e:
    print(f"❌ 啟動失敗: {e}")
    print("\n故障排除：")
    print("1. 確認所有套件已正確安裝")
    print("2. 檢查 Python 版本 >= 3.8")
    print("3. 確認有足夠的記憶體和硬碟空間")