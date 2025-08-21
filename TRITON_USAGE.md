# TIMM to TensorRT 轉換器使用指南

## 🚀 快速開始

### 1. 一鍵啟動
```bash
./start-triton.sh
```

### 2. 手動啟動
```bash
# 構建並啟動容器
docker-compose -f docker-compose.triton.yml up --build

# 或僅啟動轉換器
docker-compose -f docker-compose.triton.yml up triton-converter
```

### 3. 訪問介面
- 打開瀏覽器
- 訪問: http://localhost:7860

## 📖 轉換流程

### 步驟 1: 準備模型
- 確保您有 TIMM 訓練的 .pth 或 .pt 模型檔案
- 記住模型架構名稱（如 `efficientnet_b0`）

### 步驟 2: 上傳模型
1. 在 Web 介面點擊「上傳 .pth 模型檔案」
2. 選擇您的模型檔案

### 步驟 3: 配置參數
- **模型架構**: 選擇對應的 TIMM 模型名稱
- **類別數量**: 設定模型的輸出類別數（如分類任務的類別數）
- **批次大小**: 設定推論時的最大批次大小
- **輸入尺寸**: 設定輸入圖像的高度和寬度
- **FP16 加速**: 建議啟用以獲得更好效能

### 步驟 4: 開始轉換
1. 點擊「🚀 開始轉換」按鈕
2. 等待轉換完成（通常 2-10 分鐘）
3. 觀察轉換狀態和進度

### 步驟 5: 下載結果
1. 轉換完成後，查看轉換結果資訊
2. 點擊下載按鈕獲取 .engine 檔案
3. .engine 檔案即為 TensorRT 最佳化的模型

## 🎯 常見使用案例

### 案例 1: 火煙分類模型
```
模型架構: efficientnet_b0
類別數量: 2 (火煙 vs 非火煙)
批次大小: 4
輸入尺寸: 224x224
FP16 加速: 啟用
```

### 案例 2: 大型分類模型
```
模型架構: efficientnet_b4
類別數量: 1000 (ImageNet)
批次大小: 1
輸入尺寸: 380x380
FP16 加速: 啟用
```

### 案例 3: 邊緣設備模型
```
模型架構: mobilenetv3_small_100
類別數量: 10
批次大小: 1
輸入尺寸: 224x224
FP16 加速: 啟用
```

## ⚡ 效能最佳化建議

### 1. 模型選擇
- **邊緣設備**: MobileNetV3, EfficientNet-B0~B2
- **平衡效能**: EfficientNet-B3~B4, ResNet-50
- **高精度**: EfficientNet-B5+, ConvNeXt, Swin Transformer

### 2. 批次大小設定
- **單張推論**: batch_size = 1
- **批次推論**: 根據 GPU 記憶體調整（通常 4-32）
- **記憶體限制**: 較小的批次大小可減少記憶體使用

### 3. 精度設定
- **FP16 加速**: 可提供 1.5-3x 加速，精度損失極小
- **FP32 精度**: 保持最高精度，但速度較慢

## 🔧 進階配置

### 環境變數自訂
```bash
# 在 docker-compose.triton.yml 中修改
environment:
  - CUDA_VISIBLE_DEVICES=0,1    # 使用多個 GPU
  - TRT_LOG_LEVEL=WARNING       # 調整日誌等級
```

### 記憶體設定
```python
# 在 model_converter.py 中調整
max_workspace_size = 2 << 30  # 2GB 工作空間
```

## 🚨 故障排除

### 1. 轉換失敗
**問題**: 模型轉換過程中出現錯誤
**解決**:
- 檢查模型架構名稱是否正確
- 確認模型檔案完整性
- 嘗試較小的批次大小
- 檢查 GPU 記憶體是否足夠

### 2. 記憶體不足
**問題**: CUDA out of memory
**解決**:
- 減少 max_batch_size
- 減少 max_workspace_size
- 關閉其他 GPU 進程

### 3. 轉換時間過長
**問題**: 轉換超過 30 分鐘
**解決**:
- 檢查 GPU 利用率
- 確認沒有其他高負載任務
- 考慮使用較簡單的模型架構

## 📊 支援的模型列表

### 輕量級模型（< 50MB）
- mobilenetv3_small_100
- mobilenetv3_large_100
- efficientnet_b0
- efficientnet_b1
- ghostnet_100

### 中等模型（50-200MB）
- efficientnet_b2
- efficientnet_b3
- efficientnet_b4
- resnet50
- resnet101
- convnext_tiny

### 大型模型（> 200MB）
- efficientnet_b5
- efficientnet_b6
- efficientnet_b7
- convnext_base
- swin_base_patch4_window7_224
- vit_base_patch16_224

## 📱 API 使用（進階）

如需程式化使用，可直接調用轉換器：

```python
from utils.model_converter import TimmToTensorRTConverter

converter = TimmToTensorRTConverter()
results = converter.convert_pth_to_tensorrt(
    pth_path="model.pth",
    output_path="model.engine",
    model_name="efficientnet_b0",
    num_classes=2
)
```

## 📞 技術支援

遇到問題時，請提供：
1. 錯誤訊息截圖
2. 模型資訊（架構、大小）
3. 系統配置（GPU型號、驅動版本）
4. Docker 日誌輸出