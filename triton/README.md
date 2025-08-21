# TIMM to TensorRT 轉換器

這是一個基於 NVIDIA DeepStream 和 Triton 的容器化應用程式，專門用於將 TIMM 的 .pth 模型轉換為 TensorRT 的 .engine/.plan 格式。

## 🚀 主要功能

- **模型轉換**: 將 TIMM .pth 模型轉換為 TensorRT .engine/.plan 格式
- **Web 介面**: 提供直觀的 Gradio Web 介面
- **效能最佳化**: 支援 FP16 精度、動態批次大小
- **模型管理**: 支援模型上傳、轉換、下載、刪除
- **架構支援**: 支援所有 TIMM 預訓練模型架構

## 📋 系統需求

### 硬體要求
- NVIDIA GPU（支援 CUDA 12.1+）
- 至少 8GB GPU 記憶體
- 至少 16GB 系統記憶體

### 軟體要求
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Container Toolkit
- NVIDIA Driver 535+

## 🛠️ 安裝與配置

### 1. 克隆專案
```bash
git clone <repository-url>
cd fire-smoke-validator
```

### 2. 建立必要目錄
```bash
mkdir -p triton/{models,converted,temp,triton_models}
```

### 3. 構建並啟動容器
```bash
# 僅啟動轉換器
docker-compose -f docker-compose.triton.yml up triton-converter

# 啟動轉換器和 Triton 推論伺服器
docker-compose -f docker-compose.triton.yml up
```

### 4. 訪問 Web 介面
打開瀏覽器並訪問: http://localhost:7860

## 📖 使用說明

### 模型轉換流程

1. **上傳模型**
   - 在 Web 介面上傳 .pth 或 .pt 檔案
   - 支援的模型大小：最大 10GB

2. **配置參數**
   - 選擇對應的 TIMM 模型架構（如 efficientnet_b0）
   - 設定類別數量（預設: 2）
   - 設定最大批次大小（預設: 1）
   - 選擇是否啟用 FP16 精度

3. **輸入尺寸設定**
   - 設定輸入圖像高度（預設: 224）
   - 設定輸入圖像寬度（預設: 224）

4. **開始轉換**
   - 點擊「開始轉換」按鈕
   - 等待轉換完成（通常需要 2-10 分鐘）

5. **下載結果**
   - 轉換完成後下載 .engine 檔案
   - 檔案可用於 TensorRT 推論

### 支援的模型架構

#### 輕量級模型（適合邊緣設備）
- `mobilenetv3_small_100`, `mobilenetv3_large_100`
- `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`
- `ghostnet_100`

#### 平衡型模型（推薦）
- `efficientnet_b3`, `efficientnet_b4`
- `efficientnetv2_s`, `efficientnetv2_m`
- `resnet50`, `resnet101`
- `convnext_tiny`, `convnext_small`

#### 高精度模型
- `efficientnet_b5`, `efficientnet_b6`, `efficientnet_b7`
- `efficientnetv2_l`
- `convnext_base`, `convnext_large`
- `swin_base_patch4_window7_224`
- `vit_base_patch16_224`, `vit_large_patch16_224`

## 🔧 進階配置

### 環境變數
```bash
# GPU 設定
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0

# 記憶體設定
TENSORRT_WORKSPACE_SIZE=1073741824  # 1GB

# 日誌等級
TRT_LOG_LEVEL=INFO
```

### 目錄結構
```
triton/
├── app.py                 # 主應用程式
├── utils/
│   └── model_converter.py # 模型轉換工具
├── models/                # 輸入模型目錄
├── converted/             # 轉換後模型目錄
├── temp/                  # 臨時檔案目錄
└── triton_models/         # Triton 模型倉庫
```

## 🚨 注意事項

### 轉換限制
1. **記憶體需求**: 大型模型可能需要更多 GPU 記憶體
2. **轉換時間**: 複雜模型轉換時間較長（5-30分鐘）
3. **精度設定**: FP16 可能影響精度，建議先測試

### 常見問題解決

#### 1. GPU 記憶體不足
```bash
# 減少批次大小
max_batch_size = 1

# 減少工作空間大小
max_workspace_size = 512 * 1024 * 1024  # 512MB
```

#### 2. 轉換失敗
```bash
# 檢查模型架構是否正確
# 確認模型檔案完整性
# 檢查 CUDA 驅動版本
```

#### 3. 容器啟動失敗
```bash
# 檢查 NVIDIA Container Toolkit
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi

# 檢查權限
sudo chmod -R 755 triton/
```

## 📊 效能基準

### 轉換時間（參考）
| 模型架構 | 模型大小 | 轉換時間 | FP16加速 |
|---------|---------|---------|---------|
| EfficientNet-B0 | 21MB | ~2分鐘 | 2.1x |
| EfficientNet-B4 | 75MB | ~5分鐘 | 2.3x |
| ResNet-50 | 98MB | ~3分鐘 | 1.8x |
| ConvNeXt-Tiny | 109MB | ~4分鐘 | 2.0x |

### 推論效能提升
- 比原始 PyTorch: **2-10x** 加速
- FP16 vs FP32: **1.5-3x** 加速
- 批次處理: **線性擴展**

## 🔍 監控與除錯

### 日誌查看
```bash
# 轉換器日誌
docker-compose -f docker-compose.triton.yml logs triton-converter

# Triton 伺服器日誌
docker-compose -f docker-compose.triton.yml logs triton-server
```

### 效能監控
```bash
# GPU 使用率
nvidia-smi -l 1

# 容器資源使用
docker stats triton-converter
```

### 健康檢查
```bash
# Gradio 介面
curl http://localhost:7860

# Triton 伺服器
curl http://localhost:8100/v2/health/ready
```

## 🛡️ 安全考量

1. **網路安全**: 生產環境請配置適當的防火牆規則
2. **檔案權限**: 確保模型檔案的適當存取權限
3. **資源限制**: 配置適當的容器資源限制

## 📞 技術支援

如遇到問題，請提供以下資訊：
1. 系統配置（GPU型號、驅動版本）
2. 模型資訊（架構、大小）
3. 錯誤日誌
4. 轉換參數

---

**版本**: 1.0.0  
**更新時間**: 2025-08-20  
**相容性**: CUDA 12.1+, TensorRT 8.6+