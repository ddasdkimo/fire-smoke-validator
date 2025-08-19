# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

這是一個**火煙誤判標註系統** - 簡化的網頁應用程式，用於快速建立訓練資料集。上傳影片以使用 best.pt 模型自動掃描，使用 ReID 分組相似事件，並快速標註真實火煙與誤判。

## 開發指令

### 環境設定
```bash
# 安裝所有依賴套件
pip install -r requirements.txt

# 開發模式安裝套件（選擇性）
pip install -e .
```

### 程式碼品質
```bash
# 使用 black 格式化程式碼
black app.py tools/

# 使用 flake8 檢查程式碼  
flake8 app.py tools/

# 執行測試（如果有的話）
pytest tests/
```

### 主要應用程式
```bash
# 啟動網頁介面
python app.py

# 使用啟動腳本
./run.sh

# 替代的 ReID 標註介面
python start_reid_labeling.py
```

## 架構總覽

### 核心應用程式
- **主程式**: `app.py` - 基於 Gradio 的影片分析和標註網頁介面（1,181 行）
- **影片分析**: 使用 best.pt YOLO 模型自動掃描上傳的影片
- **ReID 分組**: 將跨幀的相似偵測結果分組為事件
- **快速標註**: 二分類介面（真實火煙 vs 誤判）
- **資料集匯出**: 匯出標註資料為結構化訓練資料集

### 關鍵技術
- **偵測**: Ultralytics YOLO（預訓練模型位於 `best.pt`）
- **追蹤**: Supervision 函式庫用於物件追蹤和 ReID 分組
- **介面**: Gradio 網頁框架
- **視覺**: OpenCV 處理影片
- **加速**: 支援 Mac MPS、CUDA 和 CPU 推論

### 資料流程
1. 上傳影片 → 擷取幀
2. 對幀執行 YOLO 偵測
3. 透過 ReID 相似度分組偵測結果
4. 呈現事件縮圖供標註
5. 匯出標註資料集為 ZIP

### 輸出結構
```
dataset/
└── export_YYYYMMDD_HHMMSS.zip
    ├── true_positive/          # 真實火煙事件
    │   ├── event_0/
    │   │   ├── frame_000_1.2s.jpg
    │   │   ├── frame_001_1.7s.jpg  
    │   │   └── metadata.json
    │   └── event_1/
    └── false_positive/         # 誤判事件
        ├── event_2/
        └── event_3/
```

## 關鍵考量

- 從原始複雜的時序分類專案簡化而來
- 專注於快速誤判資料集建立
- 使用現有的 best.pt 模型進行偵測
- 二分類標註針對速度最佳化
- 輕量化架構，移除不必要的元件
- 支援 Mac MPS 加速模型推論
- 內建記憶體管理（每 500 幀自動釋放）
- 支援並行分析多個影片
- 自動裝置偵測（CPU/CUDA/MPS）

## 調整參數

在 `app.py` 中可調整的重要參數：
- `conf=0.3`: 偵測信心度閾值（第 144 行）
- `sample_interval`: 影片採樣間隔（第 80 行） 
- ReID 分組閾值（第 234 行）

## 環境變數

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # 允許 MPS 回退到 CPU
export OMP_NUM_THREADS=4              # 限制執行緒數量
```

## 工具目錄

- `tools/reid_labeling_interface.py`: 進階時序標註工具，包含 ReID 功能
- `tools/data_labeling_interface.py`: 簡單的火煙分類工具

## 目前狀態

- ✅ 簡化的網頁應用程式完成
- ✅ 影片上傳和分析功能正常
- ✅ ReID 事件分組已實作  
- ✅ 快速二分類標註介面就緒
- ✅ 資料集匯出功能完成
- ⚠️ 無正式測試框架（雖然 requirements.txt 包含 pytest）