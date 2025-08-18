# 火煙誤判標註系統 (Fire/Smoke False Positive Labeling System)

一個基於深度學習的影片分析工具，用於快速標註火災和煙霧偵測中的誤判案例，建立高品質的訓練資料集。

## 功能特點

- 🎥 **影片自動分析**：使用 YOLO 模型自動偵測影片中的火災和煙霧
- 🎯 **ReID 事件分組**：利用物件追蹤技術將相似偵測結果分組為事件
- ⚡ **快速標註介面**：直觀的網頁介面，支援二分類快速標註（真實/誤判）
- 🖼️ **自動幀輪播**：自動輪播事件中的所有幀，便於判斷動態特徵
- 📦 **資料集匯出**：一鍵匯出結構化的訓練資料集（ZIP 格式）
- 🚀 **Mac MPS 加速**：支援 Apple Silicon 的 MPS 加速

## 系統需求

- Python 3.8+ (建議 3.10+)
- macOS (支援 MPS 加速) / Linux / Windows
- 4GB+ RAM
- 支援的影片格式：MP4, MOV, AVI, MKV, WebM

## 安裝

1. Clone 專案
```bash
git clone [your-repo-url]
cd temporal_classification_model
```

2. 安裝依賴套件
```bash
pip install -r requirements.txt
```

3. 下載或準備 YOLO 模型檔案 `best.pt`（放在專案根目錄）

## 使用方法

### 快速開始

```bash
python app.py
```

或使用啟動腳本：
```bash
./run.sh
```

然後在瀏覽器中開啟 `http://localhost:7860`

### 標註流程

1. **上傳影片**：點擊上傳區域選擇影片檔案
2. **開始分析**：點擊「開始分析」按鈕，系統會自動偵測火煙事件
3. **標註事件**：
   - 觀察自動輪播的事件幀
   - 點擊「✅ 真實火煙」或「❌ 誤判」進行標註
   - 系統會自動跳到下一個未標註事件
4. **導航控制**：
   - ⬅️ 上一個：查看前一個事件
   - ➡️ 下一個：查看下一個事件  
   - ⏭️ 跳過：跳過當前事件不標註
5. **匯出資料**：完成標註後點擊「匯出標註資料集」

### 輸出格式

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

## 專案結構

```
.
├── app.py                  # 主應用程式
├── requirements.txt        # Python 依賴套件
├── best.pt                # YOLO 模型檔案（需自行準備）
├── run.sh                 # 啟動腳本
├── README.md              # 專案說明
├── CLAUDE.md              # Claude Code 使用指南
└── tools/                 # 工具模組（保留的舊版工具）
```

## 進階設定

### 調整偵測參數

在 `app.py` 中可以調整：
- `conf=0.3`：偵測信心度閾值（第 144 行）
- `sample_interval`：影片採樣間隔（第 80 行）
- ReID 分組閾值（第 234 行）

### 環境變數

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # 允許 MPS 回退到 CPU
export OMP_NUM_THREADS=4              # 限制執行緒數量
```

## 常見問題

**Q: 影片分析很慢？**
A: 可以調整採樣間隔，減少處理的幀數。預設每秒採樣一幀。

**Q: 記憶體不足？**
A: 程式已內建記憶體管理，每 500 幀會自動釋放記憶體。

**Q: 沒有偵測到任何事件？**
A: 檢查 best.pt 模型是否正確載入，或降低信心度閾值。

## 授權

MIT License

## 貢獻

歡迎提交 Issue 和 Pull Request！