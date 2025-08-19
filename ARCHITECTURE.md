# 🏗️ 系統架構說明

## 📁 專案結構

重構後的模組化架構，讓代碼更清晰易維護：

```
fire-smoke-validator/
│
├── 📄 app.py                    # 主程式 (簡化版，僅 63 行)
├── 📄 app_old.py                # 原始版本備份 (1200+ 行)
├── 📄 app_new.py                # 新版本原始檔案
│
├── 📂 core/                     # 核心功能模組
│   ├── __init__.py
│   ├── analyzer.py              # 視頻分析器 (主要邏輯)
│   └── labeling.py              # 標註和資料匯出管理
│
├── 📂 ui/                       # 使用者介面模組
│   ├── __init__.py
│   ├── interface.py             # 介面控制器
│   └── builder.py               # Gradio 介面建構器
│
├── 📂 tools/                    # 額外工具 (保持原有)
├── 📂 docs/                     # 文檔
├── 🐳 Dockerfile               # Docker 配置
├── 🐳 docker-compose.yaml      # 服務編排 (包含代碼掛載)
└── 🐳 docker-compose.dev.yaml  # 開發環境覆蓋
```

## 🧩 模組功能說明

### 1. 📄 `app.py` (主程式)
**63 行** → 原來 **1200+ 行**，**95% 代碼量減少**

```python
# 僅負責系統初始化和啟動
from core.analyzer import VideoAnalyzer
from core.labeling import LabelingManager  
from ui.interface import InterfaceController
from ui.builder import create_interface
```

### 2. 🧠 `core/analyzer.py` (核心分析器)
**責任**: 視頻處理、物件偵測、事件分組
- 視頻並發分析
- YOLO 模型管理
- ReID 事件分組
- 記憶體優化
- 進度追蹤

### 3. 🏷️ `core/labeling.py` (標註管理)
**責任**: 標註邏輯、進度統計、資料匯出
- 事件標註
- 進度計算
- 按影片分組統計
- ZIP 格式匯出

### 4. 🎮 `ui/interface.py` (介面控制器)  
**責任**: 使用者互動邏輯
- 模型載入控制
- 分析流程控制
- 事件導航邏輯
- 標註操作處理
- 彈出提醒管理

### 5. 🖼️ `ui/builder.py` (介面建構器)
**責任**: Gradio 元件建立和佈局
- 介面元件定義
- 事件綁定
- 佈局設計

## ✨ 重構優勢

### 📊 代碼品質提升
- **可讀性**: 每個文件專注單一職責
- **可維護性**: 模組化設計，易於修改和擴展
- **可測試性**: 各模組可獨立測試
- **可重用性**: 核心功能可在其他項目中重用

### 🔧 開發體驗改善  
- **開發熱重載**: 修改任何模組都能即時生效
- **調試友好**: 錯誤堆棧更清晰
- **新功能添加**: 更容易找到相關代碼位置

### 🚀 效能最佳化
- **記憶體管理**: 集中在 analyzer.py 
- **並發處理**: 獨立的分析邏輯
- **資源清理**: 統一的清理機制

## 🔄 版本切換

### 切換到重構版
```bash
./switch-to-new.sh    # 啟用模組化版本
```

### 回到原版本
```bash  
./switch-to-old.sh    # 回到單文件版本
```

## 🐳 Docker 整合

新架構完全支援 Docker 開發模式：

```bash
# 開發模式 (代碼熱重載)
./dev-start.sh

# 或手動啟動
docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
```

**掛載的模組**:
- `./core` → `/app/core` 
- `./ui` → `/app/ui`
- `./app.py` → `/app/app.py`

## 📈 擴展計劃

模組化架構為未來擴展奠定基礎：

- **`core/models/`**: 多模型支援
- **`core/exporters/`**: 多格式匯出 
- **`ui/themes/`**: 自定義主題
- **`api/`**: REST API 介面
- **`plugins/`**: 插件系統

---

**總結**: 重構後的代碼更清晰、更好維護，同時保持所有原有功能完整性。開發體驗大幅提升！