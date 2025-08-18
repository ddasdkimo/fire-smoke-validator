#!/bin/bash
# 火煙誤判標註系統啟動腳本

echo "🔥 火煙誤判標註系統啟動腳本"
echo "================================"

# 檢查 Python 版本
python3 --version

# 檢查必要套件
echo ""
echo "檢查必要套件..."
python3 -c "import gradio; print(f'✅ Gradio {gradio.__version__}')" 2>/dev/null || echo "❌ Gradio 未安裝"
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__}')" 2>/dev/null || echo "❌ OpenCV 未安裝"
python3 -c "import torch; print(f'✅ PyTorch {torch.__version__}')" 2>/dev/null || echo "❌ PyTorch 未安裝"

# 檢查 best.pt 模型檔案
echo ""
if [ -f "best.pt" ]; then
    echo "✅ 找到 best.pt 模型檔案"
else
    echo "⚠️  未找到 best.pt 模型檔案，將使用模擬偵測"
fi

# 設定環境變數以優化效能
export PYTORCH_ENABLE_MPS_FALLBACK=1  # 允許 MPS 回退到 CPU
export OMP_NUM_THREADS=4              # 限制執行緒數量

echo ""
echo "啟動應用程式..."
echo "================================"
echo "請在瀏覽器中開啟 http://localhost:7860"
echo ""

# 啟動應用程式
python3 app.py