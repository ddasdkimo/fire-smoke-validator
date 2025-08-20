#!/bin/bash

# 切換到新版本的腳本

echo "🔄 切換到重構版本..."

# 備份舊版本
if [ ! -f "archive/app_old.py" ]; then
    echo "📁 備份舊版本..."
    mkdir -p archive
    cp app.py archive/app_old.py
fi

# 啟用新版本
echo "✨ 啟用新版本..."
if [ -f "archive/app_new.py" ]; then
    cp archive/app_new.py app.py
else
    echo "⚠️ 使用當前模組化版本 app.py"
fi

echo "✅ 切換完成！"
echo ""
echo "📋 變更說明:"
echo "  - 核心分析功能: core/analyzer.py"
echo "  - 標註管理功能: core/labeling.py" 
echo "  - 介面控制器: ui/interface.py"
echo "  - 介面建構器: ui/builder.py"
echo "  - 主程式: app.py (簡化版)"
echo ""
echo "🚀 現在可以啟動應用:"
echo "  python app.py"
echo "  或使用開發模式: ./dev-start.sh"