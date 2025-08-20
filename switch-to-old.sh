#!/bin/bash

# 切換回舊版本的腳本

echo "🔄 切換回舊版本..."

if [ -f "archive/app_old.py" ]; then
    echo "📁 還原舊版本..."
    cp archive/app_old.py app.py
    echo "✅ 已還原到舊版本 (app_old.py 來自 archive/)"
else
    echo "❌ 找不到備份的舊版本檔案 (請檢查 archive/app_old.py)"
fi