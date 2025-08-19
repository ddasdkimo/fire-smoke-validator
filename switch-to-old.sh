#!/bin/bash

# 切換回舊版本的腳本

echo "🔄 切換回舊版本..."

if [ -f "app_old.py" ]; then
    echo "📁 還原舊版本..."
    cp app_old.py app.py
    echo "✅ 已還原到舊版本"
else
    echo "❌ 找不到備份的舊版本檔案"
fi