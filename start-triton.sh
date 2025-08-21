#!/bin/bash
set -e

echo "🚀 TIMM to TensorRT 轉換器啟動腳本"
echo "=================================="

# 檢查系統要求
echo "📋 檢查系統要求..."

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安裝，請先安裝 Docker"
    exit 1
fi

# 檢查 Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose 未安裝，請先安裝 Docker Compose"
    exit 1
fi

# 檢查 NVIDIA Docker
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkit 未正確配置"
    echo "請參考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

echo "✅ 系統要求檢查通過"

# 創建必要目錄
echo "📁 創建必要目錄..."
mkdir -p triton/{models,converted,temp,triton_models}
echo "✅ 目錄創建完成"

# 設定權限
echo "🔧 設定目錄權限..."
chmod -R 755 triton/
echo "✅ 權限設定完成"

# 詢問啟動模式
echo ""
echo "請選擇啟動模式："
echo "1) 僅啟動轉換器 (推薦)"
echo "2) 啟動轉換器 + Triton 推論伺服器"
echo ""
read -p "請輸入選項 [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo "🔄 啟動 TIMM to TensorRT 轉換器..."
        docker-compose -f docker-compose.triton.yml up --build triton-converter
        ;;
    2)
        echo "🔄 啟動轉換器 + Triton 推論伺服器..."
        docker-compose -f docker-compose.triton.yml up --build
        ;;
    *)
        echo "❌ 無效選項，預設啟動轉換器"
        docker-compose -f docker-compose.triton.yml up --build triton-converter
        ;;
esac

echo ""
echo "🎉 服務已啟動！"
echo ""
echo "📱 訪問方式："
echo "- Gradio Web 介面: http://localhost:7860"
if [ "$choice" = "2" ]; then
    echo "- Triton 推論伺服器: http://localhost:8100"
fi
echo ""
echo "💡 使用說明請參考: triton/README.md"