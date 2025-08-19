# 使用支援 CUDA 12.1 的 PyTorch 基礎映像
# 使用 PyTorch 2.3.0 與 CUDA 12.1 以確保與主機 CUDA 12.8 相容
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

# 設定非互動模式，避免時區等提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 設定工作目錄
WORKDIR /app

# 設定時區並安裝系統依賴
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y \
    tzdata \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libnotify-dev \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    ffmpeg \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt
COPY requirements.txt .

# 安裝 Python 依賴
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式碼
COPY . .

# 建立必要的目錄
RUN mkdir -p dataset uploads

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# 暴露 Gradio 預設端口
EXPOSE 7860

# 啟動應用程式 (預設使用三頁籤版本)
CMD ["python", "start.py", "three-tabs"]