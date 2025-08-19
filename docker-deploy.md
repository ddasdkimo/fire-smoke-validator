# Docker 部署指南 - NVIDIA GPU 主機

## 前置需求

1. **NVIDIA GPU 驅動程式**
   ```bash
   nvidia-smi  # 確認 GPU 驅動已安裝
   ```

2. **Docker 和 Docker Compose**
   ```bash
   docker --version
   docker-compose --version
   ```

3. **NVIDIA Container Toolkit**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

## 部署步驟

### 生產環境部署

1. **準備模型檔案**
   ```bash
   # 將 best.pt 模型檔案放在專案根目錄
   ls -la best.pt
   ```

2. **建立必要目錄**
   ```bash
   mkdir -p dataset uploads videos
   ```

3. **啟動服務**
   ```bash
   # 建構並啟動
   docker-compose up -d --build
   
   # 查看日誌
   docker-compose logs -f
   ```

4. **訪問應用程式**
   - 開啟瀏覽器訪問: `http://<主機IP>:7860`

### 開發環境部署 (代碼熱重載)

如果您需要頻繁修改代碼，建議使用開發模式，避免每次都重新編譯：

1. **準備開發環境**
   ```bash
   # 將 best.pt 模型檔案放在專案根目錄
   ls -la best.pt
   ```

2. **使用開發模式啟動**
   ```bash
   # 方法 1: 使用開發腳本 (推薦)
   ./dev-start.sh
   
   # 方法 2: 手動啟動
   docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up
   ```

3. **開發模式特點**
   - ✅ **代碼熱重載**: 修改 `app.py` 和 `tools/` 目錄後自動生效
   - ✅ **無需重建**: 不用執行 `--build` 參數
   - ✅ **調試模式**: 啟用詳細的錯誤信息
   - ✅ **快速啟動**: 跳過健康檢查以加快啟動速度

4. **開發常用指令**
   ```bash
   # 快速重啟 (保持代碼掛載)
   ./dev-restart.sh
   
   # 查看即時日誌
   docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml logs -f
   
   # 停止開發環境
   docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml down
   ```

## 常用指令

```bash
# 停止服務
docker-compose down

# 重新啟動
docker-compose restart

# 查看容器狀態
docker-compose ps

# 進入容器
docker-compose exec fire-smoke-validator bash

# 更新後重新部署
docker-compose down
docker-compose up -d --build

# 清理未使用的映像
docker system prune -a
```

## 配置調整

### GPU 配置
在 `docker-compose.yaml` 中調整:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # 使用多個 GPU
```

### 記憶體限制
```yaml
mem_limit: 16g  # 增加記憶體限制
shm_size: 4g    # 增加共享記憶體
```

### 端口變更
```yaml
ports:
  - "8080:7860"  # 改為 8080 端口
```

## 效能優化

1. **使用 GPU 加速**
   - 確保 `CUDA_VISIBLE_DEVICES` 設定正確
   - 監控 GPU 使用: `nvidia-smi -l 1`

2. **調整批次大小**
   - 在 `app.py` 中調整推論批次大小以充分利用 GPU

3. **使用持久化卷**
   - 避免重複下載模型和處理檔案

## 故障排除

1. **GPU 無法使用**
   ```bash
   # 檢查 NVIDIA runtime
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. **記憶體不足**
   - 增加 Docker 的記憶體限制
   - 調整 `shm_size` 參數

3. **連線問題**
   - 檢查防火牆設定
   - 確認端口 7860 已開放

## 生產環境建議

1. **使用反向代理**
   ```nginx
   location / {
       proxy_pass http://localhost:7860;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection 'upgrade';
       proxy_set_header Host $host;
       proxy_cache_bypass $http_upgrade;
   }
   ```

2. **啟用 HTTPS**
   - 使用 Let's Encrypt 或其他 SSL 憑證

3. **監控和日誌**
   - 整合 Prometheus/Grafana 監控
   - 使用 ELK Stack 收集日誌

4. **備份策略**
   - 定期備份 `dataset/` 目錄
   - 使用卷快照功能