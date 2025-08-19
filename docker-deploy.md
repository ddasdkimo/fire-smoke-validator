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