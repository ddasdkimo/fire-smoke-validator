# HPWREN FIgLib 手動下載指南

由於 HPWREN 的 CDN 伺服器有防護機制，無法透過程式直接下載。請按照以下步驟手動下載資料：

## 步驟 1：訪問資料集頁面

1. 開啟瀏覽器，訪問：https://www.hpwren.ucsd.edu/FIgLib/
2. 點擊 "Data Set" 連結

## 步驟 2：下載影像序列

1. 訪問 Tar 目錄：https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/index.html
2. 選擇想要的火災序列下載（建議先下載 5-10 個作為測試）
3. 檔名格式說明：
   - `YYYYMMDD_FIRE_location.tgz`
   - 例如：`20190514_FIRE_mw-e-mobo-c.tgz`
4. 將下載的 .tgz 檔案放到 `data/HPWREN_FIgLib/images/` 目錄

## 步驟 3：下載標註檔案

### CSV 標註
1. 訪問：https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Miscellaneous/Labels/HPWREN-BB/CSV/index.html
2. 下載對應的 .csv 檔案
3. 放到 `data/HPWREN_FIgLib/labels/CSV/` 目錄

### XML 標註（PASCAL VOC 格式）
1. 訪問：https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Miscellaneous/Labels/HPWREN-BB/VOC/index.html
2. 下載對應的 .xml 檔案
3. 放到 `data/HPWREN_FIgLib/labels/XML/` 目錄

## 步驟 4：解壓縮資料

在專案根目錄執行：
```bash
cd data/HPWREN_FIgLib/images
for file in *.tgz; do tar -xzf "$file"; done
```

## 資料集結構

解壓縮後，每個火災序列資料夾包含：
- 連續的 JPEG 影像（火災前後各約 20 分鐘）
- 一個 MP4 延時影片
- 檔名包含時間戳記和相對於火災出現的時間偏移

## 建議下載清單（2019-2020 年的火災）

優先下載這些較新的資料集：
- 20190514_FIRE_mw-e-mobo-c.tgz
- 20190717_FIRE_sm-n-mobo-c.tgz
- 20190906_FIRE_hp-w-mobo-c.tgz
- 20191006_FIRE_sm-e-mobo-c.tgz
- 20200820_FIRE_lp-s-mobo-c.tgz

## 注意事項

1. 完整資料集約 30GB，建議先下載部分測試
2. 確保有足夠的硬碟空間
3. 下載速度可能較慢，請耐心等待
4. 如需完整資料集，可能需要聯繫 HPWREN 團隊