#!/usr/bin/env python3
"""
訓練模組
負責模型訓練功能
"""

import os
import zipfile
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class ModelTrainer:
    """模型訓練器"""
    
    def __init__(self):
        self.training_dir = Path("training_workspace")
        self.training_dir.mkdir(exist_ok=True)
        
        # 支援的模型類型
        self.supported_models = {
            "yolov8n.pt": "YOLOv8 Nano - 輕量級，速度快",
            "yolov8s.pt": "YOLOv8 Small - 平衡性能與速度", 
            "yolov8m.pt": "YOLOv8 Medium - 較高精度",
            "yolov8l.pt": "YOLOv8 Large - 高精度",
            "yolov8x.pt": "YOLOv8 Extra Large - 最高精度"
        }
        
        # 訓練狀態
        self.is_training = False
        self.training_progress = ""
        self.training_results = None
    
    def upload_and_extract_dataset(self, zip_files):
        """上傳並解壓多個標註資料集"""
        try:
            if not zip_files:
                return "請選擇標註資料集ZIP檔案"
            
            # 確保是列表格式
            if not isinstance(zip_files, list):
                zip_files = [zip_files]
            
            # 建立新的合併資料集目錄
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            merged_dataset_dir = self.training_dir / f"merged_dataset_{timestamp}"
            merged_dataset_dir.mkdir(exist_ok=True)
            
            # 建立合併目錄結構
            merged_true_positive_dir = merged_dataset_dir / "true_positive"
            merged_false_positive_dir = merged_dataset_dir / "false_positive"
            merged_true_positive_dir.mkdir(exist_ok=True)
            merged_false_positive_dir.mkdir(exist_ok=True)
            
            total_stats = {
                "true_positive": 0,
                "false_positive": 0,
                "total_images": 0,
                "processed_files": 0,
                "failed_files": []
            }
            
            # 處理每個ZIP檔案
            for i, zip_file in enumerate(zip_files):
                try:
                    print(f"處理第 {i+1}/{len(zip_files)} 個ZIP檔案: {Path(zip_file.name).name}")
                    
                    # 建立臨時解壓目錄
                    temp_extract_dir = merged_dataset_dir / f"temp_extract_{i}"
                    temp_extract_dir.mkdir(exist_ok=True)
                    
                    # 解壓檔案
                    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                        zip_ref.extractall(temp_extract_dir)
                    
                    # 驗證資料集結構
                    validation_result = self._validate_dataset_structure(temp_extract_dir)
                    if not validation_result["valid"]:
                        total_stats["failed_files"].append({
                            "filename": Path(zip_file.name).name,
                            "error": validation_result["error"]
                        })
                        shutil.rmtree(temp_extract_dir)
                        continue
                    
                    # 合併到主資料集
                    self._merge_dataset_to_main(temp_extract_dir, merged_dataset_dir, i)
                    
                    # 累計統計
                    total_stats["true_positive"] += validation_result["stats"]["true_positive"]
                    total_stats["false_positive"] += validation_result["stats"]["false_positive"]
                    total_stats["total_images"] += validation_result["stats"]["total_images"]
                    total_stats["processed_files"] += 1
                    
                    # 清理臨時目錄
                    shutil.rmtree(temp_extract_dir)
                    
                except Exception as e:
                    total_stats["failed_files"].append({
                        "filename": Path(zip_file.name).name,
                        "error": str(e)
                    })
                    # 清理可能的臨時目錄
                    temp_dir = merged_dataset_dir / f"temp_extract_{i}"
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
            
            # 檢查是否有成功處理的檔案
            if total_stats["processed_files"] == 0:
                shutil.rmtree(merged_dataset_dir)
                error_details = "\n".join([f"- {f['filename']}: {f['error']}" for f in total_stats["failed_files"]])
                return f"❌ 所有檔案處理失敗:\n\n{error_details}"
            
            # 轉換為YOLO格式
            yolo_dataset_dir = self._convert_to_yolo_format(merged_dataset_dir, total_stats)
            
            # 生成結果報告
            result_text = f"""✅ 多資料集合併成功！
            
📊 處理結果:
- 成功處理: {total_stats['processed_files']} 個ZIP檔案
- 失敗檔案: {len(total_stats['failed_files'])} 個

📈 合併後統計:
- 真實火煙事件: {total_stats['true_positive']} 個
- 誤判事件: {total_stats['false_positive']} 個  
- 總影像數: {total_stats['total_images']} 張

📁 YOLO資料集路徑: {yolo_dataset_dir}
            """
            
            # 如果有失敗檔案，添加詳細信息
            if total_stats["failed_files"]:
                result_text += "\n\n⚠️ 失敗檔案詳情:\n"
                for failed in total_stats["failed_files"]:
                    result_text += f"- {failed['filename']}: {failed['error']}\n"
            
            return result_text
            
        except Exception as e:
            return f"❌ 資料集處理失敗: {str(e)}"
    
    def _validate_dataset_structure(self, dataset_dir):
        """驗證資料集結構"""
        try:
            true_positive_dir = dataset_dir / "true_positive"
            false_positive_dir = dataset_dir / "false_positive"
            
            if not true_positive_dir.exists() or not false_positive_dir.exists():
                return {"valid": False, "error": "缺少 true_positive 或 false_positive 資料夾"}
            
            # 統計事件和影像數量
            true_events = list(true_positive_dir.iterdir()) if true_positive_dir.exists() else []
            false_events = list(false_positive_dir.iterdir()) if false_positive_dir.exists() else []
            
            total_images = 0
            for event_dir in true_events + false_events:
                if event_dir.is_dir():
                    images = list(event_dir.glob("*.jpg"))
                    total_images += len(images)
            
            stats = {
                "true_positive": len(true_events),
                "false_positive": len(false_events),
                "total_images": total_images
            }
            
            if total_images == 0:
                return {"valid": False, "error": "未找到任何影像檔案"}
            
            return {"valid": True, "stats": stats}
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def _merge_dataset_to_main(self, source_dataset_dir, target_dataset_dir, dataset_index):
        """將單個資料集合併到主資料集"""
        try:
            source_true_dir = source_dataset_dir / "true_positive"
            source_false_dir = source_dataset_dir / "false_positive"
            target_true_dir = target_dataset_dir / "true_positive"
            target_false_dir = target_dataset_dir / "false_positive"
            
            # 合併 true_positive 事件
            if source_true_dir.exists():
                for event_dir in source_true_dir.iterdir():
                    if event_dir.is_dir():
                        # 重命名事件目錄避免衝突
                        new_event_name = f"dataset_{dataset_index}_{event_dir.name}"
                        target_event_dir = target_true_dir / new_event_name
                        shutil.copytree(event_dir, target_event_dir)
            
            # 合併 false_positive 事件
            if source_false_dir.exists():
                for event_dir in source_false_dir.iterdir():
                    if event_dir.is_dir():
                        # 重命名事件目錄避免衝突
                        new_event_name = f"dataset_{dataset_index}_{event_dir.name}"
                        target_event_dir = target_false_dir / new_event_name
                        shutil.copytree(event_dir, target_event_dir)
                        
        except Exception as e:
            print(f"合併資料集時發生錯誤: {e}")
            raise e
    
    def _convert_to_yolo_format(self, dataset_dir, stats):
        """轉換為YOLO訓練格式"""
        # 這裡建立基本的YOLO資料集結構
        # 實際實作時會根據具體需求調整
        yolo_dir = dataset_dir.parent / f"yolo_{dataset_dir.name}"
        yolo_dir.mkdir(exist_ok=True)
        
        # 建立基本目錄結構
        (yolo_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 建立dataset.yaml配置檔
        dataset_config = {
            "train": str(yolo_dir / "images" / "train"),
            "val": str(yolo_dir / "images" / "val"),
            "nc": 2,  # 類別數量: 火煙, 無火煙
            "names": ["fire_smoke", "no_fire_smoke"]
        }
        
        with open(yolo_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            if YAML_AVAILABLE:
                yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
            else:
                # 如果沒有 yaml，手動寫入配置
                f.write(f"train: {dataset_config['train']}\n")
                f.write(f"val: {dataset_config['val']}\n")
                f.write(f"nc: {dataset_config['nc']}\n")
                f.write(f"names: {dataset_config['names']}\n")
        
        return yolo_dir
    
    def get_available_models(self):
        """取得可用的預訓練模型列表"""
        return list(self.supported_models.keys())
    
    def get_model_info(self, model_name):
        """取得模型資訊"""
        return self.supported_models.get(model_name, "未知模型")
    
    def start_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始訓練模型"""
        try:
            if not ULTRALYTICS_AVAILABLE:
                return "❌ 未安裝 ultralytics 套件，無法進行訓練"
            
            if self.is_training:
                return "⚠️ 已有訓練正在進行中"
            
            # 驗證參數
            if not Path(dataset_path).exists():
                return "❌ 資料集路徑不存在"
            
            # 設定訓練參數
            self.is_training = True
            self.training_progress = "🚀 準備開始訓練..."
            
            # 實際訓練邏輯會在這裡實作
            # 這裡先建立框架
            
            return f"""✅ 訓練任務已啟動！
            
🎯 訓練設定:
- 資料集: {dataset_path}
- 基礎模型: {model_name}
- 訓練輪數: {epochs}
- 批次大小: {batch_size}
- 影像尺寸: {image_size}

📊 請關注下方進度顯示...
            """
            
        except Exception as e:
            self.is_training = False
            return f"❌ 訓練啟動失敗: {str(e)}"
    
    def get_training_progress(self):
        """取得訓練進度"""
        if not self.is_training:
            return "等待開始訓練..."
        return self.training_progress
    
    def stop_training(self):
        """停止訓練"""
        if self.is_training:
            self.is_training = False
            self.training_progress = "⏹️ 訓練已停止"
            return "✅ 訓練已停止"
        return "⚠️ 沒有正在進行的訓練"
    
    def list_trained_models(self):
        """列出已訓練的模型"""
        runs_dir = Path("runs/detect")
        if not runs_dir.exists():
            return []
        
        models = []
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir():
                weights_dir = run_dir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        models.append({
                            "name": run_dir.name,
                            "path": str(best_pt),
                            "created": best_pt.stat().st_mtime
                        })
        
        # 按建立時間排序
        models.sort(key=lambda x: x["created"], reverse=True)
        return models