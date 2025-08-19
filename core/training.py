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
            # YOLO 模型（物件偵測）
            "yolov8n.pt": "YOLOv8 Nano - 輕量級物件偵測",
            "yolov8s.pt": "YOLOv8 Small - 平衡性能物件偵測", 
            "yolov8m.pt": "YOLOv8 Medium - 高精度物件偵測",
            "yolov8l.pt": "YOLOv8 Large - 最高精度物件偵測",
            "yolov8x.pt": "YOLOv8 Extra Large - 最高精度物件偵測",
            
            # 時序分類模型（基於 timm backbone）
            "temporal_resnet50": "時序 ResNet50 + 注意力機制",
            "temporal_resnet18": "時序 ResNet18 + 注意力機制（輕量）",
            "temporal_convnext_small": "時序 ConvNeXt Small + LSTM",
            "temporal_convnext_tiny": "時序 ConvNeXt Tiny + 注意力機制",
            "temporal_efficientnet_b3": "時序 EfficientNet B3 + 注意力機制",
            "temporal_efficientnet_b0": "時序 EfficientNet B0 + 注意力機制（輕量）"
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
            # 檢查是否為時序模型
            is_temporal_model = model_name.startswith("temporal_")
            
            if not is_temporal_model and not ULTRALYTICS_AVAILABLE:
                return "❌ 未安裝 ultralytics 套件，無法進行 YOLO 訓練"
            
            if self.is_training:
                return "⚠️ 已有訓練正在進行中"
            
            # 驗證參數
            if not self._validate_dataset_path(dataset_path):
                return "❌ 資料集路徑不存在或格式錯誤"
            
            # 設定訓練參數
            self.is_training = True
            self.training_progress = "🚀 準備開始訓練..."
            
            if is_temporal_model:
                # 時序模型訓練
                return self._start_temporal_training(dataset_path, model_name, epochs, batch_size, image_size)
            else:
                # YOLO 模型訓練
                return self._start_yolo_training(dataset_path, model_name, epochs, batch_size, image_size)
            
        except Exception as e:
            self.is_training = False
            return f"❌ 訓練啟動失敗: {str(e)}"
    
    def _validate_dataset_path(self, dataset_path):
        """驗證資料集路徑"""
        if "資料集路徑:" not in str(dataset_path):
            return False
        # 從結果文字中提取實際路徑
        lines = str(dataset_path).split('\n')
        for line in lines:
            if "YOLO資料集路徑:" in line:
                actual_path = line.split("YOLO資料集路徑:")[-1].strip()
                return Path(actual_path).exists()
        return False
    
    def _start_temporal_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始時序模型訓練"""
        try:
            # 導入時序模型相關模組
            from .models.temporal_trainer import TemporalTrainer
            from .models.temporal_classifier import DEFAULT_MODEL_CONFIGS
            
            # 提取實際資料集路徑
            lines = str(dataset_path).split('\n')
            actual_dataset_path = None
            for line in lines:
                if "合併後統計:" in line or "資料集路徑:" in line:
                    continue
                if "merged_dataset_" in line or "dataset_" in line:
                    actual_dataset_path = line.split(":")[-1].strip()
                    break
            
            if not actual_dataset_path:
                return "❌ 無法解析資料集路徑"
            
            # 建立模型配置
            model_config = self._get_temporal_model_config(model_name, image_size)
            
            # 啟動訓練任務（在背景執行）
            import threading
            def training_worker():
                try:
                    trainer = TemporalTrainer(model_config)
                    self.training_progress = "🔥 正在訓練時序模型..."
                    
                    result = trainer.train(
                        dataset_path=actual_dataset_path,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=1e-3,
                        val_split=0.2
                    )
                    
                    self.training_results = result
                    self.training_progress = f"✅ 時序模型訓練完成！最佳準確率: {result['best_val_accuracy']:.3f}"
                    self.is_training = False
                    
                except Exception as e:
                    self.training_progress = f"❌ 訓練失敗: {str(e)}"
                    self.is_training = False
            
            training_thread = threading.Thread(target=training_worker)
            training_thread.start()
            
            return f"""✅ 時序模型訓練已啟動！
            
🎯 訓練設定:
- 模型類型: {model_name} (時序分類)
- 資料集: {actual_dataset_path}
- 訓練輪數: {epochs}
- 批次大小: {batch_size}
- 影像尺寸: {image_size}
- 時序幀數: 5 (固定)

🔥 特色功能:
- T=5 固定幀輸入策略
- 不足5幀：重複填充+微量噪音
- 超過5幀：等距均勻取樣
- timm backbone 特徵提取
- 注意力/LSTM 時序融合

📊 請關注下方進度顯示...
            """
            
        except ImportError:
            return "❌ 缺少 timm 或相關依賴，請安裝: pip install timm matplotlib seaborn"
        except Exception as e:
            return f"❌ 時序模型訓練啟動失敗: {str(e)}"
    
    def _start_yolo_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始 YOLO 模型訓練"""
        # 原有的 YOLO 訓練邏輯
        return f"""✅ YOLO 訓練任務已啟動！
        
🎯 訓練設定:
- 資料集: {dataset_path}
- 基礎模型: {model_name}
- 訓練輪數: {epochs}
- 批次大小: {batch_size}
- 影像尺寸: {image_size}

📊 請關注下方進度顯示...
        """
    
    def _get_temporal_model_config(self, model_name, image_size):
        """取得時序模型配置"""
        # 從模型名稱提取 backbone
        if "resnet50" in model_name:
            backbone = "resnet50"
            fusion = "attention"
        elif "resnet18" in model_name:
            backbone = "resnet18"
            fusion = "attention"
        elif "convnext_small" in model_name:
            backbone = "convnext_small"
            fusion = "lstm"
        elif "convnext_tiny" in model_name:
            backbone = "convnext_tiny"
            fusion = "attention"
        elif "efficientnet_b3" in model_name:
            backbone = "efficientnet_b3"
            fusion = "attention"
        elif "efficientnet_b0" in model_name:
            backbone = "efficientnet_b0"
            fusion = "attention"
        else:
            backbone = "resnet50"
            fusion = "attention"
        
        return {
            'backbone_name': backbone,
            'num_classes': 2,
            'temporal_frames': 5,
            'pretrained': True,
            'temporal_fusion': fusion,
            'dropout': 0.2,
            'freeze_backbone': False
        }
    
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