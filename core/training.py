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
        
        # 支援的時序分類模型（按照用途分類）
        self.supported_models = {
            # 低延遲（Edge/即時）- 適合顆粒小、ROI多、GPU預算緊的場景
            "temporal_mobilenetv3_large": "🚀 MobileNetV3-Large + 注意力 - 移動端優化、極低延遲",
            "temporal_ghostnet_100": "🚀 GhostNet-100 + 注意力 - 輕量高效、參數少",
            "temporal_repvgg_b0": "🚀 RepVGG-B0 + 注意力 - 推論優化、部署友善",
            "temporal_efficientnet_b0": "🚀 EfficientNet-B0 + 注意力 - 經典輕量、穩定",
            "temporal_efficientnet_b1": "🚀 EfficientNet-B1 + 注意力 - 輕量升級版",
            
            # 均衡（速度/準度平衡，建議起手）
            "temporal_convnext_tiny": "⚖️ ConvNeXt-Tiny + 注意力 - 常用穩定、首選平衡",
            "temporal_efficientnetv2_s": "⚖️ EfficientNetV2-S + 注意力 - 速度準度平衡",
            "temporal_resnet50d": "⚖️ ResNet50D + 注意力 - 改良版ResNet、可靠",
            "temporal_resnet50": "⚖️ ResNet50 + 注意力 - 經典骨幹、廣泛驗證",
            "temporal_regnety_032": "⚖️ RegNetY-032 + 注意力 - 高效網路架構",
            
            # 準度優先（算力可接受）- 適合離線/準即時場景
            "temporal_convnext_base": "🎯 ConvNeXt-Base + LSTM - 高準度、大模型",
            "temporal_efficientnetv2_m": "🎯 EfficientNetV2-M + LSTM - 中大型高精度",
            "temporal_swin_tiny": "🎯 Swin-Tiny + 注意力 - Transformer視覺模型",
            "temporal_vit_small": "🎯 ViT-Small + 注意力 - 純Transformer（需充足資料）"
        }
        
        # 訓練狀態 - 使用持久化狀態文件
        self.state_file = Path("training_workspace/training_state.json")
        self._load_training_state()
        
        # 檢查並重置僵死的訓練狀態
        self._check_and_reset_training_state()
    
    def _load_training_state(self):
        """載入持久化的訓練狀態"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self.is_training = state.get('is_training', False)
                self.training_progress = state.get('training_progress', '')
                self.training_results = state.get('training_results', None)
                print(f"✅ [STATE] 載入訓練狀態: is_training={self.is_training}")
            else:
                # 預設狀態
                self.is_training = False
                self.training_progress = ""
                self.training_results = None
                print("🔧 [STATE] 使用預設訓練狀態")
        except Exception as e:
            print(f"⚠️ [STATE] 載入狀態失敗: {e}")
            self.is_training = False
            self.training_progress = ""
            self.training_results = None
    
    def _save_training_state(self):
        """保存訓練狀態到文件"""
        try:
            state = {
                'is_training': self.is_training,
                'training_progress': self.training_progress,
                'training_results': self.training_results
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"💾 [STATE] 保存訓練狀態: is_training={self.is_training}")
        except Exception as e:
            print(f"⚠️ [STATE] 保存狀態失敗: {e}")
    
    def _check_and_reset_training_state(self):
        """檢查並重置僵死的訓練狀態"""
        try:
            import psutil
            import os
            
            # 檢查是否有實際的 Python 訓練進程在運行
            training_process_found = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if cmdline and any('training' in str(cmd).lower() for cmd in cmdline):
                            training_process_found = True
                            print(f"🔍 [DEBUG-INIT] 發現訓練進程: PID {proc.info['pid']}")
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # 如果沒有找到實際的訓練進程，重置狀態
            if not training_process_found and hasattr(self, 'is_training'):
                if self.is_training:
                    print("🔧 [DEBUG-INIT] 重置僵死的訓練狀態")
                    self.is_training = False
                    self.training_progress = ""
                else:
                    print("✅ [DEBUG-INIT] 訓練狀態正常")
            else:
                print(f"🔍 [DEBUG-INIT] 訓練進程檢查: 找到={training_process_found}")
                
        except ImportError:
            print("⚠️ [DEBUG-INIT] psutil 不可用，跳過進程檢查")
        except Exception as e:
            print(f"⚠️ [DEBUG-INIT] 狀態檢查失敗: {e}")
    
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
            
            # 轉換為時序分類格式
            temporal_dataset_dir = self._convert_to_temporal_format(merged_dataset_dir, total_stats)
            
            # 生成結果報告
            result_text = f"""✅ 多資料集合併成功！
            
📊 處理結果:
- 成功處理: {total_stats['processed_files']} 個ZIP檔案
- 失敗檔案: {len(total_stats['failed_files'])} 個

📈 合併後統計:
- 真實火煙事件: {total_stats['true_positive']} 個
- 誤判事件: {total_stats['false_positive']} 個  
- 總影像數: {total_stats['total_images']} 張

📁 時序分類資料集路徑: {temporal_dataset_dir}
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
    
    def _convert_to_temporal_format(self, dataset_dir, stats):
        """轉換為時序分類訓練格式"""
        # 這裡建立基本的時序分類資料集結構
        # 實際實作時會根據具體需求調整
        temporal_dir = dataset_dir.parent / f"temporal_{dataset_dir.name}"
        temporal_dir.mkdir(exist_ok=True)
        
        # 建立基本目錄結構
        (temporal_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (temporal_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (temporal_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (temporal_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # 建立dataset.yaml配置檔
        dataset_config = {
            "train": str(temporal_dir / "images" / "train"),
            "val": str(temporal_dir / "images" / "val"),
            "nc": 2,  # 類別數量: 火煙, 無火煙
            "names": ["fire_smoke", "no_fire_smoke"]
        }
        
        with open(temporal_dir / "dataset.yaml", "w", encoding="utf-8") as f:
            if YAML_AVAILABLE:
                yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
            else:
                # 如果沒有 yaml，手動寫入配置
                f.write(f"train: {dataset_config['train']}\n")
                f.write(f"val: {dataset_config['val']}\n")
                f.write(f"nc: {dataset_config['nc']}\n")
                f.write(f"names: {dataset_config['names']}\n")
        
        return temporal_dir
    
    def get_available_models(self):
        """取得可用的預訓練模型列表"""
        return list(self.supported_models.keys())
    
    def get_model_info(self, model_name):
        """取得模型資訊"""
        base_info = self.supported_models.get(model_name, "未知模型")
        
        # 獲取推薦的輸入尺寸
        temp_config = self._get_temporal_model_config(model_name, 0)  # 傳入0讓它使用預設值
        recommended_size = temp_config.get('default_input_size', 224)
        
        return f"{base_info}\n📐 推薦輸入尺寸: {recommended_size}x{recommended_size}"
    
    def get_recommended_input_size(self, model_name):
        """取得模型推薦的輸入尺寸"""
        temp_config = self._get_temporal_model_config(model_name, 0)
        return temp_config.get('default_input_size', 224)
    
    def update_model_selection(self, model_name):
        """當模型選擇改變時更新資訊和建議尺寸"""
        model_info = self.get_model_info(model_name)
        recommended_size = self.get_recommended_input_size(model_name)
        return model_info, recommended_size
    
    def start_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始時序模型訓練"""
        try:
            print(f"🔍 [DEBUG-CORE] start_training 被調用:")
            print(f"   dataset_path: {repr(dataset_path)}")
            print(f"   model_name: {model_name}")
            print(f"   is_training: {self.is_training}")
            
            if self.is_training:
                print("⚠️ [DEBUG-CORE] 已有訓練正在進行中")
                return "⚠️ 已有訓練正在進行中"
            
            # 驗證參數
            print(f"🔍 [DEBUG-CORE] 開始驗證資料集路徑...")
            path_valid = self._validate_dataset_path(dataset_path)
            print(f"🔍 [DEBUG-CORE] 路徑驗證結果: {path_valid}")
            
            if not path_valid:
                print("❌ [DEBUG-CORE] 資料集路徑驗證失敗")
                return "❌ 資料集路徑不存在或格式錯誤"
            
            # 設定訓練參數
            self.is_training = True
            self.training_progress = "🚀 準備開始訓練..."
            self._save_training_state()  # 保存狀態
            
            # 時序模型訓練
            return self._start_temporal_training(dataset_path, model_name, epochs, batch_size, image_size)
            
        except Exception as e:
            self.is_training = False
            return f"❌ 訓練啟動失敗: {str(e)}"
    
    def _validate_dataset_path(self, dataset_path):
        """驗證資料集路徑"""
        print(f"🔍 [DEBUG-VALIDATE] _validate_dataset_path 被調用:")
        print(f"   dataset_path 類型: {type(dataset_path)}")
        print(f"   dataset_path 內容: {repr(dataset_path)}")
        
        dataset_str = str(dataset_path)
        has_temporal = "時序分類資料集路徑:" in dataset_str
        has_old = "資料集路徑:" in dataset_str
        
        print(f"🔍 [DEBUG-VALIDATE] 字串檢查:")
        print(f"   含有 '時序分類資料集路徑:': {has_temporal}")
        print(f"   含有 '資料集路徑:': {has_old}")
        
        if not has_temporal and not has_old:
            print("❌ [DEBUG-VALIDATE] 沒有找到任何資料集路徑標記")
            return False
        
        # 從結果文字中提取實際路徑
        lines = dataset_str.split('\n')
        print(f"🔍 [DEBUG-VALIDATE] 分割成 {len(lines)} 行")
        
        for i, line in enumerate(lines):
            print(f"   行 {i}: {repr(line)}")
            
            if "時序分類資料集路徑:" in line:
                actual_path = line.split("時序分類資料集路徑:")[-1].strip()
                print(f"🔍 [DEBUG-VALIDATE] 提取到路徑: {repr(actual_path)}")
                exists = Path(actual_path).exists()
                print(f"🔍 [DEBUG-VALIDATE] 路徑存在: {exists}")
                return exists
            elif "資料集路徑:" in line:
                actual_path = line.split("資料集路徑:")[-1].strip()
                print(f"🔍 [DEBUG-VALIDATE] 提取到路徑: {repr(actual_path)}")
                exists = Path(actual_path).exists()
                print(f"🔍 [DEBUG-VALIDATE] 路徑存在: {exists}")
                return exists
        
        print("❌ [DEBUG-VALIDATE] 沒有找到有效的路徑行")
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
            
            print(f"🔍 [DEBUG-TEMPORAL] 開始解析資料集路徑")
            print(f"   總行數: {len(lines)}")
            
            for line in lines:
                print(f"   檢查行: {repr(line)}")
                
                # 檢查新格式：時序分類資料集路徑: xxx
                if "時序分類資料集路徑:" in line:
                    temporal_path = line.split("時序分類資料集路徑:")[-1].strip()
                    print(f"✅ [DEBUG-TEMPORAL] 找到時序路徑: {temporal_path}")
                    # 從時序路徑推導出原始標註資料集路徑
                    if temporal_path.startswith("training_workspace/temporal_"):
                        # temporal_merged_dataset_xxx -> merged_dataset_xxx
                        original_path = temporal_path.replace("temporal_", "")
                        print(f"🔍 [DEBUG-TEMPORAL] 推導原始路徑: {original_path}")
                        if Path(original_path).exists():
                            actual_dataset_path = original_path
                            print(f"✅ [DEBUG-TEMPORAL] 使用原始標註資料集: {actual_dataset_path}")
                            break
                    actual_dataset_path = temporal_path
                    break
                # 檢查舊格式：資料集路徑: xxx (如果有的話)
                elif "資料集路徑:" in line and "時序分類" not in line:
                    # 只處理包含實際路徑的行
                    potential_path = line.split("資料集路徑:")[-1].strip()
                    if potential_path and "/" in potential_path:
                        actual_dataset_path = potential_path
                        print(f"✅ [DEBUG-TEMPORAL] 找到舊格式路徑: {actual_dataset_path}")
                        break
            
            if not actual_dataset_path:
                print(f"❌ [DEBUG-TEMPORAL] 無法解析出有效路徑")
                return "❌ 無法解析資料集路徑"
            
            print(f"🎯 [DEBUG-TEMPORAL] 最終路徑: {actual_dataset_path}")
            
            # 建立模型配置
            model_config = self._get_temporal_model_config(model_name, image_size)
            
            # 啟動訓練任務（在背景執行）
            import threading
            import os
            def training_worker():
                try:
                    print(f"🚀 [TRAINING-WORKER] 訓練執行緒啟動")
                    print(f"   進程 PID: {os.getpid()}")
                    print(f"   執行緒 ID: {threading.current_thread().ident}")
                    print(f"   模型配置: {model_config}")
                    
                    # 創建訓練器
                    print(f"🏗️ [TRAINING-WORKER] 正在創建 TemporalTrainer...")
                    trainer = TemporalTrainer(model_config)
                    print(f"✅ [TRAINING-WORKER] TemporalTrainer 創建成功")
                    
                    self.training_progress = "🔥 正在訓練時序模型..."
                    print(f"🔥 [TRAINING-WORKER] 開始訓練...")
                    print(f"   資料集路徑: {actual_dataset_path}")
                    print(f"   訓練參數: epochs={epochs}, batch_size={batch_size}")
                    
                    result = trainer.train(
                        dataset_path=actual_dataset_path,
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=1e-3,
                        val_split=0.2
                    )
                    
                    print(f"✅ [TRAINING-WORKER] 訓練完成！結果: {result}")
                    self.training_results = result
                    self.training_progress = f"✅ 時序模型訓練完成！最佳準確率: {result['best_val_accuracy']:.3f}"
                    self.is_training = False
                    self._save_training_state()  # 保存最終狀態
                    
                except Exception as e:
                    print(f"❌ [TRAINING-WORKER] 訓練失敗: {str(e)}")
                    import traceback
                    print(f"🔍 [TRAINING-WORKER] 錯誤堆疊:")
                    traceback.print_exc()
                    self.training_progress = f"❌ 訓練失敗: {str(e)}"
                    self.is_training = False
                    self._save_training_state()  # 保存錯誤狀態
            
            training_thread = threading.Thread(target=training_worker)
            training_thread.daemon = False  # 確保執行緒不會被提前終止
            training_thread.start()
            print(f"🚀 [TRAINING-MAIN] 訓練執行緒已啟動: {training_thread.ident}")
            
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
    
    
    def _get_temporal_model_config(self, model_name, image_size):
        """取得時序模型配置"""
        # 從模型名稱提取 backbone 和融合策略
        model_configs = {
            # 低延遲（Edge/即時）
            "temporal_mobilenetv3_large": ("mobilenetv3_large_100", "attention"),
            "temporal_ghostnet_100": ("ghostnet_100", "attention"),
            "temporal_repvgg_b0": ("repvgg_b0", "attention"),
            "temporal_efficientnet_b0": ("efficientnet_b0", "attention"),
            "temporal_efficientnet_b1": ("efficientnet_b1", "attention"),
            
            # 均衡（速度/準度平衡）
            "temporal_convnext_tiny": ("convnext_tiny", "attention"),
            "temporal_efficientnetv2_s": ("efficientnetv2_s", "attention"),
            "temporal_resnet50d": ("resnet50d", "attention"),
            "temporal_resnet50": ("resnet50", "attention"),
            "temporal_regnety_032": ("regnety_032", "attention"),
            
            # 準度優先
            "temporal_convnext_base": ("convnext_base", "lstm"),
            "temporal_efficientnetv2_m": ("efficientnetv2_m", "lstm"),
            "temporal_swin_tiny": ("swin_tiny_patch4_window7_224", "attention"),
            "temporal_vit_small": ("vit_small_patch16_224", "attention"),
        }
        
        # 獲取配置，預設使用 convnext_tiny + attention（均衡推薦）
        backbone, fusion = model_configs.get(model_name, ("convnext_tiny", "attention"))
        
        # 根據模型類型調整 dropout（低延遲模型用較低dropout，大模型用較高）
        dropout_configs = {
            # 低延遲模型 - 較低dropout以保持輕量
            "mobilenetv3_large_100": 0.1,
            "ghostnet_100": 0.1,
            "repvgg_b0": 0.12,
            "efficientnet_b0": 0.12,
            "efficientnet_b1": 0.15,
            
            # 均衡模型 - 中等dropout
            "convnext_tiny": 0.15,
            "efficientnetv2_s": 0.18,
            "resnet50d": 0.2,
            "resnet50": 0.2,
            "regnety_032": 0.18,
            
            # 準度優先模型 - 較高dropout防過擬合
            "convnext_base": 0.3,
            "efficientnetv2_m": 0.25,
            "swin_tiny_patch4_window7_224": 0.2,
            "vit_small_patch16_224": 0.15,  # ViT通常較敏感
        }
        
        dropout = dropout_configs.get(backbone, 0.18)  # 預設為中等值
        
        # 根據timm模型設定對應的輸入圖片大小
        input_size_configs = {
            # 低延遲模型 - 通常使用較小輸入尺寸
            "mobilenetv3_large_100": 224,
            "ghostnet_100": 224,
            "repvgg_b0": 224,
            "efficientnet_b0": 224,
            "efficientnet_b1": 240,
            
            # 均衡模型 - 標準224或稍大
            "convnext_tiny": 224,
            "efficientnetv2_s": 224,  # EfficientNetV2可以adaptive
            "resnet50d": 224,
            "resnet50": 224,
            "regnety_032": 224,
            
            # 準度優先模型 - 較大輸入尺寸
            "convnext_base": 224,  # ConvNeXt系列通常224
            "efficientnetv2_m": 288,  # EfficientNetV2-M建議使用更大尺寸
            "swin_tiny_patch4_window7_224": 224,  # Swin名稱中就有224
            "vit_small_patch16_224": 224,  # ViT名稱中就有224
        }
        
        # 取得對應的輸入尺寸，如果user有指定則使用user的設定
        default_input_size = input_size_configs.get(backbone, 224)
        final_input_size = image_size if image_size and image_size > 0 else default_input_size
        
        return {
            'backbone_name': backbone,
            'num_classes': 2,
            'temporal_frames': 5,
            'pretrained': True,
            'temporal_fusion': fusion,
            'dropout': dropout,
            'freeze_backbone': False,
            'default_input_size': default_input_size  # 保存預設值供參考
        }
    
    def get_training_progress(self):
        """取得訓練進度"""
        # 如果訓練已完成且有進度信息，顯示最終結果
        if not self.is_training and self.training_progress:
            return self.training_progress
        # 如果沒有開始訓練且沒有進度信息
        elif not self.is_training:
            return "等待開始訓練..."
        # 如果正在訓練
        return self.training_progress
    
    def stop_training(self):
        """停止訓練"""
        if self.is_training:
            self.is_training = False
            self.training_progress = "⏹️ 訓練已停止"
            self._save_training_state()  # 保存停止狀態
            return "✅ 訓練已停止"
        return "⚠️ 沒有正在進行的訓練"
    
    def list_trained_models(self):
        """列出已訓練的模型"""
        models = []
        
        # 檢查時序分類模型 (新的主要模型類型)
        temporal_runs_dir = Path("runs/temporal_training")
        if temporal_runs_dir.exists():
            for run_dir in temporal_runs_dir.iterdir():
                if run_dir.is_dir():
                    best_model = run_dir / "best_model.pth"
                    if best_model.exists():
                        models.append({
                            "name": f"時序模型 - {run_dir.name}",
                            "path": str(best_model),
                            "created": best_model.stat().st_mtime,
                            "type": "temporal"
                        })
        
        # 檢查 YOLO 模型 (如果存在)
        yolo_runs_dir = Path("runs/detect")
        if yolo_runs_dir.exists():
            for run_dir in yolo_runs_dir.iterdir():
                if run_dir.is_dir():
                    weights_dir = run_dir / "weights"
                    if weights_dir.exists():
                        best_pt = weights_dir / "best.pt"
                        if best_pt.exists():
                            models.append({
                                "name": f"YOLO模型 - {run_dir.name}",
                                "path": str(best_pt),
                                "created": best_pt.stat().st_mtime,
                                "type": "yolo"
                            })
        
        # 按建立時間排序
        models.sort(key=lambda x: x["created"], reverse=True)
        return models
    
    def delete_model(self, model_path):
        """刪除指定的模型"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                return f"❌ 模型不存在: {model_path}"
            
            # 如果是時序模型，刪除整個目錄
            if model_path.suffix == '.pth':
                # 刪除整個訓練目錄
                model_dir = model_path.parent
                if model_dir.name.startswith('temporal_'):
                    import shutil
                    shutil.rmtree(model_dir)
                    print(f"🗑️ 已刪除時序模型目錄: {model_dir}")
                    return f"✅ 已刪除時序模型: {model_dir.name}"
                else:
                    # 只刪除模型文件
                    model_path.unlink()
                    return f"✅ 已刪除模型文件: {model_path.name}"
            
            # 如果是 YOLO 模型
            elif model_path.suffix == '.pt':
                # 刪除整個 YOLO 訓練目錄
                weights_dir = model_path.parent
                if weights_dir.name == 'weights':
                    run_dir = weights_dir.parent
                    import shutil
                    shutil.rmtree(run_dir)
                    print(f"🗑️ 已刪除 YOLO 模型目錄: {run_dir}")
                    return f"✅ 已刪除 YOLO 模型: {run_dir.name}"
                else:
                    model_path.unlink()
                    return f"✅ 已刪除模型文件: {model_path.name}"
            
            return f"❌ 不支援的模型格式: {model_path.suffix}"
            
        except Exception as e:
            return f"❌ 刪除失敗: {str(e)}"