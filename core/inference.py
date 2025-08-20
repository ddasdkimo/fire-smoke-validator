#!/usr/bin/env python3
"""
推論模組
負責模型推論功能
"""

import cv2
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False




class ModelInference:
    """模型推論器"""
    
    def __init__(self):
        self.inference_dir = Path("inference_workspace")
        self.inference_dir.mkdir(exist_ok=True)
        
        # 當前載入的模型
        self.current_model = None
        self.current_model_path = None
        self.current_model_info = ""
        
        # 推論結果
        self.inference_results = []
        
    
    def get_available_models(self):
        """取得可用的時序模型列表"""
        models = []
        
        # 時序訓練模型 - 從 runs/temporal_training 目錄
        temporal_runs_dir = Path("runs/temporal_training")
        if temporal_runs_dir.exists():
            for run_dir in temporal_runs_dir.iterdir():
                if run_dir.is_dir():
                    # 尋找 best_model.pth
                    best_model = run_dir / "best_model.pth"
                    tensorboard_dir = run_dir / "tensorboard"
                    training_history = run_dir / "training_history.json"
                    
                    if best_model.exists():
                        model_info = {
                            "name": f"時序模型 - {run_dir.name}",
                            "path": str(best_model),
                            "type": "temporal_best",
                            "created": best_model.stat().st_mtime,
                            "run_dir": str(run_dir)
                        }
                        
                        # 添加TensorBoard信息
                        if tensorboard_dir.exists():
                            model_info["tensorboard_path"] = str(tensorboard_dir)
                            model_info["has_tensorboard"] = True
                        else:
                            model_info["has_tensorboard"] = False
                        
                        # 添加訓練歷史信息
                        if training_history.exists():
                            try:
                                import json
                                with open(training_history, 'r') as f:
                                    history = json.load(f)
                                    model_info["training_metrics"] = {
                                        "best_val_acc": max(history.get("val_acc", [])) if history.get("val_acc") else None,
                                        "final_train_acc": history.get("train_acc", [])[-1] if history.get("train_acc") else None,
                                        "total_epochs": len(history.get("train_acc", []))
                                    }
                            except Exception as e:
                                model_info["training_metrics"] = None
                        
                        models.append(model_info)
                    
                    # 尋找 final_model.pth
                    final_model = run_dir / "final_model.pth"
                    if final_model.exists():
                        model_info = {
                            "name": f"時序模型 (最終) - {run_dir.name}",
                            "path": str(final_model),
                            "type": "temporal_final",
                            "created": final_model.stat().st_mtime,
                            "run_dir": str(run_dir)
                        }
                        
                        # 添加TensorBoard信息
                        if tensorboard_dir.exists():
                            model_info["tensorboard_path"] = str(tensorboard_dir)
                            model_info["has_tensorboard"] = True
                        else:
                            model_info["has_tensorboard"] = False
                        
                        models.append(model_info)
        
        # 如果沒有找到任何模型，提供預設說明
        if not models:
            models.append({
                "name": "尚無已訓練模型 (請先進行訓練)",
                "path": "",
                "type": "placeholder",
                "created": 0
            })
        
        # 按時間排序，最新的在前面
        models.sort(key=lambda x: -x.get("created", 0))
        return models
    
    def get_model_tensorboard_info(self, model_path):
        """取得模型的TensorBoard資訊"""
        try:
            model_dir = Path(model_path).parent
            tensorboard_dir = model_dir / "tensorboard"
            training_history_path = model_dir / "training_history.json"
            
            info = {
                "has_tensorboard": tensorboard_dir.exists(),
                "tensorboard_path": str(tensorboard_dir) if tensorboard_dir.exists() else None,
                "tensorboard_command": f"tensorboard --logdir={tensorboard_dir}" if tensorboard_dir.exists() else None,
                "training_metrics": None
            }
            
            # 讀取訓練歷史
            if training_history_path.exists():
                try:
                    with open(training_history_path, 'r') as f:
                        history = json.load(f)
                        
                    metrics = {
                        "total_epochs": len(history.get("train_acc", [])),
                        "best_val_accuracy": max(history.get("val_acc", [])) if history.get("val_acc") else None,
                        "final_train_accuracy": history.get("train_acc", [])[-1] if history.get("train_acc") else None,
                        "final_val_loss": history.get("val_loss", [])[-1] if history.get("val_loss") else None,
                        "final_train_loss": history.get("train_loss", [])[-1] if history.get("train_loss") else None,
                    }
                    
                    info["training_metrics"] = metrics
                    
                except Exception as e:
                    print(f"讀取訓練歷史失敗: {e}")
            
            return info
            
        except Exception as e:
            print(f"取得TensorBoard資訊失敗: {e}")
            return {"has_tensorboard": False, "tensorboard_path": None, "training_metrics": None}
    
    def load_model(self, model_path, device='auto'):
        """載入時序推論模型"""
        try:
            if not model_path or model_path == "":
                return "❌ 請先選擇有效的模型"
            
            if not Path(model_path).exists():
                return f"❌ 模型檔案不存在: {model_path}"
            
            # 載入時序模型
            return self._load_temporal_model(model_path, device)
            
        except Exception as e:
            self.current_model = None
            self.current_model_path = None
            return f"❌ 模型載入失敗: {str(e)}"
    
    
    def _load_temporal_model(self, model_path, device):
        """載入時序模型"""
        try:
            if not TORCH_AVAILABLE:
                return "❌ PyTorch 未安裝，無法載入時序模型"
                
            from .models.temporal_trainer import load_temporal_model
            
            # 設定設備
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.current_model = load_temporal_model(model_path, device)
            self.current_model_path = model_path
            
            
            # 取得模型資訊和 TensorBoard 資訊
            model_info = self.current_model.get_model_info()
            tensorboard_info = self.get_model_tensorboard_info(model_path)
            
            # 構建基本資訊
            base_info = f"""✅ 時序模型載入成功！

📁 模型路徑: {model_path}
⚡ 運算設備: {device}
🎯 模型類型: 時序火煙分類器
🧠 Backbone: {model_info['backbone']}
⏱️ 時序幀數: {model_info['temporal_frames']}
🔄 融合策略: {model_info['temporal_fusion']}
📊 參數量: {model_info['total_parameters']:,}
🏃 可訓練參數: {model_info['trainable_parameters']:,}"""

            # 添加訓練指標資訊
            if tensorboard_info.get("training_metrics"):
                metrics = tensorboard_info["training_metrics"]
                base_info += f"""

📈 訓練指標:
- 總訓練輪數: {metrics.get('total_epochs', 'N/A')}
- 最佳驗證準確率: {metrics.get('best_val_accuracy', 'N/A'):.4f}
- 最終訓練準確率: {metrics.get('final_train_accuracy', 'N/A'):.4f}
- 最終驗證損失: {metrics.get('final_val_loss', 'N/A'):.4f}"""

            # 添加 TensorBoard 資訊
            if tensorboard_info.get("has_tensorboard"):
                base_info += f"""

📊 TensorBoard 可用:
- 路徑: {tensorboard_info['tensorboard_path']}
- 啟動指令: {tensorboard_info['tensorboard_command']}"""
            else:
                base_info += f"""

⚠️ 此模型沒有 TensorBoard 記錄"""

            self.current_model_info = base_info
            
            return self.current_model_info
            
        except ImportError:
            return "❌ 缺少 timm 或相關依賴，無法載入時序模型"
        except Exception as e:
            return f"❌ 時序模型載入失敗: {str(e)}"
    
    
    def _get_model_info(self, model_path):
        """取得模型詳細資訊"""
        try:
            # 基本資訊
            file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            info = f"📊 檔案大小: {file_size:.1f} MB\n"
            
            # 如果有模型的話，可以取得更多資訊
            if self.current_model:
                # 類別數量
                if hasattr(self.current_model.model, 'nc'):
                    info += f"🎯 類別數量: {self.current_model.model.nc}\n"
                
                # 類別名稱
                if hasattr(self.current_model.model, 'names'):
                    names = self.current_model.model.names
                    info += f"📝 類別名稱: {list(names.values())}\n"
            
            return info
            
        except Exception as e:
            return f"⚠️ 無法取得詳細資訊: {str(e)}\n"
    
    def inference_batch_images(self, image_files, confidence_threshold=0.5):
        """時序模型批次推論"""
        try:
            if not self.current_model:
                return "❌ 請先載入時序模型", []
            
            if not image_files:
                return "❌ 請上傳時序影像檔案（建議上傳同一事件的多幀影像）", []
            
            # 執行時序模型推論
            return self._inference_temporal_model(image_files, confidence_threshold)
            
        except Exception as e:
            return f"❌ 時序推論失敗: {str(e)}", []
    
    def _inference_temporal_model(self, image_files, confidence_threshold=0.5):
        """時序模型推論"""
        try:
            if not TORCH_AVAILABLE:
                return "❌ PyTorch 未安裝，無法進行時序推論", []
                
            from .models.data_utils import prepare_temporal_frames
            
            results = []
            processed_count = 0
            
            # 確保是列表格式
            if not isinstance(image_files, list):
                image_files = [image_files]
            
            # 時序模型需要將多張影像作為一個序列處理
            # 這裡假設用戶上傳的是一個事件的多個幀
            print(f"⏱️ 時序模型推論: 處理 {len(image_files)} 張影像作為一個時序序列...")
            
            try:
                # 載入所有影像作為一個時序序列
                frames = []
                valid_files = []
                
                for img_file in image_files:
                    image = cv2.imread(img_file.name)
                    if image is not None:
                        frames.append(image)
                        valid_files.append(img_file)
                
                if not frames:
                    return "❌ 無法讀取任何有效影像", []
                
                # 準備時序輸入 (T=5)
                temporal_input = prepare_temporal_frames(frames, target_frames=5, training=False)
                temporal_input = temporal_input.unsqueeze(0)  # [1, T, C, H, W] 加入 batch 維度
                
                # 移動到正確設備
                device = next(self.current_model.parameters()).device
                temporal_input = temporal_input.to(device)
                
                # 進行推論
                with torch.no_grad():
                    outputs = self.current_model(temporal_input)  # [1, num_classes]
                    probabilities = torch.softmax(outputs, dim=1)  # 轉換為機率
                    
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                
                # 類別映射
                class_names = {0: "false_positive", 1: "true_positive"}
                predicted_label = class_names.get(predicted_class, f"class_{predicted_class}")
                
                # 建立結果圖像（將所有幀組合成網格）
                grid_image = self._create_temporal_result_grid(frames, predicted_label, confidence)
                
                # 儲存結果圖像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"temporal_inference_{timestamp}.jpg"
                output_path = self.inference_dir / output_filename
                cv2.imwrite(str(output_path), grid_image)
                
                
                sequence_result = {
                    "sequence_id": f"temporal_sequence_{timestamp}",
                    "input_frames": [Path(f.name).name for f in valid_files],
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                    "confidence": float(confidence),
                    "probabilities": {
                        "false_positive": float(probabilities[0][0]),
                        "true_positive": float(probabilities[0][1])
                    },
                    "result_image_path": str(output_path),
                    "total_frames": len(frames),
                    "processed_frames": 5  # 固定處理5幀
                }
                
                results.append(sequence_result)
                
                # 生成詳細摘要
                status_emoji = "🔥" if predicted_label == "true_positive" else "✅"
                result_name = "真實火煙事件" if predicted_label == "true_positive" else "非火煙事件"
                
                
                summary = f"""{status_emoji} 時序模型推論完成！

🎯 分析結果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 預測類別: {result_name}
📈 信心度: {confidence:.3f} ({confidence*100:.1f}%)

🔍 詳細機率分佈:
- 🔥 真實火煙: {probabilities[0][1]:.3f} ({probabilities[0][1]*100:.1f}%)
- ❌ 誤報事件: {probabilities[0][0]:.3f} ({probabilities[0][0]*100:.1f}%)

⚙️ 處理參數:
- 輸入幀數: {len(frames)} 張影像
- 時序長度: T=5 (固定策略)
- 模型架構: {self.current_model.backbone_name}
- 融合方式: {self.current_model.temporal_fusion}

💾 結果檔案:
- 視覺化結果: {Path(output_path).name}
- 完整路徑: {output_path}

📅 分析時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                """
                
                self.inference_results = results
                return summary, results
                
            except Exception as e:
                results.append({
                    "sequence_id": "error_sequence",
                    "error": str(e)
                })
                return f"❌ 時序推論失敗: {str(e)}", results
                
        except ImportError:
            return "❌ 缺少時序模型相關依賴", []
    
    
    
    
    
    
    def _create_temporal_result_grid(self, frames, predicted_label, confidence):
        """建立時序結果網格圖像"""
        try:
            if not PIL_AVAILABLE:
                # 如果 PIL 不可用，使用英文版本
                return self._create_temporal_result_grid_english(frames, predicted_label, confidence)
            
            # 限制顯示的幀數
            display_frames = frames[:5] if len(frames) > 5 else frames
            
            # 調整每幀大小
            target_size = (180, 180)
            resized_frames = []
            for frame in display_frames:
                resized = cv2.resize(frame, target_size)
                # 將 BGR 轉換為 RGB (PIL 使用 RGB)
                resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                resized_frames.append(resized_rgb)
            
            # 建立網格
            cols = len(resized_frames)
            padding = 10
            
            # 建立空白畫布
            header_height = 80
            footer_height = 60
            canvas_height = target_size[1] + header_height + footer_height
            canvas_width = target_size[0] * cols + padding * (cols + 1)
            
            # 使用 PIL 創建圖像
            canvas = Image.new('RGB', (canvas_width, canvas_height), color=(240, 240, 240))
            draw = ImageDraw.Draw(canvas)
            
            # 嘗試載入中文字體
            try:
                # 常見的中文字體路徑
                font_paths = [
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                    "/System/Library/Fonts/Arial.ttf",  # macOS
                    "C:\\Windows\\Fonts\\Arial.ttf",    # Windows
                ]
                
                title_font = None
                text_font = None
                small_font = None
                
                for font_path in font_paths:
                    if Path(font_path).exists():
                        title_font = ImageFont.truetype(font_path, 24)
                        text_font = ImageFont.truetype(font_path, 18)
                        small_font = ImageFont.truetype(font_path, 14)
                        break
                
                # 如果找不到字體，使用預設字體
                if not title_font:
                    title_font = ImageFont.load_default()
                    text_font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                    
            except Exception:
                title_font = ImageFont.load_default()
                text_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # 添加標題背景
            draw.rectangle([(0, 0), (canvas_width, header_height)], fill=(50, 50, 50))
            
            # 添加預測結果標題
            title_text = "Temporal Fire/Smoke Classification Result"
            draw.text((20, 15), title_text, fill=(255, 255, 255), font=title_font)
            
            # 添加預測結果
            if predicted_label == "true_positive":
                result_text = "Prediction: Fire/Smoke Detected"
                result_color = (0, 255, 0)  # 綠色
                status_text = "Fire/Smoke Event Detected"
            else:
                result_text = "Prediction: No Fire/Smoke"
                result_color = (255, 165, 0)  # 橙色
                status_text = "Non Fire/Smoke Event"
            
            confidence_text = f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)"
            
            draw.text((20, 45), result_text, fill=result_color, font=text_font)
            draw.text((350, 45), confidence_text, fill=(255, 255, 255), font=text_font)
            
            # 放置幀
            for i, frame in enumerate(resized_frames):
                x_start = padding + i * (target_size[0] + padding)
                y_start = header_height + 10
                
                # 添加白色邊框
                border_box = [(x_start-2, y_start-2), (x_start+target_size[0]+2, y_start+target_size[1]+2)]
                draw.rectangle(border_box, outline=(255, 255, 255), width=2)
                
                # 將 numpy 陣列轉換為 PIL 圖像
                frame_img = Image.fromarray(frame)
                canvas.paste(frame_img, (x_start, y_start))
                
                # 添加幀編號
                frame_text = f"Frame {i+1}"
                draw.text((x_start + 5, y_start - 20), frame_text, fill=(100, 100, 100), font=small_font)
            
            # 添加底部狀態
            footer_y = header_height + target_size[1] + 20
            footer_box = [(0, footer_y), (canvas_width, canvas_height)]
            draw.rectangle(footer_box, fill=(240, 240, 240))
            
            draw.text((20, footer_y + 15), status_text, fill=result_color, font=text_font)
            
            # 添加處理信息
            info_text = f"Processed {len(display_frames)} frames, Temporal Length: T=5"
            draw.text((20, footer_y + 35), info_text, fill=(100, 100, 100), font=small_font)
            
            # 將 PIL 圖像轉換回 OpenCV 格式 (RGB -> BGR)
            canvas_array = np.array(canvas)
            canvas_bgr = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2BGR)
            
            return canvas_bgr
            
        except Exception as e:
            print(f"建立結果網格時發生錯誤: {e}")
            # 回傳英文版本作為備選
            return self._create_temporal_result_grid_english(frames, predicted_label, confidence)
    
    def _create_temporal_result_grid_english(self, frames, predicted_label, confidence):
        """建立時序結果網格圖像 (英文版本)"""
        try:
            # 限制顯示的幀數
            display_frames = frames[:5] if len(frames) > 5 else frames
            
            # 調整每幀大小
            target_size = (180, 180)
            resized_frames = []
            for frame in display_frames:
                resized = cv2.resize(frame, target_size)
                resized_frames.append(resized)
            
            # 建立網格
            cols = len(resized_frames)
            padding = 10
            
            # 建立空白畫布
            header_height = 80
            footer_height = 60
            canvas_height = target_size[1] + header_height + footer_height
            canvas_width = target_size[0] * cols + padding * (cols + 1)
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 240
            
            # 添加標題背景
            cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), (50, 50, 50), -1)
            
            # 添加預測結果標題
            title_text = "Temporal Fire/Smoke Classification"
            cv2.putText(canvas, title_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加預測結果
            if predicted_label == "true_positive":
                result_text = "Fire/Smoke Detected"
                result_color = (0, 255, 0)  # 綠色
                status_text = "Fire/Smoke Event"
            else:
                result_text = "No Fire/Smoke"
                result_color = (0, 165, 255)  # 橙色
                status_text = "Non Fire/Smoke Event"
            
            confidence_text = f"Confidence: {confidence:.3f}"
            
            cv2.putText(canvas, result_text, (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
            cv2.putText(canvas, confidence_text, (300, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 放置幀
            for i, frame in enumerate(resized_frames):
                x_start = padding + i * (target_size[0] + padding)
                y_start = header_height + 10
                
                # 添加白色邊框
                cv2.rectangle(canvas, (x_start-2, y_start-2), 
                             (x_start+target_size[0]+2, y_start+target_size[1]+2), 
                             (255, 255, 255), 2)
                
                canvas[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = frame
                
                # 添加幀編號
                cv2.putText(canvas, f"Frame {i+1}", (x_start + 5, y_start - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            # 添加底部狀態
            footer_y = header_height + target_size[1] + 20
            cv2.rectangle(canvas, (0, footer_y), (canvas_width, canvas_height), (240, 240, 240), -1)
            cv2.putText(canvas, status_text, (20, footer_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
            
            # 添加處理信息
            info_text = f"Processed {len(display_frames)} frames, T=5"
            cv2.putText(canvas, info_text, (20, footer_y + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            return canvas
            
        except Exception as e:
            print(f"建立結果網格時發生錯誤: {e}")
            # 回傳第一幀作為備選
            return frames[0] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
    
    def get_inference_results_summary(self):
        """取得推論結果摘要"""
        if not self.inference_results:
            return "尚未進行推論"
        
        successful_results = [r for r in self.inference_results if "error" not in r]
        error_results = [r for r in self.inference_results if "error" in r]
        
        summary_lines = [
            f"📊 推論結果摘要:",
            f"",
            f"✅ 成功處理: {len(successful_results)} 張影像",
            f"❌ 處理失敗: {len(error_results)} 張影像",
            f""
        ]
        
        if successful_results:
            total_detections = sum(r["detection_count"] for r in successful_results)
            avg_detections = total_detections / len(successful_results)
            summary_lines.extend([
                f"🎯 總偵測數: {total_detections} 個物件",
                f"📈 平均每張: {avg_detections:.1f} 個物件",
                f""
            ])
            
            # 顯示各影像的結果
            summary_lines.append("📋 詳細結果:")
            for i, result in enumerate(successful_results[:10]):  # 只顯示前10個
                filename = result["filename"][:30] + "..." if len(result["filename"]) > 30 else result["filename"]
                summary_lines.append(f"  {i+1}. {filename}: {result['detection_count']} 個物件")
            
            if len(successful_results) > 10:
                summary_lines.append(f"  ... 還有 {len(successful_results) - 10} 個結果")
        
        if error_results:
            summary_lines.extend([
                f"",
                f"❌ 處理失敗的檔案:",
            ])
            for result in error_results[:5]:  # 只顯示前5個錯誤
                filename = result["filename"][:30] + "..." if len(result["filename"]) > 30 else result["filename"]
                error_msg = result["error"][:50] + "..." if len(result["error"]) > 50 else result["error"]
                summary_lines.append(f"  - {filename}: {error_msg}")
        
        return "\n".join(summary_lines)
    
    def get_detection_gallery(self):
        """取得偵測結果影像畫廊"""
        if not self.inference_results:
            return []
        
        gallery_paths = []
        for result in self.inference_results:
            # 檢查時序模型結果
            if "result_image_path" in result and Path(result["result_image_path"]).exists():
                gallery_paths.append(result["result_image_path"])
            # 檢查其他模型結果
            elif "annotated_image_path" in result and Path(result["annotated_image_path"]).exists():
                gallery_paths.append(result["annotated_image_path"])
        
        return gallery_paths
    
    
    def clear_inference_results(self):
        """清除推論結果"""
        self.inference_results = []
        
        # 清除暫存檔案（保留最近的結果）
        try:
            if self.inference_dir.exists():
                inference_files = list(self.inference_dir.glob("inference_result_*.jpg"))
                # 保留最新的20個檔案，刪除其餘的
                if len(inference_files) > 20:
                    inference_files.sort(key=lambda x: x.stat().st_mtime)
                    for old_file in inference_files[:-20]:
                        old_file.unlink()
        except Exception as e:
            print(f"清理推論結果時發生錯誤: {e}")
        
        return "✅ 已清除推論結果"