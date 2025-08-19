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
        """取得可用的模型列表"""
        models = []
        
        # 預設模型
        default_model = Path("best.pt")
        if default_model.exists():
            models.append({
                "name": "best.pt (預設模型)",
                "path": str(default_model),
                "type": "default"
            })
        
        # 已訓練的模型
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    weights_dir = run_dir / "weights"
                    if weights_dir.exists():
                        best_pt = weights_dir / "best.pt"
                        if best_pt.exists():
                            models.append({
                                "name": f"{run_dir.name}/best.pt",
                                "path": str(best_pt),
                                "type": "trained",
                                "created": best_pt.stat().st_mtime
                            })
        
        # 按類型和時間排序
        models.sort(key=lambda x: (x["type"] != "default", -x.get("created", 0)))
        return models
    
    def load_model(self, model_path, device='auto'):
        """載入推論模型"""
        try:
            if not Path(model_path).exists():
                return f"❌ 模型檔案不存在: {model_path}"
            
            # 判斷模型類型
            is_temporal_model = self._is_temporal_model(model_path)
            
            if is_temporal_model:
                # 載入時序模型
                return self._load_temporal_model(model_path, device)
            else:
                # 載入 YOLO 模型
                return self._load_yolo_model(model_path, device)
            
        except Exception as e:
            self.current_model = None
            self.current_model_path = None
            return f"❌ 模型載入失敗: {str(e)}"
    
    def _is_temporal_model(self, model_path):
        """判斷是否為時序模型"""
        try:
            # 嘗試載入檢查點
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            # 如果有 model_config 且包含 backbone_name，視為時序模型
            return 'model_config' in checkpoint and 'backbone_name' in checkpoint.get('model_config', {})
        except:
            return False
    
    def _load_temporal_model(self, model_path, device):
        """載入時序模型"""
        try:
            from .models.temporal_trainer import load_temporal_model
            
            # 設定設備
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.current_model = load_temporal_model(model_path, device)
            self.current_model_path = model_path
            
            # 取得模型資訊
            model_info = self.current_model.get_model_info()
            self.current_model_info = f"""✅ 時序模型載入成功！

📁 模型路徑: {model_path}
⚡ 運算設備: {device}
🎯 模型類型: 時序火煙分類器
🧠 Backbone: {model_info['backbone']}
⏱️ 時序幀數: {model_info['temporal_frames']}
🔄 融合策略: {model_info['temporal_fusion']}
📊 參數量: {model_info['total_parameters']:,}
🏃 可訓練參數: {model_info['trainable_parameters']:,}
            """
            
            return self.current_model_info
            
        except ImportError:
            return "❌ 缺少 timm 或相關依賴，無法載入時序模型"
        except Exception as e:
            return f"❌ 時序模型載入失敗: {str(e)}"
    
    def _load_yolo_model(self, model_path, device):
        """載入 YOLO 模型"""
        try:
            if not ULTRALYTICS_AVAILABLE:
                return "❌ 未安裝 ultralytics 套件，無法載入 YOLO 模型"
            
            # 載入模型
            self.current_model = YOLO(model_path)
            
            # 設定設備
            if device == 'auto':
                # 自動檢測最佳設備
                if hasattr(self.current_model, 'device'):
                    device = str(self.current_model.device)
                else:
                    device = 'cpu'
            
            self.current_model.to(device)
            self.current_model_path = model_path
            
            # 取得模型資訊
            model_info = self._get_model_info(model_path)
            self.current_model_info = f"""✅ YOLO 模型載入成功！

📁 模型路徑: {model_path}
⚡ 運算設備: {device}
🎯 模型類型: YOLO 物件偵測
{model_info}
            """
            
            return self.current_model_info
            
        except Exception as e:
            return f"❌ YOLO 模型載入失敗: {str(e)}"
    
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
        """批次推論多張影像"""
        try:
            if not self.current_model:
                return "❌ 請先載入模型", []
            
            if not image_files:
                return "❌ 請上傳影像檔案", []
            
            # 判斷模型類型
            is_temporal_model = self._is_temporal_model(self.current_model_path)
            
            if is_temporal_model:
                return self._inference_temporal_model(image_files, confidence_threshold)
            else:
                return self._inference_yolo_model(image_files, confidence_threshold)
            
        except Exception as e:
            return f"❌ 推論失敗: {str(e)}", []
    
    def _inference_temporal_model(self, image_files, confidence_threshold=0.5):
        """時序模型推論"""
        try:
            from .models.data_utils import prepare_temporal_frames
            import torch
            
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
                
                # 生成摘要
                summary = f"""✅ 時序模型推論完成！

🎯 模型預測:
- 預測類別: {predicted_label}
- 信心度: {confidence:.3f}
- 假陽性機率: {probabilities[0][0]:.3f}
- 真火煙機率: {probabilities[0][1]:.3f}

📊 處理結果:
- 輸入幀數: {len(frames)} 張
- 處理幀數: 5 張 (T=5 固定策略)
- 時序融合: {self.current_model.temporal_fusion}

📁 結果儲存在: {output_path}
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
    
    def _inference_yolo_model(self, image_files, confidence_threshold=0.5):
        """YOLO 模型推論"""
        results = []
        processed_count = 0
        
        # 確保是列表格式
        if not isinstance(image_files, list):
            image_files = [image_files]
        
        for img_file in image_files:
            try:
                # 讀取影像
                image = cv2.imread(img_file.name)
                if image is None:
                    results.append({
                        "filename": img_file.name,
                        "error": "無法讀取影像檔案"
                    })
                    continue
                
                # 進行推論
                prediction_results = self.current_model(image, conf=confidence_threshold, verbose=False)
                
                # 處理結果
                detections = []
                annotated_image = image.copy()
                
                for result in prediction_results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            # 取得類別名稱
                            class_name = "unknown"
                            if hasattr(self.current_model.model, 'names'):
                                class_name = self.current_model.model.names.get(cls, f"class_{cls}")
                            
                            detections.append({
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "confidence": float(conf),
                                "class": int(cls),
                                "class_name": class_name
                            })
                            
                            # 在影像上繪製框線
                            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(annotated_image, f"{class_name}: {conf:.2f}",
                                      (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # 儲存標註後的影像
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"yolo_inference_{processed_count:03d}_{timestamp}.jpg"
                output_path = self.inference_dir / output_filename
                cv2.imwrite(str(output_path), annotated_image)
                
                results.append({
                    "filename": Path(img_file.name).name,
                    "detections": detections,
                    "detection_count": len(detections),
                    "annotated_image_path": str(output_path),
                    "original_size": image.shape[:2]  # (height, width)
                })
                
                processed_count += 1
                
            except Exception as e:
                results.append({
                    "filename": Path(img_file.name).name,
                    "error": str(e)
                })
        
        # 儲存推論結果
        self.inference_results = results
        
        # 生成摘要
        total_detections = sum(r.get("detection_count", 0) for r in results)
        successful_count = len([r for r in results if "error" not in r])
        error_count = len([r for r in results if "error" in r])
        
        summary = f"""✅ YOLO 批次推論完成！

📊 處理結果:
- 成功處理: {successful_count} 張影像
- 處理失敗: {error_count} 張影像  
- 總偵測數: {total_detections} 個物件

📁 結果儲存在: {self.inference_dir}
        """
        
        return summary, results
    
    def _create_temporal_result_grid(self, frames, predicted_label, confidence):
        """建立時序結果網格圖像"""
        try:
            # 限制顯示的幀數
            display_frames = frames[:5] if len(frames) > 5 else frames
            
            # 調整每幀大小
            target_size = (200, 200)
            resized_frames = []
            for frame in display_frames:
                resized = cv2.resize(frame, target_size)
                resized_frames.append(resized)
            
            # 建立網格
            rows = 1
            cols = len(resized_frames)
            
            # 建立空白畫布
            canvas_height = target_size[1] + 100  # 額外空間用於文字
            canvas_width = target_size[0] * cols
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # 放置幀
            for i, frame in enumerate(resized_frames):
                x_start = i * target_size[0]
                y_start = 50  # 為頂部文字留空間
                canvas[y_start:y_start+target_size[1], x_start:x_start+target_size[0]] = frame
                
                # 添加幀編號
                cv2.putText(canvas, f"Frame {i+1}", (x_start + 5, y_start - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 添加預測結果
            result_text = f"Prediction: {predicted_label} (Conf: {confidence:.3f})"
            text_color = (0, 255, 0) if predicted_label == "true_positive" else (0, 0, 255)
            cv2.putText(canvas, result_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
            
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
            if "annotated_image_path" in result and Path(result["annotated_image_path"]).exists():
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