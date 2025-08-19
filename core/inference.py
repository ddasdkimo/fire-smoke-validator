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
            if not ULTRALYTICS_AVAILABLE:
                return "❌ 未安裝 ultralytics 套件，無法載入模型"
            
            if not Path(model_path).exists():
                return f"❌ 模型檔案不存在: {model_path}"
            
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
            self.current_model_info = f"""✅ 模型載入成功！

📁 模型路徑: {model_path}
⚡ 運算設備: {device}
{model_info}
            """
            
            return self.current_model_info
            
        except Exception as e:
            self.current_model = None
            self.current_model_path = None
            return f"❌ 模型載入失敗: {str(e)}"
    
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
                    output_filename = f"inference_result_{processed_count:03d}_{timestamp}.jpg"
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
            
            summary = f"""✅ 批次推論完成！

📊 處理結果:
- 成功處理: {successful_count} 張影像
- 處理失敗: {error_count} 張影像  
- 總偵測數: {total_detections} 個物件

📁 結果儲存在: {self.inference_dir}
            """
            
            return summary, results
            
        except Exception as e:
            return f"❌ 推論失敗: {str(e)}", []
    
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