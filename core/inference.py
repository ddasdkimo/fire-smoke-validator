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


class GradCAMVisualizer:
    """Grad-CAM 視覺化工具"""
    
    def __init__(self, model):
        self.model = model
        self.target_layers = []
        self.gradients = []
        self.activations = []
        
        # 註冊 hook 到目標層
        self._register_hooks()
    
    def _register_hooks(self):
        """註冊 forward 和 backward hooks"""
        # 找到backbone的最後一個卷積層
        target_layer = None
        
        # 遍歷模型尋找合適的特徵層
        for name, module in self.model.named_modules():
            if 'backbone' in name and hasattr(module, 'weight') and len(module.weight.shape) == 4:
                target_layer = module
        
        if target_layer is not None:
            # 註冊 forward hook
            target_layer.register_forward_hook(self._forward_hook)
            # 註冊 backward hook
            target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Forward hook 儲存激活值"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Backward hook 儲存梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """生成 Class Activation Map"""
        try:
            self.model.eval()
            
            # Forward pass
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = torch.argmax(output, dim=1)
            
            # Backward pass
            self.model.zero_grad()
            class_loss = output[0, target_class]
            class_loss.backward(retain_graph=True)
            
            # 計算 Grad-CAM
            if len(self.gradients) > 0 and len(self.activations) > 0:
                # 池化梯度得到權重
                weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
                
                # 加權組合特徵圖
                cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
                cam = F.relu(cam)  # ReLU activation
                
                # 正歸化到 0-1
                cam = cam - cam.min()
                cam = cam / (cam.max() + 1e-8)
                
                return cam.squeeze().cpu().numpy()
            
        except Exception as e:
            print(f"生成 Grad-CAM 時發生錯誤: {e}")
            
        return None


class AttentionVisualizer:
    """注意力權重視覺化工具"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        
        # 註冊 hook 到注意力層
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """註冊注意力層的 hooks"""
        for name, module in self.model.named_modules():
            # 尋找注意力模組
            if 'attention' in name.lower() or 'attn' in name.lower():
                module.register_forward_hook(
                    lambda module, input, output, name=name: self._attention_hook(name, output)
                )
    
    def _attention_hook(self, name, output):
        """儲存注意力權重"""
        if isinstance(output, tuple):
            # 通常注意力模組會返回 (output, attention_weights)
            if len(output) > 1:
                self.attention_weights[name] = output[1].detach().cpu()
        elif hasattr(output, 'attention_weights'):
            self.attention_weights[name] = output.attention_weights.detach().cpu()
    
    def get_attention_maps(self):
        """取得注意力地圖"""
        return self.attention_weights


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
        
        # 視覺化工具
        self.gradcam_visualizer = None
        self.attention_visualizer = None
    
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
                    if best_model.exists():
                        models.append({
                            "name": f"時序模型 - {run_dir.name}",
                            "path": str(best_model),
                            "type": "temporal_best",
                            "created": best_model.stat().st_mtime
                        })
                    
                    # 尋找 final_model.pth
                    final_model = run_dir / "final_model.pth"
                    if final_model.exists():
                        models.append({
                            "name": f"時序模型 (最終) - {run_dir.name}",
                            "path": str(final_model),
                            "type": "temporal_final",
                            "created": final_model.stat().st_mtime
                        })
        
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
            
            # 初始化視覺化工具
            self.gradcam_visualizer = GradCAMVisualizer(self.current_model)
            self.attention_visualizer = AttentionVisualizer(self.current_model)
            
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
                
                # 生成熱區圖 (需要梯度，所以重新計算)
                heatmaps = self._generate_heatmaps(temporal_input, predicted_class)
                
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
                
                # 生成和儲存熱區圖視覺化
                heatmap_paths = self._save_heatmap_visualizations(
                    frames, heatmaps, predicted_label, confidence, timestamp
                )
                
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
                    "heatmap_paths": heatmap_paths,  # 新增熱區圖路徑
                    "has_heatmaps": len(heatmap_paths) > 0,  # 標記是否有熱區圖
                    "total_frames": len(frames),
                    "processed_frames": 5  # 固定處理5幀
                }
                
                results.append(sequence_result)
                
                # 生成詳細摘要
                status_emoji = "🔥" if predicted_label == "true_positive" else "✅"
                result_name = "真實火煙事件" if predicted_label == "true_positive" else "非火煙事件"
                
                # 熱區圖狀態
                heatmap_status = "✅ 已生成" if len(heatmap_paths) > 0 else "❌ 未生成"
                heatmap_info = f"- 🔥 熱區圖: {heatmap_status} ({len(heatmap_paths)} 個檔案)" if len(heatmap_paths) > 0 else "- 🔥 熱區圖: 生成失敗或不支援"
                
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
{heatmap_info}

🔬 視覺化說明:
- Grad-CAM 熱區圖顯示模型關注的影像區域
- 紅色區域表示高關注度，藍色區域表示低關注度
- 組合圖展示原始幀與熱區圖的對比

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
    
    def _generate_heatmaps(self, temporal_input, predicted_class):
        """生成熱區圖"""
        heatmaps = {}
        
        try:
            if self.gradcam_visualizer and TORCH_AVAILABLE:
                # 生成 Grad-CAM
                cam = self.gradcam_visualizer.generate_cam(temporal_input, predicted_class)
                if cam is not None:
                    heatmaps['gradcam'] = cam
            
            if self.attention_visualizer:
                # 生成注意力地圖
                attention_maps = self.attention_visualizer.get_attention_maps()
                if attention_maps:
                    heatmaps['attention'] = attention_maps
        
        except Exception as e:
            print(f"生成熱區圖時發生錯誤: {e}")
        
        return heatmaps
    
    def _create_heatmap_overlay(self, image, heatmap, alpha=0.4, colormap='jet'):
        """將熱區圖疊加到原始影像上"""
        try:
            if not MATPLOTLIB_AVAILABLE:
                return image
            
            # 調整熱區圖大小到影像尺寸
            h, w = image.shape[:2]
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # 使用 matplotlib colormap
            cmap = cm.get_cmap(colormap)
            heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # 移除 alpha 通道
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            # 將 RGB 轉為 BGR (OpenCV 格式)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
            
            # 混合原始影像和熱區圖
            overlay = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)
            
            return overlay
            
        except Exception as e:
            print(f"建立熱區圖疊加時發生錯誤: {e}")
            return image
    
    def _save_heatmap_visualizations(self, frames, heatmaps, predicted_label, confidence, timestamp):
        """儲存熱區圖視覺化結果"""
        visualization_paths = []
        
        try:
            # 建立熱區圖目錄
            heatmap_dir = self.inference_dir / f"heatmaps_{timestamp}"
            heatmap_dir.mkdir(exist_ok=True)
            
            if 'gradcam' in heatmaps and len(frames) > 0:
                cam = heatmaps['gradcam']
                
                # 為每個幀生成熱區圖
                for i, frame in enumerate(frames[:5]):  # 限制5幀
                    # 建立疊加圖
                    overlay = self._create_heatmap_overlay(frame, cam, alpha=0.4)
                    
                    # 儲存單獨的熱區圖
                    heatmap_path = heatmap_dir / f"gradcam_frame_{i+1}.jpg"
                    cv2.imwrite(str(heatmap_path), overlay)
                    visualization_paths.append(str(heatmap_path))
                
                # 建立組合視覺化
                combined_viz = self._create_combined_heatmap_visualization(
                    frames, cam, predicted_label, confidence, timestamp
                )
                combined_path = heatmap_dir / "combined_heatmap.jpg"
                cv2.imwrite(str(combined_path), combined_viz)
                visualization_paths.append(str(combined_path))
            
            return visualization_paths
            
        except Exception as e:
            print(f"儲存熱區圖視覺化時發生錯誤: {e}")
            return []
    
    def _create_combined_heatmap_visualization(self, frames, heatmap, predicted_label, confidence, timestamp):
        """建立組合的熱區圖視覺化"""
        try:
            # 限制顯示的幀數
            display_frames = frames[:3] if len(frames) > 3 else frames
            
            # 調整每幀大小
            target_size = (200, 200)
            
            # 建立原始幀和熱區圖疊加幀
            original_frames = []
            heatmap_frames = []
            
            for frame in display_frames:
                # 原始幀
                resized_original = cv2.resize(frame, target_size)
                original_frames.append(resized_original)
                
                # 熱區圖疊加幀
                overlay = self._create_heatmap_overlay(resized_original, heatmap, alpha=0.5)
                heatmap_frames.append(overlay)
            
            # 建立組合畫布
            padding = 15
            header_height = 80
            footer_height = 40
            
            canvas_width = target_size[0] * len(display_frames) * 2 + padding * (len(display_frames) * 2 + 1)
            canvas_height = target_size[1] + header_height + footer_height
            
            canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 245
            
            # 添加標題背景
            cv2.rectangle(canvas, (0, 0), (canvas_width, header_height), (60, 60, 60), -1)
            
            # 添加標題
            title_text = "Grad-CAM Heatmap Visualization"
            cv2.putText(canvas, title_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 添加預測結果
            result_color = (0, 255, 0) if predicted_label == "true_positive" else (0, 165, 255)
            result_text = f"Prediction: {'Fire/Smoke' if predicted_label == 'true_positive' else 'No Fire/Smoke'}"
            confidence_text = f"Confidence: {confidence:.3f}"
            
            cv2.putText(canvas, result_text, (20, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, result_color, 2)
            cv2.putText(canvas, confidence_text, (400, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 放置幀 - 原始幀和熱區圖並排
            y_start = header_height + 10
            
            for i in range(len(display_frames)):
                # 原始幀位置
                orig_x = padding + i * (target_size[0] * 2 + padding)
                # 熱區圖位置  
                heat_x = orig_x + target_size[0] + 5
                
                # 放置原始幀
                canvas[y_start:y_start+target_size[1], orig_x:orig_x+target_size[0]] = original_frames[i]
                
                # 放置熱區圖
                canvas[y_start:y_start+target_size[1], heat_x:heat_x+target_size[0]] = heatmap_frames[i]
                
                # 添加標籤
                cv2.putText(canvas, f"Original {i+1}", (orig_x, y_start-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                cv2.putText(canvas, f"Grad-CAM {i+1}", (heat_x, y_start-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                
                # 添加白色邊框
                cv2.rectangle(canvas, (orig_x-1, y_start-1), 
                             (orig_x+target_size[0]+1, y_start+target_size[1]+1), 
                             (255, 255, 255), 2)
                cv2.rectangle(canvas, (heat_x-1, y_start-1), 
                             (heat_x+target_size[0]+1, y_start+target_size[1]+1), 
                             (255, 255, 255), 2)
            
            # 添加底部說明
            footer_y = y_start + target_size[1] + 15
            info_text = "Left: Original frames, Right: Grad-CAM heatmap overlay"
            cv2.putText(canvas, info_text, (20, footer_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            return canvas
            
        except Exception as e:
            print(f"建立組合熱區圖視覺化時發生錯誤: {e}")
            # 回傳第一幀作為備選
            return frames[0] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
    
    
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
            
            # 添加熱區圖路徑
            if "heatmap_paths" in result and result["heatmap_paths"]:
                for heatmap_path in result["heatmap_paths"]:
                    if Path(heatmap_path).exists():
                        gallery_paths.append(heatmap_path)
            
            # 檢查其他模型結果
            elif "annotated_image_path" in result and Path(result["annotated_image_path"]).exists():
                gallery_paths.append(result["annotated_image_path"])
        
        return gallery_paths
    
    def get_heatmap_gallery(self):
        """取得熱區圖專用畫廊"""
        if not self.inference_results:
            return []
        
        heatmap_paths = []
        for result in self.inference_results:
            if "heatmap_paths" in result and result["heatmap_paths"]:
                for path in result["heatmap_paths"]:
                    if Path(path).exists():
                        heatmap_paths.append(path)
        
        return heatmap_paths
    
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