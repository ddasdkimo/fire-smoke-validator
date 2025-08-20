#!/usr/bin/env python3
"""
Gradio 資料標記介面
用於標記火災偵測模型的輸出結果，區分真實火災與誤判
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
import os
import tempfile
from collections import defaultdict

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Ultralytics 未安裝，將使用模擬偵測")

class DataLabelingTool:
    def __init__(self, model_path="best.pt", data_dir="data/raw/HPWREN/extracted"):
        self.data_dir = Path(data_dir)
        self.model_path = model_path
        self.output_dir = Path("data/labeled")
        self.output_dir.mkdir(exist_ok=True)
        
        # 建立分類資料夾
        self.categories = {
            "true_fire": self.output_dir / "true_fire",      # 真實火災
            "true_smoke": self.output_dir / "true_smoke",    # 真實煙霧
            "false_cloud": self.output_dir / "false_cloud",  # 雲朵誤判
            "false_light": self.output_dir / "false_light",  # 燈光誤判
            "false_static": self.output_dir / "false_static", # 靜態物體誤判
            "uncertain": self.output_dir / "uncertain"       # 不確定
        }
        
        for category_path in self.categories.values():
            category_path.mkdir(exist_ok=True)
        
        # 載入模型
        self.model = None
        if ULTRALYTICS_AVAILABLE and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                # 在 Mac 上使用 MPS 設備
                import torch
                if torch.backends.mps.is_available():
                    self.model.to('mps')
                    print(f"✅ 成功載入模型: {model_path} (使用 MPS 加速)")
                else:
                    print(f"✅ 成功載入模型: {model_path} (使用 CPU)")
            except Exception as e:
                print(f"❌ 載入模型失敗: {e}")
        
        # 標記歷史
        self.labeling_history = []
        self.current_detections = {}
        
    def get_mp4_files(self):
        """獲取所有 MP4 檔案列表"""
        mp4_files = []
        for mp4_path in self.data_dir.rglob("*.mp4"):
            relative_path = mp4_path.relative_to(self.data_dir)
            mp4_files.append(str(relative_path))
        return sorted(mp4_files)
    
    def process_video(self, video_path):
        """處理影片並進行火災偵測"""
        if not video_path:
            return "請選擇一個影片檔案", None, []
        
        full_video_path = self.data_dir / video_path
        if not full_video_path.exists():
            return f"檔案不存在: {video_path}", None, []
        
        try:
            # 讀取影片
            cap = cv2.VideoCapture(str(full_video_path))
            if not cap.isOpened():
                return f"無法開啟影片: {video_path}", None, []
            
            # 獲取影片資訊
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 進行偵測（每隔幾幀處理一次以節省時間）
            detections = []
            frame_idx = 0
            sample_interval = max(1, fps // 2)  # 每 0.5 秒採樣一次
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    # 進行火災偵測
                    detection_result = self.detect_fire_smoke(frame, frame_idx, fps)
                    if detection_result['has_detection']:
                        detections.append(detection_result)
                
                frame_idx += 1
            
            cap.release()
            
            # 儲存偵測結果
            self.current_detections[video_path] = detections
            
            info = f"影片資訊：\n"
            info += f"- 檔案: {video_path}\n"
            info += f"- 時長: {duration:.1f} 秒\n"
            info += f"- 幀率: {fps} FPS\n"
            info += f"- 總幀數: {frame_count}\n"
            info += f"- 偵測到的目標: {len(detections)} 個\n"
            
            # 建立偵測結果的選項
            detection_options = []
            for i, det in enumerate(detections):
                time_str = f"{det['timestamp']:.1f}s"
                conf_str = f"{det['confidence']:.2f}"
                class_name = det['class_name']
                detection_options.append(f"[{i+1}] {time_str} - {class_name} ({conf_str})")
            
            return info, str(full_video_path), detection_options
            
        except Exception as e:
            return f"處理影片時發生錯誤: {e}", None, []
    
    def detect_fire_smoke(self, frame, frame_idx, fps):
        """對單一幀進行火災煙霧偵測"""
        timestamp = frame_idx / fps
        
        if self.model is not None:
            try:
                # 使用 YOLO 模型進行偵測
                results = self.model(frame, verbose=False)
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        # 取第一個偵測結果
                        box = result.boxes[0]
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        # 取得邊界框座標
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        return {
                            'has_detection': True,
                            'timestamp': timestamp,
                            'frame_idx': frame_idx,
                            'confidence': confidence,
                            'class_name': class_name,
                            'class_id': class_id,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'frame': frame.copy()
                        }
                
            except Exception as e:
                print(f"偵測時發生錯誤: {e}")
        
        # 模擬偵測結果（如果沒有模型）
        if np.random.random() < 0.1:  # 10% 機率產生偵測
            class_names = ['fire', 'smoke']
            class_name = np.random.choice(class_names)
            h, w = frame.shape[:2]
            
            return {
                'has_detection': True,
                'timestamp': timestamp,
                'frame_idx': frame_idx,
                'confidence': np.random.uniform(0.5, 0.95),
                'class_name': class_name,
                'class_id': 0 if class_name == 'fire' else 1,
                'bbox': [
                    np.random.randint(0, w//2),
                    np.random.randint(0, h//2),
                    np.random.randint(w//2, w),
                    np.random.randint(h//2, h)
                ],
                'frame': frame.copy()
            }
        
        return {'has_detection': False}
    
    def show_detection(self, video_path, detection_idx):
        """顯示特定的偵測結果"""
        if not video_path or video_path not in self.current_detections:
            return None, "請先處理影片"
        
        detections = self.current_detections[video_path]
        if not detections or detection_idx < 0 or detection_idx >= len(detections):
            return None, "偵測索引無效"
        
        detection = detections[detection_idx]
        frame = detection['frame']
        bbox = detection['bbox']
        
        # 在幀上繪製邊界框
        annotated_frame = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加標籤
        label = f"{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 轉換為 RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        info = f"時間: {detection['timestamp']:.1f}s\n"
        info += f"類別: {detection['class_name']}\n"
        info += f"信心度: {detection['confidence']:.2f}\n"
        info += f"邊界框: {bbox}"
        
        return annotated_frame, info
    
    def label_detection(self, video_path, detection_idx, category):
        """標記偵測結果"""
        if not video_path or video_path not in self.current_detections:
            return "❌ 請先處理影片"
        
        detections = self.current_detections[video_path]
        if not detections or detection_idx < 0 or detection_idx >= len(detections):
            return "❌ 偵測索引無效"
        
        if category not in self.categories:
            return f"❌ 無效的類別: {category}"
        
        try:
            detection = detections[detection_idx]
            frame = detection['frame']
            bbox = detection['bbox']
            
            # 建立檔案名稱
            video_name = Path(video_path).stem
            timestamp = detection['timestamp']
            filename = f"{video_name}_{timestamp:.1f}s_{detection['class_name']}.jpg"
            
            # 儲存標記的影像
            save_path = self.categories[category] / filename
            
            # 在影像上繪製邊界框
            annotated_frame = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加標籤
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(save_path), annotated_frame)
            
            # 記錄標記歷史
            label_record = {
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'detection_idx': detection_idx,
                'category': category,
                'detection_info': {
                    'timestamp': detection['timestamp'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': bbox
                },
                'saved_path': str(save_path)
            }
            
            self.labeling_history.append(label_record)
            
            # 儲存標記歷史
            history_path = self.output_dir / "labeling_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.labeling_history, f, indent=2, ensure_ascii=False)
            
            return f"✅ 已標記為 {category}，儲存至 {save_path}"
            
        except Exception as e:
            return f"❌ 標記時發生錯誤: {e}"
    
    def get_labeling_stats(self):
        """獲取標記統計"""
        stats = defaultdict(int)
        for record in self.labeling_history:
            stats[record['category']] += 1
        
        total = sum(stats.values())
        stats_text = f"標記統計 (總計: {total}):\n"
        for category, count in stats.items():
            percentage = (count / total * 100) if total > 0 else 0
            stats_text += f"- {category}: {count} ({percentage:.1f}%)\n"
        
        return stats_text

def create_interface():
    """建立 Gradio 介面"""
    labeling_tool = DataLabelingTool()
    
    with gr.Blocks(title="火災偵測資料標記工具", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔥 火災偵測資料標記工具")
        gr.Markdown("選擇 MP4 影片進行火災偵測，然後標記偵測結果的正確性")
        
        with gr.Tab("影片處理"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 影片選擇
                    video_dropdown = gr.Dropdown(
                        choices=labeling_tool.get_mp4_files(),
                        label="選擇 MP4 影片",
                        value=None
                    )
                    
                    refresh_btn = gr.Button("🔄 重新整理影片列表", variant="secondary")
                    process_btn = gr.Button("🎬 處理影片", variant="primary")
                    
                    # 影片資訊
                    video_info = gr.Textbox(
                        label="影片資訊",
                        lines=6,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    # 偵測結果列表
                    detection_list = gr.Dropdown(
                        choices=[],
                        label="偵測結果",
                        value=None
                    )
                    
                    show_detection_btn = gr.Button("👁️ 顯示偵測結果", variant="secondary")
                    
                    # 顯示偵測影像
                    detection_image = gr.Image(
                        label="偵測結果",
                        type="numpy"
                    )
                    
                    detection_info = gr.Textbox(
                        label="偵測詳情",
                        lines=4,
                        interactive=False
                    )
        
        with gr.Tab("標記"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🏷️ 選擇正確的類別")
                    
                    with gr.Row():
                        true_fire_btn = gr.Button("✅ 真實火災", variant="primary")
                        true_smoke_btn = gr.Button("✅ 真實煙霧", variant="primary")
                    
                    with gr.Row():
                        false_cloud_btn = gr.Button("❌ 雲朵誤判", variant="secondary")
                        false_light_btn = gr.Button("❌ 燈光誤判", variant="secondary")
                    
                    with gr.Row():
                        false_static_btn = gr.Button("❌ 靜態物體", variant="secondary")
                        uncertain_btn = gr.Button("❓ 不確定", variant="stop")
                    
                    label_result = gr.Textbox(
                        label="標記結果",
                        lines=3,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### 📊 標記統計")
                    stats_display = gr.Textbox(
                        label="統計資訊",
                        lines=8,
                        interactive=False
                    )
                    
                    refresh_stats_btn = gr.Button("🔄 更新統計", variant="secondary")
        
        # 隱藏狀態
        current_video = gr.State(value="")
        current_detection_idx = gr.State(value=-1)
        
        # 事件處理
        def refresh_video_list():
            return gr.Dropdown(choices=labeling_tool.get_mp4_files())
        
        def process_video_wrapper(video_path):
            info, full_path, detections = labeling_tool.process_video(video_path)
            return info, video_path, gr.Dropdown(choices=detections), detections
        
        def show_detection_wrapper(video_path, detection_selection):
            if not detection_selection:
                return None, "請選擇一個偵測結果"
            
            # 解析選擇的索引
            try:
                detection_idx = int(detection_selection.split(']')[0].split('[')[1]) - 1
                image, info = labeling_tool.show_detection(video_path, detection_idx)
                return image, info, detection_idx
            except:
                return None, "解析偵測索引失敗", -1
        
        def label_wrapper(video_path, detection_idx, category):
            result = labeling_tool.label_detection(video_path, detection_idx, category)
            stats = labeling_tool.get_labeling_stats()
            return result, stats
        
        # 綁定事件
        refresh_btn.click(refresh_video_list, outputs=[video_dropdown])
        
        process_btn.click(
            process_video_wrapper,
            inputs=[video_dropdown],
            outputs=[video_info, current_video, detection_list, gr.State()]
        )
        
        show_detection_btn.click(
            show_detection_wrapper,
            inputs=[current_video, detection_list],
            outputs=[detection_image, detection_info, current_detection_idx]
        )
        
        # 標記按鈕
        for category, btn in [
            ("true_fire", true_fire_btn),
            ("true_smoke", true_smoke_btn),
            ("false_cloud", false_cloud_btn),
            ("false_light", false_light_btn),
            ("false_static", false_static_btn),
            ("uncertain", uncertain_btn)
        ]:
            btn.click(
                lambda video, idx, cat=category: label_wrapper(video, idx, cat),
                inputs=[current_video, current_detection_idx],
                outputs=[label_result, stats_display]
            )
        
        refresh_stats_btn.click(
            labeling_tool.get_labeling_stats,
            outputs=[stats_display]
        )
    
    return interface

def main():
    """啟動標記介面"""
    print("🚀 啟動資料標記介面...")
    
    # 檢查必要套件
    if not ULTRALYTICS_AVAILABLE:
        print("⚠️  請安裝 ultralytics: pip install ultralytics")
    
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 啟動介面失敗: {e}")

if __name__ == "__main__":
    main()