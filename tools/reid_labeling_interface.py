#!/usr/bin/env python3
"""
基於 ReID 的時序分組標記介面
使用物件重新識別將偵測結果分組，進行時序標記
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from datetime import datetime
import os
from collections import defaultdict
import tempfile

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  Ultralytics 未安裝，將使用模擬偵測")

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("⚠️  Supervision 未安裝，ReID 功能受限")

class ReIDLabelingTool:
    def __init__(self, model_path="best.pt", data_dir="data/raw/HPWREN/extracted"):
        self.data_dir = Path(data_dir)
        self.model_path = model_path
        self.output_dir = Path("data/labeled_sequences")
        self.output_dir.mkdir(exist_ok=True)
        
        # 可調整的過濾參數
        self.confidence_threshold = 0.5      # 信心度閾值
        self.min_track_frames = 3            # 最小追蹤幀數
        self.min_crop_size = 32              # 最小裁切尺寸
        self.sample_fps = 2                  # 每秒採樣幀數
        
        # 建立分類資料夾
        self.categories = {
            "true_dynamic_fire": self.output_dir / "true_dynamic_fire",
            "true_dynamic_smoke": self.output_dir / "true_dynamic_smoke", 
            "false_static_cloud": self.output_dir / "false_static_cloud",
            "false_static_light": self.output_dir / "false_static_light",
            "false_static_object": self.output_dir / "false_static_object",
            "uncertain": self.output_dir / "uncertain"
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
        
        # ReID 追蹤器
        self.tracker = None
        if SUPERVISION_AVAILABLE:
            try:
                self.tracker = sv.ByteTrack()
                print("✅ 成功初始化 ByteTrack")
            except Exception as e:
                print(f"❌ 初始化追蹤器失敗: {e}")
        
        # 當前處理狀態
        self.current_tracks = {}
        self.current_video_path = None
        self.labeling_history = []
        
    def upload_video(self, uploaded_file):
        """處理上傳的影片檔案"""
        if uploaded_file is None:
            return "請上傳一個 MP4 影片檔案", None
        
        try:
            # 建立臨時檔案目錄
            temp_dir = Path("temp_uploads")
            temp_dir.mkdir(exist_ok=True)
            
            # 處理上傳檔案 (適配不同 Gradio 版本)
            if hasattr(uploaded_file, 'name'):
                # 檔案路徑格式
                source_path = uploaded_file.name
                file_name = Path(source_path).name
            else:
                # 假設是檔案路徑字符串
                source_path = str(uploaded_file)
                file_name = Path(source_path).name
            
            # 支援的影片格式
            supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
            file_ext = Path(file_name).suffix.lower()
            
            if file_ext not in supported_formats:
                return f"請上傳支援的影片格式: {', '.join(supported_formats)}", None
            
            temp_video_path = temp_dir / file_name
            
            # 複製檔案到臨時位置
            import shutil
            shutil.copy2(source_path, temp_video_path)
            
            self.current_video_path = str(temp_video_path)
            
            return f"✅ 成功上傳影片: {file_name}", str(temp_video_path)
            
        except Exception as e:
            return f"❌ 上傳失敗: {e}", None
    
    def process_video_with_reid(self, video_path):
        """處理影片並使用 ReID 進行分組"""
        if not video_path:
            return "請上傳一個影片檔案", [], {}
        
        full_video_path = Path(video_path)
        if not full_video_path.exists():
            return f"檔案不存在: {video_path}", [], {}
        
        try:
            cap = cv2.VideoCapture(str(full_video_path))
            if not cap.isOpened():
                return f"無法開啟影片: {video_path}", [], {}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # 重置追蹤器
            if self.tracker:
                self.tracker = sv.ByteTrack()
            
            tracks_data = defaultdict(list)
            frame_idx = 0
            sample_interval = max(1, fps // self.sample_fps)  # 根據設定的採樣頻率
            
            print(f"開始處理影片: {video_path}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / fps
                    
                    # 進行偵測
                    detections = self.detect_objects(frame)
                    
                    if detections and len(detections) > 0:
                        # 使用 ReID 追蹤
                        if self.tracker:
                            tracked_objects = self.track_objects(detections, frame)
                        else:
                            # 模擬追蹤
                            tracked_objects = self.simulate_tracking(detections, timestamp)
                        
                        # 記錄每個追蹤目標
                        for track in tracked_objects:
                            track_id = track['track_id']
                            track_data = {
                                'frame_idx': frame_idx,
                                'timestamp': timestamp,
                                'bbox': track['bbox'],
                                'confidence': track['confidence'],
                                'class_name': track['class_name'],
                                'frame': frame.copy()
                            }
                            tracks_data[track_id].append(track_data)
                
                frame_idx += 1
                
                # 每處理 100 幀顯示進度
                if frame_idx % 100 == 0:
                    print(f"已處理 {frame_idx}/{frame_count} 幀")
            
            cap.release()
            
            # 篩選有效的追蹤序列（根據最小幀數設定）
            valid_tracks = {
                track_id: frames 
                for track_id, frames in tracks_data.items() 
                if len(frames) >= self.min_track_frames
            }
            
            self.current_tracks = valid_tracks
            self.current_video_path = video_path
            
            # 生成追蹤序列選項
            track_options = []
            for track_id, frames in valid_tracks.items():
                duration_seconds = frames[-1]['timestamp'] - frames[0]['timestamp']
                class_name = frames[0]['class_name']
                track_options.append(
                    f"Track {track_id}: {class_name} ({len(frames)}幀, {duration_seconds:.1f}s)"
                )
            
            info = f"影片處理完成：\n"
            info += f"- 檔案: {video_path}\n"
            info += f"- 時長: {duration:.1f} 秒\n"
            info += f"- 幀率: {fps} FPS\n"
            info += f"- 總追蹤序列: {len(valid_tracks)} 個\n"
            info += f"- 總幀數: {sum(len(frames) for frames in valid_tracks.values())}\n"
            
            return info, track_options, valid_tracks
            
        except Exception as e:
            return f"處理影片時發生錯誤: {e}", [], {}
    
    def detect_objects(self, frame):
        """對單一幀進行物件偵測"""
        if self.model is not None:
            try:
                results = self.model(frame, verbose=False)
                detections = []
                
                for result in results:
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            confidence = float(box.conf[0])
                            
                            # 只保留高於閾值的偵測
                            if confidence >= self.confidence_threshold:
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': confidence,
                                    'class_name': class_name,
                                    'class_id': class_id
                                })
                
                return detections
                
            except Exception as e:
                print(f"偵測時發生錯誤: {e}")
                return []
        else:
            # 模擬偵測
            if np.random.random() < 0.3:  # 30% 機率產生偵測
                h, w = frame.shape[:2]
                return [{
                    'bbox': [
                        np.random.randint(0, w//2),
                        np.random.randint(0, h//2),
                        np.random.randint(w//2, w),
                        np.random.randint(h//2, h)
                    ],
                    'confidence': np.random.uniform(0.5, 0.95),
                    'class_name': np.random.choice(['fire', 'smoke']),
                    'class_id': np.random.randint(0, 2)
                }]
            return []
    
    def track_objects(self, detections, frame):
        """使用 ReID 追蹤物件"""
        if not self.tracker or not detections:
            return []
        
        try:
            # 轉換為 supervision 格式
            boxes = np.array([det['bbox'] for det in detections])
            confidences = np.array([det['confidence'] for det in detections])
            class_ids = np.array([det['class_id'] for det in detections])
            
            # 建立 Detections 物件
            detection_sv = sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
            
            # 進行追蹤
            tracked_detections = self.tracker.update_with_detections(detection_sv)
            
            # 轉換回我們的格式
            tracked_objects = []
            for i, track_id in enumerate(tracked_detections.tracker_id):
                if track_id is not None:
                    tracked_objects.append({
                        'track_id': int(track_id),
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_name': detections[i]['class_name'],
                        'class_id': int(class_ids[i])
                    })
            
            return tracked_objects
            
        except Exception as e:
            print(f"追蹤時發生錯誤: {e}")
            return self.simulate_tracking(detections, 0)
    
    def simulate_tracking(self, detections, timestamp):
        """模擬追蹤（當 ReID 不可用時）"""
        tracked_objects = []
        for i, detection in enumerate(detections):
            # 簡單的模擬追蹤 ID
            track_id = hash(str(detection['bbox'])) % 1000
            tracked_objects.append({
                'track_id': track_id,
                **detection
            })
        return tracked_objects
    
    def show_track_sequence(self, track_selection):
        """顯示選中的追蹤序列"""
        if not track_selection or not self.current_tracks:
            return [], "請先處理影片並選擇追蹤序列", []
        
        try:
            # 解析追蹤 ID
            track_id = int(track_selection.split()[1].rstrip(':'))
            
            if track_id not in self.current_tracks:
                return [], "追蹤序列不存在", []
            
            frames = self.current_tracks[track_id]
            
            # 生成序列影像和選擇選項
            sequence_images = []
            frame_options = []
            for i, frame_data in enumerate(frames):
                frame = frame_data['frame']
                bbox = frame_data['bbox']
                timestamp = frame_data['timestamp']
                confidence = frame_data['confidence']
                class_name = frame_data['class_name']
                
                # 在幀上繪製邊界框
                annotated_frame = frame.copy()
                x1, y1, x2, y2 = bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加標籤和幀號
                label = f"[{i}] T{track_id} {class_name} {confidence:.3f} @{timestamp:.1f}s"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 轉換為 RGB
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                sequence_images.append(annotated_frame)
                
                # 建立幀選擇選項
                frame_options.append(f"[{i}] {timestamp:.1f}s - {class_name} ({confidence:.3f})")
            
            info = f"追蹤序列 {track_id}:\n"
            info += f"- 類別: {frames[0]['class_name']}\n"
            info += f"- 幀數: {len(frames)}\n"
            info += f"- 時間範圍: {frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s\n"
            info += f"- 持續時間: {frames[-1]['timestamp'] - frames[0]['timestamp']:.1f}s\n"
            info += f"- 平均信心度: {np.mean([f['confidence'] for f in frames]):.2f}\n"
            
            return sequence_images, info, frame_options
            
        except Exception as e:
            return [], f"顯示序列時發生錯誤: {e}", []
    
    def remove_frames_from_track(self, track_selection, frame_indices_str):
        """從追蹤序列中移除指定幀"""
        if not track_selection or not self.current_tracks:
            return "請先選擇追蹤序列"
        
        try:
            # 解析追蹤 ID
            track_id = int(track_selection.split()[1].rstrip(':'))
            
            if track_id not in self.current_tracks:
                return "追蹤序列不存在"
            
            # 解析要刪除的幀索引
            if not frame_indices_str.strip():
                return "請輸入要刪除的幀編號"
            
            remove_indices = []
            for part in frame_indices_str.strip().split(','):
                part = part.strip()
                if '-' in part:
                    # 處理範圍 (例如: 0-3)
                    try:
                        start, end = part.split('-')
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        remove_indices.extend(range(start_idx, end_idx + 1))
                    except ValueError:
                        continue
                else:
                    # 處理單個索引
                    try:
                        idx = int(part)
                        remove_indices.append(idx)
                    except ValueError:
                        continue
            
            if not remove_indices:
                return "無效的幀編號格式"
            
            frames = self.current_tracks[track_id]
            original_count = len(frames)
            
            # 移除指定的幀 (從後往前刪除避免索引變化)
            remove_indices = sorted(set(remove_indices), reverse=True)
            removed_count = 0
            
            for idx in remove_indices:
                if 0 <= idx < len(frames):
                    del frames[idx]
                    removed_count += 1
            
            # 更新追蹤序列
            if len(frames) >= self.min_track_frames:  # 根據最小幀數設定
                self.current_tracks[track_id] = frames
                return f"✅ 已刪除 {removed_count} 幀，剩餘 {len(frames)} 幀"
            else:
                # 如果幀數太少，刪除整個追蹤序列
                del self.current_tracks[track_id]
                return f"⚠️ 幀數不足 (< {self.min_track_frames})，已刪除整個追蹤序列 {track_id}"
            
        except Exception as e:
            return f"❌ 刪除幀時發生錯誤: {e}"
    
    def create_download_package(self):
        """建立可下載的標記資料包"""
        if not self.labeling_history:
            return None, "尚無標記資料可下載"
        
        try:
            import zipfile
            from datetime import datetime
            
            # 建立下載資料夾
            download_dir = Path("downloads")
            download_dir.mkdir(exist_ok=True)
            
            # 建立 ZIP 檔案
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"labeled_sequences_{timestamp}.zip"
            zip_path = download_dir / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 添加所有標記的序列資料
                for category_name, category_path in self.categories.items():
                    if category_path.exists():
                        for sequence_dir in category_path.iterdir():
                            if sequence_dir.is_dir():
                                # 添加序列中的所有檔案
                                for file_path in sequence_dir.rglob("*"):
                                    if file_path.is_file():
                                        # 建立 ZIP 內的相對路徑
                                        arc_path = f"{category_name}/{sequence_dir.name}/{file_path.name}"
                                        zipf.write(file_path, arc_path)
                
                # 添加標記歷史記錄
                history_path = self.output_dir / "labeling_history.json"
                if history_path.exists():
                    zipf.write(history_path, "labeling_history.json")
            
            # 檢查檔案大小
            file_size = zip_path.stat().st_size / (1024 * 1024)  # MB
            
            info = f"下載套件建立成功:\\n"
            info += f"- 檔案: {zip_filename}\\n"
            info += f"- 大小: {file_size:.1f} MB\\n"
            info += f"- 包含: {len(self.labeling_history)} 個標記序列\\n"
            
            return str(zip_path), info
            
        except Exception as e:
            return None, f"❌ 建立下載套件失敗: {e}"
    
    def clear_session_data(self):
        """清除當前會話的所有資料"""
        try:
            # 清除追蹤資料
            self.current_tracks = {}
            self.current_video_path = None
            self.labeling_history = []
            
            # 清除臨時上傳檔案
            temp_dir = Path("temp_uploads")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # 清除標記的序列資料
            for category_path in self.categories.values():
                if category_path.exists():
                    import shutil
                    shutil.rmtree(category_path, ignore_errors=True)
                    category_path.mkdir(exist_ok=True)
            
            # 清除標記歷史檔案
            history_path = self.output_dir / "labeling_history.json"
            if history_path.exists():
                history_path.unlink()
            
            # 清除下載檔案
            download_dir = Path("downloads")
            if download_dir.exists():
                import shutil
                shutil.rmtree(download_dir, ignore_errors=True)
            
            return "✅ 已清除所有會話資料\\n包括：上傳影片、追蹤序列、標記資料和下載檔案"
            
        except Exception as e:
            return f"❌ 清除資料時發生錯誤: {e}"
    
    def update_filter_settings(self, confidence_threshold, min_track_frames, min_crop_size, sample_fps):
        """更新所有過濾設定"""
        self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))
        self.min_track_frames = max(1, int(min_track_frames))
        self.min_crop_size = max(16, int(min_crop_size))
        self.sample_fps = max(1, int(sample_fps))
        
        info = "過濾設定已更新:\n"
        info += f"- 信心度閾值: {self.confidence_threshold:.3f}\n"
        info += f"- 最小追蹤幀數: {self.min_track_frames}\n"
        info += f"- 最小裁切尺寸: {self.min_crop_size}px\n"
        info += f"- 採樣頻率: {self.sample_fps} FPS"
        
        return info
    
    def label_track_sequence(self, track_selection, category):
        """標記整個追蹤序列"""
        if not track_selection or not self.current_tracks:
            return "❌ 請先選擇追蹤序列"
        
        if category not in self.categories:
            return f"❌ 無效的類別: {category}"
        
        try:
            # 解析追蹤 ID
            track_id = int(track_selection.split()[1].rstrip(':'))
            
            if track_id not in self.current_tracks:
                return "❌ 追蹤序列不存在"
            
            frames = self.current_tracks[track_id]
            video_name = Path(self.current_video_path).stem
            
            # 建立序列資料夾
            sequence_dir = self.categories[category] / f"{video_name}_track_{track_id}"
            sequence_dir.mkdir(exist_ok=True)
            
            # 儲存序列中的每一幀（裁切邊界框區域）
            saved_files = []
            for i, frame_data in enumerate(frames):
                frame = frame_data['frame']
                bbox = frame_data['bbox']
                timestamp = frame_data['timestamp']
                
                # 建立檔案名稱
                filename = f"{i:03d}_{timestamp:.1f}s.jpg"
                save_path = sequence_dir / filename
                
                x1, y1, x2, y2 = bbox
                
                # 裁切邊界框區域
                h, w = frame.shape[:2]
                # 確保邊界框在圖片範圍內
                x1 = max(0, x1)
                y1 = max(0, y1) 
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # 裁切物件區域
                cropped_frame = frame[y1:y2, x1:x2]
                
                # 如果裁切區域太小，保存帶邊界框的原圖
                if cropped_frame.shape[0] < self.min_crop_size or cropped_frame.shape[1] < self.min_crop_size:
                    # 保存帶邊界框的原圖
                    annotated_frame = frame.copy()
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"T{track_id} {frame_data['class_name']} {frame_data['confidence']:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imwrite(str(save_path), annotated_frame)
                else:
                    # 保存裁切的物件區域
                    cv2.imwrite(str(save_path), cropped_frame)
                
                saved_files.append(str(save_path))
            
            # 儲存序列的元資料
            metadata = {
                'video_path': self.current_video_path,
                'track_id': track_id,
                'category': category,
                'frame_count': len(frames),
                'duration': frames[-1]['timestamp'] - frames[0]['timestamp'],
                'class_name': frames[0]['class_name'],
                'avg_confidence': float(np.mean([f['confidence'] for f in frames])),
                'timestamp_range': [frames[0]['timestamp'], frames[-1]['timestamp']],
                'saved_files': saved_files,
                'label_time': datetime.now().isoformat()
            }
            
            metadata_path = sequence_dir / "metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 記錄標記歷史
            self.labeling_history.append(metadata)
            
            # 儲存標記歷史
            history_path = self.output_dir / "labeling_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.labeling_history, f, indent=2, ensure_ascii=False)
            
            return f"✅ 序列已標記為 {category}\n儲存至: {sequence_dir}\n包含 {len(frames)} 幀影像"
            
        except Exception as e:
            return f"❌ 標記時發生錯誤: {e}"
    
    def get_labeling_stats(self):
        """獲取標記統計"""
        stats = defaultdict(int)
        total_frames = 0
        
        for record in self.labeling_history:
            category = record['category']
            frame_count = record['frame_count']
            stats[category] += 1
            total_frames += frame_count
        
        total_sequences = sum(stats.values())
        stats_text = f"序列標記統計:\n"
        stats_text += f"- 總序列數: {total_sequences}\n"
        stats_text += f"- 總幀數: {total_frames}\n\n"
        
        for category, count in stats.items():
            percentage = (count / total_sequences * 100) if total_sequences > 0 else 0
            stats_text += f"- {category}: {count} 序列 ({percentage:.1f}%)\n"
        
        return stats_text

def create_reid_interface():
    """建立 ReID 標記介面"""
    labeling_tool = ReIDLabelingTool()
    
    with gr.Blocks(title="ReID 時序標記工具", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔥 ReID 時序火災偵測標記工具")
        gr.Markdown("使用物件重新識別技術，將偵測結果分組成時序序列進行標記")
        
        with gr.Tab("影片上傳與處理"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📹 上傳影片")
                    video_upload = gr.File(
                        label="上傳影片檔案 (支援 MP4, MOV, AVI, MKV, WebM)",
                        file_count="single",
                        type="file"
                    )
                    
                    with gr.Row():
                        upload_btn = gr.Button("📤 上傳影片", variant="secondary")
                        clear_btn = gr.Button("🗑️ 清除資料", variant="stop")
                    
                    gr.Markdown("### ⚙️ 過濾設定")
                    with gr.Column():
                        confidence_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.500,
                            step=0.001,
                            label="信心度閾值",
                            info="只保留高於此閾值的偵測結果"
                        )
                        
                        min_frames_slider = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=3,
                            step=1,
                            label="最小追蹤幀數",
                            info="追蹤序列最少需要的幀數"
                        )
                        
                        min_crop_slider = gr.Slider(
                            minimum=16,
                            maximum=128,
                            value=32,
                            step=8,
                            label="最小裁切尺寸 (像素)",
                            info="小於此尺寸的物件不單獨裁切"
                        )
                        
                        sample_fps_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label="採樣頻率 (FPS)",
                            info="每秒採樣的幀數"
                        )
                        
                        update_filters_btn = gr.Button("🔧 更新過濾設定", variant="secondary")
                        
                        filter_info = gr.Textbox(
                            label="過濾設定狀態",
                            lines=5,
                            interactive=False,
                            value="過濾設定已更新:\n- 信心度閾值: 0.500\n- 最小追蹤幀數: 3\n- 最小裁切尺寸: 32px\n- 採樣頻率: 2 FPS"
                        )
                    
                    process_btn = gr.Button("🎬 處理影片 (ReID)", variant="primary")
                    
                    video_info = gr.Textbox(
                        label="處理結果",
                        lines=8,
                        interactive=False
                    )
                
                with gr.Column(scale=2):
                    track_dropdown = gr.Dropdown(
                        choices=[],
                        label="追蹤序列",
                        value=None
                    )
                    
                    show_sequence_btn = gr.Button("👁️ 顯示時序序列", variant="secondary")
                    
                    sequence_info = gr.Textbox(
                        label="序列資訊",
                        lines=6,
                        interactive=False
                    )
        
        with gr.Tab("時序序列預覽"):
            sequence_gallery = gr.Gallery(
                label="時序序列影像",
                show_label=True,
                elem_id="sequence_gallery",
                columns=4,
                rows=2,
                height="auto"
            )
            
            # 幀管理控制項
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🗑️ 移除誤判幀")
                    gr.Markdown("輸入要刪除的幀編號 (例如: 0,2,5 或 0-3)")
                    
                    remove_frames_input = gr.Textbox(
                        label="要刪除的幀編號",
                        placeholder="例如: 0,2,5 或 0-3",
                        lines=1
                    )
                    
                    remove_frames_btn = gr.Button("🗑️ 刪除選中幀", variant="stop")
                    
                    remove_result = gr.Textbox(
                        label="刪除結果",
                        lines=2,
                        interactive=False
                    )
        
        with gr.Tab("序列標記"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🏷️ 選擇序列類型")
                    gr.Markdown("**動態類型（真實時序變化）:**")
                    
                    with gr.Row():
                        true_dynamic_fire_btn = gr.Button("✅ 動態火災", variant="primary")
                        true_dynamic_smoke_btn = gr.Button("✅ 動態煙霧", variant="primary")
                    
                    gr.Markdown("**靜態類型（誤判或靜態）:**")
                    with gr.Row():
                        false_static_cloud_btn = gr.Button("❌ 靜態雲朵", variant="secondary")
                        false_static_light_btn = gr.Button("❌ 靜態燈光", variant="secondary")
                    
                    with gr.Row():
                        false_static_object_btn = gr.Button("❌ 靜態物體", variant="secondary")
                        uncertain_btn = gr.Button("❓ 不確定", variant="stop")
                    
                    label_result = gr.Textbox(
                        label="標記結果",
                        lines=4,
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
                    
                    gr.Markdown("### 📦 下載資料")
                    download_info = gr.Textbox(
                        label="下載資訊",
                        lines=4,
                        interactive=False
                    )
                    
                    create_download_btn = gr.Button("📦 建立下載套件", variant="primary")
                    download_file = gr.File(
                        label="下載標記資料",
                        visible=False
                    )
        
        # 隱藏狀態
        current_tracks = gr.State(value={})
        current_track_selection = gr.State(value="")
        current_video_path = gr.State(value="")
        
        # 事件處理
        def upload_video_wrapper(uploaded_file):
            upload_info, video_path = labeling_tool.upload_video(uploaded_file)
            return upload_info, video_path
        
        def process_video_wrapper(video_path):
            if not video_path:
                return "請先上傳影片檔案", gr.Dropdown(choices=[]), {}
            info, track_options, tracks = labeling_tool.process_video_with_reid(video_path)
            return info, gr.Dropdown(choices=track_options), tracks
        
        def show_sequence_wrapper(tracks, track_selection):
            images, info, frame_options = labeling_tool.show_track_sequence(track_selection)
            return images, info, track_selection
        
        def label_wrapper(track_selection, category):
            result = labeling_tool.label_track_sequence(track_selection, category)
            stats = labeling_tool.get_labeling_stats()
            return result, stats
        
        def remove_frames_wrapper(track_selection, frame_indices, tracks):
            # 刪除幀
            result = labeling_tool.remove_frames_from_track(track_selection, frame_indices)
            
            # 如果刪除成功，重新顯示序列
            if "✅" in result:
                images, info, frame_options = labeling_tool.show_track_sequence(track_selection)
                return result, images, info
            else:
                # 如果序列被刪除或出錯，清空顯示
                return result, [], ""
        
        def create_download_wrapper():
            zip_path, info = labeling_tool.create_download_package()
            if zip_path:
                return info, gr.File(value=zip_path, visible=True)
            else:
                return info, gr.File(visible=False)
        
        def clear_data_wrapper():
            result = labeling_tool.clear_session_data()
            # 重置所有介面元素
            empty_stats = "標記統計 (總計: 0):\\n目前沒有標記資料"
            return (
                result,  # video_info
                "",      # current_video_path
                gr.Dropdown(choices=[], value=None),  # track_dropdown
                {},      # current_tracks
                [],      # sequence_gallery
                "",      # sequence_info
                "",      # current_track_selection
                "",      # remove_result
                "",      # label_result
                empty_stats,  # stats_display
                "",      # download_info
                gr.File(visible=False)  # download_file
            )
        
        def update_filters_wrapper(confidence, min_frames, min_crop, sample_fps):
            result = labeling_tool.update_filter_settings(confidence, min_frames, min_crop, sample_fps)
            return result
        
        # 綁定事件
        update_filters_btn.click(
            update_filters_wrapper,
            inputs=[confidence_slider, min_frames_slider, min_crop_slider, sample_fps_slider],
            outputs=[filter_info]
        )
        
        upload_btn.click(
            upload_video_wrapper,
            inputs=[video_upload],
            outputs=[video_info, current_video_path]
        )
        
        process_btn.click(
            process_video_wrapper,
            inputs=[current_video_path],
            outputs=[video_info, track_dropdown, current_tracks]
        )
        
        show_sequence_btn.click(
            show_sequence_wrapper,
            inputs=[current_tracks, track_dropdown],
            outputs=[sequence_gallery, sequence_info, current_track_selection]
        )
        
        remove_frames_btn.click(
            remove_frames_wrapper,
            inputs=[current_track_selection, remove_frames_input, current_tracks],
            outputs=[remove_result, sequence_gallery, sequence_info]
        )
        
        # 標記按鈕
        for category, btn in [
            ("true_dynamic_fire", true_dynamic_fire_btn),
            ("true_dynamic_smoke", true_dynamic_smoke_btn),
            ("false_static_cloud", false_static_cloud_btn),
            ("false_static_light", false_static_light_btn),
            ("false_static_object", false_static_object_btn),
            ("uncertain", uncertain_btn)
        ]:
            btn.click(
                lambda track_sel, cat=category: label_wrapper(track_sel, cat),
                inputs=[current_track_selection],
                outputs=[label_result, stats_display]
            )
        
        refresh_stats_btn.click(
            labeling_tool.get_labeling_stats,
            outputs=[stats_display]
        )
        
        create_download_btn.click(
            create_download_wrapper,
            outputs=[download_info, download_file]
        )
        
        clear_btn.click(
            clear_data_wrapper,
            outputs=[
                video_info, current_video_path, track_dropdown, current_tracks,
                sequence_gallery, sequence_info, current_track_selection,
                remove_result, label_result, stats_display, download_info, download_file
            ]
        )
    
    return interface

def main():
    """啟動 ReID 標記介面"""
    print("🚀 啟動 ReID 時序標記介面...")
    
    if not ULTRALYTICS_AVAILABLE:
        print("⚠️  請安裝 ultralytics: pip install ultralytics")
    
    if not SUPERVISION_AVAILABLE:
        print("⚠️  請安裝 supervision: pip install supervision")
    
    try:
        interface = create_reid_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ 啟動介面失敗: {e}")

if __name__ == "__main__":
    main()