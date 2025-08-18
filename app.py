#!/usr/bin/env python3
"""
火煙誤判標註系統
上傳影片後使用 best.pt 模型掃描，按 ReID 分組事件，快速標註有/無誤判
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime
import tempfile
import os
from collections import defaultdict
import zipfile

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

class VideoAnalyzer:
    def __init__(self):
        self.model_path = "best.pt"
        self.work_dir = Path("temp_analysis")
        self.work_dir.mkdir(exist_ok=True)
        self.dataset_dir = Path("dataset")
        self.dataset_dir.mkdir(exist_ok=True)
        
        # 初始化模型
        if ULTRALYTICS_AVAILABLE and Path(self.model_path).exists():
            import torch
            # 檢查並使用 MPS
            if torch.backends.mps.is_available():
                device = 'mps'
                print("✅ 使用 Mac MPS 加速")
            else:
                device = 'cpu'
                print("ℹ️  使用 CPU 模式")
            
            self.model = YOLO(self.model_path)
            self.model.to(device)
            print(f"✅ 已載入 best.pt 模型到 {device}")
        else:
            self.model = None
            print("⚠️  未找到 best.pt，將使用模擬偵測")
        
        # 初始化追蹤器
        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack()
        else:
            self.tracker = None
        
        self.current_events = []
        self.session_id = None
    
    def analyze_videos(self, video_paths):
        """分析多個影片，提取事件"""
        try:
            if not video_paths:
                return "請上傳影片檔案", [], ""
            
            # 確保是串列
            if not isinstance(video_paths, list):
                video_paths = [video_paths]
            
            self.session_id = str(uuid.uuid4())[:8]
            session_dir = self.work_dir / self.session_id
            session_dir.mkdir(exist_ok=True)
            
            all_video_detections = []
            video_summaries = []
            
            print(f"準備分析 {len(video_paths)} 個影片...")
            
            # 處理每個影片
            for video_idx, video_path in enumerate(video_paths):
                video_name = Path(video_path).name
                print(f"\n分析影片 {video_idx+1}/{len(video_paths)}: {video_name}")
                
                # 分析單個影片
                video_detections, summary = self._analyze_single_video(video_path, video_idx, video_name)
                video_summaries.append(summary)
                all_video_detections.extend(video_detections)
            
            print(f"\n總共偵測到 {len(all_video_detections)} 個物件，開始分組...")
            
            # 按 ReID 分組事件
            events = self._group_detections_by_reid(all_video_detections, session_dir)
            
            print(f"分析完成！找到 {len(events)} 個事件")
            
            self.current_events = events
            
            # 生成分析結果
            result_text = f"""
分析完成！
影片數量：{len(video_paths)} 個
"""
            for summary in video_summaries:
                result_text += f"\n{summary['name']}: {summary['detections']} 個偵測, {summary['duration']:.1f} 秒"
            
            result_text += f"""

總偵測數：{len(all_video_detections)} 個
分組事件數：{len(events)} 個

使用上方輪播介面進行標註
            """.strip()
            
            # 生成事件縮圖供選擇
            event_gallery = []
            for i, event in enumerate(events):
                if event['frames']:
                    # 使用第一幀作為代表圖像
                    first_frame_path = event['frames'][0]['image_path']
                    event_gallery.append(first_frame_path)
            
            return result_text, event_gallery, f"找到 {len(events)} 個事件"
            
        except Exception as e:
            print(f"分析影片時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"分析失敗: {str(e)}", [], "錯誤"
    
    def _analyze_single_video(self, video_path, video_idx, video_name):
        """分析單個影片"""
        # 打開影片
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"無法打開影片檔案: {video_name}")
            return [], {'name': video_name, 'detections': 0, 'duration': 0}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"影片資訊: {total_frames} 幀, {fps:.1f} FPS, {duration:.1f} 秒")
        
        # 收集偵測結果
        video_detections = []
        frame_idx = 0
        sample_interval = max(1, int(fps * 1.0))  # 每 1 秒採樣一幀
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % 100 == 0:  # 每 100 幀顯示進度
                print(f"分析進度: {frame_idx}/{total_frames} 幀 ({frame_idx/total_frames*100:.1f}%)")
            
            # 採樣策略：每秒取一幀
            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                detections = self._detect_objects(frame, frame_idx, timestamp)
                if detections:
                    # 加上影片來源資訊
                    for det in detections:
                        det['video_name'] = video_name
                        det['video_idx'] = video_idx
                    video_detections.extend(detections)
            
            frame_idx += 1
            
            # 釋放記憶體
            if frame_idx % 500 == 0:
                import gc
                gc.collect()
        
        cap.release()
        
        summary = {
            'name': video_name,
            'detections': len(video_detections),
            'duration': duration
        }
        
        return video_detections, summary
    
    def _detect_objects(self, frame, frame_idx, timestamp):
        """在幀中偵測物件"""
        try:
            if self.model:
                # 使用真實的 YOLO 模型（會自動使用已設定的 device）
                results = self.model(frame, conf=0.3, verbose=False)
                detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # 裁切檢測區域
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        if crop.size > 0:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class': int(cls),
                                'frame_idx': frame_idx,
                                'timestamp': timestamp,
                                'crop': crop,
                                'full_frame': frame.copy()  # 保存完整畫面
                            })
                return detections
            else:
                # 模擬偵測（用於測試）
                h, w = frame.shape[:2]
                if frame_idx % 10 == 0:  # 每10幀模擬一個偵測
                    x1, y1 = w//4, h//4
                    x2, y2 = w*3//4, h*3//4
                    crop = frame[y1:y2, x1:x2]
                    return [{
                        'bbox': [x1, y1, x2, y2],
                        'confidence': 0.8,
                        'class': 0,
                        'frame_idx': frame_idx,
                        'timestamp': timestamp,
                        'crop': crop,
                        'full_frame': frame.copy()
                    }]
                return []
        except Exception as e:
            print(f"偵測物件時發生錯誤: {str(e)}")
            return []
    
    def _group_detections_by_reid(self, detections, session_dir):
        """使用 ReID 分組偵測結果"""
        try:
            if not detections:
                return []
            
            # 簡化版分組：按位置和時間接近度分組
            events = []
            grouped = defaultdict(list)
            
            for det in detections:
                # 計算中心點
                x1, y1, x2, y2 = det['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 找到最近的組別
                best_group = None
                min_distance = float('inf')
                
                for group_id, group_dets in grouped.items():
                    # 計算與組別中最後一個偵測的距離
                    last_det = group_dets[-1]
                    last_x1, last_y1, last_x2, last_y2 = last_det['bbox']
                    last_center_x = (last_x1 + last_x2) / 2
                    last_center_y = (last_y1 + last_y2) / 2
                    
                    distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                    time_diff = abs(det['timestamp'] - last_det['timestamp'])
                    
                    # 如果位置接近且時間差不大，視為同一事件
                    if distance < 100 and time_diff < 10:  # 可調整的閾值
                        if distance < min_distance:
                            min_distance = distance
                            best_group = group_id
                
                if best_group is not None:
                    grouped[best_group].append(det)
                else:
                    # 創建新組別
                    new_group_id = len(grouped)
                    grouped[new_group_id].append(det)
            
            # 轉換為事件格式
            for group_id, group_dets in grouped.items():
                if len(group_dets) >= 2:  # 至少需要 2 幀才算一個事件
                    # 保存事件幀
                    event_dir = session_dir / f"event_{group_id}"
                    event_dir.mkdir(exist_ok=True)
                    
                    frames = []
                    for i, det in enumerate(group_dets):
                        # 保存裁切圖片
                        crop_path = event_dir / f"crop_{i:03d}_{det['timestamp']:.1f}s.jpg"
                        cv2.imwrite(str(crop_path), det['crop'])
                        
                        # 在完整畫面上繪製框線
                        full_frame = det['full_frame'].copy()
                        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                        cv2.rectangle(full_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(full_frame, f"Conf: {det['confidence']:.2f}", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
                        # 保存帶框線的完整畫面
                        full_path = event_dir / f"full_{i:03d}_{det['timestamp']:.1f}s.jpg"
                        cv2.imwrite(str(full_path), full_frame)
                        
                        frames.append({
                            'crop_path': str(crop_path),
                            'full_path': str(full_path),
                            'image_path': str(crop_path),  # 保持相容性
                            'timestamp': det['timestamp'],
                            'confidence': det['confidence'],
                            'bbox': det['bbox'],
                            'video_name': det.get('video_name', 'unknown'),
                            'video_idx': det.get('video_idx', 0)
                        })
                    
                    # 獲取主要影片來源
                    video_names = list(set(det.get('video_name', 'unknown') for det in group_dets))
                    primary_video = video_names[0] if len(video_names) == 1 else 'multiple'
                    
                    event = {
                        'id': group_id,
                        'frames': frames,
                        'start_time': float(group_dets[0]['timestamp']),
                        'end_time': float(group_dets[-1]['timestamp']),
                        'duration': float(group_dets[-1]['timestamp'] - group_dets[0]['timestamp']),
                        'frame_count': len(group_dets),
                        'avg_confidence': float(np.mean([d['confidence'] for d in group_dets])),
                        'video_name': primary_video,
                        'video_names': video_names,
                        'label': None  # 待標註
                    }
                    events.append(event)
            
            return events
        except Exception as e:
            print(f"分組偵測結果時發生錯誤: {str(e)}")
            return []
    
    def label_event(self, event_idx, label):
        """標註事件"""
        if 0 <= event_idx < len(self.current_events):
            self.current_events[event_idx]['label'] = label
            self.current_events[event_idx]['labeled_at'] = datetime.now().isoformat()
            
            event = self.current_events[event_idx]
            
            # 統計已標註和未標註的數量
            labeled_count = sum(1 for e in self.current_events if e['label'] is not None)
            total_count = len(self.current_events)
            
            return f"事件 {event_idx+1} 已標註為：{label}   進度：{labeled_count}/{total_count}"
        return "標註失敗"
    
    def export_dataset(self):
        """匯出標註好的資料集"""
        if not self.current_events:
            return None, "尚未分析影片"
        
        labeled_events = [e for e in self.current_events if e['label'] is not None]
        if not labeled_events:
            return None, "尚未標註任何事件"
        
        # 創建資料集結構
        export_dir = self.dataset_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        export_dir.mkdir(exist_ok=True)
        
        # 按標籤分類
        true_positive_dir = export_dir / "true_positive"  # 真實火煙
        false_positive_dir = export_dir / "false_positive"  # 誤判
        true_positive_dir.mkdir(exist_ok=True)
        false_positive_dir.mkdir(exist_ok=True)
        
        stats = {'true_positive': 0, 'false_positive': 0}
        
        for event in labeled_events:
            if event['label'] == '真實火煙':
                target_dir = true_positive_dir
                stats['true_positive'] += 1
            else:  # '誤判'
                target_dir = false_positive_dir
                stats['false_positive'] += 1
            
            # 複製事件檔案
            event_target_dir = target_dir / f"event_{event['id']}"
            event_target_dir.mkdir(exist_ok=True)
            
            for frame in event['frames']:
                src_path = Path(frame['image_path'])
                if src_path.exists():
                    dst_path = event_target_dir / src_path.name
                    cv2.imwrite(str(dst_path), cv2.imread(str(src_path)))
            
            # 保存事件元資料（轉換 numpy 類型為 Python 原生類型）
            metadata = {
                'id': int(event['id']),
                'label': event['label'],
                'start_time': float(event['start_time']),
                'end_time': float(event['end_time']),
                'duration': float(event['duration']),
                'frame_count': int(event['frame_count']),
                'avg_confidence': float(event['avg_confidence']),
                'labeled_at': event.get('labeled_at')
            }
            
            with open(event_target_dir / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # 保存統計資訊
        summary = {
            'export_date': datetime.now().isoformat(),
            'total_events': len(labeled_events),
            'true_positive_count': stats['true_positive'],
            'false_positive_count': stats['false_positive'],
            'unlabeled_count': len(self.current_events) - len(labeled_events)
        }
        
        with open(export_dir / "dataset_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 創建 ZIP 檔案
        zip_path = export_dir.with_suffix('.zip')
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file_path in export_dir.rglob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.relative_to(export_dir))
        
        result_text = f"""
資料集匯出完成！
真實火煙：{stats['true_positive']} 個事件
誤判：{stats['false_positive']} 個事件
檔案位置：{zip_path}
        """.strip()
        
        return str(zip_path), result_text

def create_interface():
    analyzer = VideoAnalyzer()
    
    with gr.Blocks(title="火煙誤判標註系統", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔥 火煙誤判標註系統")
        gr.Markdown("上傳多個影片 → 批次分析火煙事件 → 統一標註真實/誤判 → 匯出訓練資料集")
        
        # 狀態變數
        current_event_idx = gr.State(0)
        frame_idx = gr.State(0)
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="上傳影片檔案（支援多檔案）",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"]
                )
                analyze_btn = gr.Button("🔍 開始分析", variant="primary")
                
                analysis_result = gr.Textbox(
                    label="分析結果",
                    lines=10,
                    placeholder="上傳影片檔案（支援多個檔案）並點擊分析按鈕"
                )
                
            with gr.Column(scale=2):
                # 當前事件顯示區
                gr.Markdown("## 🎯 當前事件")
                current_event_info = gr.Textbox(
                    label="事件資訊",
                    lines=2,
                    interactive=False
                )
                
                with gr.Row():
                    # 左邊：完整畫面與框線
                    full_frame_display = gr.Image(
                        label="完整畫面（含偵測框線）",
                        type="filepath",
                        height=400,
                        scale=2
                    )
                    
                    # 右邊：裁切區域輪播
                    crop_frame_display = gr.Image(
                        label="事件區域（放大檢視）",
                        type="filepath",
                        height=400,
                        scale=1
                    )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📝 快速標註")
                progress_info = gr.Textbox(
                    label="標註進度",
                    value="等待分析完成...",
                    lines=2,
                    interactive=False
                )
                
                with gr.Row():
                    label_true_btn = gr.Button("✅ 真實火煙", variant="primary", scale=1)
                    label_false_btn = gr.Button("❌ 誤判", variant="secondary", scale=1)
                
                with gr.Row():
                    prev_btn = gr.Button("⬅️ 上一個", scale=1)
                    next_btn = gr.Button("➡️ 下一個", scale=1)
                    skip_btn = gr.Button("⏭️ 跳過", scale=1)
                
            with gr.Column():
                gr.Markdown("## 📦 匯出資料集")
                export_btn = gr.Button("💾 匯出標註資料集", variant="secondary")
                export_result = gr.Textbox(label="匯出結果", lines=3)
                export_file = gr.File(label="下載資料集")
        
        # 自動輪播計時器
        timer = gr.Timer(value=0.5, active=False)
        
        # 更新當前事件顯示
        def update_event_display(event_idx, frame_idx):
            try:
                if not analyzer.current_events or event_idx >= len(analyzer.current_events):
                    return "無事件", None, None, f"進度: 0/0", event_idx, 0
                
                event = analyzer.current_events[event_idx]
                if not event['frames']:
                    return "無幀資料", None, None, f"進度: {event_idx+1}/{len(analyzer.current_events)}", event_idx, 0
                
                # 循環顯示幀
                frame_idx = frame_idx % len(event['frames'])
                frame_info = event['frames'][frame_idx]
                
                # 事件資訊
                info_text = f"事件 {event_idx+1}/{len(analyzer.current_events)} | " \
                           f"時長: {event['duration']:.1f}秒 | " \
                           f"幀數: {event['frame_count']} | " \
                           f"信心度: {event['avg_confidence']:.2f}"
                
                # 顯示影片來源
                if 'video_name' in event and event['video_name'] != 'multiple':
                    info_text += f" | 來源: {event['video_name']}"
                elif 'video_names' in event and len(event['video_names']) > 1:
                    info_text += f" | 來源: {len(event['video_names'])} 個影片"
                
                if event['label']:
                    info_text += f" | 已標註: {event['label']}"
                
                # 進度資訊
                labeled_count = sum(1 for e in analyzer.current_events if e['label'] is not None)
                progress_text = f"進度: {labeled_count}/{len(analyzer.current_events)} 已標註"
                
                # 返回完整畫面和裁切區域
                full_path = frame_info.get('full_path', frame_info['image_path'])
                crop_path = frame_info.get('crop_path', frame_info['image_path'])
                
                return info_text, full_path, crop_path, progress_text, event_idx, (frame_idx + 1)
            except Exception as e:
                print(f"更新事件顯示時發生錯誤: {str(e)}")
                return "錯誤", None, None, "錯誤", 0, 0
        
        # 標註並移到下一個
        def label_and_next(event_idx, label):
            if 0 <= event_idx < len(analyzer.current_events):
                analyzer.current_events[event_idx]['label'] = label
                analyzer.current_events[event_idx]['labeled_at'] = datetime.now().isoformat()
                
                # 找下一個未標註的事件
                next_idx = event_idx + 1
                while next_idx < len(analyzer.current_events):
                    if analyzer.current_events[next_idx]['label'] is None:
                        break
                    next_idx += 1
                
                if next_idx >= len(analyzer.current_events):
                    # 如果沒有未標註的，回到第一個未標註的
                    for i in range(len(analyzer.current_events)):
                        if analyzer.current_events[i]['label'] is None:
                            next_idx = i
                            break
                    else:
                        next_idx = 0  # 全部都標註完了
                
                return next_idx, 0
            return event_idx, 0
        
        # 導航函數
        def go_prev(event_idx):
            new_idx = max(0, event_idx - 1)
            return new_idx, 0
        
        def go_next(event_idx):
            new_idx = min(len(analyzer.current_events) - 1, event_idx + 1) if analyzer.current_events else 0
            return new_idx, 0
        
        def skip_current(event_idx):
            return label_and_next(event_idx, None)[0], 0
        
        # 處理上傳的檔案
        def process_uploaded_files(files):
            if not files:
                return "請上傳影片檔案", [], ""
            
            # 提取檔案路徑
            video_paths = [file.name for file in files] if isinstance(files, list) else [files.name]
            return analyzer.analyze_videos(video_paths)
        
        # 分析影片完成後的處理
        def on_analysis_complete(result, gallery, status):
            if analyzer.current_events:
                # 啟動計時器開始輪播
                return result, 0, 0, gr.Timer(active=True)
            return result, 0, 0, gr.Timer(active=False)
        
        # 分析影片
        analyze_btn.click(
            process_uploaded_files,
            inputs=[video_input],
            outputs=[analysis_result, gr.Gallery(visible=False), gr.Textbox(visible=False)]
        ).then(
            on_analysis_complete,
            inputs=[analysis_result, gr.Gallery(visible=False), gr.Textbox(visible=False)],
            outputs=[analysis_result, current_event_idx, frame_idx, timer]
        )
        
        # 計時器觸發更新
        timer.tick(
            update_event_display,
            inputs=[current_event_idx, frame_idx],
            outputs=[current_event_info, full_frame_display, crop_frame_display, progress_info, current_event_idx, frame_idx]
        )
        
        # 標註事件
        label_true_btn.click(
            lambda idx: label_and_next(idx, "真實火煙"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx]
        )
        
        label_false_btn.click(
            lambda idx: label_and_next(idx, "誤判"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx]
        )
        
        # 導航按鈕
        prev_btn.click(go_prev, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        next_btn.click(go_next, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        skip_btn.click(skip_current, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        
        # 匯出資料集
        def export_and_return_file():
            zip_path, result_text = analyzer.export_dataset()
            return result_text, zip_path
        
        export_btn.click(
            export_and_return_file,
            outputs=[export_result, export_file]
        )
    
    return app

if __name__ == "__main__":
    print("🔥 啟動火煙誤判標註系統...")
    print("=" * 50)
    print("功能：")
    print("✅ 多影片批次上傳和分析")
    print("✅ 使用 best.pt 進行火煙偵測")
    print("✅ ReID 技術自動分組事件")
    print("✅ 統一標註流程處理多影片事件")
    print("✅ 快速標註真實/誤判")
    print("✅ 匯出結構化訓練資料集")
    print("=" * 50)
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )