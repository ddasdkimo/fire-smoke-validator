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
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

try:
    import psutil
    import gc
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False
    print("⚠️  psutil 未安裝，記憶體監控功能受限")

class VideoAnalyzer:
    def __init__(self):
        self.model_path = "best.pt"
        self.work_dir = Path("temp_analysis")
        self.work_dir.mkdir(exist_ok=True)
        self.dataset_dir = Path("dataset")
        self.dataset_dir.mkdir(exist_ok=True)
        
        # 進度追蹤
        self.progress_queue = queue.Queue()
        self.analysis_status = {}
        self.max_workers = min(4, os.cpu_count() or 1)  # 限制並發數
        
        # 檢測可用設備
        self.available_devices = self._detect_available_devices()
        self.current_device = self.available_devices['default']
        
        # 延遲模型載入，等待用戶選擇設備
        self.model = None
        
        # 初始化追蹤器
        if SUPERVISION_AVAILABLE:
            self.tracker = sv.ByteTrack()
        else:
            self.tracker = None
        
        self.current_events = []
        self.session_id = None
        
        # 記憶體管理
        self._cleanup_old_sessions()
    
    def _detect_available_devices(self):
        """檢測可用的計算設備"""
        devices = {
            'options': ['cpu'],
            'default': 'cpu',
            'status': {'cpu': '✅ 可用'}
        }
        
        if ULTRALYTICS_AVAILABLE:
            try:
                import torch
                
                # 檢查 CUDA
                if torch.cuda.is_available():
                    devices['options'].insert(0, 'cuda')
                    devices['default'] = 'cuda'
                    device_name = torch.cuda.get_device_name()
                    devices['status']['cuda'] = f'✅ 可用 ({device_name})'
                    print(f"✅ 檢測到 CUDA: {device_name}")
                else:
                    devices['status']['cuda'] = '❌ 不可用'
                
                # 檢查 MPS (Apple Silicon)
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    if 'cuda' not in devices['options']:  # 如果沒有 CUDA，MPS 為優先
                        devices['options'].insert(0, 'mps')
                        devices['default'] = 'mps'
                    else:
                        devices['options'].insert(1, 'mps')
                    devices['status']['mps'] = '✅ 可用 (Apple Silicon)'
                    print("✅ 檢測到 MPS (Apple Silicon)")
                else:
                    devices['status']['mps'] = '❌ 不可用'
                    
            except Exception as e:
                print(f"設備檢測錯誤: {e}")
        
        print(f"可用設備: {devices['options']}, 預設: {devices['default']}")
        return devices
    
    def load_model(self, device='auto'):
        """載入或重新載入模型到指定設備"""
        if device == 'auto':
            device = self.current_device
        
        try:
            if ULTRALYTICS_AVAILABLE and Path(self.model_path).exists():
                print(f"正在載入模型到 {device}...")
                
                # 釋放舊模型
                if self.model is not None:
                    del self.model
                    if MEMORY_MONITORING_AVAILABLE:
                        gc.collect()
                
                self.model = YOLO(self.model_path)
                self.model.to(device)
                self.current_device = device
                print(f"✅ 已載入 best.pt 模型到 {device}")
                return f"✅ 模型已載入到 {device}"
            else:
                self.model = None
                print("⚠️  未找到 best.pt，將使用模擬偵測")
                return "⚠️  未找到 best.pt，使用模擬偵測"
        except Exception as e:
            print(f"模型載入失敗: {e}")
            # 回退到 CPU
            if device != 'cpu':
                print("嘗試回退到 CPU...")
                return self.load_model('cpu')
            else:
                self.model = None
                return f"❌ 模型載入失敗: {e}"
        
        # 記憶體管理
        self._cleanup_old_sessions()
    
    def analyze_videos(self, video_paths, confidence_threshold=0.300, progress_callback=None):
        """分析多個影片，提取事件（支援並發處理）"""
        try:
            if not video_paths:
                return "請上傳影片檔案", [], ""
            
            # 確保是串列
            if not isinstance(video_paths, list):
                video_paths = [video_paths]
            
            self.session_id = str(uuid.uuid4())[:8]
            session_dir = self.work_dir / self.session_id
            session_dir.mkdir(exist_ok=True)
            
            # 初始化進度追蹤
            self.analysis_status = {}
            for i, path in enumerate(video_paths):
                video_name = Path(path).name
                self.analysis_status[i] = {
                    'name': video_name,
                    'status': '等待中',
                    'progress': 0,
                    'detections': 0
                }
            
            all_video_detections = []
            video_summaries = []
            
            print(f"準備並發分析 {len(video_paths)} 個影片（最多 {self.max_workers} 個並發）...")
            
            if progress_callback:
                progress_callback(self._format_progress_text())
            
            # 使用線程池並發處理
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任務
                future_to_video = {
                    executor.submit(self._analyze_single_video_with_progress, video_path, video_idx, Path(video_path).name, confidence_threshold, progress_callback): (video_idx, video_path)
                    for video_idx, video_path in enumerate(video_paths)
                }
                
                # 收集結果
                for future in as_completed(future_to_video):
                    video_idx, video_path = future_to_video[future]
                    try:
                        video_detections, summary = future.result()
                        video_summaries.append(summary)
                        all_video_detections.extend(video_detections)
                        
                        # 更新狀態
                        self.analysis_status[video_idx]['status'] = '完成'
                        if progress_callback:
                            progress_callback(self._format_progress_text())
                            
                    except Exception as e:
                        print(f"分析影片 {Path(video_path).name} 時發生錯誤: {str(e)}")
                        self.analysis_status[video_idx]['status'] = '錯誤'
                        self.analysis_status[video_idx]['error'] = str(e)
                        if progress_callback:
                            progress_callback(self._format_progress_text())
            
            print(f"\n總共偵測到 {len(all_video_detections)} 個物件，開始分組...")
            if progress_callback:
                progress_callback(self._format_progress_text() + "\n\n🔄 正在分組事件...")
            
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
                detections = self._detect_objects(frame, frame_idx, timestamp, confidence_threshold)
                if detections:
                    # 加上影片來源資訊
                    for det in detections:
                        det['video_name'] = video_name
                        det['video_idx'] = video_idx
                    video_detections.extend(detections)
            
            frame_idx += 1
            
            # 釋放記憶體
            if frame_idx % 500 == 0 and MEMORY_MONITORING_AVAILABLE:
                gc.collect()
        
        cap.release()
        
        summary = {
            'name': video_name,
            'detections': len(video_detections),
            'duration': duration
        }
        
        return video_detections, summary
    
    def _analyze_single_video_with_progress(self, video_path, video_idx, video_name, confidence_threshold, progress_callback):
        """帶進度回饋的單個影片分析"""
        try:
            # 更新狀態為處理中
            self.analysis_status[video_idx]['status'] = '處理中'
            if progress_callback:
                progress_callback(self._format_progress_text())
            
            # 打開影片
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.analysis_status[video_idx]['status'] = '錯誤'
                self.analysis_status[video_idx]['error'] = '無法打開影片'
                return [], {'name': video_name, 'detections': 0, 'duration': 0}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # 收集偵測結果
            video_detections = []
            frame_idx = 0
            sample_interval = max(1, int(fps * 1.0))
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 更新進度
                progress_percent = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
                self.analysis_status[video_idx]['progress'] = progress_percent
                
                # 每100幀或進度變化時更新介面
                if frame_idx % 100 == 0 or progress_percent != self.analysis_status[video_idx].get('last_progress', -1):
                    self.analysis_status[video_idx]['last_progress'] = progress_percent
                    if progress_callback:
                        progress_callback(self._format_progress_text())
                
                # 採樣策略：每秒取一幀
                if frame_idx % sample_interval == 0:
                    timestamp = frame_idx / fps
                    detections = self._detect_objects(frame, frame_idx, timestamp, confidence_threshold)
                    if detections:
                        # 加上影片來源資訊
                        for det in detections:
                            det['video_name'] = video_name
                            det['video_idx'] = video_idx
                        video_detections.extend(detections)
                        
                        # 更新偵測數量
                        self.analysis_status[video_idx]['detections'] = len(video_detections)
                
                frame_idx += 1
                
                # 釋放記憶體和優化
                if frame_idx % 500 == 0:
                    memory_usage = self._optimize_memory_usage()
                    if frame_idx % 1000 == 0:  # 每1000幀顯示一次記憶體使用情況
                        print(f"記憶體使用: {memory_usage:.1f} MB")
            
            cap.release()
            
            # 完成狀態
            self.analysis_status[video_idx]['progress'] = 100
            self.analysis_status[video_idx]['detections'] = len(video_detections)
            
            summary = {
                'name': video_name,
                'detections': len(video_detections),
                'duration': duration
            }
            
            return video_detections, summary
            
        except Exception as e:
            self.analysis_status[video_idx]['status'] = '錯誤'
            self.analysis_status[video_idx]['error'] = str(e)
            if progress_callback:
                progress_callback(self._format_progress_text())
            raise e
    
    def _format_progress_text(self):
        """格式化進度文字"""
        lines = [f"📊 分析進度 ({len(self.analysis_status)} 個影片):"]
        lines.append("")
        
        for idx, status in self.analysis_status.items():
            name = status['name'][:30] + "..." if len(status['name']) > 30 else status['name']
            
            if status['status'] == '等待中':
                lines.append(f"⏳ {name}: 等待中")
            elif status['status'] == '處理中':
                progress = status.get('progress', 0)
                detections = status.get('detections', 0)
                lines.append(f"🔄 {name}: {progress}% ({detections} 個偵測)")
            elif status['status'] == '完成':
                detections = status.get('detections', 0)
                lines.append(f"✅ {name}: 完成 ({detections} 個偵測)")
            elif status['status'] == '錯誤':
                error = status.get('error', '未知錯誤')
                lines.append(f"❌ {name}: 錯誤 - {error}")
        
        return "\n".join(lines)
    
    def _cleanup_old_sessions(self):
        """清理舊的分析會話以節省磁碟空間"""
        try:
            if not self.work_dir.exists():
                return
                
            import time
            current_time = time.time()
            
            for session_path in self.work_dir.iterdir():
                if session_path.is_dir():
                    # 檢查資料夾修改時間，刪除超過1小時的舊會話
                    if current_time - session_path.stat().st_mtime > 3600:  # 1小時
                        import shutil
                        shutil.rmtree(session_path, ignore_errors=True)
                        print(f"清理舊會話: {session_path.name}")
        except Exception as e:
            print(f"清理舊會話時發生錯誤: {e}")
    
    def _optimize_memory_usage(self):
        """優化記憶體使用"""
        if not MEMORY_MONITORING_AVAILABLE:
            return 0
            
        try:
            # 強制垃圾回收
            gc.collect()
            
            # 檢查記憶體使用情況
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            if memory_usage > 2000:  # 如果超過2GB
                print(f"⚠️  記憶體使用量較高: {memory_usage:.1f} MB")
                # 清理可能的大型變數
                gc.collect()
                
            return memory_usage
        except Exception as e:
            print(f"記憶體監控錯誤: {e}")
            return 0
    
    def _detect_objects(self, frame, frame_idx, timestamp, confidence_threshold=0.300):
        """在幀中偵測物件"""
        try:
            if self.model:
                # 使用真實的 YOLO 模型（會自動使用已設定的 device）
                results = self.model(frame, conf=confidence_threshold, verbose=False)
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
            
            # 統計按影片分組的進度
            video_progress = self._get_video_labeling_progress()
            
            # 構建詳細進度信息
            progress_text = f"事件 {event_idx+1} 已標註為：{label}\n"
            progress_text += f"總體進度：{labeled_count}/{total_count} ({labeled_count/total_count*100:.1f}%)\n"
            progress_text += f"來源檔案：{event.get('video_name', 'unknown')}\n"
            progress_text += video_progress
            
            return progress_text
        return "標註失敗"
    
    def _get_video_labeling_progress(self):
        """獲取按影片分組的標註進度"""
        video_stats = {}
        
        # 按影片統計事件
        for i, event in enumerate(self.current_events):
            video_name = event.get('video_name', 'unknown')
            if video_name not in video_stats:
                video_stats[video_name] = {'total': 0, 'labeled': 0, 'events': []}
            
            video_stats[video_name]['total'] += 1
            video_stats[video_name]['events'].append(i)
            
            if event.get('label') is not None:
                video_stats[video_name]['labeled'] += 1
        
        # 構建進度文本
        progress_lines = []
        completed_videos = 0
        
        for video_name, stats in video_stats.items():
            labeled = stats['labeled']
            total = stats['total']
            percentage = (labeled / total * 100) if total > 0 else 0
            
            status = "✅ 完成" if labeled == total else f"📝 進行中"
            if labeled == total:
                completed_videos += 1
                
            progress_lines.append(f"  {status} {video_name}: {labeled}/{total} ({percentage:.0f}%)")
        
        total_videos = len(video_stats)
        summary = f"影片進度：{completed_videos}/{total_videos} 個檔案完成\n"
        
        return summary + "\n".join(progress_lines)
    
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
        
        # 為 analyzer 添加進度追蹤屬性
        analyzer.current_progress = ""
        analyzer.analysis_complete = False
        analyzer.analysis_result = None
        analyzer.analysis_error = None
        
        # 載入模型到指定設備
        def load_model_to_device(device):
            status_message = analyzer.load_model(device)
            
            # 同時顯示設備可用性信息
            device_info = []
            for dev in ['cuda', 'mps', 'cpu']:
                if dev in analyzer.available_devices['status']:
                    status = analyzer.available_devices['status'][dev]
                    device_info.append(f"{dev.upper()}: {status}")
            
            full_status = f"{status_message}\n\n設備狀態:\n" + "\n".join(device_info)
            return full_status
        
        # 初始載入模型到預設設備
        initial_status = load_model_to_device(analyzer.available_devices['default'])
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(
                    label="上傳影片檔案（支援多檔案）",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"]
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.001,
                    maximum=1.0,
                    value=0.300,
                    step=0.001,
                    label="🎯 偵測信心度閾值",
                    info="調整YOLO模型的偵測閾值，數值越低偵測越敏感"
                )
                
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=analyzer.available_devices['options'],
                        value=analyzer.available_devices['default'],
                        label="⚡ 計算設備",
                        info="選擇模型運行設備"
                    )
                    load_model_btn = gr.Button("🔄 載入模型", variant="secondary", size="sm")
                
                model_status = gr.Textbox(
                    label="📊 模型狀態",
                    value=initial_status,
                    lines=4,
                    interactive=False
                )
                
                analyze_btn = gr.Button("🔍 開始分析", variant="primary")
                
                # 即時進度顯示
                progress_display = gr.Textbox(
                    label="📊 即時分析進度",
                    lines=8,
                    placeholder="等待上傳影片檔案...",
                    interactive=False,
                    max_lines=15
                )
                
                analysis_result = gr.Textbox(
                    label="分析結果",
                    lines=6,
                    placeholder="分析完成後顯示結果",
                    interactive=False
                )
                
            with gr.Column(scale=2):
                # 當前事件顯示區
                gr.Markdown("## 🎯 當前事件 & 快速標註")
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
                
                # 標註進度顯示
                progress_info = gr.Textbox(
                    label="📊 標註進度",
                    value="等待分析完成...",
                    lines=3,
                    interactive=False
                )
                
                # 快速標註按鈕
                gr.Markdown("### 🏷️ 快速標註")
                with gr.Row():
                    label_true_btn = gr.Button("✅ 真實火煙", variant="primary", scale=2, size="lg")
                    label_false_btn = gr.Button("❌ 誤判", variant="secondary", scale=2, size="lg")
                
                # 導航按鈕
                with gr.Row():
                    prev_btn = gr.Button("⬅️ 上一個", scale=1)
                    next_btn = gr.Button("➡️ 下一個", scale=1)
                    skip_btn = gr.Button("⏭️ 跳過", scale=1)
        
        # 播放控制和幀資訊顯示
        with gr.Row():
            with gr.Column(scale=3):
                frame_info_display = gr.Textbox(
                    label="📸 幀播放資訊",
                    lines=1,
                    interactive=False,
                    placeholder="等待事件載入..."
                )
            with gr.Column(scale=2):
                playback_speed = gr.Slider(
                    minimum=0.1,
                    maximum=5.0,
                    value=1.0,
                    step=0.1,
                    label="⚡ 播放速度",
                    info="調整幀切換速度（倍數）"
                )
        
        # 匯出資料集區域
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📦 匯出資料集")
                export_btn = gr.Button("💾 匯出標註資料集", variant="secondary")
                export_result = gr.Textbox(label="匯出結果", lines=3)
                export_file = gr.File(label="下載資料集")
        
        # 自動輪播計時器（初始值1.0秒，會根據播放速度動態調整）
        timer = gr.Timer(value=1.0, active=False)
        
        # 進度更新計時器  
        progress_timer = gr.Timer(value=1.0, active=False)
        
        # 播放速度狀態
        current_speed = gr.State(1.0)
        
        # 更新播放速度
        def update_playback_speed(speed):
            """根據播放速度調整計時器間隔"""
            # 基礎間隔是1.0秒，速度越快間隔越短
            base_interval = 1.0
            new_interval = base_interval / speed
            # 限制間隔範圍在0.2秒到10秒之間
            new_interval = max(0.2, min(10.0, new_interval))
            
            # 檢查計時器是否活躍，如果是則更新間隔並保持活躍狀態
            is_active = analyzer.current_events and len(analyzer.current_events) > 0
            return gr.Timer(value=new_interval, active=is_active), speed
        
        # 更新當前事件顯示
        def update_event_display(event_idx, frame_idx):
            try:
                if not analyzer.current_events or event_idx >= len(analyzer.current_events):
                    return "無事件", None, None, f"進度: 0/0", "等待事件載入...", event_idx, 0
                
                event = analyzer.current_events[event_idx]
                if not event['frames']:
                    return "無幀資料", None, None, f"進度: {event_idx+1}/{len(analyzer.current_events)}", "無幀資料", event_idx, 0
                
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
                
                # 詳細進度資訊
                labeled_count = sum(1 for e in analyzer.current_events if e['label'] is not None)
                total_count = len(analyzer.current_events)
                video_progress = analyzer._get_video_labeling_progress()
                
                progress_text = f"總體進度: {labeled_count}/{total_count} ({labeled_count/total_count*100:.1f}%)\n"
                progress_text += f"當前事件來源: {event.get('video_name', 'unknown')}\n"
                progress_text += video_progress
                
                # 幀播放資訊
                frame_info_text = f"第 {frame_idx + 1} 幀 / 共 {len(event['frames'])} 幀 | " \
                                f"時間: {frame_info['timestamp']:.1f}s | " \
                                f"信心度: {frame_info['confidence']:.3f}"
                
                # 返回完整畫面和裁切區域
                full_path = frame_info.get('full_path', frame_info['image_path'])
                crop_path = frame_info.get('crop_path', frame_info['image_path'])
                
                return info_text, full_path, crop_path, progress_text, frame_info_text, event_idx, (frame_idx + 1)
            except Exception as e:
                print(f"更新事件顯示時發生錯誤: {str(e)}")
                return "錯誤", None, None, "錯誤", "錯誤", 0, 0
        
        # 標註並移到下一個
        def label_and_next(event_idx, label):
            if 0 <= event_idx < len(analyzer.current_events):
                # 使用analyzer的label_event方法獲取詳細進度
                progress_message = analyzer.label_event(event_idx, label)
                
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
                
                return next_idx, 0, progress_message
            return event_idx, 0, "標註失敗"
        
        # 導航函數
        def go_prev(event_idx):
            new_idx = max(0, event_idx - 1)
            return new_idx, 0
        
        def go_next(event_idx):
            new_idx = min(len(analyzer.current_events) - 1, event_idx + 1) if analyzer.current_events else 0
            return new_idx, 0
        
        def skip_current(event_idx):
            # 跳過當前事件，移到下一個
            new_idx = min(len(analyzer.current_events) - 1, event_idx + 1) if analyzer.current_events else 0
            return new_idx, 0
        
        # 進度更新函數
        def update_progress():
            """更新分析進度"""
            if hasattr(analyzer, 'current_progress') and analyzer.current_progress:
                progress_text = analyzer.current_progress
            else:
                progress_text = "等待上傳影片檔案..."
            
            return progress_text
        
        # 檢查分析完成狀態
        def check_analysis_complete(current_speed_val):
            """檢查分析是否完成並返回結果"""
            if hasattr(analyzer, 'analysis_complete') and analyzer.analysis_complete:
                if hasattr(analyzer, 'analysis_error') and analyzer.analysis_error:
                    return f"❌ 分析失敗: {analyzer.analysis_error}", [], "", gr.Timer(active=False), 0, 0, gr.Timer(active=False)
                elif hasattr(analyzer, 'analysis_result') and analyzer.analysis_result:
                    result_text, event_gallery, status = analyzer.analysis_result
                    # 重置完成狀態
                    analyzer.analysis_complete = False
                    analyzer.analysis_result = None
                    
                    # 啟動輪播計時器，根據播放速度設定間隔
                    if analyzer.current_events:
                        base_interval = 1.0
                        new_interval = base_interval / current_speed_val
                        new_interval = max(0.2, min(10.0, new_interval))
                        return result_text, event_gallery, status, gr.Timer(active=False), 0, 0, gr.Timer(value=new_interval, active=True)
                    else:
                        return result_text, event_gallery, status, gr.Timer(active=False), 0, 0, gr.Timer(active=False)
            
            # 如果還沒完成，保持進度計時器運行
            return gr.update(), gr.update(), gr.update(), gr.Timer(active=True), gr.update(), gr.update(), gr.update()
        
        # 非阻塞的分析處理
        def start_analysis(files, confidence):
            if not files:
                return "請上傳影片檔案", gr.Timer(active=False)
            
            # 檢查模型是否已載入
            if analyzer.model is None and ULTRALYTICS_AVAILABLE and Path(analyzer.model_path).exists():
                return "⚠️  請先載入模型再開始分析", gr.Timer(active=False)
            
            video_paths = [file.name for file in files] if isinstance(files, list) else [files.name]
            
            # 啟動後台分析任務
            def analysis_worker():
                def progress_callback(progress_text):
                    analyzer.current_progress = progress_text
                
                try:
                    result = analyzer.analyze_videos(video_paths, confidence, progress_callback)
                    analyzer.analysis_result = result
                    analyzer.analysis_complete = True
                except Exception as e:
                    analyzer.analysis_error = str(e)
                    analyzer.analysis_complete = True
            
            # 重置狀態
            analyzer.analysis_complete = False
            analyzer.analysis_error = None
            analyzer.current_progress = "🚀 準備分析影片..."
            
            # 啟動背景任務
            thread = threading.Thread(target=analysis_worker)
            thread.daemon = True
            thread.start()
            
            return "🚀 開始分析影片，請查看即時進度...", gr.Timer(active=True)
        
        # 載入模型按鈕點擊
        load_model_btn.click(
            load_model_to_device,
            inputs=[device_dropdown],
            outputs=[model_status]
        )
        
        # 分析影片按鈕點擊
        analyze_btn.click(
            start_analysis,
            inputs=[video_input, confidence_slider],
            outputs=[analysis_result, progress_timer]
        )
        
        # 進度計時器更新
        progress_timer.tick(
            update_progress,
            outputs=[progress_display]
        )
        
        # 同時檢查分析完成狀態
        progress_timer.tick(
            check_analysis_complete,
            inputs=[current_speed],
            outputs=[analysis_result, gr.Gallery(visible=False), gr.Textbox(visible=False), progress_timer, current_event_idx, frame_idx, timer]
        )
        
        # 計時器觸發更新
        timer.tick(
            update_event_display,
            inputs=[current_event_idx, frame_idx],
            outputs=[current_event_info, full_frame_display, crop_frame_display, progress_info, frame_info_display, current_event_idx, frame_idx]
        )
        
        # 標註事件
        label_true_btn.click(
            lambda idx: label_and_next(idx, "真實火煙"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx, analysis_result]
        )
        
        label_false_btn.click(
            lambda idx: label_and_next(idx, "誤判"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx, analysis_result]
        )
        
        # 導航按鈕
        prev_btn.click(go_prev, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        next_btn.click(go_next, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        skip_btn.click(skip_current, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        
        # 播放速度控制
        playback_speed.change(
            update_playback_speed,
            inputs=[playback_speed],
            outputs=[timer, current_speed]
        )
        
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