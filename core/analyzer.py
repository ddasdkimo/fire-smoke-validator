#!/usr/bin/env python3
"""
視頻分析器核心模組
負責影片處理、物件偵測、事件分組等核心功能
"""

import cv2
import numpy as np
from pathlib import Path
import json
import uuid
from datetime import datetime
import tempfile
import os
from collections import defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

try:
    import psutil
    import gc
    MEMORY_MONITORING_AVAILABLE = True
except ImportError:
    MEMORY_MONITORING_AVAILABLE = False


class VideoAnalyzer:
    """視頻分析器主類"""
    
    def __init__(self):
        self.model_path = "best.pt"
        self.work_dir = Path("temp_analysis")
        self.work_dir.mkdir(exist_ok=True)
        self.dataset_dir = Path("dataset")
        self.dataset_dir.mkdir(exist_ok=True)
        
        # 進度追蹤
        self.progress_queue = queue.Queue()
        self.analysis_status = {}
        self.max_workers = min(4, os.cpu_count() or 1)
        
        # 檢測可用設備
        self.available_devices = self._detect_available_devices()
        self.current_device = self.available_devices['default']
        
        # 延遲模型載入
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
                    if 'cuda' not in devices['options']:
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
            if device != 'cpu':
                print("嘗試回退到 CPU...")
                return self.load_model('cpu')
            else:
                self.model = None
                return f"❌ 模型載入失敗: {e}"
        
        # 記憶體管理
        self._cleanup_old_sessions()
    
    def analyze_videos(self, video_paths, confidence_threshold=0.300, min_frames=2, max_frames=30, progress_callback=None):
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
            events = self._group_detections_by_reid(all_video_detections, session_dir, min_frames, max_frames)
            
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
                    first_frame_path = event['frames'][0]['image_path']
                    event_gallery.append(first_frame_path)
            
            return result_text, event_gallery, f"找到 {len(events)} 個事件"
            
        except Exception as e:
            print(f"分析影片時發生錯誤: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"分析失敗: {str(e)}", [], "錯誤"
    
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
                    if frame_idx % 1000 == 0:
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
    
    def _detect_objects(self, frame, frame_idx, timestamp, confidence_threshold=0.300):
        """在幀中偵測物件"""
        try:
            if self.model:
                # 使用真實的 YOLO 模型
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
                                    'full_frame': frame.copy()
                                })
                return detections
            else:
                # 模擬偵測（用於測試）
                h, w = frame.shape[:2]
                if frame_idx % 10 == 0:
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
    
    def _group_detections_by_reid(self, detections, session_dir, min_frames=2, max_frames=30):
        """使用 ReID 分組偵測結果，長事件會自動分割成多個子事件"""
        try:
            if not detections:
                return []
            
            # 第一階段：按位置和時間分組（不限制幀數）
            initial_groups = defaultdict(list)
            
            # 按時間排序偵測結果
            sorted_detections = sorted(detections, key=lambda d: d['timestamp'])
            
            for det in sorted_detections:
                # 計算中心點
                x1, y1, x2, y2 = det['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # 找到最近的組別
                best_group = None
                min_distance = float('inf')
                
                for group_id, group_dets in initial_groups.items():
                    # 計算與組別中最後一個偵測的距離
                    last_det = group_dets[-1]
                    last_x1, last_y1, last_x2, last_y2 = last_det['bbox']
                    last_center_x = (last_x1 + last_x2) / 2
                    last_center_y = (last_y1 + last_y2) / 2
                    
                    distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                    time_diff = abs(det['timestamp'] - last_det['timestamp'])
                    
                    # 如果位置接近且時間差不大，視為同一 ReID 追蹤
                    if distance < 100 and time_diff < 10:
                        if distance < min_distance:
                            min_distance = distance
                            best_group = group_id
                
                if best_group is not None:
                    initial_groups[best_group].append(det)
                else:
                    # 創建新組別
                    new_group_id = len(initial_groups)
                    initial_groups[new_group_id].append(det)
            
            # 第二階段：將長事件分割成符合幀數限制的子事件
            events = []
            event_counter = 0
            
            for group_id, group_dets in initial_groups.items():
                print(f"處理原始組別 {group_id}：共 {len(group_dets)} 幀")
                
                # 如果組別少於最少幀數，直接忽略
                if len(group_dets) < min_frames:
                    print(f"  忽略：只有 {len(group_dets)} 幀，少於最少要求 {min_frames} 幀")
                    continue
                
                # 將長事件分割成多個子事件
                num_chunks = (len(group_dets) + max_frames - 1) // max_frames  # 向上取整
                
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * max_frames
                    end_idx = min(start_idx + max_frames, len(group_dets))
                    chunk_dets = group_dets[start_idx:end_idx]
                    
                    # 檢查子事件是否滿足最少幀數要求
                    if len(chunk_dets) >= min_frames:
                        event_dir = session_dir / f"event_{event_counter}"
                        event_dir.mkdir(exist_ok=True)
                        self._process_event_chunk(chunk_dets, event_dir, event_counter, events)
                        print(f"  創建子事件 {event_counter}：{len(chunk_dets)} 幀 (第 {chunk_idx+1}/{num_chunks} 段)")
                        event_counter += 1
                    else:
                        print(f"  跳過子事件：只有 {len(chunk_dets)} 幀，少於最少要求 {min_frames} 幀")
            
            # 統計信息
            total_detections = len(detections)
            total_groups = len(initial_groups)
            total_events = len(events)
            
            print(f"分組完成：")
            print(f"  - 總偵測數: {total_detections}")
            print(f"  - 初始 ReID 組: {total_groups}")
            print(f"  - 最終事件數: {total_events}")
            
            # 檢查事件幀數分布
            frame_counts = [len(event['frames']) for event in events]
            if frame_counts:
                print(f"  - 事件幀數範圍: {min(frame_counts)} - {max(frame_counts)} 幀")
                print(f"  - 平均幀數: {sum(frame_counts)/len(frame_counts):.1f} 幀")
                
                # 顯示分割統計
                original_group_sizes = [len(group_dets) for group_dets in initial_groups.values()]
                if original_group_sizes:
                    large_groups = [size for size in original_group_sizes if size > max_frames]
                    if large_groups:
                        print(f"  - 大型組別分割: {len(large_groups)} 個組別被分割")
                        print(f"  - 分割前最大組別: {max(original_group_sizes)} 幀")
            
            return events
        except Exception as e:
            print(f"分組偵測結果時發生錯誤: {str(e)}")
            return []
    
    def _process_event_chunk(self, group_dets, event_dir, event_id, events):
        """處理一個事件塊，將其保存並添加到事件列表"""
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
            'id': event_id,
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
            
            if memory_usage > 8000:  # 如果超過8GB
                print(f"⚠️  記憶體使用量較高: {memory_usage:.1f} MB")
                gc.collect()
                
            return memory_usage
        except Exception as e:
            print(f"記憶體監控錯誤: {e}")
            return 0