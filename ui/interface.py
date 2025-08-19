#!/usr/bin/env python3
"""
使用者介面控制器
處理 Gradio 介面的邏輯和事件處理
"""

import gradio as gr
from pathlib import Path
import threading


class InterfaceController:
    """介面控制器"""
    
    def __init__(self, analyzer, labeling_manager):
        self.analyzer = analyzer
        self.labeling_manager = labeling_manager
    
    def load_model_to_device(self, device):
        """載入模型到指定設備"""
        status_message = self.analyzer.load_model(device)
        
        # 同時顯示設備可用性信息
        device_info = []
        for dev in ['cuda', 'mps', 'cpu']:
            if dev in self.analyzer.available_devices['status']:
                status = self.analyzer.available_devices['status'][dev]
                device_info.append(f"{dev.upper()}: {status}")
        
        full_status = f"{status_message}\n\n設備狀態:\n" + "\n".join(device_info)
        return full_status
    
    def start_analysis(self, files, confidence, min_frames, max_frames):
        """啟動影片分析"""
        if not files:
            return "請上傳影片檔案", gr.Timer(active=False)
        
        # 檢查模型是否已載入
        try:
            from ultralytics import YOLO
            ULTRALYTICS_AVAILABLE = True
        except ImportError:
            ULTRALYTICS_AVAILABLE = False
            
        if self.analyzer.model is None and ULTRALYTICS_AVAILABLE and Path(self.analyzer.model_path).exists():
            return "⚠️  請先載入模型再開始分析", gr.Timer(active=False)
        
        video_paths = [file.name for file in files] if isinstance(files, list) else [files.name]
        
        # 啟動後台分析任務
        def analysis_worker():
            def progress_callback(progress_text):
                self.analyzer.current_progress = progress_text
            
            try:
                result = self.analyzer.analyze_videos(video_paths, confidence, min_frames, max_frames, progress_callback)
                self.analyzer.analysis_result = result
                self.analyzer.analysis_complete = True
            except Exception as e:
                self.analyzer.analysis_error = str(e)
                self.analyzer.analysis_complete = True
        
        # 重置狀態
        self.analyzer.analysis_complete = False
        self.analyzer.analysis_error = None
        self.analyzer.current_progress = "🚀 準備分析影片..."
        
        # 啟動分析線程
        thread = threading.Thread(target=analysis_worker)
        thread.start()
        
        # 啟動進度計時器
        return "🔄 開始分析...", gr.Timer(active=True)
    
    def update_progress(self):
        """更新分析進度"""
        if hasattr(self.analyzer, 'current_progress') and self.analyzer.current_progress:
            progress_text = self.analyzer.current_progress
        else:
            progress_text = "等待上傳影片檔案..."
        
        return progress_text
    
    def check_analysis_complete(self, current_speed_val):
        """檢查分析是否完成並返回結果"""
        if hasattr(self.analyzer, 'analysis_complete') and self.analyzer.analysis_complete:
            if hasattr(self.analyzer, 'analysis_error') and self.analyzer.analysis_error:
                return f"❌ 分析失敗: {self.analyzer.analysis_error}", [], "", gr.Timer(active=False), 0, 0, gr.Timer(active=False)
            elif hasattr(self.analyzer, 'analysis_result') and self.analyzer.analysis_result:
                result_text, event_gallery, status = self.analyzer.analysis_result
                # 重置完成狀態
                self.analyzer.analysis_complete = False
                self.analyzer.analysis_result = None
                
                # 啟動輪播計時器
                if self.analyzer.current_events:
                    base_interval = 1.0
                    new_interval = base_interval / current_speed_val
                    new_interval = max(0.2, min(10.0, new_interval))
                    return result_text, event_gallery, status, gr.Timer(active=False), 0, 0, gr.Timer(value=new_interval, active=True)
                else:
                    return result_text, event_gallery, status, gr.Timer(active=False), 0, 0, gr.Timer(active=False)
        
        # 如果還沒完成，保持進度計時器運行
        return gr.update(), gr.update(), gr.update(), gr.Timer(active=True), gr.update(), gr.update(), gr.update()
    
    def update_playback_speed(self, speed):
        """更新播放速度"""
        base_interval = 1.0
        new_interval = base_interval / speed
        new_interval = max(0.2, min(10.0, new_interval))
        
        is_active = self.analyzer.current_events and len(self.analyzer.current_events) > 0
        return gr.Timer(value=new_interval, active=is_active), speed
    
    def update_event_display(self, event_idx, frame_idx):
        """更新當前事件顯示"""
        try:
            if not self.analyzer.current_events or event_idx >= len(self.analyzer.current_events):
                return "無事件", None, None, f"進度: 0/0", "等待事件載入...", event_idx, 0
            
            event = self.analyzer.current_events[event_idx]
            if not event['frames']:
                return "無幀資料", None, None, f"進度: {event_idx+1}/{len(self.analyzer.current_events)}", "無幀資料", event_idx, 0
            
            # 循環顯示幀
            frame_idx = frame_idx % len(event['frames'])
            frame_info = event['frames'][frame_idx]
            
            # 事件資訊
            info_text = f"事件 {event_idx+1}/{len(self.analyzer.current_events)} | " \
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
            labeled_count = sum(1 for e in self.analyzer.current_events if e['label'] is not None)
            total_count = len(self.analyzer.current_events)
            video_progress = self.labeling_manager.get_video_labeling_progress()
            
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
    
    def label_and_next(self, event_idx, label):
        """標註並移到下一個"""
        if 0 <= event_idx < len(self.analyzer.current_events):
            # 使用標註管理器
            progress_message = self.labeling_manager.label_event(event_idx, label)
            
            # 找下一個未標註的事件
            next_idx = event_idx + 1
            while next_idx < len(self.analyzer.current_events):
                if self.analyzer.current_events[next_idx]['label'] is None:
                    break
                next_idx += 1
            
            if next_idx >= len(self.analyzer.current_events):
                # 如果沒有未標註的，回到第一個未標註的
                for i in range(len(self.analyzer.current_events)):
                    if self.analyzer.current_events[i]['label'] is None:
                        next_idx = i
                        progress_message += "\n\n⚠️ 已到達最後一個事件，跳回第一個未標註事件"
                        # 使用 gr.Info 顯示提醒
                        gr.Info("已到達最後一個事件！\n即將跳回第一個未標註的事件。")
                        break
                else:
                    # 全部都標註完了
                    next_idx = len(self.analyzer.current_events) - 1  # 停在最後一個
                    labeled_count = sum(1 for e in self.analyzer.current_events if e['label'] is not None)
                    total_count = len(self.analyzer.current_events)
                    progress_message += f"\n\n🎉 恭喜！所有 {total_count} 個事件都已標註完成！"
                    progress_message += "\n💾 請點擊下方「匯出標註資料集」按鈕保存結果"
                    # 使用 gr.Info 顯示完成提醒
                    gr.Info(f"🎉 恭喜！所有 {total_count} 個事件都已標註完成！\n請點擊「匯出標註資料集」按鈕保存結果。")
            
            return next_idx, 0, progress_message
        return event_idx, 0, "標註失敗"
    
    def go_prev(self, event_idx):
        """上一個事件"""
        if event_idx <= 0:
            gr.Info("已經是第一個事件了！")
        new_idx = max(0, event_idx - 1)
        return new_idx, 0
    
    def go_next(self, event_idx):
        """下一個事件"""
        if self.analyzer.current_events and event_idx >= len(self.analyzer.current_events) - 1:
            gr.Info("已經是最後一個事件了！")
        new_idx = min(len(self.analyzer.current_events) - 1, event_idx + 1) if self.analyzer.current_events else 0
        return new_idx, 0
    
    def skip_current(self, event_idx):
        """跳過當前事件"""
        if self.analyzer.current_events and event_idx >= len(self.analyzer.current_events) - 1:
            gr.Info("已經是最後一個事件了！")
        new_idx = min(len(self.analyzer.current_events) - 1, event_idx + 1) if self.analyzer.current_events else 0
        return new_idx, 0
    
    def export_and_return_file(self):
        """匯出資料集並返回檔案"""
        zip_path, result_text = self.labeling_manager.export_dataset()
        return result_text, zip_path