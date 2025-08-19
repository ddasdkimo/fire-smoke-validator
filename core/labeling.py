#!/usr/bin/env python3
"""
標註和資料匯出模組
處理事件標註、進度追蹤、資料集匯出等功能
"""

import json
import zipfile
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime


class LabelingManager:
    """標註管理器"""
    
    def __init__(self, analyzer):
        self.analyzer = analyzer
    
    def label_event(self, event_idx, label):
        """標註事件"""
        if 0 <= event_idx < len(self.analyzer.current_events):
            self.analyzer.current_events[event_idx]['label'] = label
            self.analyzer.current_events[event_idx]['labeled_at'] = datetime.now().isoformat()
            
            event = self.analyzer.current_events[event_idx]
            
            # 統計已標註和未標註的數量
            labeled_count = sum(1 for e in self.analyzer.current_events if e['label'] is not None)
            total_count = len(self.analyzer.current_events)
            
            # 統計按影片分組的進度
            video_progress = self.get_video_labeling_progress()
            
            # 構建詳細進度信息
            progress_text = f"事件 {event_idx+1} 已標註為：{label}\n"
            progress_text += f"總體進度：{labeled_count}/{total_count} ({labeled_count/total_count*100:.1f}%)\n"
            progress_text += f"來源檔案：{event.get('video_name', 'unknown')}\n"
            progress_text += video_progress
            
            return progress_text
        return "標註失敗"
    
    def get_video_labeling_progress(self):
        """獲取按影片分組的標註進度"""
        video_stats = {}
        
        # 按影片統計事件
        for i, event in enumerate(self.analyzer.current_events):
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
        if not self.analyzer.current_events:
            return None, "尚未分析影片"
        
        labeled_events = [e for e in self.analyzer.current_events if e['label'] is not None]
        if not labeled_events:
            return None, "尚未標註任何事件"
        
        # 創建資料集結構
        export_dir = self.analyzer.dataset_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            
            # 保存事件元資料
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
            'unlabeled_count': len(self.analyzer.current_events) - len(labeled_events)
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