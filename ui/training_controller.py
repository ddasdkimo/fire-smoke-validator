#!/usr/bin/env python3
"""
訓練功能控制器
處理模型訓練相關的 UI 邏輯
"""

import gradio as gr
import threading
from pathlib import Path


class TrainingController:
    """訓練控制器"""
    
    def __init__(self, trainer):
        self.trainer = trainer
    
    def upload_and_extract_dataset(self, zip_file):
        """上傳並解析資料集"""
        try:
            result = self.trainer.upload_and_extract_dataset(zip_file)
            return result
        except Exception as e:
            return f"❌ 上傳失敗: {str(e)}"
    
    def get_available_models(self):
        """取得可用的預訓練模型"""
        return self.trainer.get_available_models()
    
    def get_model_info(self, model_name):
        """取得模型資訊"""
        return self.trainer.get_model_info(model_name)
    
    def start_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始訓練（框架功能）"""
        try:
            # 檢查資料集路徑
            if not dataset_path or "資料集路徑:" not in dataset_path:
                return "❌ 請先上傳並解析資料集", gr.Timer(active=False)
            
            # 啟動訓練任務
            result = self.trainer.start_training(dataset_path, model_name, epochs, batch_size, image_size)
            
            # 如果訓練啟動成功，啟動進度計時器
            if result.startswith("✅"):
                return result, gr.Timer(active=True)
            else:
                return result, gr.Timer(active=False)
                
        except Exception as e:
            return f"❌ 訓練啟動失敗: {str(e)}", gr.Timer(active=False)
    
    def stop_training(self):
        """停止訓練"""
        result = self.trainer.stop_training()
        return result, gr.Timer(active=False)
    
    def get_training_progress(self):
        """取得訓練進度"""
        return self.trainer.get_training_progress()
    
    def list_trained_models(self):
        """列出已訓練模型"""
        return self.trainer.list_trained_models()
    
    def refresh_models_list(self):
        """重新整理模型列表"""
        models = self.trainer.list_trained_models()
        if not models:
            return "🗂️ 尚無已訓練模型"
        
        model_lines = ["🗂️ 已訓練模型列表:", ""]
        for i, model in enumerate(models[:10]):  # 只顯示前10個
            name = model["name"]
            path = model["path"]
            import datetime
            created_time = datetime.datetime.fromtimestamp(model["created"]).strftime("%Y-%m-%d %H:%M")
            model_lines.append(f"{i+1}. {name}")
            model_lines.append(f"   路徑: {path}")
            model_lines.append(f"   建立時間: {created_time}")
            model_lines.append("")
        
        if len(models) > 10:
            model_lines.append(f"... 還有 {len(models) - 10} 個模型")
        
        return "\n".join(model_lines)