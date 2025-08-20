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
    
    def upload_and_extract_dataset(self, zip_files):
        """上傳並解析多個資料集"""
        try:
            result = self.trainer.upload_and_extract_dataset(zip_files)
            return result
        except Exception as e:
            return f"❌ 上傳失敗: {str(e)}"
    
    def get_available_models(self):
        """取得可用的預訓練模型"""
        return self.trainer.get_available_models()
    
    def get_model_info(self, model_name):
        """取得模型資訊"""
        return self.trainer.get_model_info(model_name)
    
    def get_recommended_input_size(self, model_name):
        """取得模型推薦的輸入尺寸"""
        return self.trainer.get_recommended_input_size(model_name)
    
    def update_model_selection(self, model_name):
        """當模型選擇改變時更新資訊和建議尺寸"""
        return self.trainer.update_model_selection(model_name)
    
    def start_training(self, dataset_path, model_name, epochs, batch_size, image_size):
        """開始訓練（框架功能）"""
        try:
            # 調試日誌 - 檢查接收到的參數
            print(f"🔍 [DEBUG] 接收到的參數:")
            print(f"   dataset_path 類型: {type(dataset_path)}")
            print(f"   dataset_path 長度: {len(str(dataset_path)) if dataset_path else 0}")
            print(f"   dataset_path 內容: {repr(dataset_path)}")
            print(f"   model_name: {model_name}")
            print(f"   epochs: {epochs}, batch_size: {batch_size}, image_size: {image_size}")
            
            # 檢查資料集路徑
            has_temporal_path = "時序分類資料集路徑:" in str(dataset_path) if dataset_path else False
            has_old_path = "資料集路徑:" in str(dataset_path) if dataset_path else False
            
            print(f"🔍 [DEBUG] 路徑檢查結果:")
            print(f"   has_temporal_path: {has_temporal_path}")
            print(f"   has_old_path: {has_old_path}")
            print(f"   dataset_path 為空: {not dataset_path}")
            
            if not dataset_path or (not has_temporal_path and not has_old_path):
                print("❌ [DEBUG] 路徑檢查失敗，返回錯誤")
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
            model_type = model.get("type", "unknown")
            import datetime
            created_time = datetime.datetime.fromtimestamp(model["created"]).strftime("%Y-%m-%d %H:%M")
            
            # 根據模型類型添加圖標
            icon = "🔥" if model_type == "temporal" else "🎯" if model_type == "yolo" else "📦"
            
            model_lines.append(f"{i+1}. {icon} {name}")
            model_lines.append(f"   📁 路徑: {path}")
            model_lines.append(f"   ⏰ 建立時間: {created_time}")
            model_lines.append("")
        
        if len(models) > 10:
            model_lines.append(f"... 還有 {len(models) - 10} 個模型")
        
        return "\n".join(model_lines)
    
    def delete_model(self, model_index):
        """刪除選定的模型"""
        try:
            if not model_index:
                return "❌ 請輸入要刪除的模型編號", self.refresh_models_list()
            
            try:
                index = int(model_index) - 1  # 轉換為 0 基索引
            except ValueError:
                return "❌ 請輸入有效的數字編號", self.refresh_models_list()
            
            models = self.trainer.list_trained_models()
            if not models:
                return "❌ 沒有可刪除的模型", self.refresh_models_list()
            
            if index < 0 or index >= len(models):
                return f"❌ 模型編號無效（1-{len(models)}）", self.refresh_models_list()
            
            # 取得要刪除的模型
            model_to_delete = models[index]
            model_name = model_to_delete["name"]
            model_path = model_to_delete["path"]
            
            # 執行刪除
            result = self.trainer.delete_model(model_path)
            
            # 刷新模型列表
            updated_list = self.refresh_models_list()
            
            return f"{result}\n🗑️ 已刪除: {model_name}", updated_list
            
        except Exception as e:
            return f"❌ 刪除失敗: {str(e)}", self.refresh_models_list()