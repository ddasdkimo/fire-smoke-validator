#!/usr/bin/env python3
"""
推論功能控制器
處理模型推論相關的 UI 邏輯
"""

import gradio as gr
from pathlib import Path


class InferenceController:
    """推論控制器"""
    
    def __init__(self, inferencer):
        self.inferencer = inferencer
    
    def get_available_models(self):
        """取得可用模型列表"""
        return self.inferencer.get_available_models()
    
    def load_model(self, model_path, device='auto'):
        """載入推論模型"""
        try:
            result = self.inferencer.load_model(model_path, device)
            return result
        except Exception as e:
            return f"❌ 載入模型失敗: {str(e)}"
    
    def inference_batch_images(self, image_files, confidence_threshold=0.5):
        """批次推論多張影像"""
        try:
            summary, results = self.inferencer.inference_batch_images(image_files, confidence_threshold)
            return summary, results
        except Exception as e:
            return f"❌ 推論失敗: {str(e)}", []
    
    def get_inference_results_summary(self):
        """取得推論結果摘要"""
        return self.inferencer.get_inference_results_summary()
    
    def get_detection_gallery(self):
        """取得偵測結果影像畫廊"""
        return self.inferencer.get_detection_gallery()
    
    def clear_inference_results(self):
        """清除推論結果"""
        result = self.inferencer.clear_inference_results()
        return result, [], []  # 清除摘要、畫廊和詳細結果
    
    def get_model_info_by_choice(self, model_choice, model_choices, model_paths, device):
        """根據選擇取得模型資訊"""
        if not model_choice or model_choice not in model_choices:
            return "❌ 請選擇有效的模型"
        
        try:
            model_path = model_paths[model_choices.index(model_choice)]
            return self.inferencer.load_model(model_path, device)
        except Exception as e:
            return f"❌ 載入失敗: {str(e)}"