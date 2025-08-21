#!/usr/bin/env python3
"""
TIMM 模型到 TensorRT 轉換 Gradio 介面
支援 .pth 模型轉換為 .engine/.plan 格式
"""

import gradio as gr
import os
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import zipfile

from utils.model_converter import TimmToTensorRTConverter, get_supported_timm_models

class TensorRTConverterApp:
    """TensorRT 轉換應用程式"""
    
    def __init__(self):
        self.converter = TimmToTensorRTConverter(verbose=True)
        self.models_dir = Path("/app/models")
        self.converted_dir = Path("/app/converted_models")
        self.temp_dir = Path("/app/temp")
        
        # 確保目錄存在
        for dir_path in [self.models_dir, self.converted_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def convert_model(self,
                     model_file: gr.File,
                     model_name: str,
                     num_classes: int,
                     max_batch_size: int,
                     enable_fp16: bool,
                     input_height: int,
                     input_width: int) -> Tuple[str, str, Optional[str]]:
        """模型轉換主函數"""
        
        if model_file is None:
            return "❌ 請上傳 .pth 模型檔案", "", None
        
        try:
            # 生成時間戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 複製上傳的檔案到臨時目錄
            temp_model_path = self.temp_dir / f"temp_model_{timestamp}.pth"
            shutil.copy2(model_file.name, temp_model_path)
            
            # 設定輸出路徑
            model_basename = Path(model_file.name).stem
            output_name = f"{model_basename}_{model_name}_{timestamp}"
            output_path = self.converted_dir / f"{output_name}.engine"
            
            # 執行轉換
            status_msg = f"🔄 開始轉換模型...\n"
            status_msg += f"📁 模型名稱: {model_name}\n"
            status_msg += f"🎯 類別數量: {num_classes}\n"
            status_msg += f"📊 批次大小: {max_batch_size}\n"
            status_msg += f"⚡ FP16 加速: {'啟用' if enable_fp16 else '停用'}\n"
            status_msg += f"🖼️ 輸入尺寸: {input_height}x{input_width}\n\n"
            
            input_shape = (max_batch_size, 3, input_height, input_width)
            
            results = self.converter.convert_pth_to_tensorrt(
                pth_path=str(temp_model_path),
                output_path=str(output_path),
                model_name=model_name,
                num_classes=num_classes,
                input_shape=input_shape,
                max_batch_size=max_batch_size,
                fp16_mode=enable_fp16
            )
            
            # 清理臨時檔案
            if temp_model_path.exists():
                temp_model_path.unlink()
            
            if results['success']:
                # 獲取引擎資訊
                engine_info = self.converter.get_model_info(results['engine_path'])
                
                success_msg = f"🎉 模型轉換成功！\n\n"
                success_msg += f"📁 輸出檔案: {Path(results['engine_path']).name}\n"
                success_msg += f"💾 檔案大小: {self._get_file_size(results['engine_path'])}\n\n"
                
                # 模型資訊
                if 'model_info' in results:
                    info = results['model_info']
                    success_msg += f"🤖 模型資訊:\n"
                    success_msg += f"- 模型架構: {info.get('model_name', 'unknown')}\n"
                    success_msg += f"- 總參數量: {info.get('total_parameters', 0):,}\n"
                    success_msg += f"- 可訓練參數: {info.get('trainable_parameters', 0):,}\n"
                    success_msg += f"- 類別數量: {info.get('num_classes', num_classes)}\n\n"
                
                # TensorRT 引擎資訊
                if 'error' not in engine_info:
                    success_msg += f"⚡ TensorRT 引擎資訊:\n"
                    success_msg += f"- 最大批次大小: {engine_info.get('max_batch_size', 'N/A')}\n"
                    success_msg += f"- 設備記憶體: {engine_info.get('device_memory_size', 0) / (1024**2):.1f} MB\n"
                    success_msg += f"- 工作空間大小: {engine_info.get('workspace_size', 0) / (1024**2):.1f} MB\n"
                    success_msg += f"- 繫結數量: {engine_info.get('num_bindings', 0)}\n"
                
                return success_msg, self._create_download_info(results['engine_path']), results['engine_path']
            else:
                error_msg = f"❌ 轉換失敗:\n{results.get('error', '未知錯誤')}"
                return error_msg, "", None
                
        except Exception as e:
            error_msg = f"❌ 轉換過程發生錯誤: {str(e)}"
            return error_msg, "", None
    
    def _get_file_size(self, file_path: str) -> str:
        """取得檔案大小的可讀格式"""
        try:
            size_bytes = os.path.getsize(file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024**2:
                return f"{size_bytes/1024:.1f} KB"
            elif size_bytes < 1024**3:
                return f"{size_bytes/(1024**2):.1f} MB"
            else:
                return f"{size_bytes/(1024**3):.1f} GB"
        except:
            return "Unknown"
    
    def _create_download_info(self, engine_path: str) -> str:
        """建立下載資訊"""
        file_name = Path(engine_path).name
        file_size = self._get_file_size(engine_path)
        
        info = f"""📦 下載資訊:
        
檔案名稱: {file_name}
檔案大小: {file_size}
格式: TensorRT Engine (.engine)

✅ 檔案已準備就緒，可以下載使用！

💡 使用說明:
1. 下載 .engine 檔案
2. 在您的 TensorRT 應用程式中載入此引擎
3. 確保推論時使用相同的輸入尺寸和批次大小
"""
        return info
    
    def get_converted_models(self) -> List[Tuple[str, str]]:
        """取得已轉換的模型列表"""
        models = []
        try:
            for file_path in self.converted_dir.glob("*.engine"):
                file_size = self._get_file_size(str(file_path))
                models.append((file_path.name, f"{file_path.name} ({file_size})"))
        except Exception as e:
            print(f"獲取模型列表失敗: {e}")
        
        return models
    
    def delete_model(self, model_name: str) -> str:
        """刪除已轉換的模型"""
        if not model_name:
            return "❌ 請選擇要刪除的模型"
        
        try:
            model_path = self.converted_dir / model_name
            if model_path.exists():
                model_path.unlink()
                return f"✅ 已刪除模型: {model_name}"
            else:
                return f"❌ 模型不存在: {model_name}"
        except Exception as e:
            return f"❌ 刪除失敗: {str(e)}"
    
    def create_interface(self) -> gr.Interface:
        """建立 Gradio 介面"""
        
        # 取得支援的模型列表
        supported_models = get_supported_timm_models()
        
        with gr.Blocks(
            title="TIMM to TensorRT Converter",
            theme=gr.themes.Soft(),
            css="""
            .main-header { text-align: center; margin-bottom: 2rem; }
            .conversion-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                 padding: 2rem; border-radius: 1rem; margin: 1rem 0; }
            .status-box { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; 
                         border-left: 4px solid #28a745; margin: 1rem 0; }
            """
        ) as interface:
            
            # 標題
            gr.HTML("""
            <div class="main-header">
                <h1>🚀 TIMM to TensorRT Converter</h1>
                <p>將 TIMM 的 .pth 模型轉換為 TensorRT .engine/.plan 格式，提供高效能推論加速</p>
            </div>
            """)
            
            with gr.Tab("🔄 模型轉換", elem_id="conversion-tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📥 輸入設定</h3>")
                        
                        model_file = gr.File(
                            label="上傳 .pth 模型檔案",
                            file_types=[".pth", ".pt"],
                            type="filepath"
                        )
                        
                        model_name = gr.Dropdown(
                            label="選擇模型架構",
                            choices=supported_models,
                            value="efficientnet_b0",
                            allow_custom_value=True,
                            info="選擇對應的 TIMM 模型架構"
                        )
                        
                        with gr.Row():
                            num_classes = gr.Number(
                                label="類別數量",
                                value=2,
                                minimum=1,
                                maximum=10000,
                                step=1
                            )
                            
                            max_batch_size = gr.Number(
                                label="最大批次大小",
                                value=1,
                                minimum=1,
                                maximum=64,
                                step=1
                            )
                        
                        gr.HTML("<h4>🖼️ 輸入尺寸設定</h4>")
                        with gr.Row():
                            input_height = gr.Number(
                                label="輸入高度",
                                value=224,
                                minimum=32,
                                maximum=1024,
                                step=32
                            )
                            
                            input_width = gr.Number(
                                label="輸入寬度",
                                value=224,
                                minimum=32,
                                maximum=1024,
                                step=32
                            )
                        
                        enable_fp16 = gr.Checkbox(
                            label="啟用 FP16 精度（加速推論）",
                            value=True,
                            info="在支援的 GPU 上可提供 2x 加速"
                        )
                        
                        convert_btn = gr.Button(
                            "🚀 開始轉換",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("<h3>📊 轉換結果</h3>")
                        
                        status_output = gr.Textbox(
                            label="轉換狀態",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        download_info = gr.Textbox(
                            label="下載資訊",
                            lines=8,
                            interactive=False
                        )
                        
                        download_file = gr.File(
                            label="下載轉換後的 .engine 檔案",
                            interactive=False
                        )
            
            with gr.Tab("📁 管理模型", elem_id="management-tab"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.HTML("<h3>📂 已轉換的模型</h3>")
                        
                        model_list = gr.Dropdown(
                            label="選擇模型",
                            choices=[],
                            interactive=True
                        )
                        
                        refresh_btn = gr.Button("🔄 重新整理列表")
                        delete_btn = gr.Button("🗑️ 刪除選中模型", variant="stop")
                        
                        delete_result = gr.Textbox(
                            label="操作結果",
                            lines=3,
                            interactive=False
                        )
                    
                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div class="status-box">
                            <h4>💡 使用說明</h4>
                            <ul>
                                <li>支援所有 TIMM 預訓練模型</li>
                                <li>自動進行 FP16 最佳化</li>
                                <li>支援動態批次大小</li>
                                <li>輸出 TensorRT .engine 格式</li>
                                <li>可直接用於 Triton 推論伺服器</li>
                            </ul>
                        </div>
                        """)
            
            with gr.Tab("ℹ️ 系統資訊", elem_id="info-tab"):
                gr.HTML("""
                <div class="status-box">
                    <h3>🔧 支援的功能</h3>
                    <ul>
                        <li><strong>模型格式</strong>: TIMM .pth/.pt 模型</li>
                        <li><strong>輸出格式</strong>: TensorRT .engine/.plan</li>
                        <li><strong>最佳化</strong>: FP16 精度、動態形狀、記憶體最佳化</li>
                        <li><strong>支援架構</strong>: ResNet, EfficientNet, ConvNeXt, Swin, ViT, 等</li>
                    </ul>
                    
                    <h3>⚡ 效能優勢</h3>
                    <ul>
                        <li>比原始 PyTorch 模型快 2-10 倍</li>
                        <li>支援 FP16 混合精度加速</li>
                        <li>記憶體使用量最佳化</li>
                        <li>支援批次處理最佳化</li>
                    </ul>
                </div>
                """)
            
            # 事件綁定
            convert_btn.click(
                fn=self.convert_model,
                inputs=[
                    model_file, model_name, num_classes, max_batch_size,
                    enable_fp16, input_height, input_width
                ],
                outputs=[status_output, download_info, download_file]
            )
            
            refresh_btn.click(
                fn=lambda: gr.Dropdown(choices=[name for name, _ in self.get_converted_models()]),
                outputs=model_list
            )
            
            delete_btn.click(
                fn=self.delete_model,
                inputs=model_list,
                outputs=delete_result
            )
            
            # 初始化模型列表
            interface.load(
                fn=lambda: gr.Dropdown(choices=[name for name, _ in self.get_converted_models()]),
                outputs=model_list
            )
        
        return interface


def main():
    """主程序"""
    print("🚀 啟動 TIMM to TensorRT 轉換器...")
    
    app = TensorRTConverterApp()
    interface = app.create_interface()
    
    # 啟動 Gradio 介面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        show_tips=True
    )


if __name__ == "__main__":
    main()