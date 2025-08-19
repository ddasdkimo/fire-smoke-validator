#!/usr/bin/env python3
"""
Gradio 介面建構器
負責建立和配置使用者介面
"""

import gradio as gr


def create_interface(controller):
    """創建 Gradio 介面"""
    
    with gr.Blocks(title="火煙誤判標註系統", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔥 火煙誤判標註系統")
        gr.Markdown("上傳多個影片 → 批次分析火煙事件 → 統一標註真實/誤判 → 匯出訓練資料集")
        
        # 狀態變數
        current_event_idx = gr.State(0)
        frame_idx = gr.State(0)
        current_speed = gr.State(1.0)
        
        # 初始載入模型
        initial_status = controller.load_model_to_device(controller.analyzer.available_devices['default'])
        
        with gr.Row():
            with gr.Column(scale=1):
                # 上傳區域
                video_input = gr.File(
                    label="上傳影片檔案（支援多檔案）",
                    file_count="multiple",
                    file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm"]
                )
                
                # 偵測設定
                confidence_slider = gr.Slider(
                    minimum=0.001,
                    maximum=1.0,
                    value=0.300,
                    step=0.001,
                    label="🎯 偵測信心度閾值",
                    info="調整YOLO模型的偵測閾值，數值越低偵測越敏感"
                )
                
                # 事件分組設定
                gr.Markdown("### 📐 事件分組設定")
                with gr.Row():
                    min_frames_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="🔽 最少幀數",
                        info="事件需要的最少幀數，低於此數量將被忽略"
                    )
                    max_frames_slider = gr.Slider(
                        minimum=5,
                        maximum=100,
                        value=5,
                        step=1,
                        label="🔼 最多幀數",
                        info="事件達到此幀數後會自動結束，避免事件過長"
                    )
                
                # 設備選擇
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=controller.analyzer.available_devices['options'],
                        value=controller.analyzer.available_devices['default'],
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
        
        # 計時器
        timer = gr.Timer(value=1.0, active=False)
        progress_timer = gr.Timer(value=1.0, active=False)
        
        # === 事件綁定 ===
        
        # 載入模型按鈕點擊
        load_model_btn.click(
            controller.load_model_to_device,
            inputs=[device_dropdown],
            outputs=[model_status]
        )
        
        # 分析影片按鈕點擊
        analyze_btn.click(
            controller.start_analysis,
            inputs=[video_input, confidence_slider, min_frames_slider, max_frames_slider],
            outputs=[analysis_result, progress_timer]
        )
        
        # 進度計時器更新
        progress_timer.tick(
            controller.update_progress,
            outputs=[progress_display]
        )
        
        # 同時檢查分析完成狀態
        progress_timer.tick(
            controller.check_analysis_complete,
            inputs=[current_speed],
            outputs=[analysis_result, gr.Gallery(visible=False), gr.Textbox(visible=False), progress_timer, current_event_idx, frame_idx, timer]
        )
        
        # 計時器觸發更新
        timer.tick(
            controller.update_event_display,
            inputs=[current_event_idx, frame_idx],
            outputs=[current_event_info, full_frame_display, crop_frame_display, progress_info, frame_info_display, current_event_idx, frame_idx]
        )
        
        # 標註事件
        label_true_btn.click(
            lambda idx: controller.label_and_next(idx, "真實火煙"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx, analysis_result]
        )
        
        label_false_btn.click(
            lambda idx: controller.label_and_next(idx, "誤判"),
            inputs=[current_event_idx],
            outputs=[current_event_idx, frame_idx, analysis_result]
        )
        
        # 導航按鈕
        prev_btn.click(controller.go_prev, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        next_btn.click(controller.go_next, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        skip_btn.click(controller.skip_current, inputs=[current_event_idx], outputs=[current_event_idx, frame_idx])
        
        # 播放速度控制
        playback_speed.change(
            controller.update_playback_speed,
            inputs=[playback_speed],
            outputs=[timer, current_speed]
        )
        
        # 匯出資料集
        export_btn.click(
            controller.export_and_return_file,
            outputs=[export_result, export_file]
        )
    
    return app