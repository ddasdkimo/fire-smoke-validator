#!/usr/bin/env python3
"""
新的三頁籤 Gradio 介面建構器
包含標註、訓練、推論三大功能
"""

import gradio as gr


def create_labeling_tab(controller):
    """建立標註頁籤"""
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


def create_training_tab(training_controller):
    """建立訓練頁籤"""
    
    gr.Markdown("## 🎓 時序模型訓練")
    gr.Markdown("上傳標註資料集，選擇時序分類模型架構，訓練火煙事件時序分類器（T=5幀輸入）")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 資料集上傳
            gr.Markdown("### 📁 資料集準備")
            gr.Markdown("""
**支援多資料集合併：**
- 可同時上傳多個標註資料集ZIP檔案
- 系統會自動合併所有資料集
- 每個ZIP應包含 `true_positive` 和 `false_positive` 資料夾
- 合併後轉換為統一的YOLO訓練格式
            """)
            dataset_upload = gr.File(
                label="上傳標註資料集（支援多個ZIP檔案）",
                file_count="multiple",
                file_types=[".zip"],
                type="filepath"
            )
            
            upload_btn = gr.Button("📤 上傳並合併多個資料集", variant="secondary")
            upload_result = gr.Textbox(
                label="上傳結果",
                lines=6,
                interactive=False,
                placeholder="等待上傳資料集..."
            )
            
            # 模型設定
            gr.Markdown("### 🤖 時序模型設定")
            
            base_model_dropdown = gr.Dropdown(
                choices=training_controller.get_available_models(),
                value="temporal_convnext_tiny",
                label="🎯 時序分類架構",
                info="選擇timm backbone + 時序融合策略（推薦ConvNeXt-Tiny平衡款）"
            )
            
            model_info_display = gr.Textbox(
                label="模型資訊",
                value=training_controller.get_model_info("temporal_convnext_tiny"),
                lines=3,
                interactive=False
            )
            
            with gr.Row():
                epochs_slider = gr.Slider(
                    minimum=10,
                    maximum=300,
                    value=50,
                    step=10,
                    label="🔄 訓練輪數",
                    info="更多輪數通常會提高精度但需要更多時間"
                )
                batch_size_slider = gr.Slider(
                    minimum=4,
                    maximum=32,
                    value=16,
                    step=4,
                    label="📦 批次大小",
                    info="根據GPU記憶體調整"
                )
            
            image_size_dropdown = gr.Dropdown(
                choices=[224, 240, 288, 320, 384, 416, 512],
                value=224,
                label="📐 影像尺寸",
                info="建議使用模型推薦尺寸，或根據GPU記憶體調整"
            )
            
            # 訓練控制
            with gr.Row():
                start_training_btn = gr.Button("🚀 開始訓練", variant="primary", scale=2)
                stop_training_btn = gr.Button("⏹️ 停止訓練", variant="secondary", scale=1)
        
        with gr.Column(scale=2):
            # 訓練進度和結果
            gr.Markdown("### 📊 訓練進度")
            
            training_progress = gr.Textbox(
                label="即時訓練進度",
                lines=10,
                interactive=False,
                placeholder="等待開始訓練...",
                max_lines=20
            )
            
            # 訓練結果展示
            gr.Markdown("### 📈 訓練結果")
            
            training_results = gr.Textbox(
                label="訓練結果摘要",
                lines=6,
                interactive=False,
                placeholder="訓練完成後顯示結果..."
            )
            
            # 已訓練模型列表
            gr.Markdown("### 🗂️ 已訓練模型")
            
            trained_models_display = gr.Textbox(
                label="可用模型",
                lines=8,
                interactive=False,
                placeholder="尚無已訓練模型..."
            )
            
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 重新整理模型列表", variant="secondary")
                
                # 刪除模型功能
                with gr.Column(scale=2):
                    model_delete_index = gr.Number(
                        label="模型編號",
                        placeholder="輸入要刪除的模型編號",
                        minimum=1,
                        precision=0
                    )
                with gr.Column(scale=1):
                    delete_model_btn = gr.Button("🗑️ 刪除模型", variant="stop")
            
            delete_result = gr.Textbox(
                label="刪除結果",
                lines=2,
                interactive=False,
                visible=False
            )
    
    # 訓練進度計時器
    training_timer = gr.Timer(value=2.0, active=False)
    
    # === 事件綁定 ===
    
    # 上傳資料集
    upload_btn.click(
        training_controller.upload_and_extract_dataset,
        inputs=[dataset_upload],
        outputs=[upload_result]
    )
    
    # 模型選擇變化
    base_model_dropdown.change(
        training_controller.update_model_selection,
        inputs=[base_model_dropdown],
        outputs=[model_info_display, image_size_dropdown]
    )
    
    # 開始訓練（框架功能）
    start_training_btn.click(
        lambda dataset, model, epochs, batch, size: training_controller.start_training(
            dataset, model, epochs, batch, size
        ),
        inputs=[upload_result, base_model_dropdown, epochs_slider, batch_size_slider, image_size_dropdown],
        outputs=[training_results, training_timer]
    )
    
    # 停止訓練
    stop_training_btn.click(
        training_controller.stop_training,
        outputs=[training_results, training_timer]
    )
    
    # 訓練進度更新
    training_timer.tick(
        training_controller.get_training_progress,
        outputs=[training_progress]
    )
    
    # 重新整理模型列表
    refresh_models_btn.click(
        training_controller.refresh_models_list,
        outputs=[trained_models_display]
    )
    
    # 刪除模型
    delete_model_btn.click(
        training_controller.delete_model,
        inputs=[model_delete_index],
        outputs=[delete_result, trained_models_display]
    ).then(
        lambda: gr.update(visible=True),
        outputs=[delete_result]
    )


def create_inference_tab(inference_controller):
    """建立推論頁籤"""
    
    gr.Markdown("## 🔮 時序模型推論")  
    gr.Markdown("載入訓練好的時序分類模型，對影像序列進行火煙事件分類推論（T=5幀輸入）")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 模型載入
            gr.Markdown("### 🤖 模型載入")
            
            # 初始載入模型列表
            available_models = inference_controller.get_available_models()
            model_choices = [f"{m['name']} ({m['type']})" for m in available_models]
            model_paths = {f"{m['name']} ({m['type']})": m['path'] for m in available_models}
            
            inference_model_dropdown = gr.Dropdown(
                choices=model_choices,
                value=model_choices[0] if model_choices else None,
                label="🎯 選擇時序模型",
                info="選擇要用於時序分類的訓練好模型"
            )
            
            # 儲存模型路徑映射
            model_paths_state = gr.State(model_paths)
            
            device_dropdown = gr.Dropdown(
                choices=['auto', 'cuda', 'cpu'],
                value='auto',
                label="⚡ 運算設備",
                info="選擇推論使用的設備"
            )
            
            with gr.Row():
                load_inference_model_btn = gr.Button("📥 載入模型", variant="primary")
                refresh_inference_models_btn = gr.Button("🔄 重新整理列表", variant="secondary")
            
            model_info = gr.Textbox(
                label="📊 模型資訊",
                lines=8,
                interactive=False,
                placeholder="請選擇並載入模型..."
            )
            
            # 推論設定
            gr.Markdown("### ⚙️ 時序推論設定")
            
            inference_confidence = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="🎯 分類信心度閾值",
                info="顯示分類信心度參考（時序模型輸出二元分類結果）"
            )
            
            # 影像上傳
            gr.Markdown("### 📷 時序影像上傳")
            gr.Markdown("""
**上傳同一事件的多幀影像：**
- 建議上傳同一火煙事件的連續幀
- 系統會自動處理為T=5固定輸入
- 支援格式：JPG, PNG, BMP, TIFF
            """)
            
            inference_images = gr.File(
                label="上傳時序影像檔案（同一事件的多幀）",
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            )
            
            inference_btn = gr.Button("🚀 開始推論", variant="primary")
            clear_results_btn = gr.Button("🗑️ 清除結果", variant="secondary")
            
        with gr.Column(scale=2):
            # 推論結果顯示
            gr.Markdown("### 📊 推論結果")
            
            inference_summary = gr.Textbox(
                label="推論結果摘要",
                lines=12,
                interactive=False,
                placeholder="等待開始推論..."
            )
            
            # 結果影像畫廊
            gr.Markdown("### 🖼️ 偵測結果展示")
            
            results_gallery = gr.Gallery(
                label="偵測結果（含標註）",
                show_label=True,
                elem_id="inference_gallery",
                columns=3,
                rows=2,
                height="auto"
            )
            
            # 詳細結果
            detailed_results = gr.JSON(
                label="📋 詳細偵測資料",
                visible=False  # 預設隱藏，可選顯示
            )
    
    # === 事件綁定 ===
    
    # 重新整理模型列表
    def refresh_model_list():
        """重新整理推論模型列表"""
        available_models = inference_controller.get_available_models()
        model_choices = [f"{m['name']} ({m['type']})" for m in available_models]
        model_paths = {f"{m['name']} ({m['type']})": m['path'] for m in available_models}
        
        # 返回更新的下拉選單和路徑映射
        return (
            gr.update(choices=model_choices, value=model_choices[0] if model_choices else None),
            model_paths
        )
    
    refresh_inference_models_btn.click(
        refresh_model_list,
        outputs=[inference_model_dropdown, model_paths_state]
    )
    
    # 載入推論模型
    def load_selected_model(model_choice, device, model_paths):
        """載入選擇的模型"""
        if not model_choice or not model_paths:
            return "❌ 請先選擇模型"
        
        model_path = model_paths.get(model_choice)
        if not model_path:
            return "❌ 找不到模型路徑"
        
        return inference_controller.load_model(model_path, device)
    
    load_inference_model_btn.click(
        load_selected_model,
        inputs=[inference_model_dropdown, device_dropdown, model_paths_state],
        outputs=[model_info]
    )
    
    # 開始推論
    def run_inference_and_update(images, conf):
        """執行推論並更新所有結果"""
        # 執行推論
        summary, results = inference_controller.inference_batch_images(images, conf)
        
        # 獲取畫廊圖片
        gallery_images = inference_controller.get_detection_gallery()
        
        return summary, results, gallery_images
    
    inference_btn.click(
        run_inference_and_update,
        inputs=[inference_images, inference_confidence],
        outputs=[inference_summary, detailed_results, results_gallery]
    )
    
    # 清除結果
    clear_results_btn.click(
        inference_controller.clear_inference_results,
        outputs=[inference_summary, results_gallery, detailed_results]
    )


def create_interface(controller, training_controller, inference_controller):
    """創建三頁籤 Gradio 介面"""
    
    with gr.Blocks(title="火煙偵測系統", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🔥 火煙偵測系統")
        gr.Markdown("全功能火煙偵測平台：資料標註 → 模型訓練 → 推論應用")
        
        with gr.Tabs():
            # 標註頁籤
            with gr.TabItem("🏷️ 資料標註", elem_id="labeling_tab"):
                gr.Markdown("### 影片分析與事件標註")
                create_labeling_tab(controller)
            
            # 訓練頁籤  
            with gr.TabItem("🎓 模型訓練", elem_id="training_tab"):
                create_training_tab(training_controller)
            
            # 推論頁籤
            with gr.TabItem("🔮 模型推論", elem_id="inference_tab"):
                create_inference_tab(inference_controller)
    
    return app