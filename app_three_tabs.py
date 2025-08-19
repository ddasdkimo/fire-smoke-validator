#!/usr/bin/env python3
"""
火煙偵測系統 - 三頁籤版本
包含標註、訓練、推論三大功能
"""

from core.analyzer import VideoAnalyzer
from core.labeling import LabelingManager
from core.training import ModelTrainer
from core.inference import ModelInference
from ui.interface import InterfaceController
from ui.training_controller import TrainingController
from ui.inference_controller import InferenceController
from ui.builder_new import create_interface


def create_app():
    """建立三頁籤應用程式"""
    print("🔥 初始化火煙偵測系統...")
    
    # 初始化核心組件
    analyzer = VideoAnalyzer()
    labeling_manager = LabelingManager(analyzer)
    trainer = ModelTrainer()
    inferencer = ModelInference()
    
    # 初始化控制器
    interface_controller = InterfaceController(analyzer, labeling_manager)
    training_controller = TrainingController(trainer)
    inference_controller = InferenceController(inferencer)
    
    # 建立三頁籤介面
    app = create_interface(interface_controller, training_controller, inference_controller)
    
    print("✅ 三頁籤系統初始化完成")
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("🔥 火煙偵測系統 v3.0 (三頁籤版)")
    print("=" * 60)
    print("🆕 全新功能:")
    print("  🏷️ 資料標註 - 影片分析與事件標註")
    print("  🎓 模型訓練 - 上傳資料集，訓練自定義模型")
    print("  🔮 模型推論 - 載入模型，批次影像推論")
    print("  🚀 完整的端到端機器學習流程")
    print("=" * 60)
    print()
    
    try:
        app = create_app()
        
        # 啟動應用
        print("🚀 啟動三頁籤 Gradio 服務...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ 啟動失敗: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n👋 使用者中斷，正在關閉系統...")
    finally:
        print("🔚 系統已關閉")