#!/usr/bin/env python3
"""
火煙誤判標註系統 - 重構版
簡潔的主程式，使用模組化架構
"""

from core.analyzer import VideoAnalyzer
from core.labeling import LabelingManager
from ui.interface import InterfaceController
from ui.builder import create_interface


def create_app():
    """建立應用程式"""
    print("🔥 初始化火煙誤判標註系統...")
    
    # 初始化核心組件
    analyzer = VideoAnalyzer()
    labeling_manager = LabelingManager(analyzer)
    controller = InterfaceController(analyzer, labeling_manager)
    
    # 建立介面
    app = create_interface(controller)
    
    print("✅ 系統初始化完成")
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("🔥 火煙誤判標註系統 v2.0 (重構版)")
    print("=" * 60)
    print("✨ 新功能:")
    print("  🎯 彈出視窗提醒標註進度")
    print("  📐 可調整事件幀數範圍")  
    print("  🚀 模組化架構，更易維護")
    print("  💡 智能事件分組")
    print("=" * 60)
    print()
    
    try:
        app = create_app()
        
        # 啟動應用
        print("🚀 啟動 Gradio 服務...")
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