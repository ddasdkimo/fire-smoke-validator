#!/usr/bin/env python3
"""
啟動腳本 - 可選擇原版或三頁籤版本
"""

import sys
import os
from pathlib import Path

def show_usage():
    print("""
🔥 火煙偵測系統啟動器

使用方法:
  python start.py [版本]

版本選項:
  original    - 原版單頁面標註系統
  three-tabs  - 新版三頁籤系統 (標註 + 訓練 + 推論)
  
如果不指定版本，將啟動三頁籤版本。

範例:
  python start.py original      # 啟動原版
  python start.py three-tabs    # 啟動三頁籤版本
  python start.py              # 預設啟動三頁籤版本
    """)

def main():
    version = "three-tabs"  # 預設使用三頁籤版本
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ["-h", "--help", "help"]:
            show_usage()
            return
        version = sys.argv[1]
    
    if version == "original":
        print("🔥 啟動原版單頁面標註系統...")
        from app import create_app
        app_file = "app.py"
    elif version == "three-tabs":
        print("🔥 啟動三頁籤系統 (標註 + 訓練 + 推論)...")
        from app_three_tabs import create_app
        app_file = "app_three_tabs.py"
    else:
        print(f"❌ 不支援的版本: {version}")
        print("請使用 'original' 或 'three-tabs'")
        show_usage()
        return
    
    print(f"📁 使用檔案: {app_file}")
    print("=" * 60)
    
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

if __name__ == "__main__":
    main()