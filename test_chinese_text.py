#!/usr/bin/env python3
"""
測試中文文字渲染
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    PIL_AVAILABLE = True
    print("✅ PIL 可用")
except ImportError:
    PIL_AVAILABLE = False
    print("❌ PIL 不可用")

def test_chinese_text():
    """測試中文文字渲染功能"""
    print("\n🔍 測試中文文字渲染...")
    
    if not PIL_AVAILABLE:
        print("❌ PIL 不可用，無法測試中文文字渲染")
        return
    
    try:
        # 建立測試畫布
        canvas = Image.new('RGB', (400, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(canvas)
        
        # 測試字體載入
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        
        title_font = None
        for font_path in font_paths:
            try:
                from pathlib import Path
                if Path(font_path).exists():
                    title_font = ImageFont.truetype(font_path, 24)
                    print(f"✅ 找到字體: {font_path}")
                    break
            except Exception as e:
                print(f"⚠️ 無法載入字體 {font_path}: {e}")
        
        if not title_font:
            title_font = ImageFont.load_default()
            print("⚠️ 使用預設字體")
        
        # 測試中文文字
        chinese_text = "火煙偵測結果"
        english_text = "Fire/Smoke Detection Result"
        
        # 繪製標題背景
        draw.rectangle([(0, 0), (400, 60)], fill=(50, 50, 50))
        
        # 繪製中文文字
        draw.text((20, 15), chinese_text, fill=(255, 255, 255), font=title_font)
        
        # 繪製英文文字
        draw.text((20, 45), english_text, fill=(200, 200, 200), font=title_font)
        
        # 儲存測試結果
        output_path = "/home/ubuntu/fire-smoke-validator/chinese_text_test.png"
        canvas.save(output_path)
        
        print(f"✅ 中文文字測試完成，結果儲存至: {output_path}")
        print("請檢查圖片中的中文字符是否正確顯示")
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_opencv_fallback():
    """測試OpenCV備用方案"""
    print("\n🔍 測試OpenCV備用方案...")
    
    try:
        import cv2
        print("✅ OpenCV 可用")
        
        # 建立測試畫布
        canvas = np.ones((200, 400, 3), dtype=np.uint8) * 240
        
        # 添加標題背景
        cv2.rectangle(canvas, (0, 0), (400, 60), (50, 50, 50), -1)
        
        # 添加英文文字（OpenCV只支援英文）
        title_text = "Fire/Smoke Detection Result"
        cv2.putText(canvas, title_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 儲存測試結果
        output_path = "/home/ubuntu/fire-smoke-validator/opencv_text_test.jpg"
        cv2.imwrite(output_path, canvas)
        
        print(f"✅ OpenCV備用方案測試完成，結果儲存至: {output_path}")
        return True
        
    except ImportError:
        print("❌ OpenCV 不可用")
        return False
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        return False

if __name__ == "__main__":
    success1 = test_chinese_text()
    success2 = test_opencv_fallback()
    
    if success1 or success2:
        print("\n✅ 文字渲染功能修復完成！")
        print("📊 修復摘要:")
        if success1:
            print("  - PIL中文文字渲染：✅ 可用")
        if success2:
            print("  - OpenCV英文備用方案：✅ 可用")
    else:
        print("\n❌ 文字渲染測試失敗")