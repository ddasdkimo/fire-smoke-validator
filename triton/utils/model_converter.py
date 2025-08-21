#!/usr/bin/env python3
"""
TIMM 模型到 TensorRT 轉換工具
支援將 .pth 模型轉換為 .engine 和 .plan 格式
"""

import os
import torch
import timm
import tensorrt as trt
import numpy as np
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, List, Dict, Any
import tempfile
import onnx

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimmToTensorRTConverter:
    """TIMM 模型到 TensorRT 轉換器"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.trt_logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
        
    def load_timm_model(self, 
                       model_path: str, 
                       model_name: Optional[str] = None,
                       num_classes: int = 2) -> torch.nn.Module:
        """載入 TIMM 模型"""
        try:
            if model_path.endswith('.pth') or model_path.endswith('.pt'):
                # 載入預訓練的模型
                if model_name:
                    # 創建模型架構然後載入權重
                    model = timm.create_model(model_name, num_classes=num_classes)
                    state_dict = torch.load(model_path, map_location='cpu')
                    
                    # 處理可能的狀態字典格式
                    if 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model' in state_dict:
                        state_dict = state_dict['model']
                    
                    model.load_state_dict(state_dict, strict=False)
                else:
                    # 直接載入完整模型
                    model = torch.load(model_path, map_location='cpu')
            else:
                raise ValueError(f"不支援的模型格式: {model_path}")
            
            model.eval()
            logger.info(f"✅ 成功載入模型: {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"❌ 載入模型失敗: {e}")
            raise
    
    def convert_to_onnx(self, 
                       model: torch.nn.Module,
                       input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                       output_path: str = "temp_model.onnx",
                       opset_version: int = 11) -> str:
        """將 PyTorch 模型轉換為 ONNX"""
        try:
            # 創建示例輸入
            dummy_input = torch.randn(*input_shape)
            
            # 導出為 ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # 驗證 ONNX 模型
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"✅ 成功轉換為 ONNX: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"❌ ONNX 轉換失敗: {e}")
            raise
    
    def build_tensorrt_engine(self,
                            onnx_path: str,
                            engine_path: str,
                            max_batch_size: int = 1,
                            fp16_mode: bool = True,
                            max_workspace_size: int = 1 << 30) -> str:  # 1GB
        """將 ONNX 模型轉換為 TensorRT 引擎"""
        try:
            # 創建 TensorRT builder
            builder = trt.Builder(self.trt_logger)
            
            # 創建網路定義
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # 創建 ONNX 解析器
            parser = trt.OnnxParser(network, self.trt_logger)
            
            # 解析 ONNX 模型
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("❌ ONNX 解析失敗")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("ONNX 解析失敗")
            
            # 創建構建配置
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
            
            # 啟用 FP16 精度（如果支援）
            if fp16_mode and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("🚀 啟用 FP16 精度加速")
            
            # 設定動態輸入形狀
            profile = builder.create_optimization_profile()
            input_tensor = network.get_input(0)
            
            # 設定最小、最佳、最大輸入形狀
            min_shape = (1, 3, 224, 224)
            opt_shape = (max_batch_size, 3, 224, 224)
            max_shape = (max_batch_size * 2, 3, 224, 224)
            
            profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)
            
            # 構建引擎
            logger.info("🔄 開始構建 TensorRT 引擎...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                raise RuntimeError("TensorRT 引擎構建失敗")
            
            # 儲存引擎
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            logger.info(f"✅ 成功建立 TensorRT 引擎: {engine_path}")
            return engine_path
            
        except Exception as e:
            logger.error(f"❌ TensorRT 引擎構建失敗: {e}")
            raise
    
    def convert_pth_to_tensorrt(self,
                              pth_path: str,
                              output_path: str,
                              model_name: Optional[str] = None,
                              num_classes: int = 2,
                              input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                              max_batch_size: int = 1,
                              fp16_mode: bool = True) -> Dict[str, Any]:
        """完整的 .pth 到 TensorRT 轉換流程"""
        
        results = {
            'success': False,
            'onnx_path': None,
            'engine_path': None,
            'plan_path': None,
            'error': None,
            'model_info': {}
        }
        
        temp_onnx_path = None
        
        try:
            # 1. 載入 TIMM 模型
            logger.info("📥 載入 TIMM 模型...")
            model = self.load_timm_model(pth_path, model_name, num_classes)
            
            # 獲取模型資訊
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results['model_info'] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_name': model_name or "unknown",
                'num_classes': num_classes
            }
            
            # 2. 轉換為 ONNX
            logger.info("🔄 轉換為 ONNX 格式...")
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp_file:
                temp_onnx_path = tmp_file.name
            
            self.convert_to_onnx(
                model, 
                input_shape=input_shape,
                output_path=temp_onnx_path
            )
            results['onnx_path'] = temp_onnx_path
            
            # 3. 轉換為 TensorRT 引擎
            logger.info("⚡ 轉換為 TensorRT 引擎...")
            
            # 確定輸出路徑
            if output_path.endswith('.engine') or output_path.endswith('.plan'):
                engine_path = output_path
            else:
                engine_path = f"{output_path}.engine"
            
            self.build_tensorrt_engine(
                temp_onnx_path,
                engine_path,
                max_batch_size=max_batch_size,
                fp16_mode=fp16_mode
            )
            
            results['engine_path'] = engine_path
            results['plan_path'] = engine_path  # .engine 和 .plan 本質上相同
            results['success'] = True
            
            logger.info("🎉 模型轉換完成！")
            
        except Exception as e:
            error_msg = f"模型轉換失敗: {str(e)}"
            logger.error(f"❌ {error_msg}")
            results['error'] = error_msg
            
        finally:
            # 清理臨時檔案
            if temp_onnx_path and os.path.exists(temp_onnx_path):
                try:
                    os.unlink(temp_onnx_path)
                except:
                    pass
        
        return results
    
    def get_model_info(self, engine_path: str) -> Dict[str, Any]:
        """獲取 TensorRT 引擎資訊"""
        try:
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            runtime = trt.Runtime(self.trt_logger)
            engine = runtime.deserialize_cuda_engine(engine_data)
            
            info = {
                'num_bindings': engine.num_bindings,
                'max_batch_size': engine.max_batch_size,
                'device_memory_size': engine.device_memory_size,
                'workspace_size': engine.workspace_size,
                'bindings': []
            }
            
            for i in range(engine.num_bindings):
                binding_info = {
                    'name': engine.get_binding_name(i),
                    'shape': engine.get_binding_shape(i),
                    'dtype': str(engine.get_binding_dtype(i)),
                    'is_input': engine.binding_is_input(i)
                }
                info['bindings'].append(binding_info)
            
            return info
            
        except Exception as e:
            logger.error(f"❌ 獲取引擎資訊失敗: {e}")
            return {'error': str(e)}


def get_supported_timm_models() -> List[str]:
    """獲取支援的 TIMM 模型列表"""
    # 常用的 TIMM 模型
    popular_models = [
        'resnet50', 'resnet101', 'resnet152',
        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
        'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
        'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l',
        'mobilenetv3_small_100', 'mobilenetv3_large_100',
        'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
        'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224',
        'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224',
        'vit_base_patch16_224', 'vit_large_patch16_224',
        'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224'
    ]
    
    return popular_models


if __name__ == "__main__":
    # 測試轉換器
    converter = TimmToTensorRTConverter(verbose=True)
    
    # 列出支援的模型
    models = get_supported_timm_models()
    print("🤖 支援的 TIMM 模型:")
    for i, model in enumerate(models[:10]):  # 只顯示前10個
        print(f"  {i+1}. {model}")
    print(f"  ... 總共 {len(models)} 個模型")