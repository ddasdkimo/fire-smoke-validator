#!/usr/bin/env python3
"""
時序資料處理工具
處理幀序列的採樣、填充和增強
"""

import numpy as np
import cv2
import torch
import random
from typing import List, Tuple, Optional


def prepare_temporal_frames(frames: List[np.ndarray], 
                          target_frames: int = 5,
                          image_size: Tuple[int, int] = (224, 224),
                          training: bool = True) -> torch.Tensor:
    """
    準備時序幀序列，固定長度為 T=5
    
    Args:
        frames: 輸入幀列表 [H, W, C]
        target_frames: 目標幀數 (固定為5)
        image_size: 目標影像尺寸 (H, W)
        training: 是否為訓練模式
    
    Returns:
        torch.Tensor: [T, C, H, W] 格式的張量
    """
    if not frames:
        raise ValueError("幀列表不能為空")
    
    # 調整影像尺寸
    resized_frames = []
    for frame in frames:
        if frame.shape[:2] != image_size:
            frame = cv2.resize(frame, (image_size[1], image_size[0]))
        # 確保是 RGB 格式
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frames.append(frame)
    
    # 處理幀數不足或過多的情況
    if len(resized_frames) < target_frames:
        # 不足5幀：重複最後一張 (replicate padding)
        processed_frames = _replicate_padding(resized_frames, target_frames, training)
    elif len(resized_frames) > target_frames:
        # 超過5幀：等距取樣 (uniform sampling)
        processed_frames = _uniform_sampling(resized_frames, target_frames)
    else:
        # 恰好5幀
        processed_frames = resized_frames
    
    # 轉換為張量 [T, H, W, C] -> [T, C, H, W]
    frames_tensor = torch.stack([
        torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        for frame in processed_frames
    ])
    
    return frames_tensor


def _replicate_padding(frames: List[np.ndarray], 
                      target_frames: int, 
                      training: bool = True) -> List[np.ndarray]:
    """
    重複最後一張幀進行填充
    訓練時加入微小的隨機亮度噪音避免模型記住重複樣式
    """
    if not frames:
        raise ValueError("幀列表不能為空")
    
    processed_frames = frames.copy()
    last_frame = frames[-1].copy()
    
    # 需要填充的幀數
    padding_needed = target_frames - len(frames)
    
    for i in range(padding_needed):
        if training:
            # 訓練時加入微小隨機亮度噪音
            padded_frame = _add_brightness_noise(last_frame.copy())
        else:
            # 推論時直接複製
            padded_frame = last_frame.copy()
        
        processed_frames.append(padded_frame)
    
    return processed_frames


def _uniform_sampling(frames: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
    """
    等距取樣幀序列
    """
    if len(frames) <= target_frames:
        return frames
    
    # 計算取樣索引
    indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
    
    # 確保索引不重複且有序
    indices = sorted(list(set(indices)))
    
    # 如果去重後數量不足，補充缺失的幀
    while len(indices) < target_frames:
        # 找到間隔最大的位置插入新索引
        max_gap = 0
        insert_pos = 0
        for i in range(len(indices) - 1):
            gap = indices[i + 1] - indices[i]
            if gap > max_gap:
                max_gap = gap
                insert_pos = i
        
        # 在最大間隔中間插入新索引
        new_idx = (indices[insert_pos] + indices[insert_pos + 1]) // 2
        if new_idx not in indices:
            indices.insert(insert_pos + 1, new_idx)
        else:
            # 如果中間位置已存在，嘗試其他位置
            for offset in [1, -1, 2, -2]:
                new_idx = (indices[insert_pos] + indices[insert_pos + 1]) // 2 + offset
                if 0 <= new_idx < len(frames) and new_idx not in indices:
                    indices.insert(insert_pos + 1, new_idx)
                    break
            else:
                # 如果都不行，直接複製最後一幀
                indices.append(len(frames) - 1)
        
        indices.sort()
    
    # 取前 target_frames 個
    indices = indices[:target_frames]
    
    return [frames[i] for i in indices]


def _add_brightness_noise(frame: np.ndarray, noise_range: float = 0.02) -> np.ndarray:
    """
    為幀添加微小的隨機亮度噪音
    
    Args:
        frame: 輸入幀 [H, W, C]
        noise_range: 噪音範圍 (0.0-1.0)
    
    Returns:
        帶噪音的幀
    """
    # 生成隨機亮度調整值
    brightness_factor = 1.0 + random.uniform(-noise_range, noise_range)
    
    # 應用亮度調整
    noisy_frame = frame.astype(np.float32) * brightness_factor
    
    # 裁剪到有效範圍
    noisy_frame = np.clip(noisy_frame, 0, 255).astype(np.uint8)
    
    return noisy_frame


def augment_temporal_sequence(frames: List[np.ndarray], 
                            training: bool = True) -> List[np.ndarray]:
    """
    時序序列增強
    """
    if not training:
        return frames
    
    augmented_frames = []
    
    # 隨機選擇增強策略
    aug_prob = 0.3
    
    for frame in frames:
        aug_frame = frame.copy()
        
        # 隨機水平翻轉
        if random.random() < aug_prob:
            aug_frame = cv2.flip(aug_frame, 1)
        
        # 隨機亮度調整
        if random.random() < aug_prob:
            brightness_factor = random.uniform(0.8, 1.2)
            aug_frame = np.clip(aug_frame.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
        
        # 隨機對比度調整
        if random.random() < aug_prob:
            contrast_factor = random.uniform(0.8, 1.2)
            aug_frame = np.clip((aug_frame.astype(np.float32) - 128) * contrast_factor + 128, 0, 255).astype(np.uint8)
        
        # 隨機色相調整（輕微）
        if random.random() < aug_prob / 2:  # 降低機率
            hsv = cv2.cvtColor(aug_frame, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] += random.uniform(-10, 10)  # 色相調整
            hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
            aug_frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        augmented_frames.append(aug_frame)
    
    return augmented_frames


def create_temporal_batch(frame_sequences: List[List[np.ndarray]], 
                         target_frames: int = 5,
                         image_size: Tuple[int, int] = (224, 224),
                         training: bool = True) -> torch.Tensor:
    """
    創建時序批次
    
    Args:
        frame_sequences: 多個幀序列列表
        target_frames: 目標幀數
        image_size: 影像尺寸
        training: 訓練模式
    
    Returns:
        torch.Tensor: [B, T, C, H, W] 格式的批次張量
    """
    batch_tensors = []
    
    for frames in frame_sequences:
        if training:
            frames = augment_temporal_sequence(frames, training=True)
        
        frame_tensor = prepare_temporal_frames(
            frames, target_frames, image_size, training
        )
        batch_tensors.append(frame_tensor)
    
    return torch.stack(batch_tensors)


def validate_frame_sequence(frames: List[np.ndarray]) -> bool:
    """
    驗證幀序列的有效性
    """
    if not frames:
        return False
    
    # 檢查所有幀的形狀是否一致
    first_shape = frames[0].shape
    for frame in frames[1:]:
        if frame.shape != first_shape:
            return False
    
    # 檢查是否為有效的影像格式
    if len(first_shape) != 3 or first_shape[2] not in [1, 3, 4]:
        return False
    
    return True