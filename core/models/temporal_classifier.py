#!/usr/bin/env python3
"""
時序火煙分類模型
使用 timm backbone 進行特徵提取，支援 T=5 幀輸入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class TemporalFireSmokeClassifier(nn.Module):
    """
    時序火煙分類器
    支援 ResNet / ConvNeXt / EfficientNet backbone
    """
    
    SUPPORTED_BACKBONES = {
        'resnet': [
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
            'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        ],
        'convnext': [
            'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large',
            'convnextv2_tiny', 'convnextv2_small', 'convnextv2_base'
        ],
        'efficientnet': [
            'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
            'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
            'efficientnetv2_s', 'efficientnetv2_m', 'efficientnetv2_l'
        ]
    }
    
    def __init__(self,
                 backbone_name: str = 'resnet50',
                 num_classes: int = 2,
                 temporal_frames: int = 5,
                 pretrained: bool = True,
                 temporal_fusion: str = 'attention',
                 dropout: float = 0.2,
                 freeze_backbone: bool = False):
        """
        初始化時序分類器
        
        Args:
            backbone_name: timm 模型名稱
            num_classes: 分類數量 (2: fire_smoke, no_fire_smoke)
            temporal_frames: 時序幀數 (固定為5)
            pretrained: 是否使用預訓練權重
            temporal_fusion: 時序融合策略 ('attention', 'lstm', 'avg', 'max')
            dropout: dropout 比率
            freeze_backbone: 是否凍結 backbone 參數
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("需要安裝 timm: pip install timm")
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.temporal_frames = temporal_frames
        self.temporal_fusion = temporal_fusion
        self.dropout = dropout
        
        # 驗證 backbone
        self._validate_backbone(backbone_name)
        
        # 建立 backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分類頭
            global_pool=''  # 移除全局池化
        )
        
        # 獲取特徵維度
        self.feature_dim = self.backbone.num_features
        
        # 凍結 backbone 參數（如果需要）
        if freeze_backbone:
            self._freeze_backbone()
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 時序融合層
        self.temporal_fusion_layer = self._build_temporal_fusion()
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._get_fusion_output_dim(), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def _validate_backbone(self, backbone_name: str):
        """驗證 backbone 是否支援"""
        supported = False
        for category, models in self.SUPPORTED_BACKBONES.items():
            if backbone_name in models:
                supported = True
                break
        
        if not supported:
            available_models = []
            for models in self.SUPPORTED_BACKBONES.values():
                available_models.extend(models)
            raise ValueError(f"不支援的 backbone: {backbone_name}. 支援的模型: {available_models}")
    
    def _freeze_backbone(self):
        """凍結 backbone 參數"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"✅ 已凍結 {self.backbone_name} backbone 參數")
    
    def _build_temporal_fusion(self) -> nn.Module:
        """建立時序融合層"""
        if self.temporal_fusion == 'attention':
            return TemporalAttention(self.feature_dim, self.temporal_frames)
        elif self.temporal_fusion == 'lstm':
            return TemporalLSTM(self.feature_dim, self.feature_dim)
        elif self.temporal_fusion == 'avg':
            return TemporalAverage()
        elif self.temporal_fusion == 'max':
            return TemporalMax()
        else:
            raise ValueError(f"不支援的融合策略: {self.temporal_fusion}")
    
    def _get_fusion_output_dim(self) -> int:
        """獲取融合層輸出維度"""
        if self.temporal_fusion in ['attention', 'lstm']:
            return self.feature_dim
        elif self.temporal_fusion in ['avg', 'max']:
            return self.feature_dim
        else:
            return self.feature_dim
    
    def _initialize_weights(self):
        """初始化權重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: 輸入張量 [B, T, C, H, W]
        
        Returns:
            torch.Tensor: 分類輸出 [B, num_classes]
        """
        batch_size, temporal_frames, channels, height, width = x.shape
        
        # 驗證輸入形狀
        if temporal_frames != self.temporal_frames:
            raise ValueError(f"期望 {self.temporal_frames} 幀，實際收到 {temporal_frames} 幀")
        
        # 重塑為 [B*T, C, H, W] 以便 backbone 處理
        x = x.view(batch_size * temporal_frames, channels, height, width)
        
        # 特徵提取
        features = self.backbone(x)  # [B*T, feature_dim, H', W']
        
        # 全局池化
        features = self.global_pool(features).squeeze(-1).squeeze(-1)  # [B*T, feature_dim]
        
        # 重塑回時序維度
        features = features.view(batch_size, temporal_frames, self.feature_dim)  # [B, T, feature_dim]
        
        # 時序融合
        fused_features = self.temporal_fusion_layer(features)  # [B, feature_dim]
        
        # 分類
        logits = self.classifier(fused_features)  # [B, num_classes]
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特徵（不進行分類）
        
        Args:
            x: 輸入張量 [B, T, C, H, W]
        
        Returns:
            torch.Tensor: 融合後的特徵 [B, feature_dim]
        """
        batch_size, temporal_frames, channels, height, width = x.shape
        
        x = x.view(batch_size * temporal_frames, channels, height, width)
        features = self.backbone(x)
        features = self.global_pool(features).squeeze(-1).squeeze(-1)
        features = features.view(batch_size, temporal_frames, self.feature_dim)
        fused_features = self.temporal_fusion_layer(features)
        
        return fused_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """獲取模型資訊"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'backbone': self.backbone_name,
            'temporal_frames': self.temporal_frames,
            'temporal_fusion': self.temporal_fusion,
            'num_classes': self.num_classes,
            'feature_dim': self.feature_dim,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_backbone': total_params != trainable_params
        }


class TemporalAttention(nn.Module):
    """
    時序注意力融合模組
    """
    
    def __init__(self, feature_dim: int, temporal_frames: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.temporal_frames = temporal_frames
        
        # 注意力機制
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 1)
        )
        
        # 特徵變換
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feature_dim]
        Returns:
            torch.Tensor: [B, feature_dim]
        """
        batch_size, temporal_frames, feature_dim = x.shape
        
        # 計算注意力權重
        attention_scores = self.attention(x)  # [B, T, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
        
        # 特徵變換
        transformed_features = self.feature_transform(x)  # [B, T, feature_dim]
        
        # 加權融合
        fused_features = torch.sum(attention_weights * transformed_features, dim=1)  # [B, feature_dim]
        
        return fused_features


class TemporalLSTM(nn.Module):
    """
    時序 LSTM 融合模組
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # 輸出投影
        if hidden_dim != input_dim:
            self.output_projection = nn.Linear(hidden_dim, input_dim)
        else:
            self.output_projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feature_dim]
        Returns:
            torch.Tensor: [B, feature_dim]
        """
        # LSTM 處理
        lstm_out, (hidden, _) = self.lstm(x)  # lstm_out: [B, T, hidden_dim]
        
        # 使用最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]  # [B, hidden_dim]
        
        # 投影回原始維度（如果需要）
        if self.output_projection is not None:
            last_output = self.output_projection(last_output)
        
        return last_output


class TemporalAverage(nn.Module):
    """
    時序平均融合
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feature_dim]
        Returns:
            torch.Tensor: [B, feature_dim]
        """
        return torch.mean(x, dim=1)


class TemporalMax(nn.Module):
    """
    時序最大值融合
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, feature_dim]
        Returns:
            torch.Tensor: [B, feature_dim]
        """
        return torch.max(x, dim=1)[0]


def create_temporal_model(model_config: Dict[str, Any]) -> TemporalFireSmokeClassifier:
    """
    創建時序模型的工廠函數
    
    Args:
        model_config: 模型配置字典
    
    Returns:
        TemporalFireSmokeClassifier: 時序分類模型
    """
    return TemporalFireSmokeClassifier(**model_config)


# 預設模型配置
DEFAULT_MODEL_CONFIGS = {
    'resnet50_attention': {
        'backbone_name': 'resnet50',
        'temporal_fusion': 'attention',
        'dropout': 0.2
    },
    'convnext_small_lstm': {
        'backbone_name': 'convnext_small',
        'temporal_fusion': 'lstm',
        'dropout': 0.3
    },
    'efficientnet_b3_attention': {
        'backbone_name': 'efficientnet_b3',
        'temporal_fusion': 'attention',
        'dropout': 0.25
    }
}