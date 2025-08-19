#!/usr/bin/env python3
"""
時序模型訓練器
處理時序資料的載入、訓練和評估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datetime import datetime

from .temporal_classifier import TemporalFireSmokeClassifier, DEFAULT_MODEL_CONFIGS
from .data_utils import prepare_temporal_frames, create_temporal_batch


class TemporalFireSmokeDataset(Dataset):
    """
    時序火煙資料集
    從標註的事件資料載入時序幀序列
    """
    
    def __init__(self, 
                 dataset_path: str,
                 temporal_frames: int = 5,
                 image_size: Tuple[int, int] = (224, 224),
                 training: bool = True):
        """
        Args:
            dataset_path: 資料集路徑
            temporal_frames: 時序幀數
            image_size: 影像尺寸
            training: 是否為訓練模式
        """
        self.dataset_path = Path(dataset_path)
        self.temporal_frames = temporal_frames
        self.image_size = image_size
        self.training = training
        
        # 載入資料
        self.samples = self._load_dataset()
        
        # 類別映射
        self.class_to_idx = {'false_positive': 0, 'true_positive': 1}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"✅ 載入 {len(self.samples)} 個時序樣本")
        self._print_dataset_stats()
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """載入資料集樣本"""
        samples = []
        
        for class_name in ['true_positive', 'false_positive']:
            class_dir = self.dataset_path / class_name
            if not class_dir.exists():
                continue
                
            for event_dir in class_dir.iterdir():
                if not event_dir.is_dir():
                    continue
                
                # 查找事件中的所有影像
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(list(event_dir.glob(ext)))
                
                if len(image_files) == 0:
                    continue
                
                # 按檔名排序確保時序正確
                image_files.sort(key=lambda x: x.name)
                
                # 載入 metadata (如果存在)
                metadata_path = event_dir / 'metadata.json'
                metadata = {}
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                samples.append({
                    'event_dir': str(event_dir),
                    'image_files': [str(f) for f in image_files],
                    'class_name': class_name,
                    'label': self.class_to_idx[class_name],
                    'metadata': metadata
                })
        
        return samples
    
    def _print_dataset_stats(self):
        """印出資料集統計"""
        stats = {}
        for sample in self.samples:
            class_name = sample['class_name']
            if class_name not in stats:
                stats[class_name] = {'count': 0, 'total_frames': 0}
            stats[class_name]['count'] += 1
            stats[class_name]['total_frames'] += len(sample['image_files'])
        
        print("📊 資料集統計:")
        for class_name, stat in stats.items():
            avg_frames = stat['total_frames'] / stat['count']
            print(f"  {class_name}: {stat['count']} 事件, 平均 {avg_frames:.1f} 幀")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        獲取單個樣本
        
        Returns:
            Tuple[torch.Tensor, int]: (frames [T, C, H, W], label)
        """
        sample = self.samples[idx]
        
        # 載入所有幀
        frames = []
        for image_path in sample['image_files']:
            frame = cv2.imread(image_path)
            if frame is not None:
                frames.append(frame)
        
        if not frames:
            # 如果無法載入任何幀，返回黑色幀
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        # 準備時序幀序列
        temporal_frames = prepare_temporal_frames(
            frames, 
            target_frames=self.temporal_frames,
            image_size=self.image_size,
            training=self.training
        )
        
        return temporal_frames, sample['label']


class TemporalTrainer:
    """
    時序模型訓練器
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 device: str = 'auto'):
        """
        Args:
            model_config: 模型配置
            device: 運算設備
        """
        self.model_config = model_config
        self.device = self._setup_device(device)
        
        # 建立模型
        self.model = TemporalFireSmokeClassifier(**model_config)
        self.model.to(self.device)
        
        print(f"✅ 建立時序模型: {self.model.backbone_name}")
        print(f"💾 模型參數: {self.model.get_model_info()}")
        
        # 訓練狀態
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def _setup_device(self, device: str) -> torch.device:
        """設置運算設備"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        device = torch.device(device)
        print(f"🔧 使用設備: {device}")
        return device
    
    def prepare_data(self, 
                     dataset_path: str,
                     batch_size: int = 16,
                     val_split: float = 0.2,
                     num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
        """
        準備訓練和驗證資料
        
        Args:
            dataset_path: 資料集路徑
            batch_size: 批次大小
            val_split: 驗證集比例
            num_workers: 資料載入工作進程數
        
        Returns:
            Tuple[DataLoader, DataLoader]: (train_loader, val_loader)
        """
        # 建立資料集
        full_dataset = TemporalFireSmokeDataset(
            dataset_path,
            temporal_frames=self.model_config.get('temporal_frames', 5),
            training=True
        )
        
        # 分割訓練和驗證集
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 設定驗證集為非訓練模式
        val_dataset.dataset.training = False
        
        # 建立 DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"📊 資料分割: 訓練 {len(train_dataset)}, 驗證 {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train(self,
              dataset_path: str,
              epochs: int = 50,
              batch_size: int = 16,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              val_split: float = 0.2,
              save_dir: str = 'runs/temporal_training') -> Dict[str, Any]:
        """
        訓練模型
        
        Args:
            dataset_path: 資料集路徑
            epochs: 訓練輪數
            batch_size: 批次大小
            learning_rate: 學習率
            weight_decay: 權重衰減
            val_split: 驗證集比例
            save_dir: 儲存目錄
        
        Returns:
            Dict[str, Any]: 訓練結果
        """
        # 建立儲存目錄
        save_path = Path(save_dir) / f"temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 準備資料
        train_loader, val_loader = self.prepare_data(
            dataset_path, batch_size, val_split
        )
        
        # 設定優化器和損失函數
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # 訓練迴圈
        print(f"🚀 開始訓練 {epochs} 輪...")
        
        for epoch in range(epochs):
            # 訓練階段
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, criterion
            )
            
            # 驗證階段
            val_loss, val_acc = self._validate_epoch(
                val_loader, criterion
            )
            
            # 更新學習率
            scheduler.step()
            
            # 記錄歷史
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # 儲存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_model(save_path / 'best_model.pth')
            
            # 印出進度
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 儲存最終模型和訓練歷史
        self._save_model(save_path / 'final_model.pth')
        self._save_training_history(save_path)
        self._plot_training_curves(save_path)
        
        result = {
            'best_val_accuracy': self.best_val_acc,
            'final_train_accuracy': self.training_history['train_acc'][-1],
            'save_path': str(save_path),
            'model_info': self.model.get_model_info()
        }
        
        print(f"✅ 訓練完成! 最佳驗證準確率: {self.best_val_acc:.4f}")
        return result
    
    def _train_epoch(self, train_loader: DataLoader, 
                     optimizer: optim.Optimizer, 
                     criterion: nn.Module) -> Tuple[float, float]:
        """訓練一個 epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for frames, labels in pbar:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新進度條
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def _validate_epoch(self, val_loader: DataLoader, 
                        criterion: nn.Module) -> Tuple[float, float]:
        """驗證一個 epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for frames, labels in val_loader:
                frames = frames.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(frames)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(val_loader), correct / total
    
    def _save_model(self, path: Path):
        """儲存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history
        }, path)
    
    def _save_training_history(self, save_dir: Path):
        """儲存訓練歷史"""
        history_path = save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def _plot_training_curves(self, save_dir: Path):
        """繪製訓練曲線"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss 曲線
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy 曲線
        ax2.plot(self.training_history['train_acc'], label='Train Acc')
        ax2.plot(self.training_history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def load_temporal_model(model_path: str, device: str = 'auto') -> TemporalFireSmokeClassifier:
    """
    載入已訓練的時序模型
    
    Args:
        model_path: 模型路徑
        device: 運算設備
    
    Returns:
        TemporalFireSmokeClassifier: 載入的模型
    """
    # 設定設備
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # 載入檢查點
    checkpoint = torch.load(model_path, map_location=device)
    
    # 建立模型
    model = TemporalFireSmokeClassifier(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 載入時序模型: {model.backbone_name}")
    return model