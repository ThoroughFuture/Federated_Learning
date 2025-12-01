#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import copy
import pandas as pd
from tqdm import tqdm

sys.path.append('../main_camel/')
from camel.model.Vit import VIT_S


class FederatedDataset(Dataset):
    """Federated learning dataset"""
    def __init__(self, data_paths: List[str], labels: List[int] = None):
        """
        Args:
            data_paths: List of image paths
            labels: List of labels (0 or 1 for binary classification)
        """
        self.data_paths = data_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if image.size != (256, 256):
                image = image.resize((256, 256), Image.Resampling.LANCZOS)
            image = self.transform(image)
            
            if self.labels is not None:
                label = torch.tensor(self.labels[idx], dtype=torch.long)
                return image, label
            else:
                return image, img_path
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = self.transform(Image.new('RGB', (256, 256), (0, 0, 0)))
            if self.labels is not None:
                label = torch.tensor(0, dtype=torch.long)
                return image, label
            else:
                return image, img_path


class FederatedClient:
    """Federated learning client"""
    
    def __init__(self, client_id: int, data_paths: List[str], labels: List[int],
                 model: nn.Module, device: str = 'cuda:0', 
                 batch_size: int = 32, lr: float = 1e-4):
        """
        Initialize client
        
        Args:
            client_id: Client ID
            data_paths: List of local data paths
            labels: List of local data labels (0 or 1)
            model: Model instance
            device: Training device
            batch_size: Batch size
            lr: Learning rate
        """
        self.client_id = client_id
        self.device = device
        self.model = model.to(device)
        self.data_paths = data_paths
        self.labels = labels
        
        self.dataset = FederatedDataset(data_paths, labels)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, eps=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.train_accuracies = []
    
    def train_local(self, epochs: int = 1) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for images, labels in self.data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = logits.max(1)
                epoch_total += labels.size(0)
                epoch_correct += predicted.eq(labels).sum().item()
            
            avg_loss = epoch_loss / len(self.data_loader)
            accuracy = 100.0 * epoch_correct / epoch_total
            
            total_loss += avg_loss
            correct += epoch_correct
            total += epoch_total
        
        avg_loss = total_loss / epochs
        accuracy = 100.0 * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': len(self.data_paths)
        }
    
    def get_model_state_dict(self) -> OrderedDict:
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_state_dict(self, state_dict: OrderedDict):
        self.model.load_state_dict(state_dict)


class FederatedServer:
    """Federated learning server"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda:0'):
        self.device = device
        self.global_model = model.to(device)
        self.global_state_dict = copy.deepcopy(self.global_model.state_dict())
        
        self.aggregation_history = []
    
    def aggregate_models(self, client_models: Dict[int, OrderedDict], 
                        client_weights: Optional[Dict[int, float]] = None) -> OrderedDict:
        if len(client_models) == 0:
            return self.global_state_dict
        
        if client_weights is None:
            num_clients = len(client_models)
            client_weights = {cid: 1.0 / num_clients for cid in client_models.keys()}
        
        total_weight = sum(client_weights.values())
        client_weights = {cid: w / total_weight for cid, w in client_weights.items()}
        
        aggregated_state = OrderedDict()
        param_names = list(client_models[list(client_models.keys())[0]].keys())
        
        for param_name in param_names:
            param_tensors = []
            weights = []
            
            for client_id, state_dict in client_models.items():
                if param_name in state_dict:
                    param_tensors.append(state_dict[param_name])
                    weights.append(client_weights[client_id])
            
            if len(param_tensors) > 0:
                aggregated_param = torch.zeros_like(param_tensors[0])
                for param, weight in zip(param_tensors, weights):
                    aggregated_param += param * weight
                
                aggregated_state[param_name] = aggregated_param
        
        self.global_state_dict = aggregated_state
        self.global_model.load_state_dict(aggregated_state)
        
        self.aggregation_history.append({
            'round': len(self.aggregation_history) + 1,
            'num_clients': len(client_models),
            'weights': client_weights
        })
        
        return aggregated_state
    
    def get_global_model(self) -> OrderedDict:
        return copy.deepcopy(self.global_state_dict)
    
    def save_global_model(self, save_path: str):
        torch.save(self.global_state_dict, save_path)
        print(f"Global model saved to: {save_path}")
    
    def load_global_model(self, load_path: str):
        self.global_state_dict = torch.load(load_path, map_location=self.device)
        self.global_model.load_state_dict(self.global_state_dict)
        print(f"Global model loaded from {load_path}")


class FederatedLearning:
    
    def __init__(self, model_class, model_kwargs: dict, clients_data: Dict[int, Tuple[List[str], List[int]]],
                 device: str = 'cuda:0', batch_size: int = 32, lr: float = 1e-4):
        """
        Initialize federated learning system
        
        Args:
            model_class: Model class (e.g., VIT_S)
            model_kwargs: Model initialization parameters (e.g., {'Linear_only': True, 'pretrained': 'uni'})
            clients_data: Dict of {client_id: (data_paths, labels)}
            device: Training device
            batch_size: Batch size
            lr: Learning rate
        """
        self.device = device
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        
        self.server = FederatedServer(model_class(**model_kwargs), device=device)
        
        self.clients = {}
        for client_id, (data_paths, labels) in clients_data.items():
            client_model = model_class(**model_kwargs)
            self.clients[client_id] = FederatedClient(
                client_id=client_id,
                data_paths=data_paths,
                labels=labels,
                model=client_model,
                device=device,
                batch_size=batch_size,
                lr=lr
            )
        
        global_state = self.server.get_global_model()
        for client in self.clients.values():
            client.set_model_state_dict(global_state)
        
        print(f"Federated learning system initialized: {len(self.clients)} clients")
    
    def train_round(self, local_epochs: int = 1, 
                   client_weights: Optional[Dict[int, float]] = None) -> Dict:
        """
        Execute one round of federated learning training
        """
        print(f"\n{'='*60}")
        print(f"Starting federated learning round")
        print(f"{'='*60}")
        
        client_models = {}
        client_train_stats = {}
        client_sample_counts = {}
        
        for client_id, client in self.clients.items():
            print(f"\nClient {client_id} starting local training...")
            stats = client.train_local(epochs=local_epochs)
            client_train_stats[client_id] = stats
            client_sample_counts[client_id] = stats['num_samples']
            
            client_models[client_id] = client.get_model_state_dict()
            
            print(f"  Loss: {stats['loss']:.4f}, Accuracy: {stats['accuracy']:.2f}%, "
                  f"Samples: {stats['num_samples']}")
        
        if client_weights is None:
            total_samples = sum(client_sample_counts.values())
            client_weights = {
                cid: count / total_samples 
                for cid, count in client_sample_counts.items()
            }
        
        print(f"\nServer aggregating models...")
        aggregated_state = self.server.aggregate_models(client_models, client_weights)
        
        print(f"Distributing global model to all clients...")
        for client in self.clients.values():
            client.set_model_state_dict(aggregated_state)
        
        avg_loss = np.mean([s['loss'] for s in client_train_stats.values()])
        avg_accuracy = np.mean([s['accuracy'] for s in client_train_stats.values()])
        
        round_stats = {
            'avg_loss': avg_loss,
            'avg_accuracy': avg_accuracy,
            'client_stats': client_train_stats,
            'num_clients': len(self.clients)
        }
        
        print(f"\nRound completed:")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Average accuracy: {avg_accuracy:.2f}%")
        
        return round_stats
    
    def train(self, num_rounds: int, local_epochs: int = 1,
              save_dir: Optional[str] = None) -> List[Dict]:
        """
        Execute multiple rounds of federated learning training
        
        Args:
            num_rounds: Number of training rounds
            local_epochs: Number of local training epochs for each client per round
            save_dir: Model save directory (optional)
        
        Returns:
            Training statistics for all rounds
        """
        all_stats = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'#'*60}")
            print(f"Federated learning round {round_num}/{num_rounds}")
            print(f"{'#'*60}")
            
            round_stats = self.train_round(local_epochs=local_epochs)
            round_stats['round'] = round_num
            all_stats.append(round_stats)
            
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                model_path = os.path.join(save_dir, f'global_model_round_{round_num}.pt')
                self.server.save_global_model(model_path)
        
        if save_dir is not None:
            final_model_path = os.path.join(save_dir, 'temp.pt')
            self.server.save_global_model(final_model_path)
            print(f"\nFinal model saved to: {final_model_path}")
        
        return all_stats
    
    def evaluate_global_model(self, test_data_paths: List[str], 
                             test_labels: List[int]) -> Dict[str, float]:
        """
        Evaluate global model performance on test set
        """
        self.server.global_model.eval()
        
        test_dataset = FederatedDataset(test_data_paths, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.server.global_model(images)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }


def load_data_from_pos_neg(pos_dir: str, neg_dir: str) -> Tuple[List[str], List[int]]:
    """
    Load data from pos_dir and neg_dir (consistent with train_module.py format)
    """
    data_paths = []
    labels = []
    
    try:
        pos_data = pd.read_table(pos_dir, sep='\t', encoding='utf_8_sig', header=None)
        for row in pos_data.values:
            if len(row) > 0:
                line = str(row[0]).strip()
                if line:
                    if ',' in line:
                        path = line.split(',')[0].strip()
                    else:
                        path = line.strip()
                    if path:
                        data_paths.append(path)
                        labels.append(1)
    except Exception as e:
        print(f"Error reading positive sample file {pos_dir}: {e}")
    
    try:
        neg_data = pd.read_table(neg_dir, sep='\t', encoding='utf_8_sig', header=None)
        for row in neg_data.values:
            if len(row) > 0:
                line = str(row[0]).strip()
                if line:
                    if ',' in line:
                        path = line.split(',')[0].strip()
                    else:
                        path = line.strip()
                    if path:
                        data_paths.append(path)
                        labels.append(0)
    except Exception as e:
        print(f"Error reading negative sample file {neg_dir}: {e}")
    
    print(f"Loaded {len(data_paths)} samples (positive: {sum(labels)}, negative: {len(labels)-sum(labels)})")
    return data_paths, labels


def load_data_from_txt(txt_path: str) -> Tuple[List[str], List[int]]:
    """
    Load data from txt file (format: each line "path,label")
    """
    data_paths = []
    labels = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    data_paths.append(parts[0].strip())
                    label = int(parts[1].strip())
                    labels.append(1 if label > 0 else 0)
    
    print(f"Loaded {len(data_paths)} samples from {txt_path}")
    return data_paths, labels


# ==================== Configuration Section ====================

if __name__ == '__main__':
    device = 'cuda:0'
    
    model_class = VIT_S
    model_kwargs = {
        'Linear_only': True,
        'pretrained': 'uni'
    }
    
    batch_size = 32
    learning_rate = 1e-4
    num_rounds = 10
    local_epochs = 1
    
    save_dir = './models/federated_learning/'
    os.makedirs(save_dir, exist_ok=True)
    
    clients_data = {
        0: load_data_from_pos_neg(
            pos_dir='../data/pos_image.txt',
            neg_dir='../data/neg_image.txt'
        ),
    }
    
    print("="*60)
    print("Federated Learning Training")
    print("="*60)
    
    fl_system = FederatedLearning(
        model_class=model_class,
        model_kwargs=model_kwargs,
        clients_data=clients_data,
        device=device,
        batch_size=batch_size,
        lr=learning_rate
    )
    
    stats = fl_system.train(
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        save_dir=save_dir
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    print(f"Final model saved to: {os.path.join(save_dir, 'temp.pt')}")
