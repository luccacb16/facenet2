import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse
from PIL import Image
import os
import wandb
import math

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F
from torch.amp import autocast

transform = Compose([
    Resize((160, 160)),
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

aug_transform = Compose([
    RandomResizedCrop(160, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --------------------------------------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, images_df, transform=None, dtype=torch.bfloat16):
        self.labels = images_df['id'].values
        self.image_paths = images_df['path'].values
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        image = image.to(self.dtype)
        label = torch.tensor(self.labels[idx], dtype=torch.int16)
        
        return image, label
    
class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, accumulation, batch_size, epochs):
        self.dataset = dataset
        self.accumulation = accumulation
        self.batch_size = batch_size
        self.accumulation_steps = accumulation // batch_size
        self.label_set = list(set(dataset.labels))
        self.labels_to_indices = {label: np.where(np.array(dataset.labels) == label)[0] for label in self.label_set}
        self.num_batches = len(dataset) // batch_size
        self.num_accumulation_batches = self.num_batches // self.accumulation_steps
        self.epochs = epochs
        self.epoch = 0
        
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __iter__(self):
        pbar = tqdm(total=self.num_accumulation_batches, desc=f'Epoch {self.epoch+1}/{self.epochs}', unit='batch')
        for _ in range(self.num_accumulation_batches):
            selected_labels = np.random.choice(self.label_set, self.batch_size, replace=False)
            indices = []
            for _ in range(self.accumulation_steps):
                batch_indices = []
                for label in selected_labels:
                    idx = np.random.choice(self.labels_to_indices[label])
                    batch_indices.append(idx)
                indices.extend(batch_indices)
                yield batch_indices
            pbar.update(1)
        pbar.close()

    def __len__(self):
        return self.num_accumulation_batches

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:
        # Loss = max(||f(a) - f(p)||^2 - ||f(a) - f(n)||^2 + margin, 0)
        pos_diff = anchor - positive
        neg_diff = anchor - negative
        
        pos_dists = torch.norm(pos_diff, p=2, dim=1)
        neg_dists = torch.norm(neg_diff, p=2, dim=1)
        
        losses = F.relu(pos_dists - neg_dists + self.margin)
        
        return losses.mean()

# --------------------------------------------------------------------------------------------------------

# Validação

def get_val_triplets(dataframe: pd.DataFrame, n: int = 128) -> torch.Tensor:
    """
    Generates n triplets (anchor, positive, negative) from the test set.
    """
    labels = dataframe['id'].values
    num_samples = len(dataframe)

    triplets = np.zeros((n, 3), dtype=np.int32)
    for i in range(n):
        anchor_idx = np.random.randint(0, num_samples)
        anchor_label = labels[anchor_idx]

        positive_idxs = np.where(labels == anchor_label)[0]
        positive_idxs = positive_idxs[positive_idxs != anchor_idx]
        positive_idx = np.random.choice(positive_idxs)

        negative_idxs = np.where(labels != anchor_label)[0]
        negative_idx = np.random.choice(negative_idxs)

        triplets[i] = [anchor_idx, positive_idx, negative_idx]

    return torch.from_numpy(triplets).to(torch.int32)

class ValTripletsDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None, dtype = torch.bfloat16, n_triplets: int = 128):
        self.image_paths = dataframe['path'].values
        self.labels = dataframe['id'].values
        self.transform = transform
        self.dtype = dtype
        
        # Generate triplets with indices
        self.triplets = get_val_triplets(dataframe, n_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor_idx, pos_idx, neg_idx = self.triplets[index]

        # Load images from paths
        anchor_img = Image.open(self.image_paths[anchor_idx]).convert('RGB')
        pos_img = Image.open(self.image_paths[pos_idx]).convert('RGB')
        neg_img = Image.open(self.image_paths[neg_idx]).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        # Convert images to bfloat16
        anchor_img = anchor_img.to(self.dtype)
        pos_img = pos_img.to(self.dtype)
        neg_img = neg_img.to(self.dtype)
        
        return (anchor_img, pos_img, neg_img), self.triplets[index]
    
def calc_val_loss(model, val_loader, loss, device='cuda', dtype=torch.bfloat16):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for (anchors, positives, negatives), _ in val_loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
        
            with autocast(dtype=dtype, device_type=device):
                anchor_embeddings = model(anchors)
                positive_embeddings = model(positives)
                negative_embeddings = model(negatives)

                loss = loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                total_loss += loss.item()
                num_batches += 1

    val_loss = total_loss / num_batches
    model.train()
    
    return val_loss
    
class WarmUpCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1):
        self.epochs = epochs
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.min_lr + (self.max_lr - self.min_lr) * alpha for _ in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (self.max_lr - self.min_lr) * cosine_decay for _ in self.base_lrs]
    
def save_model_artifact(checkpoint_path, epoch):
    artifact = wandb.Artifact(f'epoch_{epoch}', type='model')
    artifact.add_file(os.path.join(checkpoint_path, f'epoch_{epoch}.pt'))
    wandb.log_artifact(artifact)
    
# --------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural com triplet loss")
    parser.add_argument('--model', type=str, default='faceresnet50', help='Modelo a ser utilizado (default: faceresnet50)')
    parser.add_argument('--num_val_triplets', type=int, default=128, help='Número de triplets para validação (default: 128)')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamanho do batch e quantidade de identidades por batch (default: 32)')
    parser.add_argument('--accumulation', type=int, default=512, help='Acumulação de gradientes e amostras para triplet mining (default: 512)')
    parser.add_argument('--change_mining_step', type=int, default=8, help='Número de epochs para mudar o método de triplet mining, de semi-hard para negative hard (default: 8)')
    parser.add_argument('--emb_size', type=int, default=512, help='Tamanho do vetor de embeddings (default: 512)')
    parser.add_argument('--epochs', type=int, default=20, help='Número de epochs (default: 20)')
    parser.add_argument('--margin', type=float, default=0.2, help='Margem para triplet loss (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=1, help='Número de workers para o DataLoader (default: 1)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data/)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    parser.add_argument('--device', type=str, default='cuda', help='Dispositivo para treinamento (default: cuda)')
    parser.add_argument('--compile', action='store_true', help='Se deve compilar o modelo (default: False)')
    parser.add_argument('--wandb', action='store_true', help='Se está rodando com o Weights & Biases (default: False)')
    parser.add_argument('--restore', type=str, default=None, help='Caminho para um checkpoint para restaurar o treinamento (default: None)')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Taxa de aprendizado mínima (default: 1e-5)')
    parser.add_argument('--max_lr', type=float, default=3e-4, help='Taxa de aprendizado máxima (default: 3e-4)')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Número de epochs para warmup (default: 5)')
    parser.add_argument('--hardest', type='store_true', help='Hardest no semi-hard quanto não acha (default: False)')
    
    return parser.parse_args()