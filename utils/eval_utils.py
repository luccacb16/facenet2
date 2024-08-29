import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

transform = Compose(
    [
    Resize((160, 160)), 
    ToTensor(), 
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------------------------------

# EVAL
# carregar dataset de teste -> get_pairs -> eval_epochs 

def get_pairs(ids_df: pd.DataFrame, n_pairs: int | None = None) -> pd.DataFrame:
    if n_pairs is None:
        n_pairs = len(ids_df)

    pairs = []
    i = 0
    while i < n_pairs // 2:
        # Escolhe aleatoriamente uma linha/índice do dataframe
        row = ids_df.sample(1).iloc[0]
        
        # Encontra outras imagens com o mesmo ID
        same_id_df = ids_df[(ids_df['id'] == row['id']) & ~(ids_df.index == row.name)]
        if same_id_df.empty:
            continue  # Se não houver outras imagens com o mesmo ID, tenta novamente
        
        # Seleciona aleatoriamente uma imagem com o mesmo ID e uma com um ID diferente
        same = same_id_df.sample(1)
        diff = ids_df[ids_df['id'] != row['id']].sample(1)

        # Adiciona os pares ao resultado final
        pairs.append([row['path'], same['path'].values[0], 1])  # 1 significa que são do mesmo ID
        pairs.append([row['path'], diff['path'].values[0], 0])  # 0 significa que são de IDs diferentes
        
        i += 1

    return pd.DataFrame(pairs, columns=['img1', 'img2', 'label'])


def plot_distribution_and_ROC(pairs: pd.DataFrame, model_name: str, target_far=1e-3) -> float:
    col_name = 'distance'
    
    # Cria uma figura com dois subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gráfico de distribuição de distâncias
    pairs[pairs['label'] == 1][col_name].plot.kde(ax=ax[0])
    pairs[pairs['label'] == 0][col_name].plot.kde(ax=ax[0])
    ax[0].set_title(f'Distances distribution ({model_name})')
    ax[0].legend(['Positive', 'Negative'])
    
    # Cálculo da curva ROC e AUC
    fpr, tpr, thresholds = roc_curve(pairs['label'], -pairs['distance'])
    roc_auc = auc(fpr, tpr)
    
    # Encontra o threshold ótimo
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Threshold para FAR 1e-3
    far_idx = np.where(fpr <= target_far)[0][-1]
    far_threshold = thresholds[far_idx]

    # Plot da curva ROC
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlim([0.0, 1.0])
    ax[1].set_ylim([0.0, 1.05])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title(f'Receiver Operating Characteristic for {model_name}')
    ax[1].legend(loc="lower right")

    # Mostra a figura completa
    plt.tight_layout()
    plt.show()
    
    return -optimal_threshold, -far_threshold

def eval_epochs(epochs_path: str, pairs: pd.DataFrame, model_class: nn.Module, batch_size: int = 32, transform = None, device: str = 'cuda', target_far=1e-2) -> None:
    models_name = os.listdir(epochs_path)
    models_name = [model for model in models_name if not model.startswith('bons')]
    models_name_sorted = sorted(models_name, key=lambda x: int(x.replace('.pt', '').split('_')[1]))
        
    modelo = model_class(emb_size=64).to(device)
    modelo.eval()
    
    for model in models_name_sorted:        
        model_path = os.path.join(epochs_path, model)
        load = torch.load(model_path)
        load = {k.replace('_orig_mod.', ''): v for k, v in load.items()}
        modelo.load_state_dict(load)
        
        pairs_dist = calculate_distances(modelo, pairs, batch_size=batch_size, transform=transform, device=device)
        
        _, low_far = plot_distribution_and_ROC(pairs_dist, model, target_far)
        
        # Low FAR threshold
        acc = accuracy(pairs_dist, low_far)
        val = VAL(pairs_dist, low_far)
        
        print(f'Target FAR: {target_far:.0e} | Threshold: {low_far:.4f}')
        print(f'[{model}] Accuracy: {acc:.4f}')
        print(f'[{model}] VAL: {val:.4f}')
        
        print()
        
        # Calcula as estatísticas das distâncias
        pos_mean, pos_std, neg_mean, neg_std = distance_stats(pairs_dist)
        print(f'[{model}] Positive mean: {pos_mean:.4f} ± {pos_std:.4f}')
        print(f'[{model}] Negative mean: {neg_mean:.4f} ± {neg_std:.4f}')
        
def distance_stats(pairs: pd.DataFrame) -> None:
    col_name = 'distance'
    pos_mean = pairs[pairs['label'] == 1][col_name].mean()
    pos_std = pairs[pairs['label'] == 1][col_name].std()

    neg_mean = pairs[pairs['label'] == 0][col_name].mean()
    neg_std = pairs[pairs['label'] == 0][col_name].std()
    
    return pos_mean, pos_std, neg_mean, neg_std
    
def accuracy(pairs: pd.DataFrame, threshold: float) -> float:
    '''
    Retorna a acurácia do modelo com base no threshold.
    
    Args:
        - pairs (pd.DataFrame): DataFrame com os pares de imagens.
        - threshold (float): Valor de corte para a distância.
        
    Returns:
        - float: Acurácia do modelo.
    '''
    return sum((pairs['distance'] < threshold) == pairs['label']) / len(pairs)

def VAL(pairs: pd.DataFrame, threshold: float) -> float:
    '''
    Retorna a Validation Rate (VAL) do modelo com base no threshold.
    
    Args:
        - pairs (pd.DataFrame): DataFrame com os pares de imagens.
        - threshold (float): Valor de corte para a distância.
        
    Returns:
        - float: VAL do modelo.
    '''
    
    # Pares verdadeiramente da mesma pessoa
    true_positives = pairs['label'] == 1
    # Pares classificados como da mesma pessoa com base no threshold
    classified_as_true = pairs['distance'] < threshold
    # Calcula VAL
    val = sum(true_positives & classified_as_true) / sum(true_positives)
    
    return val

def FAR(pairs: pd.DataFrame, threshold: float) -> float:
    '''
    Retorna a False Accept Rate (FAR) do modelo com base no threshold.
    
    Args:
        - pairs (pd.DataFrame): DataFrame com os pares de imagens.
        - threshold (float): Valor de corte para a distância.
        
    Returns:
        - float: FAR do modelo.
    '''
    
    # Pares verdadeiramente de pessoas diferentes
    true_negatives = pairs['label'] == 0
    # Pares classificados como da mesma pessoa com base no threshold
    classified_as_true = pairs['distance'] < threshold
    # Calcula FAR
    far = sum(true_negatives & classified_as_true) / sum(true_negatives)
    
    return far

class LFWPairsDataset(Dataset):
    def __init__(self, pairs_df, transform=None):
        self.pairs_df = pairs_df
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        img1 = Image.open(row['img1'])
        img2 = Image.open(row['img2'])
        label = row['label']

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, label

def calculate_distances(model: nn.Module, pairs: pd.DataFrame, batch_size=32, transform=None, device='cuda') -> pd.DataFrame:
    model.to(device)
    model.eval()
    
    dataset = LFWPairsDataset(pairs, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    distances = []

    with torch.no_grad():
        for img1_batch, img2_batch, labels_batch in tqdm(dataloader, desc='Calculating distances'):
            img1_batch, img2_batch = img1_batch.to(device), img2_batch.to(device)
            outputs1, outputs2 = model(img1_batch), model(img2_batch)
            batch_distances = torch.nn.functional.pairwise_distance(outputs1, outputs2, p=2)
            distances.extend(batch_distances.cpu().numpy())

    pairs['distance'] = distances
    return pairs

class LFWSingleDataset(Dataset):
    def __init__(self, images_df, transform=None):
        self.labels = images_df['id'].values
        self.image_paths = images_df['path'].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]