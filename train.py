import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.NN2 import FaceNet

from utils.utils import parse_args, transform, TripletDataset, BalancedBatchSampler, TripletLoss, ValTripletsDataset, calc_val_loss, save_losses
from triplet_mining import semi_hard_triplet_mining, hard_negative_triplet_mining

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
EMB_SIZE = 64
CHANGE_MINING_STRATEGY = 0.4
N_VAL_TRIPLETS = 128
DOCS_PATH = './docs/'
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    scheduler: torch.optim.lr_scheduler = None,
    scaler: GradScaler = None,
    epochs: int = 20,
    margin: float = 0.2,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
    accumulation_steps: int = 4,
    batch_size: int = 8
):
    train_losses = []
    val_losses = []
    
    model.train()
    
    total_accumulation_size = batch_size * accumulation_steps
    
    for epoch in range(epochs):
        accumulated_loss = 0.0
        accumulated_embeddings = torch.zeros((total_accumulation_size, EMB_SIZE), device=device)
        accumulated_labels = torch.zeros(total_accumulation_size, dtype=torch.int16, device=device)
        
        progress_bar = tqdm(
            total=len(dataloader),
            desc=f"Epoch [{epoch+1}/{epochs}]",
            unit='batch',
        )
        
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            #start_time = time.time()
            with autocast(dtype=DTYPE, device_type='cuda'):
                embeddings = model(imgs)
            #embeddings_time = time.time() - start_time
            #print(f"[{i}] Embedding calculation time: {embeddings_time:.3f}s")
            
            start_idx = (i % accumulation_steps) * batch_size
            end_idx = start_idx + batch_size
            
            accumulated_embeddings[start_idx:end_idx] = embeddings
            accumulated_labels[start_idx:end_idx] = labels
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                current_size = end_idx
                all_embeddings = accumulated_embeddings[:current_size]
                all_labels = accumulated_labels[:current_size]
                
                #mining_start_time = time.time()
                if (epoch+1) < (epochs+1) * CHANGE_MINING_STRATEGY:
                    triplets = semi_hard_triplet_mining(embeddings=all_embeddings, labels=all_labels, margin=margin, device=device, hardest=False)
                else:
                    triplets = hard_negative_triplet_mining(embeddings=all_embeddings, labels=all_labels, device=device)
                #mining_time = time.time() - mining_start_time
                #print(f"Triplet mining time: {mining_time:.3f}s")
                
                #loss_start_time = time.time()
                anchor_embeddings = all_embeddings[triplets[:, 0]]
                positive_embeddings = all_embeddings[triplets[:, 1]]
                negative_embeddings = all_embeddings[triplets[:, 2]]
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                loss = loss / accumulation_steps
                #loss_time = time.time() - loss_start_time
                #print(f"Loss calculation time: {loss_time:.3f}s")
                
                scaler.scale(loss).backward()
                
                update_start_time = time.time()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                #update_time = time.time() - update_start_time
                #print(f"Weight update time: {update_time:.3f}s")
                
                accumulated_loss += loss.item()
                
                # Update progress bar
                progress_bar.update(1)
                    
        progress_bar.close()

        if scheduler is not None:
            scheduler.step()
        
        val_loss = calc_val_loss(model=model, 
                                val_loader=val_dataloader,
                                loss=triplet_loss,
                                device=device,
                                dtype=DTYPE)
        
        val_losses.append(val_loss)
        train_losses.append(accumulated_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {accumulated_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.0e}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
    
    return train_losses, val_losses

# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    margin = args.margin
    num_workers = args.num_workers
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    colab = args.colab
    
    accumulation_steps = accumulation // batch_size
    
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    
    print(f'\nDevice: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}\n')
    
    # Carregando datasets
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'CASIA/casia_train.csv'))
    
    # Treino
    min_images_per_id = accumulation // batch_size
    train_df = train_df.groupby('id').filter(lambda x: len(x) >= min_images_per_id)
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'CASIA/casia-faces/', x))
    
    # Teste
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'CASIA/casia_test.csv'))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'CASIA/casia-faces/', x))
    
    # Loader de validação
    val_triplets = ValTripletsDataset(test_df, 
                                      transform=transform, 
                                      dtype=DTYPE,
                                      n_triplets=N_VAL_TRIPLETS)
    
    val_dataloader = DataLoader(val_triplets,
                                batch_size=N_VAL_TRIPLETS, 
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers)
    
    # Loader de treino
    triplet_dataset = TripletDataset(train_df, 
                                     transform=transform, 
                                     dtype=DTYPE)

    sampler = BalancedBatchSampler(dataset=triplet_dataset,
                                   accumulation=accumulation,
                                   batch_size=batch_size)
    
    dataloader = DataLoader(triplet_dataset, batch_sampler=sampler,
                            pin_memory=True, num_workers=num_workers)
    
    # Loss
    triplet_loss = TripletLoss(margin=margin)
    
    # Modelo
    model = FaceNet(emb_size=EMB_SIZE).to(device)
    
    if not colab:
        model = torch.compile(model)
    
    # Otimizador, scheduler e scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    scaler = GradScaler()
        
    train_losses, val_losses = train(
        model               = model,
        dataloader          = dataloader,
        val_dataloader      = val_dataloader,
        optimizer           = optimizer,
        triplet_loss        = triplet_loss,
        scheduler           = scheduler,
        scaler              = scaler,
        epochs              = epochs,
        checkpoint_path     = CHECKPOINT_PATH,
        margin              = margin,
        device              = device,
        accumulation_steps  = accumulation_steps,
        batch_size          = batch_size
    )
    
    # Salva a imagem com os resultados
    save_losses(train_losses, val_losses, DOCS_PATH)