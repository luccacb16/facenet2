import pandas as pd
from tqdm import tqdm
import os
import wandb

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.NN2 import FaceNet
from models.InceptionResNetV1 import InceptionResnetV1
from models.faceresnet50 import FaceResNet50

from utils.eval_utils import calc_accuracy, get_pairs, LFWPairsDataset
from utils.utils import parse_args, aug_transform, transform, TripletDataset, BalancedBatchSampler, TripletLoss, ValTripletsDataset, calc_val_loss, WarmUpCosineAnnealingLR, save_model_artifact
from triplet_mining import semi_hard_triplet_mining, hard_negative_triplet_mining

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
DTYPE = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        DTYPE = torch.float16
        
EMB_SIZE = 128
CHANGE_MINING_STRATEGY = 0
N_VAL_TRIPLETS = 128
USING_WANDB = False

model_map = {
    'nn2': FaceNet,
    'inceptionresnetv1': InceptionResnetV1,
    'faceresnet50': FaceResNet50
}
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    acc_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    scheduler = None,
    scaler: GradScaler = None,
    epochs: int = 20,
    margin: float = 0.2,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
    accumulation_steps: int = 16,
    batch_size: int = 32
):
    total_accumulation_size = accumulation_steps * batch_size

    for epoch in range(epochs):
        dataloader.batch_sampler.set_epoch(epoch)
        model.train()
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Create tensors inside the epoch loop
        accumulated_embeddings = torch.zeros((total_accumulation_size, EMB_SIZE), device=device)
        accumulated_labels = torch.zeros(total_accumulation_size, dtype=torch.int16, device=device)
        accumulated_imgs = torch.zeros((total_accumulation_size, 3, 160, 160), dtype=DTYPE, device=device)

        batch_in_accumulation = 0
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            batch_index = batch_in_accumulation * batch_size

            with torch.no_grad():
                with autocast(dtype=DTYPE, device_type=device):
                    embeddings = model(imgs)
            
            accumulated_embeddings[batch_index:batch_index + batch_size] = embeddings
            accumulated_labels[batch_index:batch_index + batch_size] = labels
            accumulated_imgs[batch_index:batch_index + batch_size] = imgs

            batch_in_accumulation += 1

            if batch_in_accumulation == accumulation_steps:
                if CHANGE_MINING_STRATEGY > 0 and (epoch + 1) > CHANGE_MINING_STRATEGY:
                    triplets = hard_negative_triplet_mining(accumulated_embeddings, accumulated_labels, device)
                else:
                    triplets = semi_hard_triplet_mining(accumulated_embeddings, accumulated_labels, margin, device, hardest=False)

                anchor_imgs = accumulated_imgs[triplets[:, 0]]
                positive_imgs = accumulated_imgs[triplets[:, 1]]
                negative_imgs = accumulated_imgs[triplets[:, 2]]
                
                with autocast(dtype=DTYPE, device_type=device):
                    anchor_embeddings = model(anchor_imgs)
                    positive_embeddings = model(positive_imgs)
                    negative_embeddings = model(negative_imgs)
                    loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad(set_to_none=True)
                
                accumulated_loss += loss.item() * accumulation_steps

                # Clean tensors after weight update
                del triplets, anchor_imgs, positive_imgs, negative_imgs
                del anchor_embeddings, positive_embeddings, negative_embeddings
                torch.cuda.empty_cache()

                batch_in_accumulation = 0

        epoch_loss = accumulated_loss / len(dataloader)

        # Log
        val_loss = calc_val_loss(model, val_dataloader, triplet_loss, device)
        
        # Acurácia
        epoch_accuracy = calc_accuracy(model, acc_dataloader, device)
        print(f"Epoch [{epoch+1}/{epochs}] | accuracy: {epoch_accuracy:.4f} | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.0e}")
        model.save_checkpoint(checkpoint_path, f'epoch_{epoch+1}.pt')
        
        if USING_WANDB:
            wandb.log({'epoch': epoch, 'accuracy': epoch_accuracy, 'train_loss': epoch_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
            save_model_artifact(checkpoint_path, epoch+1)
        
        scheduler.step()

# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    model_name = args.model
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    margin = args.margin
    num_workers = args.num_workers
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    compile = args.compile
    pretrained = args.restore
    CHANGE_MINING_STRATEGY = args.change_mining_step
    min_lr = args.min_lr
    max_lr = args.max_lr
    warmup_epochs = args.warmup_epochs
    USING_WANDB = args.wandb
    EMB_SIZE = args.emb_size
    
    config = {
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'margin': margin,
        'num_workers': num_workers,
        'change_mining_strategy': CHANGE_MINING_STRATEGY,
    }
    
    if USING_WANDB:
        wandb.login(key=os.environ['WANDB_API_KEY'])
        wandb.init(project='facenet', config=config)
    
    accumulation_steps = accumulation // batch_size
    
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    
    # Carregando datasets
    train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    
    # Treino
    min_images_per_id = accumulation // batch_size
    train_df = train_df.groupby('id').filter(lambda x: len(x) >= min_images_per_id)
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    
    # Teste
    test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_PATH, 'casia-faces/', x))
    
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
    
    # Loader de acurácia
    acc_pairs = get_pairs(test_df, N_VAL_TRIPLETS)
    acc_dataset = LFWPairsDataset(acc_pairs, transform=transform)
    acc_dataloader = DataLoader(acc_dataset, batch_size=N_VAL_TRIPLETS, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    # Loader de treino
    triplet_dataset = TripletDataset(train_df, 
                                     transform=transform, 
                                     dtype=DTYPE)

    sampler = BalancedBatchSampler(dataset=triplet_dataset,
                                   accumulation=accumulation,
                                   batch_size=batch_size,
                                   epochs=epochs)
    
    dataloader = DataLoader(triplet_dataset, batch_sampler=sampler,
                            pin_memory=True, num_workers=num_workers)
    
    # Loss
    triplet_loss = TripletLoss(margin=margin)
    
    # Modelo
    if model_name.lower() not in model_map:
        raise ValueError(f"Model {model_name} not found")

    if pretrained:
        model = model_map[model_name.lower()].load_checkpoint(pretrained).to(device)
        model.freeze()
    else:
        model = model_map[model_name.lower()](emb_size=EMB_SIZE).to(device)
    
    if compile:
        model = torch.compile(model)
    
    # Scaler, otimizador e scheduler
    scaler = GradScaler()
    optimizer = torch.optim.AdamW([
        {'params': model.parameters(), 'lr': max_lr, 'weight_decay': 1e-5, 'initial_lr': max_lr}
    ])
    
    scheduler = WarmUpCosineAnnealingLR(optimizer, epochs, warmup_epochs, min_lr, max_lr, last_epoch=-1)
    
    print(f'\nModel: {model_name} | Params: {model.num_params:,}')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}\n')
    
    print(f'\nImagens: {len(train_df)} | Identidades: {train_df["id"].nunique()} | imgs/id: {len(train_df) / train_df["id"].nunique()}')
        
    train_losses, val_losses = train(
        model               = model,
        dataloader          = dataloader,
        val_dataloader      = val_dataloader,
        acc_dataloader      = acc_dataloader,
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