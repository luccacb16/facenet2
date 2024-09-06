import pandas as pd
from tqdm import tqdm
import os
import wandb
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.NN2 import FaceNet
from models.InceptionResNetV1 import InceptionResnetV1

from utils.eval_utils import calc_accuracy, get_pairs, LFWPairsDataset
from utils.utils import parse_args, transform, TripletDataset, BalancedBatchSampler, TripletLoss, ValTripletsDataset, calc_val_loss, save_losses, adjust_learning_rate
from triplet_mining import semi_hard_triplet_mining, hard_negative_triplet_mining

load_dotenv()

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
DOCS_PATH = './docs/'

model_class_map = {
    'nn2': FaceNet,
    'inceptionresnetv1': InceptionResnetV1
}
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    acc_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    scaler: GradScaler = None,
    epochs: int = 20,
    margin: float = 0.2,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
    accumulation_steps: int = 16,
    batch_size: int = 32
):
    total_accumulation_size = accumulation_steps * batch_size
    accumulated_embeddings = torch.zeros((total_accumulation_size, EMB_SIZE), device=device)
    accumulated_labels = torch.zeros(total_accumulation_size, dtype=torch.int16, device=device)
    accumulated_imgs = torch.zeros((total_accumulation_size, 3, 112, 112), dtype=DTYPE, device=device)

    for epoch in range(epochs):
        dataloader.batch_sampler.set_epoch(epoch)
        model.train()
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
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
                
                val_loss = calc_val_loss(model, val_dataloader, triplet_loss, device)
                
                # Log
                wandb.log({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
                
                optimizer.zero_grad(set_to_none=True)
                
                accumulated_loss += loss.item() * accumulation_steps

                accumulated_embeddings.zero_()
                accumulated_labels.zero_()
                accumulated_imgs.zero_()
                
                batch_in_accumulation = 0

        # Acurácia
        epoch_accuracy = calc_accuracy(model, acc_dataloader, device)
        wandb.log({'epoch': epoch, 'accuracy': epoch_accuracy})

        # Atualiza o scheduler customizado
        adjust_learning_rate(optimizer, epoch, epochs, CHANGE_MINING_STRATEGY)

        val_loss = calc_val_loss(model, val_dataloader, triplet_loss, device, dtype=DTYPE)
        epoch_loss = accumulated_loss / len(dataloader)

        print(f"Epoch [{epoch+1}/{epochs}] | loss: {epoch_loss:.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.0e}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))

# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    model_classe = args.model
    batch_size = args.batch_size
    accumulation = args.accumulation
    epochs = args.epochs
    margin = args.margin
    num_workers = args.num_workers
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    colab = args.colab
    restore_from_checkpoint = args.restore
    CHANGE_MINING_STRATEGY = args.change_mining_step
    
    config = {
        'batch_size': batch_size,
        'accumulation': accumulation,
        'epochs': epochs,
        'margin': margin,
        'num_workers': num_workers,
        'change_mining_strategy': CHANGE_MINING_STRATEGY,
    }
    
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(project='facenet', config=config)
    
    accumulation_steps = accumulation // batch_size
    
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    
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
    #model = FaceNet(emb_size=EMB_SIZE, restore_from_checkpoint=restore_from_checkpoint).to(device)
    #model = InceptionResnetV1(emb_size=EMB_SIZE).to(device)
    if model_classe.lower() not in model_class_map:
        raise ValueError(f"Model {model_classe} not found")
    model = model_class_map[model_classe.lower()](emb_size=EMB_SIZE).to(device)
    
    if not colab:
        model = torch.compile(model)
    wandb.watch(model, log='all', log_freq=25)
    
    # Scaler, otimizador e scheduler
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print(f'\nModel: {model_classe}')
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {DTYPE}\n')
        
    train_losses, val_losses = train(
        model               = model,
        dataloader          = dataloader,
        val_dataloader      = val_dataloader,
        acc_dataloader      = acc_dataloader,
        optimizer           = optimizer,
        triplet_loss        = triplet_loss,
        scaler              = scaler,
        epochs              = epochs,
        checkpoint_path     = CHECKPOINT_PATH,
        margin              = margin,
        device              = device,
        accumulation_steps  = accumulation_steps,
        batch_size          = batch_size
    )