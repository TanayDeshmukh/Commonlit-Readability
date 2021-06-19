import torch
import numpy as np
from tqdm import tqdm
from utils import PAD
from torch.utils.data import DataLoader
from model import BertForClassification
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import CommonlitDataset, collate_fn_train, collate_fn_test
from pytorch_transformers.optimization import AdamW, WarmupCosineSchedule

def train(model, train_data_loader, optimizer, scheduler, loss_fn, epoch, device):
    
    running_loss = 0.0
    model.train()

    with tqdm(desc='Epoch %d - Train' % epoch, unit='it', total=len(train_data_loader)) as pbar:

        for idx, batch in enumerate(train_data_loader):
            # text_id, text, target, std_err
            text = batch[1].to(device)
            target = batch[2].to(device)
            std_err = batch[3].to(device)

            optimizer.zero_grad()

            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(text)

            attention_mask = (text != PAD).float() #(batch_size, caption_len) (64, 21)
            
            pred = model(text, position_ids, attention_mask)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            used_mem = torch.cuda.max_memory_allocated() / 1024.0 ** 3

            pbar.set_postfix(train_loss = running_loss /(idx + 1), mem = used_mem)
            pbar.update()

    return running_loss/len(train_data_loader)

def val(model, val_data_loader, loss_fn, epoch, device):
    running_loss = 0.0
    model.eval()

    with tqdm(desc='Epoch %d - validation' % epoch, unit='it', total=len(val_data_loader)) as pbar:

        for iteration, batch in enumerate(train_data_loader):
            # text_id, text, target, std_err
            text = batch[1].to(device)
            target = batch[2].to(device)
            std_err = batch[3].to(device)

            position_ids = torch.arange(text.size(1), dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand_as(text)

            attention_mask = (text != PAD).float() #(batch_size, caption_len) (64, 21)
            
            pred = model(text, position_ids, attention_mask)
            loss = loss_fn(pred, target)
            
            running_loss += loss.item()

            used_mem = torch.cuda.max_memory_allocated() / 1024.0 ** 3

            pbar.set_postfix(validation_loss = running_loss /(idx + 1), mem = used_mem)
            pbar.update()

    return running_loss/len(val_data_loader)

if __name__ == '__main__':

    batch_size = 16
    shuffle_dataset = True
    random_seed= 42
    num_epochs=100
    patience=10
    min_loss = np.inf
    device = "cuda"

    lr = 5e-5
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    warmup_steps = 10000
    max_steps = 100000

    # Train Dataset
    dataset = CommonlitDataset('./data', 'train')
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_data_loader = DataLoader(dataset, 
                                    batch_size=batch_size,
                                    drop_last=True,
                                    collate_fn=collate_fn_train,
                                    sampler=train_sampler,
                                    num_workers=8)

    val_data_loader = DataLoader(dataset, 
                                    batch_size=batch_size,
                                    drop_last=True,
                                    collate_fn=collate_fn_train,
                                    sampler=val_sampler,
                                    num_workers=8)

    # Test Dataset
    test_dataset = CommonlitDataset('./data', 'test')
    test_data_loader = DataLoader(test_dataset, 
                                    batch_size=1,
                                    shuffle=True,
                                    drop_last=True,
                                    collate_fn=collate_fn_train,
                                    num_workers=8)

    # Building Model

    model = BertForClassification().to(device)

    optimizer = AdamW(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas
    )

    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        t_total=max_steps
    )

    loss_fn = torch.nn.MSELoss()

    for epoch in range(num_epochs):

        is_best_model = False

        train_loss = train(model, train_data_loader, optimizer, scheduler, loss_fn, epoch, device)

        val_loss = val(model, val_data_loader, loss_fn, epoch, device)

        if val_loss < min_loss:
            is_best_model = True
            min_loss = val_loss
        else:
            patience -= 1
        
        if patience <= 0:
            break
