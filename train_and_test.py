import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import scipy.io
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
from models import MSA_UNet_skip_level_connation

class UCI_Dataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.mat_file_list = [] 
        self.file_name_list = [f"part_{i}.mat" for i in range(1, 13)]
        self.each_file_samples_number = np.zeros(12, dtype=np.uint32) 
        self.each_file_record_samples_number = np.zeros((12, 1000), dtype=np.uint32) 
        self.record_max = float('-inf')
        self.record_min = float('inf')
        for i in range(len(self.file_name_list)):
            mat_file = scipy.io.loadmat(os.path.join(self.root_path, self.file_name_list[i]))['p'][0, :] 
            for j in range(1000):
                current_record_number = mat_file[j].shape[1] - 87
                self.each_file_record_samples_number[i, j] = current_record_number
                t_max = mat_file[j][2,:].max() 
                t_min = mat_file[j][2,:].min() 
                if t_max > self.record_max:
                    self.record_max = t_max
                if t_min < self.record_min:
                    self.record_min = t_min
            self.each_file_samples_number[i] = self.each_file_record_samples_number[i, :].sum() 
            self.mat_file_list.append(mat_file) 

    def __len__(self):
        return self.each_file_samples_number.sum() 
    
    def __getitem__(self, idx):
        for i in range(12):
            if idx < self.each_file_samples_number[i]:
                mat_file_index = i
                break
            else:
                idx -= self.each_file_samples_number[i]
        
        for j in range(1000):
            if idx < self.each_file_record_samples_number[mat_file_index, j]:
                record_index = j
                break
            else:
                idx -= self.each_file_record_samples_number[mat_file_index, j]
        
        ecg = self.mat_file_list[mat_file_index][record_index][2, idx: idx+88] 
        ecg = (ecg - self.record_min) / (self.record_max - self.record_min) 
        abp = self.mat_file_list[mat_file_index][record_index][1, idx+87] 
        return torch.tensor(ecg, dtype=torch.float32).view(-1,88), torch.tensor(abp, dtype=torch.float32)
    
dataset_file_path = '/gjh/PPG_ECG_BP/UCI_Datasets'
model = MSA_UNet_skip_level_connation(deep_supervision=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
generator = torch.Generator().manual_seed(42) # 可复现
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
dataset = UCI_Dataset(dataset_file_path)
subset_size = int(0.005 * len(dataset))
sub_dataset_indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
sub_dataset = Subset(dataset, sub_dataset_indices)
train_dataset, val_dataset, test_dataset = random_split(sub_dataset, [0.6, 0.2, 0.2], generator=generator)
traindataloader = DataLoader(
    train_dataset,
    batch_size=300, 
    shuffle=False,
    num_workers=16,
    pin_memory=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=200,
    shuffle=False,
    num_workers=16,
    pin_memory=True
)
testdataloader = DataLoader(
    test_dataset,
    batch_size=200, 
    shuffle=False,
    num_workers=16
)

def train(model: nn.Module, 
          device: torch.device, 
          optimizer: torch.optim.Optimizer, 
          criterion: nn.Module, 
          traindataloader: DataLoader,
          val_dataloader: DataLoader,
          accumulation_steps: int,
          ):
    model_save_path = "./MSCA_CSSC_UNet.pth"
    epochs = 10
    loss_weight = [0.7, 0.8, 0.9, 1.0]
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        accumulated = False
        for batch_idx, (x, y) in enumerate(tqdm(traindataloader)):
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss = 0
            for i, output in enumerate(outputs):
                loss += loss_weight[i] * criterion(output, y)
            running_loss += loss.item() 
            loss = loss / accumulation_steps 
            loss.backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulated = False
            else:
                accumulated = True
        if accumulated:
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss = running_loss / len(traindataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Traindataset_Loss: {epoch_loss:.4f}')
        running_loss = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_dataloader):
                x = x.to(device)
                y = y.to(device)
                outputs = model(x)
                loss = 0
                for i, output in enumerate(outputs):
                    loss += loss_weight[i] * criterion(output, y)
                running_loss += loss.item()
            val_loss = running_loss / len(val_dataloader)
            print(f'Epoch [{epoch+1}/{epochs}], Valdataset_Loss: {val_loss:.4f}')
        if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

def test(model: nn.Module, device: torch.device, testdataloader: DataLoader):
    model.eval()
    print('评估模型在测试集上的性能：')
    with torch.no_grad():  # 不需要梯度计算
        predictions = []
        true_values = []
        for x, y in tqdm(testdataloader):
            x = x.to(device)
            outputs = model(x)[-1]
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(y.numpy())
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        try:
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
        except:
            predictions_mean = np.nanmean(predictions) 
            predictions[np.isnan(predictions)] = predictions_mean
            mse = mean_squared_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            mae = mean_absolute_error(true_values, predictions)
        print(f'MSE: {mse:.4f}')
        print(f'R2: {r2:.4f}')
        print(f'MAE: {mae:.4f}')

accumulation_steps = 10
train(model, device, optimizer, criterion, traindataloader, val_dataloader, accumulation_steps)
model.load_state_dict(torch.load('./MSCA_CSSC_UNet.pth'))
model.to(device)
test(model, device, testdataloader)


