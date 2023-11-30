import os
import tqdm
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tab_transformer_pytorch import FTTransformer
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessor
from dataset import CustomDataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score


train = pd.read_csv('D:/Side/MAIC/SyntheticAKI/DB/Train_SyntheticAKI_MAIC2023.csv')
test = pd.read_csv('D:/Side/MAIC/SyntheticAKI/DB/Test_SyntheticAKI_MAIC2023.csv')

target = ['O_AKI', 'O_Critical_AKI_90', 'O_Death_90', 'O_RRT_90', 'Synthetic_type']

preprocessor = Preprocessor()
# NaN 값을 제거
train, test = preprocessor.del_null(train, test)

# 고혈압/저혈압/정상 categorical 변수 생성 (파생 변수)
train = preprocessor.make_A_HTN(train)
test = preprocessor.make_A_HTN(test)

# label, feature 분할
label = train[['O_AKI', 'O_Critical_AKI_90', 'O_Death_90', 'O_RRT_90']]
train = train.drop(labels=target, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(train, label, test_size=0.2, shuffle=True, random_state=34)
x_train.reset_index(drop=True, inplace=True)
x_valid.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_valid.reset_index(drop=True, inplace=True)

# train, valid, and test data에 standard scaler 적용
# tr, te, tr_std, te_std
tr_x_cate, valid_x_cate, tr_std, valid_std = preprocessor.std_scaler(x_train, x_valid)
_, te_cate, _, te_std = preprocessor.std_scaler(x_train, test)

# 이상치 제거 --> train data에서만 제거해준다.
tr_X_num, tr_X_cate, tr_y = preprocessor.cut_3sigma(tr_x_cate, tr_std, y_train)

tr_X_num.reset_index(drop=True, inplace=True)
tr_X_cate.reset_index(drop=True, inplace=True)
tr_y.reset_index(drop=True, inplace=True)

categories = tr_X_cate.nunique().values

tr_categorical_features = tr_X_cate.values
tr_numerical_features = tr_X_num.values
tr_y = tr_y.values

te_categorical_features = valid_x_cate.values
te_numerical_features = valid_std.values
te_y = y_valid.values

print('-------[Model 생성]-------')
model = FTTransformer(
    categories=categories,      # tuple containing the number of unique values within each category
    num_continuous=28,                # number of continuous values
    dim=192,                           # dimension, paper set at 32
    dim_out=4,                        # binary prediction, but could be anything
    depth=3,                          # depth, paper recommended 6
    heads=8,                          # heads, paper recommends 8
    attn_dropout=0.2,                 # post-attention dropout
    ff_dropout=0.1                    # feed forward dropout
)

print('-------[Training Loop]-------')
os.makedirs(f'D:/Side/MAIC/SyntheticAKI/backup/FFTransformer/log', exist_ok=True)
os.makedirs(f'D:/Side/MAIC/SyntheticAKI/backup/FFTransformer/ckp', exist_ok=True)
summary = SummaryWriter(f'D:/Side/MAIC/SyntheticAKI/backup/FFTransformer/log')

criterion_O_AKI = nn.BCEWithLogitsLoss()
criterion_O_Critical_AKI_90 = nn.BCEWithLogitsLoss()
criterion_O_Death_90 = nn.BCEWithLogitsLoss()
criterion_O_RRT_90 = nn.BCEWithLogitsLoss()

optim_adamw = optim.AdamW(list(model.parameters()), lr=1e-4, weight_decay=1e-5)
# scheduler_cosin = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optim_adamw, T_0=20, T_mult=1, eta_min=1e-5)

tr_dataset = CustomDataset(tr_categorical_features, tr_numerical_features, tr_y)
te_dataset = CustomDataset(te_categorical_features, te_numerical_features, te_y)

tr_loader = DataLoader(dataset=tr_dataset, shuffle=True, batch_size=1024)
te_loader = DataLoader(dataset=te_dataset, shuffle=True, batch_size=1024)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)
for ep in range(50):
    train_loss = []
    train_acc = []

    test_loss = []
    test_preds = []
    test_labels = []
    model.train()
    for idx, (cate, num, label) in enumerate(tqdm.tqdm(tr_loader, desc=f'[Train {ep}/50]')):
        cate = cate.to(device)
        num = num.to(device)
        label = label.to(device)

        logits = model(cate.long(), num.long())
        loss_O_AKI = criterion_O_AKI(logits[:][0], label[:][0])
        loss_O_Critical_AKI_90 = criterion_O_Critical_AKI_90(logits[:][1], label[:][1])
        loss_O_Death_90 = criterion_O_Death_90(logits[:][2], label[:][2])
        loss_O_RRT_90 = criterion_O_RRT_90(logits[:][3], label[:][3])

        total_loss = (loss_O_AKI * 0.7) + (loss_O_Critical_AKI_90 * 0.1) + (loss_O_Death_90 * 0.1) + (loss_O_RRT_90 * 0.1)

        optim_adamw.zero_grad()
        total_loss.backward()
        optim_adamw.step()

        train_loss.append(total_loss.item())

    train_loss_avg = np.mean(train_loss)

    # print(f'TRAIN ACC : {train_acc_avg}')
    print(f'TRAIN LOSS : {train_loss_avg}')

    with torch.no_grad():
        model.eval()
        for idx, (cate, num, label) in enumerate(tqdm.tqdm(te_loader, desc=f'[Test {ep}/50]')):
            cate = cate.to(device)
            num = num.to(device)
            label = label.to(device)

            logits = model(cate.long(), num.long())
            loss_O_AKI = criterion_O_AKI(logits[:][0], label[:][0])
            loss_O_Critical_AKI_90 = criterion_O_Critical_AKI_90(logits[:][1], label[:][1])
            loss_O_Death_90 = criterion_O_Death_90(logits[:][2], label[:][2])
            loss_O_RRT_90 = criterion_O_RRT_90(logits[:][3], label[:][3])

            total_loss = (loss_O_AKI * 0.7) + (loss_O_Critical_AKI_90 * 0.1) + (loss_O_Death_90 * 0.1) + (loss_O_RRT_90 * 0.1)

            preds = nn.functional.sigmoid(logits).cpu().detach().numpy()
            preds = preds > 0.5
            label = label.cpu().detach().numpy()

            test_loss.append(total_loss.item())
            test_preds += preds.tolist()
            test_labels += label.tolist()

    test_acc_avg_0 = roc_auc_score(test_preds[:][0], test_labels[:][0]) * 0.7
    test_acc_avg_1 = roc_auc_score(test_preds[:][1], test_labels[:][1]) * 0.1
    test_acc_avg_2 = roc_auc_score(test_preds[:][2], test_labels[:][2]) * 0.1
    test_acc_avg_3 = roc_auc_score(test_preds[:][3], test_labels[:][3]) * 0.1

    test_acc_avg = test_acc_avg_0 + test_acc_avg_1 + test_acc_avg_2 + test_acc_avg_3
    test_loss_avg = np.mean(test_loss)

    print(f'O_AKI: {test_acc_avg_0}')
    print(f'O_Critical_AKI_90: {test_acc_avg_1}')
    print(f'O_Death_90: {test_acc_avg_2}')
    print(f'O_RRT_90: {test_acc_avg_3}')

    print(f'TEST AVG ACC : {test_acc_avg}')
    print(f'TEST LOSS : {test_loss_avg}')

    # summary.add_scalar('Train/acc', train_acc_avg, ep)
    summary.add_scalar('Train/loss', train_loss_avg, ep)

    summary.add_scalar('Test/O_AKI', test_acc_avg_0, ep)
    summary.add_scalar('Test/O_Critical_AKI_90', test_acc_avg_1, ep)
    summary.add_scalar('Test/O_Death_90', test_acc_avg_2, ep)
    summary.add_scalar('Test/O_RRT_90', test_acc_avg_3, ep)
    summary.add_scalar('Test/acc', test_acc_avg, ep)
    summary.add_scalar('Test/loss', test_loss_avg, ep)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": ep,
        },
        os.path.join(f"D:/Side/MAIC/SyntheticAKI/backup/FFTransformer/ckp", f"{ep}.pth"),
    )