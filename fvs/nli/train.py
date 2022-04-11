from fvs.nli.bert import BERTNli
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

if __name__ == '__main__':
    df_raw = pd.read_csv("./train.csv")

    # make balanced
    supports = df_raw[df_raw['label'].str.contains("SUPPORT") == True].sample(10000)
    refutes = df_raw[df_raw['label'].str.contains("REFUTE") == True].sample(10000)
    df = pd.concat([supports, refutes])


    class Train(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df
            self.labels = pd.Categorical(df['label']).codes

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            claim = self.df['claim'].iloc[idx]
            evidence = self.df['evidence_sentence'].iloc[idx]
            label = self.labels[idx]
            return claim, evidence, label


    loader = DataLoader(Train(df), batch_size=8, shuffle=True)

    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    bnli = BERTNli(BertConfig())
    bnli.to(device)  # puts model params on gpu

    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    # optimiser = optim.SGD(params=bnli.parameters(), lr=0.01, momentum=0.9)
    # optimiser = optim.SGD(params=bnli.parameters(), lr=0.001, momentum=0.9)
    optimiser = optim.SGD(params=bnli.parameters(), lr=0.001)
    # optimiser = optim.SGD(params=bnli.parameters(), lr=0.0001)
    # optimiser = optim.Adam(params=bnli.parameters(), lr=0.001)

    hparams = {
        'batch_size': loader.batch_size,
        'optim': str(optimiser),
    }

    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    run: str = str(datetime.now())
    writer = SummaryWriter(log_dir=f"runs/{run}")

    EPOCH = 50
    total_loss = list()
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i, data in enumerate(loader):
            claim, evi, label = data
            output = bnli(claims=claim, evidence=evi)
            label = label.view(-1, 1).float().to(device)

            optimiser.zero_grad()
            loss = criterion(output, label)
            loss.backward()  # compute grads
            optimiser.step()  # apply grads

            running_loss += loss.item()
            total_loss.append(loss.item())
            if (i + 1) % 40 == 0:
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100}")
                writer.add_scalar("Loss", running_loss / 100, (epoch + 1) * i)
                running_loss = 0

    import numpy as np

    metrics = {'avg_loss': np.mean(total_loss),
               'max_loss': np.max(total_loss),
               'min_loss': np.min(total_loss)}
    writer.add_hparams(hparams, metrics)
    print("Training complete.")
