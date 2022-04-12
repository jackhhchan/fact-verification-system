from fvs.nli.bert import BERTNli, config_train, config_model, config_optimisers
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device('cuda:{}'.format(torch.cuda.current_device()))

from string import punctuation
import re

pattern = re.compile(f"(-L.B-|-R.B-|[{punctuation}])")


def clean(s: str):
    return re.sub(' +', ' ', pattern.sub('', s).encode("ascii", "ignore").decode())


if __name__ == '__main__':
    df_raw = pd.read_csv("./train.csv")
    df_raw['claim'] = df_raw['claim'].apply(clean)
    df_raw['evidence_sentence'] = df_raw['evidence_sentence'].apply(clean)

    # make balanced
    to_sample = 10
    supports = df_raw[df_raw['label'].str.contains("SUPPORT") == True].sample(to_sample)
    refutes = df_raw[df_raw['label'].str.contains("REFUTE") == True].sample(to_sample)
    df = pd.concat([supports, refutes])

    # split train test
    train, test = train_test_split(df, test_size=0.2)
    # train = supports
    # test = supports.copy()


    ## Data Loaders ##
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


    loader = DataLoader(Train(train), batch_size=8, shuffle=True)
    test_loader = DataLoader(Train(test), batch_size=8)

    ## CRITERION ##
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    model_configs = config_model()

    EPOCH = 10000
    modulo = 1
    for config in model_configs:
        for opt_fn, opt_params in config_optimisers():
            bnli = BERTNli(config=config)
            bnli.to(device)

            opt_params['params'] = bnli.parameters()
            optimiser = opt_fn(**opt_params)

            hparams = {
                'model_config': str(config),
                'optim': str(optimiser),
                'batch_size': loader.batch_size,
                'epochs': EPOCH
            }

            run: str = f"{str(datetime.now())}"
            writer = SummaryWriter(log_dir=f"runs/{run}")

            print(f"Training started [{run}]:\n{config}\nEpoch={EPOCH}\n{optimiser}")
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
                    torch.nn.utils.clip_grad_norm_(bnli.parameters(), 1.0)
                    optimiser.step()  # apply grads

                    running_loss += loss.item()
                    total_loss.append(loss.item())

                    if (i + 1) % modulo == 0:
                        # TODO: evaluation
                        with torch.no_grad():
                            bnli.eval()
                            wrong = 0.0
                            total = 0
                            for data in test_loader:
                                claim, evi, label = data
                                y_pred = bnli(claims=claim, evidence=evi)
                                y_pred = torch.sigmoid(y_pred)
                                print(y_pred, label)
                                wrong += torch.sum(torch.abs(
                                    torch.round(y_pred).detach().cpu().view(-1) - label.detach().cpu().view(-1)))
                                total += len(y_pred)
                                # print(f"Number of wrongs: {wrong} over {len(y_pred)}")
                            test_acc = 1.0 - float(wrong / total)
                            writer.add_scalar("Test Accuracy", test_acc, (epoch + 1) * i)
                            bnli.train()

                        params = list(bnli.parameters())
                        print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / modulo} test_acc: {test_acc}")
                        writer.add_scalar("Loss", running_loss / modulo, (epoch + 1) * i)
                        # writer.add_histogram("Linear0_weights", params[-4], (epoch + 1) * i)
                        # writer.add_histogram("Linear0_grad", params[-4].grad.view(-1), (epoch + 1) * i)
                        writer.add_histogram("Linear1_weights", params[-2], (epoch + 1) * i)
                        writer.add_histogram("Linear1_grad", params[-2].grad.view(-1), (epoch + 1) * i)
                        running_loss = 0

            metrics = {'avg_loss': np.mean(total_loss),
                       'max_loss': np.max(total_loss),
                       'min_loss': np.min(total_loss)}
            writer.add_hparams(hparams, metrics, run_name=run)
            print("Training complete.")
