#!/usr/bin/env python
# coding: utf-8

# # BERT Fine-tuned

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


# ### Extending BERT in Pytorch
# https://huggingface.co/transformers/_modules/transformers/modeling_bert.html#BertModel

class BERTNli(BertModel):
    """ Fine-Tuned BERT model for natural language inference."""
    
    def __init__(self, config):
        super().__init__(config)
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
    
    def forward(self,
               input_ids=None,
               token_type_ids=None):
        x = super(BERTNli, self).forward(input_ids=input_ids, token_type_ids=token_type_ids)
        x = x.last_hidden_state    # see huggingface's doc.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == '__main__':
    # Specify BERT config
    # Load pre-trained model weights.
    model = BERTNli(BertConfig(max_length=64)).from_pretrained('bert-base-uncased')  # BertConfig returns 12 layers on default.

    # Tensorboard init
    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    # timestamp = datetime.now().strftime("%d.%m.%Y-%H.%M.%S")
    # log_dir = 'runs/{}'.format(timestamp)


    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from fvs.pytorch_tutorial.fvs.transforms import bert_transforms
    from fvs.pytorch_tutorial.fvs.wiki_datasets import WikiDataset


    EPOCHS = 5
    BATCH_SIZE = 8
    RECORD_INTERVAL = 500 # steps

    transformed_dataset = WikiDataset(ds_csv="fvs/pytorch_tutorial/fvs/train_balanced_10000_samples.csv",
                                    transform=bert_transforms)


    trainLoader = DataLoader(transformed_dataset,
                            batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4,
                            collate_fn=transformed_dataset.transform_collate_fn)


    optimizers = {
        'train_sgd': optim.SGD(model.parameters(), lr=0.001),
        'train_adam': optim.Adam(model.parameters(), lr=0.001),
        'train_sgd_momentum': optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    }


    # CUDA
    device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    model.to(device)

    # Optimizer & Loss function
    optim_name = "train_sgd"
    optimizer = optimizers.get(optim_name)
    criterion = nn.BCELoss()

    # Tensorboard
    log_dir = 'src/pytorch_tutorial/fvs/runs/{}'.format(optim_name)
    writer = SummaryWriter(log_dir)

    running_loss = 0.0
    running_steps = 0
    interval_count = 0

    for epoch in tqdm(range(EPOCHS), desc='epoch'):
        for i, data in tqdm(enumerate(trainLoader, 1), desc='step'):
            input_ids = data['input_ids'].long().to(device)
            segments = data['segments'].long().to(device)
            targets = data['targets'].float().to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, token_type_ids=segments)
            targets = targets.view(-1, 1)   # match output shape

            loss = criterion(outputs, targets)
            loss.backward()  # calculate and store gradients in model
            optimizer.step()

            running_loss += loss.item()
            running_steps += 1.0

            if i % (RECORD_INTERVAL) == 0:
                writer.add_scalar('training_loss',
                                running_loss/running_steps,
                                interval_count)


                running_loss = 0.0
                running_steps = 0.0
                interval_count += 1

    print("Training complete.")