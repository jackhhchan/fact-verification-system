# # BERT Fine-tuned

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

from pytorch_tutorial.fvs.utils.tensorboard_utils import ModelParams, TensorboardHist


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
        (_, pooled) = x    # see huggingface's doc.
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

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

from transforms import bert_transforms
from wiki_datasets import WikiDataset


EPOCHS = 5
BATCH_SIZE = 8
RECORD_INTERVAL = 500 # steps

# train dataset
transformed_dataset = WikiDataset(ds_csv="src/pytorch_tutorial/fvs/train_balanced_10000_samples.csv",
                                transform=bert_transforms)
trainLoader = DataLoader(transformed_dataset,
                        batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=4, 
                        collate_fn=transformed_dataset.transform_collate_fn)
# validation dataset
transformed_dataset = WikiDataset(ds_csv="src/pytorch_tutorial/fvs/val_balanced_3000_samples.csv",
                                transform=bert_transforms)
valLoader = DataLoader(transformed_dataset,
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
model_params = ModelParams(model)
tb_hist = TensorboardHist(model_params, writer)
to_watch = [
    'fc1'
]
tb_hist.set_watch(to_watch)
tb_hist.register_outputs_hooks()        # capture layer outputs during forward pass 



running_loss = 0.0
running_steps = 0
interval_count = 0

for epoch in tqdm(range(EPOCHS), desc='epoch'):
    for i, data in tqdm(enumerate(trainLoader, 1), desc='train-step'):
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

            tb_hist.plot_weights_hist()
            tb_hist.plot_grads_hist()
            tb_hist.plot_bias_hist()
            tb_hist.plot_outputs_hist()

            # validation loss
            with torch.no_grad():
                running_val_loss = 0.0
                running_val_steps = 0.0
                for data in tqdm(valLoader, desc='val-step'):
                    input_ids = data['input_ids'].long().to(device)
                    segments = data['segments'].long().to(device)
                    targets = data['targets'].float().to(device)

                    outputs = model(input_ids=input_ids, token_type_ids=segments)
                    targets = targets.view(-1, 1)   # match output shape

                    val_loss = criterion(outputs, targets)
                    running_val_loss += val_loss.item()
                    running_val_steps += 1.0

                writer.add_sclar("val_loss",
                                running_val_loss/running_val_steps,
                                interval_count)
            

            running_loss = 0.0
            running_steps = 0.0
            interval_count += 1
            
print("Training complete.")