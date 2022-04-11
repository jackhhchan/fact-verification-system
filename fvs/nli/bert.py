import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import torch.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from typing import Tuple

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device('cuda:{}'.format(torch.cuda.current_device()))


class BertNliConfig(object):
    """ BERTNLi Config
    NSP: bool - leverages BERT's next sentence prediction training for sentence classification
    num_embedding_layers: int - number of BERT layers of embeddings to combine (i.e. sum)
    pool_method: str - pooling method to create sentence embedding (avg OR sum)
    """

    def __init__(self, nsp: bool, num_embedding_layers: int, pool_method: str):
        self.nsp = nsp
        self.num_embedding_layers = num_embedding_layers
        self.pool_method = pool_method


class BERTNli(nn.Module):
    def __init__(self, config: BertNliConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True))
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False
        self.bn1 = nn.BatchNorm1d(1536)
        self.fc1 = nn.Linear(1536, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(768, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

        self.bert_max_layer = 12  # bert-base

    def forward(self, claims, evidence):
        claims = tokenizer(claims, return_tensors="pt", padding=True)
        for k in claims.keys():
            claims[k] = claims.get(k).to(device)
        claims = self.bert(**claims)
        claims = claims.last_hidden_state[:, 0, :]
        claims.requires_grad = True
        evidence = tokenizer(evidence, return_tensors="pt", padding=True)
        for k in evidence.keys():
            evidence[k] = evidence.get(k).to(device)
        evidence = self.bert(**evidence)
        evidence = evidence.last_hidden_state[:, 0, :]
        evidence.requires_grad = True

        x = torch.cat((claims, evidence), axis=1)
        x = self.relu1(self.fc1(self.bn1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(self.bn2(x)))
        # x = self.dropout1(x)
        x = self.fc3(self.bn3(x))
        # x = self.sigmoid(x)       # replaced with BCELossWithLogits
        return x

    def bert_outputs(self, claims, evidence):
        """ Transforms text into BERT inputs. """
        if self.config.nsp:
            # leverage bert next sentence prediction training -- combine the two sentences separated by [SEP]
            combined = [c + ' [SEP] ' + e for c in claims for e in evidence]
            tokenised = self.tokenizer(combined, return_tensors='pt', padding=True)
            embeddings = self.pool_layer_embeddings(tokenised)
            embeddings = self.pool_token_embeddings(embeddings)
            return embeddings
        else:
            # get the sentence embedding for both and concat them
            claims = self.tokenizer(claims, return_tensors='pt', padding=True)
            evidence = self.tokenizer(evidence, return_tensors='pt', padding=True)
            claims = self.pool_layer_embeddings(claims)
            evidence = self.pool_layer_embeddings(evidence)
            claims = self.pool_token_embeddings(claims)
            evidence = self.pool_token_embeddings(evidence)
            concat = torch.cat((claims, evidence), axis=1)
            return concat

    def pool_layer_embeddings(self, tokenised):
        """ Extract the word embeddings from the BERT layers"""
        assert self.config.num_embedding_layers < self.bert_max_layer, "Current BERT (bert-base-uncased) only has 12 layers."
        outputs = self.bert(**tokenised)

        hidden_states: Tuple[torch.Tensor] = outputs.get('hidden_states')

        start_from = self.bert_max_layer - self.config.num_embedding_layers
        sum = torch.zeros(hidden_states[0].shape)
        for idx in range(start_from, self.bert_max_layer + 1):
            sum += hidden_states[idx]
        return sum

    def pool_token_embeddings(self, batch_token_embeddings: torch.Tensor):
        """ Pools the token embeddings into a sentence
        Average - this is the recommended pooling method (returned on default)
        Sum - application dependent
        """
        assert self.config.pool_method in ('avg', 'sum'), "Pooling method supported are 'avg' or 'sum'."

        num_tokens = batch_token_embeddings.shape[1]
        if self.config.pool_method == 'sum':
            return torch.sum(batch_token_embeddings, dim=1)
        else:
            return torch.sum(batch_token_embeddings, dim=1) / num_tokens


def config_train():
    return [(optim.SGD(lr=0.01)),
            (optim.SGD(lr=0.01, momentum=0.9)),
            (optim.SGD(lr=0.001)),
            (optim.SGD(lr=0.001, momentum=0.9)),
            (optim.Adam(lr=0.01)),
            (optim.Adam(lr=0.001))]


if __name__ == '__main__':
    sent_1 = "Hello there"
    sent_2 = "Bye now friend"
    # check how BertTokenizer encodes the sentences
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenised = tokenizer(sent_1)
    print(tokenised)
    print()
