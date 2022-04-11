import torch
from torch.utils.data import Dataset
import pandas as pd

class WikiDataset(Dataset):
    def __init__(self,
                ds_csv='train_balanced_10000_samples.csv',
                transform=None):
        
        self.ds_csv = pd.read_csv(ds_csv)
        self.transform = transform
            
    def __len__(self):
        return len(self.ds_csv)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # handle slices
        if isinstance(idx, slice):
            claims = self.ds_csv.iloc[idx]['claim']
            sents = self.ds_csv.iloc[idx]['sentence']
            targets = self.ds_csv.iloc[idx]['label']
            pointers = zip(claims, sents, targets)
            
            samples = []
            for (claim, sent, target) in pointers:
                sample = {'data': (claim, sent)}
                if self.transform:
                    sample = self.transform(sample)
                sample['target'] = self.encoded_label(target)
                samples.append(sample)
            
            if self.transform:
                return self.transform_collate_fn(samples)
            else:
                return self.collate_fn(samples)
            
        # handle single idx & DataLoader batching
        claim = self.ds_csv.iloc[idx]['claim']
        sent = self.ds_csv.iloc[idx]['sentence']
        target = self.ds_csv.iloc[idx]['label']
        
        target = self.encoded_label(target)
        sample = {'data': (claim, sent), 'target': target}
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @staticmethod
    def encoded_label(label:str):
        return {'SUPPORTS': 1, 'REFUTES': 0}.get(label)

    @staticmethod
    def transform_collate_fn(batch):
        collated = {
            'input_ids': [], 
            'segments': [],
            'targets': []
                   }
        for sample in batch:
            data = sample['data']
            collated['input_ids'].append(data.get('input_ids'))
            collated['segments'].append(data.get('segments'))
            collated['targets'].append(sample.get('target'))
            
        collated_tensors = WikiDataset.dict_values_to_tensor(collated)

        return collated_tensors
    
    @staticmethod
    def collate_fn(batch):
        collated = {
            'data': [],
            'targets': []
        }
        for sample in batch:
            collated['data'].append(sample['data'])
            collated['targets'].append(sample['target'])
        return collated
    
    @staticmethod
    def dict_values_to_tensor(dict_:dict):
        for k in dict_:
            dict_[k] = torch.tensor(dict_[k]).long()
        return dict_


if __name__ == "__main__":
    ds = WikiDataset()
    print(ds)