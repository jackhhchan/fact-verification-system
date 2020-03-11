import torch
from torchvision import transforms
from transformers import BertTokenizer

# Transform class interface
class Transform(object):
    sample_dict_keys = ('data')
    def __init__(self):
        pass
    def __call__(self, sample: dict)->dict:
        raise NotImplementedError("__call__() must be implemented.")

class BertTokenize(Transform):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def __call__(self, sample:dict)->dict:
        (claim, evidence) = sample['data']
        claim_tokens = self.tokenizer.tokenize(claim)
        ev_tokens = self.tokenizer.tokenize(evidence)
        sample['data'] = (claim_tokens, ev_tokens)
        return sample
    
class Truncate(Transform):
    def __init__(self, max_seq_length):
        self.max_seq_length = max_seq_length
        
    def __call__(self, sample:dict)->dict:
        (claim_tokens, ev_tokens) = sample['data']
        special_tokens = 3
        (claim_trun, ev_trun) = self._truncate_seq_pair(claim_tokens, ev_tokens, 
                                                        (self.max_seq_length - special_tokens))

        if (len(claim_trun) + len(ev_trun)) > 61:
            print("failed")
            raise Exception

        sample['data'] = (claim_trun, ev_trun)

        return sample
            
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
      # This is a simple heuristic which will always truncate the longer sequence
      # one token at a time. This makes more sense than truncating an equal percent
      # of tokens from each, since if one sequence is very short then each token
      # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = (len(tokens_a) + len(tokens_b))
            if total_length <= max_length:
                if total_length > 61:
                    print(total_length)
                    print(max_length)
                    raise Exception
                if (len(tokens_a) + len(tokens_b)) > 61:
                    print("failed")
                    raise Exception
                return tokens_a, tokens_b
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

class BertSpecialTokens(Transform):
    def __call__(self, sample:dict)->dict:
        claim_trun, ev_trun = sample['data']
        if (len(claim_trun) + len(ev_trun)) > 61:
            print("failed")
            raise Exception
        bert_tokens = ["[CLS]"] + claim_trun + ["[SEP]"] + ev_trun + ["[SEP]"]
        if len(bert_tokens) > 64:
            print("failed.")
            raise Exception
        assert len(bert_tokens) <= 64, "truncated claim must be less than max seq length."
        sample['data'] = bert_tokens
        return sample
    
class BertInputEmbeds(Transform):
    def __init__(self, max_seq_length:int):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = max_seq_length
        
    def __call__(self, sample:dict)->dict:
        bert_tokens = sample['data']

        ids = self._get_ids(bert_tokens)
        assert len(ids) == self.max_seq_length, "Input_ids embedding must be of length {}.".format(self.max_seq_length)
        segments = self._get_segments(bert_tokens)
        assert len(segments) == self.max_seq_length, "segments embedding must be of length {}.".format(self.max_seq_length)
        bert_embeds = {
            'input_ids': ids,
            'segments': segments
        }
            
        sample['data'] =  bert_embeds
        return sample
    
    def _get_segments(self, bert_tokens:list)->list:
        first_SEP_idx = bert_tokens.index("[SEP]")
        segments = [0] * (first_SEP_idx+1)  # first [SEP] belongs to the first segment.
        segments += ([1] * (len(bert_tokens)-first_SEP_idx-1))
        segments = self._added_padding(segments)
        try:
            assert len(segments) == self.max_seq_length, "segment mask length should equal bert tokens length."
        except Exception as e:
            print(bert_tokens)
            print(segments)
            print(first_SEP_idx)
            raise e
            
        return segments
    
    def _get_ids(self, bert_tokens:list)->list:
        ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        ids = self._added_padding(ids)
        try:
            assert len(ids) == self.max_seq_length, "input ids must equal max seq length."
        except Exception as e:
            print(bert_tokens)
            print(ids)
            raise e
            
        return ids
    
    def _added_padding(self, tokens)->list:
        if len(tokens) < self.max_seq_length:
            padding = [0] * (self.max_seq_length - len(tokens))
            tokens += padding
        return tokens

    
class ToTensor(Transform):
    def __init__(self, dtype:torch.dtype):
        self.dtype = dtype
    
    def __call__(self, sample:dict)->dict:
        bert_embeds = sample['data']
        for key in bert_embeds.keys():
            bert_embeds[key] = torch.tensor(bert_embeds[key], dtype=torch.long)
        
        return sample
    

max_seq_length = 64
bert_transforms = transforms.Compose([
                        BertTokenize(), 
                        Truncate(max_seq_length), 
                        BertSpecialTokens(), 
                        BertInputEmbeds(max_seq_length), 
#                         ToTensor(torch.float32)
                        ])