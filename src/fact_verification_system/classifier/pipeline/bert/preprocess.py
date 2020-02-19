"""
Contains BERT preprocess specific functions
"""
import tensorflow_hub as hub
from typing import Tuple, Dict, List
import numpy as np

# from official google-research/bert github @6/2/2020
from fact_verification_system.classifier.pipeline.bert import tokenization
     

tokenizer = None
BERT_MAX_SEQ_LENGTH = 512       # largest sequence BERT can process
INPUT_TYPE = np.int32

def get_embeddings(
    sents:Tuple[str],
    max_seq_length:int,     # combined seq length for sentence pairs
    bert_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
    pythonList=False        
    ) -> Dict[str, list]:
    """ Returns a dictionary of embeddings as per (paper & tf_hub)
            - input_word_ids
            - input_mask
            - segment_ids
        each of which are numpy arrays unless pythonList arg is set as True.
    """
    if not tokenizer:
        _create_tokenizer(bert_url)

    num_sents = len(sents)
    if num_sents<= 0 or num_sents > 2:
        raise IndexError("Input must either be a single sentence or a sentence pair in a Tuple.")

    tokens = list()
    if num_sents == 1:
        tokens = preprocess_single(sents[0], max_seq_length)
    elif num_sents == 2:
        tokens = preprocess_pair(sents[0], sents[1], max_seq_length)

    if len(tokens) > BERT_MAX_SEQ_LENGTH:
        raise AttributeError(
            "Largest sequence BERT can handle is {} \
                (incld. added tokens (see implementation details of this module.))"\
                .format(BERT_MAX_SEQ_LENGTH))

    input_word_ids = np.array(_get_ids(tokens, tokenizer, max_seq_length), dtype=np.int32)
    input_mask = np.array(_get_masks(tokens, max_seq_length), dtype=np.int32)
    segment_ids = np.array(_get_segments(tokens, max_seq_length), dtype=np.int32)

    if pythonList:
        input_word_ids = input_word_ids.tolist()
        input_mask = input_mask.tolist()
        segment_ids = segment_ids.tolist()

    assert len(input_word_ids) == max_seq_length, "input_word_ids must be equal max_seq_length."
    assert len(input_mask) == max_seq_length, "input_mask must be equal max_seq_length."
    assert len(segment_ids) == max_seq_length, "segment_ids must be equal max_seq_length."

    return {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
    }


def preprocess_pair(sent_1:str, sent_2:str, max_seq_length):
    """ Tokenizes sentence pair
    
    Adds ['CLS'] to the beginning.
    Adds ['SEP'] to separate the two sentences and to the end.

    Returns:
    list of BERT tokenized tokens.
    """
    # tokenizes sentence pair
    stokens_1 = _tokenize(sent_1, tokenizer)
    stokens_2 = _tokenize(sent_2, tokenizer)
    num_bert_identifiers = 3
    stokens_1, stokens_2 = _truncate_seq_pair(stokens_1, stokens_2, 
                                        max_seq_length-num_bert_identifiers)
    return ["[CLS]"] + stokens_1 + ["[SEP]"] + stokens_2 + ["[SEP]"]


def preprocess_single(sent:str, max_seq_length):
    """ Tokenizes sentence pair
    
    Adds ['CLS'] to start of sentence.
    Adds ['SEP'] to end of sentence.

    Returns:
    list of BERT tokenized tokens.
    """
    # tokenizes sentence pair
    stokens = _tokenize(sent, tokenizer)
    num_bert_identifiers = 2
    stokens = _truncate_seq_single(stokens, max_seq_length - num_bert_identifiers)
    return ["[CLS]"] + stokens + ["[SEP]"]


### Helpers ###

# from google-research/bert
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      return tokens_a, tokens_b
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

def _truncate_seq_single(tokens, max_length):
    while True:
        if len(tokens) <= max_length:
            return tokens
        else:
            tokens.pop()




def _create_tokenizer(bert_url):
    """Creates Tensorflow's BERT tokenizer GLOBAL STATE."""
     # lazily loads tokenizer
    bert_layer = hub.KerasLayer(bert_url, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    global tokenizer
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



def _tokenize(sent:str, tokenizer):
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")
    return tokenizer.tokenize(sent)


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab, (added padding)"""
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    padding = [0] * (max_seq_length-len(token_ids))
    input_ids = token_ids + padding
    return input_ids

def _get_masks(tokens, max_seq_length):
    """Mask for padding, (added padding)"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    padding = [0] * (max_seq_length-len(tokens))
    return [1]*len(tokens) + padding


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second, (added padding)"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1      # max. 2 segments.
    
    padding = [0] * (max_seq_length - len(tokens))
    return segments + padding


if __name__ == "__main__":
    string = '-LRB-hello-RRB'
    _create_tokenizer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")
    stokens = _tokenize(string, tokenizer)
    print(stokens)