"""
Contains BERT preprocess specific functions
"""
import tensorflow_hub as hub
from typing import Tuple, Dict

import tokenization         # from official google-research/bert github @6/2/2020

tokenizer = None
BERT_MAX_SEQ_LENGTH = 512       # largest sequence BERT can process

def get_embeddings(
    sents:Tuple[str],
    max_seq_length:int,
    bert_url="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    ) -> Dict[list]:
    """ Returns a dictionary of embeddings as per (paper & tf_hub)
            - input_word_ids
            - input_mask
            - segment_ids
    """
    if not tokenizer:
        _create_tokenizer(bert_url)

    num_sents = len(sents)
    if num_sents<= 0 or num_sents > 2:
        raise IndexError("Input must either be a single sentence or a sentence pair in a Tuple.")

    tokens = list()
    if num_sents == 1:
        tokens = preprocess_single(sents[0])
    elif num_sents == 2:
        tokens = preprocess_pair(sents[0], sents[1])

    if len(tokens) > BERT_MAX_SEQ_LENGTH:
        raise RuntimeError(
            "Largest sequence BERT can handle is {} \
                (incld. added tokens (see implementation details of this module.))"\
                .format(BERT_MAX_SEQ_LENGTH))

    return {
        'input_word_ids': _get_ids(tokens, tokenizer, max_seq_length),
        'input_mask': _get_masks(tokens, max_seq_length),
        'segment_ids': _get_segments(tokens, max_seq_length)
    }


def preprocess_pair(sent_1:str, sent_2:str):
    """ Tokenizes sentence pair
    
    Adds ['CLS'] to the beginning.
    Adds ['SEP'] to separate the two sentences and to the end.

    Returns:
    list of BERT tokenized tokens.
    """
    # tokenizes sentence pair
    stokens_1 = _tokenize(sent_1, tokenizer)
    stokens_2 = _tokenize(sent_2, tokenizer)
    return ["[CLS]"] + stokens_1 + ["[SEP]"] + stokens_2 + ["[SEP]"]


def preprocess_single(sent:str):
    """ Tokenizes sentence pair
    
    Adds ['CLS'] to start of sentence.
    Adds ['SEP'] to end of sentence.

    Returns:
    list of BERT tokenized tokens.
    """
    # tokenizes sentence pair
    stokens = _tokenize(sent, tokenizer)
    return ["[CLS]"] + stokens + ["[SEP]"]


### Helpers ###

def _create_tokenizer(bert_url):
    """Creates Tensorflow's BERT tokenizer GLOBAL STATE."""
     # lazily loads tokenizer
    bert_layer = hub.KerasLayer(bert_url, trainable=False)
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)



def _tokenize(sent:str, tokenizer):
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")
    return tokenizer.tokenize(sent)


def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")

    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    padding = [0] * (max_seq_length-len(token_ids))
    input_ids = token_ids + padding
    return input_ids

def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")

    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    padding = [0] * (max_seq_length-len(tokens))
    return [1]*len(tokens) + padding


def _get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if not tokenizer:
        raise RuntimeError("No tokenizer. Call _create_tokenizer().")

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