"""
Create TFRecords from wiki text files.

ONLY for wiki text files!
"""

import json
from typing import Tuple
import tensorflow as tf
import numpy as np

from postgres.db_admin import DatabaseAdmin
from postgres.db_query import DatabaseQuery as dbq
from fact_verification_system.classifier.pipeline.bert.preprocess import get_embeddings

def main():
    with open('../dataset/devset.json', 'r') as f:
        train_json = json.load(f)
        dba = DatabaseAdmin('postgres/config.yaml')
        dba.connect()
        with dba.session() as sess:
            for i, (_, d) in enumerate(train_json.items()):
                claim = d.get('claim')
                label = d.get('label')
                for ev in d.get('evidence'):
                    page_id = ev[0]
                    sent_idx = ev[1]
                    try:
                        sent = dbq.query_sentence(sess, page_id, sent_idx)
                        if sent is not None:
                            print(sent)
                        bert_sents = (_preprocess_string(claim), _preprocess_string(sent))
                        example = _get_embeddings_example(bert_sents, label)
                        print(example)
                    except Exception as e:
                        print("[CREATE_DS] {}".format(e))
                if i % 1000 == 0:
                    print("Just ran through {} examples.".format(i))


def _preprocess_string(string:str) -> str:
    preprocessed = _remove_brackets(string)
    return preprocessed

def _remove_brackets(string:str) -> str:
    """ Removes the brackets: specific to this dataset."""
    return string.replace("-LRB-", "").replace("-RRB-", "")



def _get_embeddings_example(bert_sents: Tuple[str], label:str) -> tf.train.Example:
    bert_dict = get_embeddings(bert_sents, max_seq_length=512)      # imported function
    # create Example in TFRecords
    feature = {
        'input_word_ids': _int64_feature(bert_dict.get('input_word_ids')),
        'input_mask': _int64_feature(bert_dict.get('input_mask')),
        'segment_ids': _int64_feature(bert_dict.get('segment_ids')),
        'target': tf.one_hot(_get_target_num(label))
    }
    return tf.train.Example(features=tf.train.Feature(feature=feature))

def _int64_feature(value) -> tf.train.Feature:
    """Converts str or bytes to type compatible with tf.train.Feature.
    
    TFRecords only store tf.Examples.
    tf.Examples are mappings of {'some_string':tf.train.Feature}
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if type(value) not in (bool, int):
        raise ValueError(
            "tf.train.Feature only accepts bool, int32, uint32, in64, uint64 or enum.")
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=[value]))

def _get_target_num(label:str):
    return {
        'REFUTES': 0,
        'SUPPORTS': 1,
        'NOT ENOUGH INFO': 2
    }.get(label)


if __name__ == "__main__":
    main()