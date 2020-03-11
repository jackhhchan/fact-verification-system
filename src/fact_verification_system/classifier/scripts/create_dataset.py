"""
Create TFRecords from wiki text files.

ONLY for wiki text files!
"""

import json
from typing import Tuple
import tensorflow as tf
import numpy as np
import pandas as pd
import requests as req
from tqdm import tqdm
import asyncio
import time


from postgres.db_admin import DatabaseAdmin
from postgres.db_query import DatabaseQuery as dbq
from fact_verification_system.classifier.pipeline.bert.preprocess import get_embeddings

max_seq_length = 64        # affects _get_embeddings()     NOTE: CHANGE THIS

import cProfile, pstats, io
def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

async def main(dba:DatabaseAdmin, path:str, train_json:list):
    with tf.io.TFRecordWriter(path) as writer:
        with dba.session() as sess:
            await asyncio.gather(
                *[write_to_tfrecord(writer, sess, d, i) for i, d in enumerate(train_json)]
                )


async def write_to_tfrecord(writer, sess, d, i):
    claim = d.get('claim')
    label = d.get('label')
    for ev in d.get('evidence'):
        page_id = ev[0]
        sent_idx = ev[1]
        try:
            # evidence 
            sent = await dbq.async_query_sentence(sess, page_id, sent_idx)            
            ### processed bert input embeddings -> tfrecords
            bert_sents = (_preprocess_string(claim), _preprocess_string(sent))
            example = await _get_embeddings_example(bert_sents, label, max_seq_length)
            if sent == None:
                continue

            # bert_sents = (claim, sent)
            # example = await _get_strings_example(bert_sents, label)
            writer.write(example.SerializeToString())
        except AttributeError:
            print("[CREATE_DS] {}:{} returned None. Skipped.".format(page_id, sent_idx))
        except ValueError:
            print("[CREATE_DS] {}:{} returned None. Skipped.".format(page_id, sent_idx))
        except Exception as e:
            print("[CREATE_DS] {}".format(e))
            raise e
        
        

#### Preprocessing ####

def _preprocess_string(string:str) -> str:
    preprocessed = _remove_brackets(string)
    # add more here
    return preprocessed

def _remove_brackets(string:str) -> str:
    """ Removes the brackets: specific to this dataset."""
    return string.replace("-LRB-", "").replace("-RRB-", "")

#### Dataset Labels ####

def _get_target_num(label:str):
    return {
        'REFUTES': 0,
        'SUPPORTS': 1,
        'NOT ENOUGH INFO': 2
    }.get(label)

#### Tensorflow tf.train.Example ####

async def _get_strings_example(bert_sents: Tuple[str], label:str) -> tf.train.Example:
    target = np.array([_get_target_num(label)], dtype=np.int32)
    print(_bytes_feature(bert_sents[0]))
    exit()
    feature = {
        'claim': _bytes_feature(bert_sents[0]),
        'evidence': _bytes_feature(bert_sents[1]),
        'target': _int64_feature(target)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

async def _get_embeddings_example(bert_sents: Tuple[str], label:str, max_seq_length=512) -> tf.train.Example:
    bert_dict = get_embeddings(bert_sents, max_seq_length=max_seq_length)      # imported function
    # create Example in TFRecords    
    target = np.array([_get_target_num(label)], dtype=np.int32)
    feature = {
        'input_word_ids': _int64_feature(bert_dict.get('input_word_ids')),
        'input_mask': _int64_feature(bert_dict.get('input_mask')),
        'segment_ids': _int64_feature(bert_dict.get('segment_ids')),
        'target': _int64_feature(target)
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def _int64_feature(value) -> tf.train.Feature:
    """Converts str or bytes to type compatible with tf.train.Feature.
    
    TFRecords only store tf.Examples.
    tf.Examples are mappings of {'some_string':tf.train.Feature}
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if value.dtype not in (np.int32, np.uint32, np.int64, np.uint64, np.bool):
        raise ValueError(
            "tf.train.Feature only accepts bool, int32, uint32, in64, uint64 or enum.")
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value) -> tf.train.Feature:
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    if type(value) not in (str, bytes):
        raise ValueError(
            "tf.train.Feature only accepts string or bytes.")
    if type(value) == str:
        value = value.encode('utf-8')

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



## Data Filtering
## NOTE: Data Analysis: Labels
# 1    80035
# 2    35639
# 0    29775
# Name: label, dtype: int64

def clean_data(train_json:dict) -> pd.DataFrame:
    """ Cleans dataset json mapper 
    
    - balancing dataset
    - 
    """
    df = pd.DataFrame(train_json)
    df = df[df.label != "NOT ENOUGH LABEL"]
    df = df.sort_values(by='label', ascending=True)
    
    print("\nRaw Dataset preview:\n{}".format(df.head()))
    print("Raw Dataset info:\n{}\n".format(df['label'].value_counts()))

    print("Cleaning dataset...")
    df_SUPPORTS = df[df.label == "SUPPORTS"]
    df_REFUTES = df[df.label == "REFUTES"]

    num_supports = len(df_REFUTES)        # balanced with refutes
    # num_supports = int(len(df_SUPPORTS)*0.6)  ## Keep 60% only.
    df_SUPPORTS = df_SUPPORTS.sample(n=num_supports, random_state=44)
    
    df_balanced = pd.concat([df_SUPPORTS, df_REFUTES])
    print("Cleaned dataset observations for each label:\n{}\n".format(
                                    df_balanced['label'].value_counts()
                                    ))

    return df_balanced


def _reduced_dataset(df:pd.DataFrame, sample_size:int):
    """ reduce dataset size but keep balanced 
    
    random_state set for reproducible results.
    """
    print("Reducing dataset...")
    df_supports = df[df['label'] == 'SUPPORTS']
    df_refutes = df[df['label'] == 'REFUTES']

    size = int(sample_size/2)

    random_state = 123
    df_supports = df_supports.sample(n=size, random_state=random_state)
    df_refutes = df_refutes.sample(n=size, random_state=random_state)

    df_reduced = pd.concat([df_supports, df_refutes])

    print("Reduced dataset observations for each label:\n{}\n".format(
                                    df_reduced['label'].value_counts()
                                    ))
    
    return df_reduced
        

if __name__ == "__main__":

    import multiprocessing as mp
    from multiprocessing import Process

    # load train.json
    json_fname = 'devset.json'
    assert json_fname.endswith('.json'), "training json must end with .json."
    with open('../dataset/{}'.format(json_fname), 'r') as f:
        train_json = json.load(f)
        train_json = list(train_json.values())

    print("{} training rows.".format(len(train_json)))
    
    ##### tfrecords naming
    prefix = json_fname[:-5]

    #### database
    dba = DatabaseAdmin('postgres/config.yaml')
    dba.connect()
    
    mp_ = False
    if mp_:
        # ASYNC + Multiprocessing -- NOT TESTED, DO NOT RUN.
        processes = mp.cpu_count()
        processes = 1
        print("Spawning %d processes..." %(processes))

        num_examples = 5000
        jobs = []
        for i in range(processes):
            path = prefix + '_512_balanced_' + str(i) + '.tfrecord'
            train_json_split = train_json[i*num_examples:(i+1)*num_examples]
            
            args=(dba, path, train_json_split)
            p = Process(target=main, args=args)
            jobs.append(p)
            p.start()
    else:
        # ASYNC ONLY
        # NOTE: CHANGE THIS
        # suffix = str(max_seq_length) + '_slightly_more_supports' + '.tfrecord'            ## unbalanced full dataset
        suffix = str(max_seq_length) + '_' + 'balanced' + '.tfrecord'                       ## balanced full dataset
        reduced_sample_size = 200
        suffix = str(max_seq_length) + '_' + 'balanced' + '_' + str(reduced_sample_size) + '_samples' + '.tfrecord'        # balanced reduced dataset
        # suffix = 'raw_string.tfrecord'                                                    ## raw string only full dataset
        path = '../dataset/tfrecords/' + prefix + '_' + suffix
        
        train_json_balanced_df = clean_data(train_json)
        train_json_balanced_df = _reduced_dataset(train_json_balanced_df, sample_size=reduced_sample_size)           # NOTE: CHANGE THIS IF FULL DATASET

        train_json_balanced = train_json_balanced_df.to_dict('records')
        
        args=(dba, path, train_json_balanced)
        
        # asyncio.run(main(*args))  # for python => 3.7
        print("[IMPORTANT]: {} max sequence length used.\n".format(max_seq_length))

        print("Data from {}".format(json_fname))
        print("Writing to path: {}".format(path))
        if not input("Confirm? (y/n): ") == 'y':
            exit("Terminated.")

        
        loop = asyncio.get_event_loop()
        s = time.time()
        loop.run_until_complete(main(*args))
        e = time.time()
        print("Elapsed {}s, {:.2f}s per example.".format((e-s), (e-s)/float(len(train_json_balanced))))